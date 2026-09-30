"""
Live presentation demo: shows a real forget-set fact being "forgotten" by an
unlearned model, then recovered by a brief relearning attack -- using the
project's actual trained checkpoints, run live (not pre-recorded numbers).

Meant to be run interactively (Kaggle notebook or a script) with the real
target_model and the real npo / npo_mcu checkpoints already in memory, so
the audience sees genuine model outputs, not canned text.
"""

import copy
import json
import os

import torch
import torch.nn.functional as F
import bitsandbytes as bnb
from torch.utils.data import DataLoader

from src.data_utils import format_qa, held_out_split
from src.eval.metrics import sequence_loss


def generate_answer(model, tokenizer, question, device, max_new_tokens=30):
    model.eval()
    prompt = f"Question: {question}\nAnswer:"
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,  # silences the pad==eos ambiguity warning
            max_new_tokens=max_new_tokens, do_sample=False,
            no_repeat_ngram_size=3,  # avoids degenerate "word word word" loops in greedy decoding
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def pick_best_example(target_model, tokenizer, forget_ds, device, candidate_indices):
    """Scans a few candidate questions and returns the index where the target
    model is most confident (lowest sequence_loss) -- cheap (forward pass
    only, no generation/training), so we don't waste the expensive attack
    step on a question the fresh target model happened to undertrain on."""
    target_model.to(device)
    scored = []
    for idx in candidate_indices:
        ex = forget_ds[idx]
        loss = sequence_loss(target_model, tokenizer, ex["question"], ex["answer"], device)
        scored.append((loss, idx))
        print(f"  candidate idx={idx} target confusion={loss:.2f}: {ex['question']}")
    scored.sort(key=lambda t: t[0])
    best_loss, best_idx = scored[0]
    print(f"  picked idx={best_idx} (confusion {best_loss:.2f})\n")
    return best_idx


def relearning_attack_live(unlearned_model, tokenizer, relearn_ds, device,
                            epochs=2, lr=2e-5, batch_size=8):
    attack_model = copy.deepcopy(unlearned_model).to(device)
    attack_model.train()
    optim = bnb.optim.AdamW8bit(attack_model.parameters(), lr=lr)

    tokenized = relearn_ds.map(lambda ex: format_qa(ex, tokenizer), batched=False)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    loader = DataLoader(tokenized, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = attack_model(input_ids=ids, attention_mask=mask)
            logits = out.logits[:, :-1, :].float()
            shift_labels = ids[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), shift_labels.reshape(-1))
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(attack_model.parameters(), max_norm=1.0)
            optim.step()

    attack_model.eval()
    return attack_model


def run_live_demo(target_model, unlearned_models, tokenizer, forget_ds,
                   device="cuda", example_idx=None, out_path="results/live_demo.json"):
    """
    target_model: the real, fully-finetuned "knows everything" model.
    unlearned_models: dict like {"npo": model, "npo_mcu": model} -- the
        real post-unlearning checkpoints (already on `device` or not, both fine).
    example_idx: which forget10 question to demo. If None, scans the first
        question of each of the first 10 authors (indices 0, 20, 40, ...) and
        picks the one the target model is most confident on.
    """
    print("=" * 70)
    print("FORGETPROOF LIVE DEMO -- real target model + real unlearned checkpoints")
    print("=" * 70)

    if example_idx is None:
        print("\nScanning a few candidate questions for one the target model knows well...")
        example_idx = pick_best_example(
            target_model, tokenizer, forget_ds, device,
            candidate_indices=[i * 20 for i in range(min(10, len(forget_ds) // 20))],
        )

    example = forget_ds[example_idx]
    question, answer = example["question"], example["answer"]
    print(f"\nQuestion: {question}")
    print(f"True answer: {answer}\n")

    results = {"question": question, "answer": answer}

    target_model.to(device)
    target_loss = sequence_loss(target_model, tokenizer, question, answer, device)
    target_gen = generate_answer(target_model, tokenizer, question, device)
    print("[Target model -- knows it]")
    print(f"  confusion score: {target_loss:.2f}")
    print(f'  says: "{target_gen}"\n')
    target_model.to("cpu")
    torch.cuda.empty_cache()
    results["target"] = {"loss": target_loss, "gen": target_gen}

    relearn_ds, holdout_ds = held_out_split(forget_ds, holdout_frac=0.2, seed=42)
    in_holdout = any(ex["question"] == question for ex in holdout_ds)
    print(f"(this question is {'in' if in_holdout else 'NOT in'} the held-out set "
          f"-- the attack below never trains on it directly)\n")

    for name, model in unlearned_models.items():
        model.to(device)
        pre_loss = sequence_loss(model, tokenizer, question, answer, device)
        pre_gen = generate_answer(model, tokenizer, question, device)
        print(f"[{name} -- after \"unlearning\"]")
        print(f"  confusion score: {pre_loss:.2f}")
        print(f'  says: "{pre_gen}"')

        print(f"  running a quick retraining attack on {len(relearn_ds)} OTHER questions...")
        attacked = relearning_attack_live(model, tokenizer, relearn_ds, device)
        post_loss = sequence_loss(attacked, tokenizer, question, answer, device)
        post_gen = generate_answer(attacked, tokenizer, question, device)
        recovery = max(0.0, min(1.0, (pre_loss - post_loss) / max(pre_loss, 1e-6)))
        print(f"  AFTER the attack: confusion score {post_loss:.2f} (recovered {recovery * 100:.0f}%)")
        print(f'  now says: "{post_gen}"\n')

        results[name] = {
            "pre_loss": pre_loss, "pre_gen": pre_gen,
            "post_loss": post_loss, "post_gen": post_gen,
            "recovery": recovery,
        }

        model.to("cpu")
        del attacked
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {out_path}")

    return results
