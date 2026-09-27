"""
Runs the full audit suite against one model and writes a results JSON that
the dashboard reads.

Usage (inside a Colab notebook, after loading `model`/`tokenizer` and
building `forget_ds`, `retain_ds`, `unseen_ds`, `paraphrased_examples`,
`entities_for_probe`):

    from src.eval.run_eval import run_full_audit
    results = run_full_audit(
        model, tokenizer, method_name="grad_diff",
        forget_ds=forget_ds, retain_ds=retain_ds, unseen_ds=unseen_ds,
        paraphrased_examples=paraphrased_examples,
        entities_for_probe=entities_for_probe,
    )
    save_results(results, "results/grad_diff.json")
"""

import json
import os

from src.attacks.relearning import relearning_attack
from src.attacks.membership_inference import membership_inference_attack
from src.attacks.rephrase import rephrase_extraction_attack
from src.attacks.probe import representation_probe_attack
from src.eval.metrics import sequence_loss


def model_utility(model, tokenizer, retain_ds, device="cuda", n_samples=50):
    """Sanity check that unlearning didn't just wreck the whole model.
    Mean loss on a sample of the retain set — should stay close to the
    pre-unlearning target model's retain-set loss."""
    sample = retain_ds.select(range(min(n_samples, len(retain_ds))))
    losses = [
        sequence_loss(model, tokenizer, ex["question"], ex["answer"], device)
        for ex in sample
    ]
    return sum(losses) / len(losses)


def run_full_audit(model, tokenizer, method_name, forget_ds, retain_ds, unseen_ds,
                    paraphrased_examples=None, entities_for_probe=None,
                    device="cuda"):
    results = {"method": method_name}

    print(f"[{method_name}] running relearning attack (RTT)...")
    results["relearning_attack"] = relearning_attack(model, tokenizer, forget_ds, device=device)

    print(f"[{method_name}] running membership inference...")
    results["membership_inference"] = membership_inference_attack(
        model, tokenizer, forget_ds, unseen_ds, device=device
    )

    if paraphrased_examples:
        print(f"[{method_name}] running rephrase extraction...")
        results["rephrase_extraction"] = rephrase_extraction_attack(
            model, tokenizer, paraphrased_examples, device=device
        )

    if entities_for_probe:
        print(f"[{method_name}] running representation probe...")
        results["representation_probe"] = representation_probe_attack(
            model, tokenizer, entities_for_probe, device=device
        )

    print(f"[{method_name}] computing model utility (retain-set loss)...")
    results["model_utility_loss"] = model_utility(model, tokenizer, retain_ds, device=device)

    return results


def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o))
    print(f"saved -> {path}")
