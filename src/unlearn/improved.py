"""
ForgetProof's proposed method: Representation-Constrained NPO (RC-NPO).

Why: the audit suite (src/attacks/) targets exactly the two ways baseline
unlearning fails —
  1. relearning attack succeeds because the weight change is shallow (small
     gradient nudge on the output distribution, easy to undo with a few
     fine-tuning steps)
  2. representation probe succeeds because hidden states still encode the
     forgotten entity even when the output head has learned to suppress it

RC-NPO = NPO's bounded forget objective (keeps model utility intact, unlike
plain gradient ascent) + a representation-alignment term that actively pushes
each forget-example's hidden state AWAY from its original representation and
toward the retain-set's average representation at the same layer. This is
meant to make the change harder to undo (it's now a representation-space
edit, not just an output-probability edit) and to specifically defeat the
probe attack (entities collapse toward a shared "average" region instead of
staying linearly separable).

loss = npo_loss(forget)                                  [bounded forgetting]
     + retain_weight * retain_loss                        [utility preservation]
     + repr_weight   * ||h_forget - h_retain_centroid||^2 [representation scrub]
"""

import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data_utils import format_qa
from src.unlearn.npo import _answer_logprob


@torch.no_grad()
def _retain_centroid(model, tokenizer, retain_ds, layer=-1, device="cuda", n_samples=100):
    """Mean hidden state over a sample of retain-set questions, at `layer` —
    the 'safe' region of representation space we push forgotten entities
    toward."""
    model.eval()
    sample = retain_ds.select(range(min(n_samples, len(retain_ds))))
    feats = []
    for ex in sample:
        ids = tokenizer(ex["question"], return_tensors="pt").input_ids.to(device)
        out = model(ids, output_hidden_states=True)
        feats.append(out.hidden_states[layer][0].mean(dim=0))
    return torch.stack(feats).mean(dim=0)


def unlearn_rc_npo(model, tokenizer, forget_ds, retain_ds, lr=1e-5, epochs=3,
                    batch_size=4, beta=0.1, retain_weight=1.0, repr_weight=0.5,
                    layer=-1, device="cuda"):
    model.to(device)
    ref_model = copy.deepcopy(model).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    centroid = _retain_centroid(model, tokenizer, retain_ds, layer=layer, device=device)

    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    forget_tok = forget_ds.map(lambda ex: format_qa(ex, tokenizer), batched=False)
    forget_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
    forget_loader = DataLoader(forget_tok, batch_size=batch_size, shuffle=True)

    retain_tok = retain_ds.map(lambda ex: format_qa(ex, tokenizer), batched=False)
    retain_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
    retain_loader = DataLoader(retain_tok, batch_size=batch_size, shuffle=True)

    prompt_len_estimate = 32

    for epoch in range(epochs):
        retain_iter = iter(retain_loader)
        total_npo, total_r, total_repr = 0.0, 0.0, 0.0

        for f_batch in forget_loader:
            f_ids = f_batch["input_ids"].to(device)
            f_mask = f_batch["attention_mask"].to(device)

            # --- NPO bounded forgetting term ---
            cur_logp = _answer_logprob(model, f_ids, f_mask, prompt_len_estimate)
            with torch.no_grad():
                ref_logp = _answer_logprob(ref_model, f_ids, f_mask, prompt_len_estimate)
            log_ratio = cur_logp - ref_logp
            npo_loss = -(2.0 / beta) * F.logsigmoid(-beta * log_ratio)
            npo_loss = npo_loss.mean()

            # --- representation-scrub term ---
            out = model(input_ids=f_ids, attention_mask=f_mask, output_hidden_states=True)
            h = out.hidden_states[layer].mean(dim=1)  # (batch, hidden_dim)
            repr_loss = F.mse_loss(h, centroid.unsqueeze(0).expand_as(h))

            # --- retain-set utility term ---
            try:
                r_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                r_batch = next(retain_iter)
            r_ids = r_batch["input_ids"].to(device)
            r_mask = r_batch["attention_mask"].to(device)
            r_out = model(input_ids=r_ids, attention_mask=r_mask, labels=r_ids)

            loss = npo_loss + retain_weight * r_out.loss + repr_weight * repr_loss

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            total_npo += npo_loss.item()
            total_r += r_out.loss.item()
            total_repr += repr_loss.item()

        n = len(forget_loader)
        print(f"[rc_npo] epoch {epoch+1}/{epochs} npo {total_npo/n:.4f} "
              f"retain {total_r/n:.4f} repr {total_repr/n:.4f}")

    return model
