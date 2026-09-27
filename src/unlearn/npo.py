"""
NPO (Negative Preference Optimization) unlearning.

GradAscent/GradDiff push forget-loss up with no ceiling, which is exactly why
they overshoot and get audited badly. NPO treats each forget example as a
"dispreferred" response (like DPO's negative-only case) and bounds how hard it
pushes: once the model is already unlikely to say it, NPO backs off instead of
continuing to ascend forever. That boundedness is what's supposed to make it
harder to detect as damage.

Reference formulation (Zhang et al., NPO, and used as an unlearning baseline
in MCU/TOFU-style setups):
    loss = -(2/beta) * log( sigmoid( -beta * log(p_theta / p_ref) ) )
where p_theta / p_ref is the current-vs-reference model likelihood ratio on
the forget answer, i.e. exactly the DPO negative term with only the forget
(dispreferred) side, no positive side.
"""

import copy
import torch
import bitsandbytes as bnb
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data_utils import format_qa


def _answer_logprob(model, input_ids, attn_mask, prompt_len):
    """Mean log-prob assigned to the answer tokens (after prompt_len)."""
    out = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = out.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    logp = torch.log_softmax(logits, dim=-1)
    token_logp = torch.gather(logp, 2, labels.unsqueeze(-1)).squeeze(-1)
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[:, prompt_len - 1:] = True
    return (token_logp * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def unlearn_npo(model, tokenizer, forget_ds, retain_ds=None, lr=1e-5, epochs=3,
                 batch_size=4, beta=0.1, retain_weight=1.0, device="cuda"):
    model.to(device)
    # NPO keeps both the trained model and a frozen reference copy on the GPU
    # at once. Gradient checkpointing + 8-bit AdamW (see grad_ascent.py) claw
    # back enough headroom on a T4 for that second full-size copy to fit.
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    ref_model = copy.deepcopy(model).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    optim = bnb.optim.AdamW8bit(model.parameters(), lr=lr)

    forget_tok = forget_ds.map(lambda ex: format_qa(ex, tokenizer), batched=False)
    forget_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
    forget_loader = DataLoader(forget_tok, batch_size=batch_size, shuffle=True)

    retain_loader = None
    if retain_ds is not None:
        retain_tok = retain_ds.map(lambda ex: format_qa(ex, tokenizer), batched=False)
        retain_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
        retain_loader = DataLoader(retain_tok, batch_size=batch_size, shuffle=True)

    # rough prompt length in tokens for the "Question: ...\nAnswer:" prefix —
    # good enough since format_qa pads/truncates consistently
    prompt_len_estimate = 32

    for epoch in range(epochs):
        retain_iter = iter(retain_loader) if retain_loader else None
        total_npo, total_r = 0.0, 0.0

        for f_batch in forget_loader:
            f_ids = f_batch["input_ids"].to(device)
            f_mask = f_batch["attention_mask"].to(device)

            cur_logp = _answer_logprob(model, f_ids, f_mask, prompt_len_estimate)
            with torch.no_grad():
                ref_logp = _answer_logprob(ref_model, f_ids, f_mask, prompt_len_estimate)

            log_ratio = cur_logp - ref_logp
            npo_loss = -(2.0 / beta) * F.logsigmoid(-beta * log_ratio)
            npo_loss = npo_loss.mean()

            loss = npo_loss
            if retain_loader is not None:
                try:
                    r_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    r_batch = next(retain_iter)
                r_ids = r_batch["input_ids"].to(device)
                r_mask = r_batch["attention_mask"].to(device)
                r_out = model(input_ids=r_ids, attention_mask=r_mask, labels=r_ids)
                loss = loss + retain_weight * r_out.loss
                total_r += r_out.loss.item()

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            total_npo += npo_loss.item()

        n = len(forget_loader)
        msg = f"[npo] epoch {epoch+1}/{epochs} npo-loss {total_npo/n:.4f}"
        if retain_loader is not None:
            msg += f" retain-loss {total_r/n:.4f}"
        print(msg)

    model.config.use_cache = True
    return model
