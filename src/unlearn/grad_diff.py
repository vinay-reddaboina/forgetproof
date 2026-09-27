"""
Gradient Difference (GradDiff) unlearning.

Fixes the "damages everything" problem of plain gradient ascent by adding a
retain-set term: ascend on forget-set loss AND descend on retain-set loss at
the same time, so the model is pushed to forget the target facts while being
actively held in place on everything else.

loss = -loss(forget_batch) + lambda * loss(retain_batch)
"""

import torch
import bitsandbytes as bnb
from torch.utils.data import DataLoader
from src.data_utils import format_qa


def unlearn_grad_diff(model, tokenizer, forget_ds, retain_ds, lr=1e-5, epochs=3,
                       batch_size=4, retain_weight=1.0, device="cuda"):
    model.to(device)
    # See grad_ascent.py -- gradient checkpointing + 8-bit AdamW keep a full
    # fp16 fine-tune of a 1.3B model under a T4's ~14.56GB budget; grad_diff
    # runs two forward passes per step (forget + retain) so it needs the
    # activation savings even more than the plain-ascent baseline.
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    optim = bnb.optim.AdamW8bit(model.parameters(), lr=lr)

    forget_tok = forget_ds.map(lambda ex: format_qa(ex, tokenizer), batched=False)
    forget_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
    retain_tok = retain_ds.map(lambda ex: format_qa(ex, tokenizer), batched=False)
    retain_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])

    forget_loader = DataLoader(forget_tok, batch_size=batch_size, shuffle=True)
    # retain set is usually much bigger than forget set — cycle it
    retain_loader = DataLoader(retain_tok, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        retain_iter = iter(retain_loader)
        total_f, total_r = 0.0, 0.0

        for f_batch in forget_loader:
            try:
                r_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                r_batch = next(retain_iter)

            f_ids, f_mask = f_batch["input_ids"].to(device), f_batch["attention_mask"].to(device)
            r_ids, r_mask = r_batch["input_ids"].to(device), r_batch["attention_mask"].to(device)

            f_out = model(input_ids=f_ids, attention_mask=f_mask, labels=f_ids)
            r_out = model(input_ids=r_ids, attention_mask=r_mask, labels=r_ids)

            loss = -f_out.loss + retain_weight * r_out.loss

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            total_f += f_out.loss.item()
            total_r += r_out.loss.item()

        n = len(forget_loader)
        print(f"[grad_diff] epoch {epoch+1}/{epochs} forget-loss {total_f/n:.4f} "
              f"(want rising) retain-loss {total_r/n:.4f} (want low/flat)")

    model.config.use_cache = True
    return model
