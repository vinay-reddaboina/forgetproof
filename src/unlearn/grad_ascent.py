"""
Gradient Ascent unlearning: the simplest possible baseline.

Normal training minimizes loss on data (pulls the model toward it).
Unlearning by gradient ASCENT does the opposite on the forget set only —
it pushes the model's loss on those examples UP, i.e. tries to make it
actively bad at reproducing them. No retain-set term, so it tends to
also damage general ability if run too long (this is the point: it's the
weakest baseline, useful to show how easily audit attacks break it).
"""

import torch
import bitsandbytes as bnb
from torch.utils.data import DataLoader
from src.data_utils import format_qa


def unlearn_grad_ascent(model, tokenizer, forget_ds, lr=1e-5, epochs=3,
                         batch_size=4, device="cuda"):
    model.to(device)
    # Full fp16 fine-tuning of a 1.3B model on a T4 (14.56 GiB usable) is
    # tight: model + grads + plain AdamW's fp16 momentum/variance buffers
    # alone run ~10.4GB before a single activation is allocated. Gradient
    # checkpointing trades compute for activation memory, and bitsandbytes'
    # 8-bit AdamW cuts optimizer-state memory ~4x (5.2GB -> ~1.3GB) -- this
    # combination is what keeps every unlearning method under budget here.
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    optim = bnb.optim.AdamW8bit(model.parameters(), lr=lr)

    tokenized = forget_ds.map(lambda ex: format_qa(ex, tokenizer), batched=False)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    loader = DataLoader(tokenized, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids)
            loss = -out.loss  # ASCEND on forget-set loss: negate it before backward

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            total_loss += out.loss.item()

        print(f"[grad_ascent] epoch {epoch+1}/{epochs} mean forget-loss "
              f"{total_loss/len(loader):.4f} (want this to keep rising)")

    model.config.use_cache = True  # re-enable for downstream .generate() calls
    return model
