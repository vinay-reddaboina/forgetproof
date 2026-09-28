"""
Builds our own "target model" checkpoint for an architecture that has no
published TOFU-finetuned checkpoint -- e.g. gpt2 (124M), used as the "very
small LLM" variant of this project instead of locuslab/tofu_ft_phi-1.5
(1.3B). Every other model in this pipeline started life pre-finetuned on
TOFU by someone else; gpt2 hasn't been, so we do that step ourselves, the
same way locuslab did to produce locuslab/tofu_ft_phi-1.5 in the first
place: standard causal-LM fine-tuning on TOFU's "full" split (every author,
forget + retain combined) so the model genuinely "knows" all of them before
any unlearning method ever touches it.

This is a ONE-TIME step per model (not an unlearning method) -- everything
downstream (src/unlearn/*, src/attacks/*) is unchanged and architecture-
agnostic already (src/unlearn/mcu.py's get_mlp_module() has a GPT-2-style
fallback, and every hidden-dim-dependent shape in this codebase is read
from the actual tensor at runtime, never hardcoded), so nothing else needs
to change to switch target models.
"""

import torch
import bitsandbytes as bnb
from torch.utils.data import DataLoader
from src.data_utils import format_qa


def finetune_target_model(model, tokenizer, full_ds, lr=2e-5, epochs=5,
                           batch_size=16, device="cuda"):
    model.to(device)
    model.train()
    optim = bnb.optim.AdamW8bit(model.parameters(), lr=lr)

    tokenized = full_ds.map(lambda ex: format_qa(ex, tokenizer), batched=False)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    loader = DataLoader(tokenized, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids)
            loss = out.loss

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            total_loss += loss.item()

        print(f"[finetune_target] epoch {epoch+1}/{epochs} mean loss "
              f"{total_loss/len(loader):.4f} (want this trending down -- "
              f"model is learning the TOFU authors)")

    model.eval()
    return model
