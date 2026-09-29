"""
RMU -- Representation Misdirection for Unlearning (Li et al. 2024, the WMDP
paper), used as one of the two base losses the MCU paper builds its method on
(paper Eq 3, and Eq 12 for the MCU variant).

Steers forget-set hidden states at a chosen layer toward a fixed random
control vector u, scaled by c, disrupting the model's ability to produce
forget-set-related activations there. The retain loss is NOT a plain
cross-entropy loss here -- following the paper's Appendix F, it penalizes the
current model's retain-set representations drifting from the ORIGINAL
(pre-unlearning) model's representations at the same layer, which is the
standard RMU retain regularizer.

use_mcu=True switches on Minor Component Unlearning (paper Eq 12): the
current representation is projected onto the minor-component subspace
(src/unlearn/mcu.py) before comparing it to c*u, so the loss can only push
the model in directions the paper shows are robust to relearning attacks.
"""

import copy
import torch
import bitsandbytes as bnb
from torch.utils.data import DataLoader
from src.data_utils import format_qa
from src.unlearn.mcu import extract_principal_components, project_minor, get_mlp_module


def unlearn_rmu(model, tokenizer, forget_ds, retain_ds, layer_idx=-1, c=6.0, lr=1e-5,
                 epochs=3, batch_size=4, retain_weight=10.0, device="cuda",
                 use_mcu=False, mcu_k=8, seed=42):
    model.to(device)
    ref_model = copy.deepcopy(model).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    mcu_stats = None
    if use_mcu:
        mcu_stats = extract_principal_components(
            model, tokenizer, forget_ds, layer_idx=layer_idx, K=mcu_k, device=device
        )

    mlp_module = get_mlp_module(model, layer_idx)
    ref_mlp_module = get_mlp_module(ref_model, layer_idx)

    captured = {}
    def hook(module, inp, out):
        captured["h"] = out
    def ref_hook(module, inp, out):
        captured["h_ref"] = out.detach()
    handle = mlp_module.register_forward_hook(hook)
    ref_handle = ref_mlp_module.register_forward_hook(ref_hook)

    model.train()
    # NOTE: no gradient_checkpointing_enable() here -- this method's loss is
    # built from a hook-captured raw activation (captured["h"]), a side
    # channel outside the model's returned/tracked output, and HF's
    # checkpointing reruns the hooked region untracked on the initial
    # forward, so that captured tensor risks losing its grad_fn. Memory
    # instead comes from 8-bit AdamW (~4x smaller optimizer state than
    # plain AdamW) plus keeping base_model off the GPU between iterations
    # (see the notebook) -- rmu already carries a second full model copy
    # (ref_model) so it needs that headroom back too.
    optim = bnb.optim.AdamW8bit(model.parameters(), lr=lr)

    forget_tok = forget_ds.map(lambda ex: format_qa(ex, tokenizer), batched=False)
    forget_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
    forget_loader = DataLoader(forget_tok, batch_size=batch_size, shuffle=True)

    retain_tok = retain_ds.map(lambda ex: format_qa(ex, tokenizer), batched=False)
    retain_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
    retain_loader = DataLoader(retain_tok, batch_size=batch_size, shuffle=True)

    torch.manual_seed(seed)
    u = None  # lazily sized on first batch, once we know the hidden dim

    try:
        for epoch in range(epochs):
            retain_iter = iter(retain_loader)
            total_forget, total_retain = 0.0, 0.0

            for f_batch in forget_loader:
                f_ids = f_batch["input_ids"].to(device)
                f_mask = f_batch["attention_mask"].to(device)

                model(input_ids=f_ids, attention_mask=f_mask)
                h = captured["h"]  # (batch, seq, d)
                if u is None:
                    u = torch.randn(h.shape[-1], device=device)
                    u = u / u.norm()

                h_loss_input = project_minor(h, mcu_stats["mean"], mcu_stats["components"]) if use_mcu else h
                # mean (not sum) over the hidden dim -- summing over GPT-2's
                # 768-wide hidden state let this overflow fp16 range (max
                # ~65504): observed forget=59488 on epoch 1, then nan on
                # epoch 2 once the overflow hit inf and produced nan grads.
                # .float() upcast BEFORE squaring: h/u can individually be
                # fp16 (raw model activations), and a fp16-vs-fp16 squared
                # difference can overflow per-element before mean() even
                # runs. Without use_mcu, this got lucky by accident (u's
                # dtype defaults to fp32, so `h - c*u` auto-promoted) -- but
                # that's not something to rely on, see retain_loss below
                # which has no such accidental rescue and DID overflow to
                # inf on every epoch even after the mean() fix.
                forget_loss = ((h_loss_input.float() - c * u) ** 2).mean(dim=-1).mean()

                try:
                    r_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    r_batch = next(retain_iter)
                r_ids = r_batch["input_ids"].to(device)
                r_mask = r_batch["attention_mask"].to(device)

                model(input_ids=r_ids, attention_mask=r_mask)
                h_r = captured["h"]
                with torch.no_grad():
                    ref_model(input_ids=r_ids, attention_mask=r_mask)
                    h_r_ref = captured["h_ref"]
                # Both h_r and h_r_ref are raw fp16 model activations with no
                # fp32 constant mixed in -- no accidental dtype promotion to
                # save this one, so the .float() upcast is load-bearing here
                # (this is the retain_loss that showed up as `inf` on every
                # epoch in the Kaggle run even after the mean-not-sum fix).
                retain_loss = ((h_r.float() - h_r_ref.float()) ** 2).mean(dim=-1).mean()

                loss = forget_loss + retain_weight * retain_loss
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()

                total_forget += forget_loss.item()
                total_retain += retain_loss.item()

            n = len(forget_loader)
            tag = "rmu_mcu" if use_mcu else "rmu"
            print(f"[{tag}] epoch {epoch+1}/{epochs} forget {total_forget/n:.4f} "
                  f"retain {total_retain/n:.4f}")
    finally:
        handle.remove()
        ref_handle.remove()

    return model
