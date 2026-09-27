"""
NPO-MCU -- our extension, not in the base paper.

The MCU paper (arXiv:2605.11685) applies its minor-component projection
(src/unlearn/mcu.py) to two REPRESENTATION-level losses, RMU and MLP
Breaking (their Section 4, Eq 12 & 14) -- in both cases the projected
representation IS the loss target, so projecting it is a one-line change.

NPO is an OUTPUT-level loss: it scores a log-probability ratio at the
vocabulary head, several layers downstream of any single MLP module. The
paper's own theory (Appendix D.2, Lemma 1) shows NPO's effective gradient is
still mediated by a representation-space residual with the same dominant-
component structure -- but their Section 4 doesn't implement a projection for
this class of loss.

Our extension: force NPO's forward pass to only "see" the minor-component
part of the chosen layer's representation, by using a forward hook that
replaces that layer's output with a reconstruction where the dominant
directions are collapsed to their forget-set mean:

    h'(t) = mean + P_perp(h(t) - mean)

Everything downstream (later layers, the LM head, the NPO loss) computes
from h'(t) instead of h(t). Gradients still flow back through the projection
into earlier parameters, but the *signal* NPO's loss can act on is
constrained to directions the paper's Theorems 1-2 show resist relearning.
The hook is only active while computing the CURRENT model's forget-set
log-probability; the reference model (used as the fixed anchor in NPO's
ratio) and the retain-set pass are both left unhooked, matching how the
paper's NPO baseline computes those terms.
"""

import copy
import torch
import bitsandbytes as bnb
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data_utils import format_qa
from src.unlearn.mcu import extract_principal_components, project_minor_reconstruct, get_mlp_module
from src.unlearn.npo import _answer_logprob


class _MinorComponentHook:
    """Forward hook that reconstructs the wrapped module's output using only
    its minor-component subspace, but only while `active` is True -- so the
    same model can be used for both the (hooked) forget pass and the
    (unhooked) retain pass without swapping modules in and out."""

    def __init__(self, mean, components):
        self.mean = mean
        self.components = components
        self.active = False

    def __call__(self, module, inputs, output):
        if not self.active:
            return None  # leave output unchanged
        return project_minor_reconstruct(output, self.mean, self.components)


def unlearn_npo_mcu(model, tokenizer, forget_ds, retain_ds, layer_idx=-1, lr=1e-5,
                     epochs=3, batch_size=4, beta=0.1, retain_weight=1.0, mcu_k=8,
                     device="cuda"):
    model.to(device)
    ref_model = copy.deepcopy(model).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    mcu_stats = extract_principal_components(
        model, tokenizer, forget_ds, layer_idx=layer_idx, K=mcu_k, device=device
    )
    mlp_module = get_mlp_module(model, layer_idx)
    hook_obj = _MinorComponentHook(mcu_stats["mean"], mcu_stats["components"])
    handle = mlp_module.register_forward_hook(hook_obj)

    # NOTE: deliberately NOT using gradient_checkpointing_enable() here.
    # HF's checkpointing runs each decoder layer's forward once "for real"
    # (untracked) and only reconstructs it under grad during backward: fine
    # for out.loss/out.logits (the checkpoint's own tracked return value),
    # but risky to reason about for a hook this deep in the stack. Keeping
    # a second full fp16 model (ref_model) is what actually needs the
    # memory back, so that comes from 8-bit AdamW instead (~4x smaller
    # optimizer state than plain AdamW) plus keeping base_model off the GPU
    # between iterations (see the notebook).
    model.train()
    optim = bnb.optim.AdamW8bit(model.parameters(), lr=lr)

    forget_tok = forget_ds.map(lambda ex: format_qa(ex, tokenizer), batched=False)
    forget_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
    forget_loader = DataLoader(forget_tok, batch_size=batch_size, shuffle=True)

    retain_tok = retain_ds.map(lambda ex: format_qa(ex, tokenizer), batched=False)
    retain_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
    retain_loader = DataLoader(retain_tok, batch_size=batch_size, shuffle=True)

    prompt_len_estimate = 32

    try:
        for epoch in range(epochs):
            retain_iter = iter(retain_loader)
            total_npo, total_r = 0.0, 0.0

            for f_batch in forget_loader:
                f_ids = f_batch["input_ids"].to(device)
                f_mask = f_batch["attention_mask"].to(device)

                hook_obj.active = True
                cur_logp = _answer_logprob(model, f_ids, f_mask, prompt_len_estimate)
                hook_obj.active = False

                with torch.no_grad():
                    ref_logp = _answer_logprob(ref_model, f_ids, f_mask, prompt_len_estimate)

                log_ratio = cur_logp - ref_logp
                npo_loss = -(2.0 / beta) * F.logsigmoid(-beta * log_ratio)
                npo_loss = npo_loss.mean()

                try:
                    r_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    r_batch = next(retain_iter)
                r_ids = r_batch["input_ids"].to(device)
                r_mask = r_batch["attention_mask"].to(device)
                # hook_obj.active is already False here -- retain pass is unhooked,
                # standard cross-entropy, same as the plain NPO baseline
                r_out = model(input_ids=r_ids, attention_mask=r_mask, labels=r_ids)

                loss = npo_loss + retain_weight * r_out.loss

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()

                total_npo += npo_loss.item()
                total_r += r_out.loss.item()

            n = len(forget_loader)
            print(f"[npo_mcu] epoch {epoch+1}/{epochs} npo {total_npo/n:.4f} "
                  f"retain {total_r/n:.4f}")
    finally:
        handle.remove()

    return model
