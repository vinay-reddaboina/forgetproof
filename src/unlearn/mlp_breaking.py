"""
MLP Breaking (Sondej and Yang, 2025), the second base loss the MCU paper
builds its method on (paper Eq 4, and Eq 14 for the MCU variant).

Motivated by factual knowledge being stored in MLP parameters (Nanda et al.,
2023): drives the current MLP output at a chosen layer to become orthogonal
to the ORIGINAL (pre-unlearning) model's MLP output on the same forget-set
input, via a ReLU-hinge on their cosine-similarity-like inner product. Unlike
RMU's fixed random target, the target here is sample-specific (the model's
own original output), so nothing needs to be sampled/seeded.

Retain loss follows the same representation-distance form as RMU's (paper
Appendix F): keep retain-set representations close to the original model's.

use_mcu=True switches on the MCU variant (Eq 14): both the current and
original representations are projected onto the minor-component subspace
before the loss compares them, so the orthogonality pressure -- and hence the
"broken" direction -- only ever operates in the subspace the paper shows
resists relearning attacks.
"""

import copy
import torch
from torch.utils.data import DataLoader
from src.data_utils import format_qa
from src.unlearn.mcu import extract_principal_components, project_minor, get_mlp_module


def unlearn_mlp_breaking(model, tokenizer, forget_ds, retain_ds, layer_idx=-1, lr=1e-5,
                          epochs=3, batch_size=4, retain_weight=1.0, device="cuda",
                          use_mcu=False, mcu_k=8):
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
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    forget_tok = forget_ds.map(lambda ex: format_qa(ex, tokenizer), batched=False)
    forget_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
    forget_loader = DataLoader(forget_tok, batch_size=batch_size, shuffle=True)

    retain_tok = retain_ds.map(lambda ex: format_qa(ex, tokenizer), batched=False)
    retain_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
    retain_loader = DataLoader(retain_tok, batch_size=batch_size, shuffle=True)

    try:
        for epoch in range(epochs):
            retain_iter = iter(retain_loader)
            total_forget, total_retain = 0.0, 0.0

            for f_batch in forget_loader:
                f_ids = f_batch["input_ids"].to(device)
                f_mask = f_batch["attention_mask"].to(device)

                model(input_ids=f_ids, attention_mask=f_mask)
                h = captured["h"]
                with torch.no_grad():
                    ref_model(input_ids=f_ids, attention_mask=f_mask)
                    h_o = captured["h_ref"]

                if use_mcu:
                    h_cmp = project_minor(h, mcu_stats["mean"], mcu_stats["components"])
                    h_o_cmp = project_minor(h_o, mcu_stats["mean"], mcu_stats["components"])
                else:
                    h_cmp, h_o_cmp = h, h_o

                inner = (h_cmp * h_o_cmp).sum(dim=-1)
                norm_sq = (h_o_cmp ** 2).sum(dim=-1).clamp(min=1e-6)
                forget_loss = torch.relu(inner / norm_sq).mean()

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
                retain_loss = ((h_r - h_r_ref) ** 2).sum(dim=-1).mean()

                loss = forget_loss + retain_weight * retain_loss
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()

                total_forget += forget_loss.item()
                total_retain += retain_loss.item()

            n = len(forget_loader)
            tag = "mlp_breaking_mcu" if use_mcu else "mlp_breaking"
            print(f"[{tag}] epoch {epoch+1}/{epochs} forget {total_forget/n:.4f} "
                  f"retain {total_retain/n:.4f}")
    finally:
        handle.remove()
        ref_handle.remove()

    return model
