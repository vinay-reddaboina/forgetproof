"""
Reproduces the base paper's (arXiv:2605.11685) three core empirical
observations (their Section 3.1, Figure 2) on OUR model/dataset (phi-1.5 +
TOFU) instead of theirs (Llama-3.1-8B + WMDP/Years). This is the evidence
that our project's whole premise -- targeting minor components should improve
robustness -- actually transfers to a different model and a different
dataset, not just something specific to their setup.

Observation 1 -- explained_variance_profile(): representations concentrate
    in a handful of dominant components (paper Fig 2a).
Observation 2 -- change_ratio_per_pc(): unlearning disproportionately moves
    the dominant components (paper Fig 2b, Eq 5).
Observation 3 -- recovery_ratio_per_pc(): a relearning attack disproportion
    -ately recovers exactly those dominant-component changes (paper Fig 2c,
    Eq 6).
"""

import torch
from src.unlearn.mcu import extract_principal_components, collect_representations


def explained_variance_profile(model, tokenizer, forget_ds, layer_idx=-1, device="cuda",
                                 max_pcs=128, max_tokens=8192):
    """Observation 1. Returns a list of explained-variance-ratio values, one
    per principal component, ordered by decreasing variance."""
    stats = extract_principal_components(
        model, tokenizer, forget_ds, layer_idx=layer_idx, K=max_pcs,
        device=device, max_tokens=max_tokens,
    )
    return stats["explained_variance_ratio"][:max_pcs].tolist()


def change_ratio_per_pc(model_before, model_after, tokenizer, forget_ds, pca_stats,
                         layer_idx=-1, device="cuda", max_tokens=4096):
    """
    Observation 2 / paper Eq 5: for each principal component vk, how much of
    the TOTAL unlearning-induced representation change is concentrated there.

    change_ratio_k = mean_x |<h_after(x) - h_before(x), vk>| / sum_j (same for j)

    Values sum to 1 across components -- a value much bigger than 1/K for an
    early (dominant) PC is exactly the paper's Observation 2.
    """
    h_before = collect_representations(model_before, tokenizer, forget_ds, layer_idx, device, max_tokens)
    h_after = collect_representations(model_after, tokenizer, forget_ds, layer_idx, device, max_tokens)

    n = min(h_before.shape[0], h_after.shape[0])
    delta = (h_after[:n] - h_before[:n])

    components = pca_stats["components"].cpu()  # (K, d)
    proj = (delta @ components.T).abs()          # (n, K)
    per_pc = proj.mean(dim=0)                     # (K,)
    total = per_pc.sum().clamp(min=1e-8)
    return (per_pc / total).tolist()


def recovery_ratio_per_pc(original_model, unlearned_model, relearned_model, tokenizer,
                           holdout_ds, pca_stats, layer_idx=-1, device="cuda", max_tokens=4096):
    """
    Observation 3 / paper Eq 6: for each principal component vk, what fraction
    of the unlearning-induced change on the RTT-relearning-attack's held-out
    set got reversed by the attack.

    recovery_ratio_k = <h_unlearned - h_relearned, vk> / <h_unlearned - h_original, vk>

    ~1 means fully reversed (the paper's finding for dominant components);
    ~0 means the change survived the attack (their finding for minor ones).
    Run this on the SAME held-out split used by src/attacks/relearning.py so
    it's measuring recovery on genuinely unattacked-on data, like the paper.
    """
    h_o = collect_representations(original_model, tokenizer, holdout_ds, layer_idx, device, max_tokens)
    h_u = collect_representations(unlearned_model, tokenizer, holdout_ds, layer_idx, device, max_tokens)
    h_r = collect_representations(relearned_model, tokenizer, holdout_ds, layer_idx, device, max_tokens)

    n = min(h_o.shape[0], h_u.shape[0], h_r.shape[0])
    h_o, h_u, h_r = h_o[:n], h_u[:n], h_r[:n]

    components = pca_stats["components"].cpu()  # (K, d)
    numerator = ((h_u - h_r) @ components.T).mean(dim=0)    # (K,)
    denominator = ((h_u - h_o) @ components.T).mean(dim=0)  # (K,)

    # guard against near-zero denominators (component barely moved during
    # unlearning) without flipping the sign of a small negative value
    safe_denom = torch.where(
        denominator.abs() < 1e-6,
        torch.full_like(denominator, 1e-6),
        denominator,
    )
    ratio = numerator / safe_denom
    return ratio.tolist()
