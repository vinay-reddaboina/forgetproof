"""
Minor Component Unlearning (MCU) — Xiao et al., arXiv:2605.11685
"Robust LLM Unlearning Against Relearning Attacks: The Minor Components in
Representations Matter" (our base paper, github.com/sustech-nlp/MCU).

Core finding of the paper: unlearning methods that only change a model's
DOMINANT representation directions (the few PCA components that carry most
of the variance) are trivially undone by a relearning attack, because those
directions encode information shared across many samples and get quickly
re-established from just a handful of fine-tuning steps. Directions with
low variance ("minor components") encode sample-specific structure that
doesn't average out the same way across a batch, so changes concentrated
there resist relearning far better (paper Theorems 1-2, Section 3.2).

MCU operationalizes this: before computing an unlearning loss, project the
representation onto the subspace ORTHOGONAL to the top-K principal
components (+ the mean direction) of the forget-set's own representations,
so the unlearning update is confined to that robust subspace. This module
implements exactly that projection (paper Eq 9-11); src/unlearn/rmu.py and
src/unlearn/mlp_breaking.py wire it into the two losses the paper applies it
to (Eq 12, 14), and src/unlearn/npo_mcu.py is our own extension applying the
same idea to an output-level loss (NPO) the paper doesn't cover.
"""

import torch
from src.data_utils import format_qa


def get_mlp_module(model, layer_idx=-1):
    """Best-effort lookup of a transformer block's MLP submodule. Covers the
    Llama-style layout (model.model.layers[i].mlp) that Phi-1.5 also uses,
    and falls back to GPT-2-style (model.transformer.h[i].mlp)."""
    layers = None
    for path in (
        lambda m: m.model.layers,
        lambda m: m.transformer.h,
    ):
        try:
            layers = path(model)
            break
        except AttributeError:
            continue
    if layers is None:
        raise ValueError(
            "could not locate transformer layers on this model architecture -- "
            "add its attribute path to get_mlp_module()"
        )
    layer = layers[layer_idx]
    for attr in ("mlp", "mlp_block", "feed_forward"):
        if hasattr(layer, attr):
            return getattr(layer, attr)
    raise ValueError("could not locate an MLP submodule on this transformer layer")


@torch.no_grad()
def collect_representations(model, tokenizer, ds, layer_idx=-1, device="cuda",
                             max_tokens=8192, use_answer=True):
    """Runs `ds` through `model`, capturing the target MLP module's output at
    every token position via a forward hook. Returns a (N, d) tensor -- one
    row per token, pooled across the collected examples. Stops once
    `max_tokens` rows are collected (this can be slow on CPU/small GPUs, and
    the paper's own PCA fit doesn't need every token -- Appendix J shows the
    spectrum is stable even at 25% of the data)."""
    model.eval()
    mlp_module = get_mlp_module(model, layer_idx)

    captured = {}
    def hook(module, inp, out):
        captured["h"] = out.detach()
    handle = mlp_module.register_forward_hook(hook)

    reps = []
    total = 0
    try:
        for ex in ds:
            text = f"Question: {ex['question']}\nAnswer: {ex['answer']}" if use_answer else ex["question"]
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).input_ids.to(device)
            model(ids)
            h = captured["h"][0]  # (seq_len, d)
            reps.append(h.cpu())
            total += h.shape[0]
            if total >= max_tokens:
                break
    finally:
        handle.remove()

    return torch.cat(reps, dim=0).float()  # (N, d)


def extract_principal_components(model, tokenizer, forget_ds, layer_idx=-1, K=64,
                                  device="cuda", max_tokens=8192):
    """
    Paper Eq 9-10: center the forget-set's representations at the chosen
    layer/module, then take the top-K principal components via SVD.

    Returns dict(mean, components, singular_values, explained_variance_ratio):
      - mean: (d,) -- the "0th principal component" the paper also projects out
      - components: (K, d) -- unit vectors v_1..v_K, ordered by decreasing variance
      - singular_values / explained_variance_ratio: (min(N,d),) full spectrum,
        for the Observation-1-style variance plot in src/eval/pca_diagnostics.py
    """
    H = collect_representations(model, tokenizer, forget_ds, layer_idx, device, max_tokens)
    mean = H.mean(dim=0)
    H_centered = H - mean

    # randomized/full SVD (paper Eq 10); torch.linalg.svd is fine at this scale
    U, S, Vt = torch.linalg.svd(H_centered, full_matrices=False)
    explained_variance_ratio = (S ** 2) / (S ** 2).sum()

    K = min(K, Vt.shape[0])
    return {
        "mean": mean.to(device),
        "components": Vt[:K].to(device),
        "singular_values": S.cpu(),
        "explained_variance_ratio": explained_variance_ratio.cpu(),
        "layer_idx": layer_idx,
    }


def project_minor(h, mean, components):
    """
    P_perp(h) (paper Eq 11, mean handled as in Section 4.3): remove the mean
    direction, then remove the top-K principal directions from what's left.
    h: (..., d) — works for a single vector or a batch/sequence of them.
    components: (K, d), rows are the unit vectors to remove.

    Returns the DEVIATION with dominant directions removed (not a full
    reconstruction) -- this is what Eq 12/14 plug directly into their losses.
    """
    h_centered = h - mean
    coeffs = torch.matmul(h_centered, components.T)          # (..., K)
    dominant_part = torch.matmul(coeffs, components)          # (..., d)
    return h_centered - dominant_part


def project_minor_reconstruct(h, mean, components):
    """mean + P_perp(h) -- a full representation that keeps the forget-set's
    typical value but zeroes out variation along dominant directions. Used by
    npo_mcu.py's forward hook, where the projected tensor has to keep flowing
    through the rest of the network (unlike RMU/MLP-Breaking, where the
    projected value only ever feeds a loss, never a later layer)."""
    return mean + project_minor(h, mean, components)
