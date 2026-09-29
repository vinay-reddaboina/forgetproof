"""
Hyperparameter sweep over mcu_k (how many dominant components MCU projects
away from) and layer_idx (which transformer block's MLP the projection is
applied at), for the 3 MCU-based methods -- npo_mcu, rmu_mcu,
mlp_breaking_mcu.

Every result reported so far (results/npo_mcu.json etc.) used a single fixed
point: mcu_k=8, layer_idx=-1 (the paper's own Appendix F used K in
{1,2,4,8,16,32,64}; we never varied K or the layer at all). This sweep exists
to answer the open question from the roadmap: is the flat/negative MCU
result a real scale limitation of GPT-2-124M, or just an under-tuned
setting?

Scored on RTT recovery_score (the paper's headline robustness metric, lower
= more robust) plus retain-set utility loss (catches a config that "wins" by
collapsing the model rather than by being robust) -- not the full 4-attack
audit suite, which would be 4x the GPU cost for questions this sweep isn't
answering. The winning config, once found, is worth a full audit pass on its
own.

Checkpoints to disk after every single config (mirrors the main 8-method
loop's per-method checkpointing) so a Kaggle disconnect only costs the one
run mid-flight, not the whole grid.
"""
import copy
import json
import os

import torch

from src.attacks.relearning import relearning_attack
from src.eval.run_eval import model_utility
from src.unlearn.mlp_breaking import unlearn_mlp_breaking
from src.unlearn.npo_mcu import unlearn_npo_mcu
from src.unlearn.rmu import unlearn_rmu

MCU_K_VALUES = [2, 4, 8, 16, 32]
LAYER_IDX_VALUES = [2, 5, 8, 11]  # early, early-mid, late-mid, last (of 12 blocks)

SWEEP_METHODS = {
    "npo_mcu": lambda model, tok, forget_ds, retain_ds, device, layer_idx, mcu_k: unlearn_npo_mcu(
        model, tok, forget_ds, retain_ds, layer_idx=layer_idx, mcu_k=mcu_k, device=device
    ),
    "rmu_mcu": lambda model, tok, forget_ds, retain_ds, device, layer_idx, mcu_k: unlearn_rmu(
        model, tok, forget_ds, retain_ds, layer_idx=layer_idx, mcu_k=mcu_k, device=device, use_mcu=True
    ),
    "mlp_breaking_mcu": lambda model, tok, forget_ds, retain_ds, device, layer_idx, mcu_k: unlearn_mlp_breaking(
        model, tok, forget_ds, retain_ds, layer_idx=layer_idx, mcu_k=mcu_k, device=device, use_mcu=True
    ),
}


def _config_key(method, mcu_k, layer_idx):
    return f"{method}__k{mcu_k}__l{layer_idx}"


def seed_from_existing_results(results_dir="results"):
    """The mcu_k=8, layer_idx=11 (==-1) point for each method was already
    run as part of the main 8-method loop -- reuse those instead of
    retraining, same normalization of -1 -> 11 the sweep grid uses."""
    seeded = {}
    for method in SWEEP_METHODS:
        path = os.path.join(results_dir, f"{method}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        key = _config_key(method, 8, 11)
        seeded[key] = {
            "method": method, "mcu_k": 8, "layer_idx": 11,
            "recovery_score": d["relearning_attack"]["recovery_score"],
            "retain_loss": d["model_utility_loss"],
            "source": "main_loop_reused",
        }
    return seeded


def run_sweep(base_model, tokenizer, forget_ds, retain_ds, device="cuda",
              out_path="results/sweep.json", ckpt_results_dir=None,
              mcu_k_values=MCU_K_VALUES, layer_idx_values=LAYER_IDX_VALUES):
    """Runs every (method, mcu_k, layer_idx) combo not already checkpointed,
    writing the growing results dict to out_path (and, if given,
    ckpt_results_dir -- a Drive/Kaggle-persistent copy) after each one."""
    results = seed_from_existing_results()

    ckpt_path = os.path.join(ckpt_results_dir, "sweep.json") if ckpt_results_dir else None
    if ckpt_path and os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            results.update(json.load(f))

    configs = [
        (method, k, layer_idx)
        for method in SWEEP_METHODS
        for k in mcu_k_values
        for layer_idx in layer_idx_values
    ]
    total = len(configs)

    for i, (method, mcu_k, layer_idx) in enumerate(configs, 1):
        key = _config_key(method, mcu_k, layer_idx)
        if key in results:
            print(f"[{i}/{total}] {key} -- already done, skipping")
            continue

        print(f"[{i}/{total}] {key} -- training...")
        model_copy = copy.deepcopy(base_model)
        unlearned = SWEEP_METHODS[method](
            model_copy, tokenizer, forget_ds, retain_ds, device, layer_idx, mcu_k
        )

        rtt = relearning_attack(unlearned, tokenizer, forget_ds, device=device)
        retain_loss = model_utility(unlearned, tokenizer, retain_ds, device=device)

        results[key] = {
            "method": method, "mcu_k": mcu_k, "layer_idx": layer_idx,
            "recovery_score": rtt["recovery_score"],
            "retain_loss": retain_loss,
            "pre_attack_holdout_loss": rtt["pre_attack_holdout_loss"],
            "post_attack_holdout_loss": rtt["post_attack_holdout_loss"],
        }

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=lambda o: float(o))
        if ckpt_path:
            os.makedirs(ckpt_results_dir, exist_ok=True)
            with open(ckpt_path, "w") as f:
                json.dump(results, f, indent=2, default=lambda o: float(o))

        del unlearned, model_copy
        torch.cuda.empty_cache()
        print(f"[{i}/{total}] {key} -- recovery_score={rtt['recovery_score']:.3f} retain_loss={retain_loss:.3f}")

    print(f"sweep done -- {len(results)} configs total -> {out_path}")
    return results
