"""
Scale check: does the MCU-pair pattern (mostly flat/negative RTT recovery
seen at forget10) hold across different forget-set sizes, or is it an
artifact of one particular 20-author slice?

TOFU ships forget01/05/10 (1%/5%/10% of its 200 fictitious authors -- roughly
2/10/20 authors' worth of QA pairs). Every result reported so far only ever
used forget10. This reruns the baseline/MCU pairs on forget01 and forget05
too, at whatever (mcu_k, layer_idx) the sweep in src/sweep.py finds best (or
the original mcu_k=8/layer=-1 default, if called before the sweep finishes) --
same cheap scoring (RTT recovery_score + retain-set utility loss) as the
sweep, for the same reason: this is about the direction and consistency of
the effect across scale, not a full audit re-run.

If the MCU-vs-baseline gap flips sign or shrinks toward zero as the forget
set gets smaller/bigger, that's real evidence the earlier result was scale-
dependent, not MCU just "not working." If it stays the same sign and rough
magnitude across forget01/05/10, that points the other way -- MCU's flat
result on this model isn't an artifact of forget-set size.
"""
import copy
import json
import os

import torch

from src.attacks.relearning import relearning_attack
from src.data_utils import load_tofu
from src.eval.run_eval import model_utility
from src.unlearn.mlp_breaking import unlearn_mlp_breaking
from src.unlearn.npo_mcu import unlearn_npo_mcu
from src.unlearn.npo import unlearn_npo
from src.unlearn.rmu import unlearn_rmu

FORGET_SPLITS = ["forget01", "forget05", "forget10"]

# (baseline_fn, mcu_fn) pairs -- baseline never takes mcu_k/layer_idx, mcu
# variant does. mlp_breaking omitted deliberately: it was already the one
# pair with a (marginal) positive result at forget10, so it's lower priority
# than confirming/refuting the two pairs that got WORSE with MCU.
SCALE_CHECK_PAIRS = {
    "npo": {
        "baseline": lambda model, tok, forget_ds, retain_ds, device, layer_idx, mcu_k:
            unlearn_npo(model, tok, forget_ds, retain_ds, device=device),
        "mcu": lambda model, tok, forget_ds, retain_ds, device, layer_idx, mcu_k:
            unlearn_npo_mcu(model, tok, forget_ds, retain_ds, layer_idx=layer_idx, mcu_k=mcu_k, device=device),
    },
    "rmu": {
        "baseline": lambda model, tok, forget_ds, retain_ds, device, layer_idx, mcu_k:
            unlearn_rmu(model, tok, forget_ds, retain_ds, layer_idx=layer_idx, device=device, use_mcu=False),
        "mcu": lambda model, tok, forget_ds, retain_ds, device, layer_idx, mcu_k:
            unlearn_rmu(model, tok, forget_ds, retain_ds, layer_idx=layer_idx, mcu_k=mcu_k, device=device, use_mcu=True),
    },
}


def best_config_from_sweep(sweep_path="results/sweep.json", method="npo_mcu",
                            default_mcu_k=8, default_layer_idx=11):
    """Reads the sweep's own results (if it's finished and committed) and
    picks the (mcu_k, layer_idx) with the lowest recovery_score for `method`.
    Falls back to the original fixed point if the sweep hasn't landed yet."""
    if not os.path.exists(sweep_path):
        return default_mcu_k, default_layer_idx
    with open(sweep_path) as f:
        sweep = json.load(f)
    candidates = [v for v in sweep.values() if v.get("method") == method]
    if not candidates:
        return default_mcu_k, default_layer_idx
    best = min(candidates, key=lambda v: v["recovery_score"])
    return best["mcu_k"], best["layer_idx"]


def run_scale_check(base_model, tokenizer, device="cuda", out_path="results/scale_check.json",
                     ckpt_results_dir=None, forget_splits=FORGET_SPLITS,
                     configs_override=None):
    """configs_override lets a caller pin a fixed (mcu_k, layer_idx) instead of
    reading the sweep per pair, e.g. {"npo": (32, 2), "rmu": (32, 8)} -- by
    default each pair uses ITS OWN best config from the sweep (pair_name +
    "_mcu" as the sweep's method key), not one config shared across pairs."""
    per_pair_config = {}
    for pair_name in SCALE_CHECK_PAIRS:
        if configs_override and pair_name in configs_override:
            per_pair_config[pair_name] = configs_override[pair_name]
        else:
            per_pair_config[pair_name] = best_config_from_sweep(method=f"{pair_name}_mcu")
        k, l = per_pair_config[pair_name]
        print(f"scale check for '{pair_name}': using mcu_k={k}, layer_idx={l} (that pair's own sweep-best)")

    results = {}
    ckpt_path = os.path.join(ckpt_results_dir, "scale_check.json") if ckpt_results_dir else None
    if ckpt_path and os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            results.update(json.load(f))

    configs = [
        (split, pair_name, variant)
        for split in forget_splits
        for pair_name in SCALE_CHECK_PAIRS
        for variant in ("baseline", "mcu")
    ]

    for split, pair_name, variant in configs:
        key = f"{split}__{pair_name}__{variant}"
        if key in results:
            print(f"{key} -- already done, skipping")
            continue

        mcu_k, layer_idx = per_pair_config[pair_name]
        print(f"{key} -- loading {split} and training...")
        forget_ds, retain_ds = load_tofu(split)
        model_copy = copy.deepcopy(base_model)
        unlearn_fn = SCALE_CHECK_PAIRS[pair_name][variant]
        unlearned = unlearn_fn(model_copy, tokenizer, forget_ds, retain_ds, device, layer_idx, mcu_k)

        rtt = relearning_attack(unlearned, tokenizer, forget_ds, device=device)
        retain_loss = model_utility(unlearned, tokenizer, retain_ds, device=device)

        results[key] = {
            "forget_split": split, "pair": pair_name, "variant": variant,
            "mcu_k": mcu_k if variant == "mcu" else None,
            "layer_idx": layer_idx if variant == "mcu" else None,
            "n_forget_examples": len(forget_ds),
            "recovery_score": rtt["recovery_score"],
            "retain_loss": retain_loss,
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
        print(f"{key} -- recovery_score={rtt['recovery_score']:.3f} retain_loss={retain_loss:.3f}")

    print(f"scale check done -- {len(results)} configs -> {out_path}")
    return results
