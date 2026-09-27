# ForgetProof — Adversarial Verification of Machine Unlearning in LLMs

Team B2 (CSE-B, Vasavi): Rohit Kumar Gangala, Sriman Miryala, Vinay Reddaboina. Guide: Dr. E. Shailaja.

## What this is

LLMs sometimes need to "forget" data they were trained on (privacy or copyright
requests). Retraining from scratch is too expensive, so people use unlearning
shortcuts instead (gradient ascent on the forget set, preference-based methods,
etc). This project checks whether those shortcuts actually erase the
information or just suppress it — by attacking the "unlearned" model and
trying to pull the data back out.

**Base paper: MCU** — Xiao et al., *Robust LLM Unlearning Against Relearning
Attacks: The Minor Components in Representations Matter*, arXiv:2605.11685
(https://github.com/sustech-nlp/MCU). Their core finding: unlearning methods
mostly change a representation's *dominant* directions (the few PCA
components carrying most of the variance) — and those are exactly the
directions a relearning attack recovers fastest, because they encode
information shared across many samples. Directions with low variance ("minor
components") encode sample-specific structure that doesn't average out the
same way, so changes concentrated there survive relearning far better. Their
method, **MCU**, projects a representation onto the minor-component subspace
before computing the unlearning loss, so the unlearning update is confined to
directions that are hard to undo.

Benchmark: TOFU (fictitious-author QA pairs, has a clean forget/retain split)

## Pipeline

1. **Target model** — a model already fine-tuned to "know" the TOFU authors
   (`locuslab/tofu_ft_phi-1.5`, 1.3B params — small enough for a free Colab T4).
2. **Unlearn** — 8 configurations, in 4 baseline/+MCU pairs. `src/unlearn/`
   | Baseline | + MCU |
   |---|---|
   | `grad_ascent`, `grad_diff` | (weaker reference baselines, no MCU variant) |
   | `npo` | `npo_mcu` — **our extension**, see below |
   | `rmu` (Li et al.) | `rmu_mcu` — paper's method, Eq 12 |
   | `mlp_breaking` (Sondej & Yang) | `mlp_breaking_mcu` — paper's method, Eq 14 |
3. **Audit** — attack every resulting model 4 ways and measure how much
   forgotten data leaks back. `src/attacks/`
   - Relearning attack (RTT): fine-tune on 80% of the forget set, test recovery
     on the held-out 20%. Fast recovery = knowledge was hidden, not erased.
     Same protocol the base paper uses to evaluate MCU.
   - Membership inference: compare loss/perplexity on forget-set examples vs
     genuinely unseen data.
   - Rephrasing extraction: ask the same forgotten fact with paraphrased
     prompts, check if the model still answers correctly.
   - Representation probing: train a linear probe on hidden states to see if
     the "forgotten" entity is still linearly decodable internally.
4. **PCA diagnostics** — reproduce the paper's own Observations 1–3 (their
   Figure 2: explained variance, per-PC change ratio, per-PC recovery ratio)
   on our model/dataset, for the `npo` vs `npo_mcu` pair specifically — the
   comparison the paper itself never ran. `src/eval/pca_diagnostics.py`
5. **Dashboard** — before/after leak rate per method per attack. `dashboard/`

### Our extension: NPO-MCU

The base paper only wires its minor-component projection into two
*representation-level* losses (RMU, MLP Breaking) — cases where the projected
representation directly *is* the loss target. NPO is an *output-level* loss
(it scores a log-probability ratio at the vocabulary head, several layers
downstream of any one module), and the paper's Section 4 doesn't cover that
case, though its own theory (Appendix D.2) suggests the same dominant/minor
structure applies there too.

`src/unlearn/npo_mcu.py` extends MCU to NPO: a forward hook reconstructs a
chosen layer's output using only its minor-component subspace
(`mean + P_perp(h - mean)`) while computing the current model's forget-set
log-probability, so gradients can only push through directions the paper's
theorems say resist relearning. The reference model and the retain-set pass
stay unhooked, matching how the paper computes the plain NPO baseline.

## Where things run

This repo is built and versioned here. Actual fine-tuning / unlearning runs
happen on Google Colab (free T4) — see `notebooks/`. Each notebook clones this
repo, installs `requirements.txt`, and calls into `src/`.

## Repo layout

```
src/
  data_utils.py        load TOFU forget/retain splits, author grouping
  unlearn/
    grad_ascent.py      GA baseline
    grad_diff.py         GA + retain-preserving term
    npo.py                NPO baseline (paper's Eq 2)
    npo_mcu.py            our extension -- NPO + minor-component projection hook
    rmu.py                RMU baseline + RMU-MCU (paper Eq 3, 12)
    mlp_breaking.py       MLP Breaking baseline + MLPBreaking-MCU (paper Eq 4, 14)
    mcu.py                shared PCA extraction + P_perp projection (paper Eq 9-11)
  attacks/              relearning.py, membership_inference.py, rephrase.py, probe.py
  eval/
    metrics.py           forget quality, model utility
    run_eval.py           runs the full audit suite, saves results/*.json
    pca_diagnostics.py    reproduces the paper's Observations 1-3 on our setup
notebooks/              Colab notebooks, one per pipeline stage
dashboard/               React app reading results/*.json
results/                 metric + diagnostic outputs (json/png)
```

## Running it

1. Open `notebooks/01_setup_and_baseline.ipynb` in Colab (T4 GPU), run it top to
   bottom. It saves the target model to your Google Drive.
2. Open `notebooks/02_unlearn_and_audit.ipynb` in Colab, run it top to bottom.
   It unlearns with all 8 configurations, runs the full audit suite on each,
   runs the PCA diagnostics for npo vs npo_mcu, and writes `results/*.json` +
   `results/pca_diagnostics.png`.
3. Pull the results back into this repo (the notebook's last cell pushes them
   via git — or copy `results/*.json` down manually), then:
   ```
   python dashboard/build_data.py
   ```
4. Open `dashboard/index.html` directly in a browser (no server needed).

## Status

- [x] Repo scaffolded
- [x] Baseline load + sanity check (notebook 01)
- [x] Unlearning methods implemented: grad_ascent, grad_diff, npo, rmu, mlp_breaking
- [x] MCU implemented faithfully (paper Eq 9-14): rmu_mcu, mlp_breaking_mcu
- [x] Our extension: npo_mcu (MCU applied to an output-level loss)
- [x] Audit attacks implemented (RTT, MIA, rephrase, probe)
- [x] PCA diagnostics reproducing paper's Observations 1-3
- [x] Dashboard (`dashboard/index.html`, reads `dashboard/data.js`)
- [ ] Run the real Colab pipeline and replace placeholder numbers in `dashboard/data.js`
- [ ] Write up results / report
