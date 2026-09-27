# ForgetProof — Adversarial Verification of Machine Unlearning in LLMs

Team B2 (CSE-B, Vasavi): Rohit Kumar Gangala, Sriman Miryala, Vinay Reddaboina. Guide: Dr. E. Shailaja.

## What this is

LLMs sometimes need to "forget" data they were trained on (privacy or copyright
requests). Retraining from scratch is too expensive, so people use unlearning
shortcuts instead (gradient ascent on the forget set, preference-based methods,
etc). This project checks whether those shortcuts actually erase the
information or just suppress it — by attacking the "unlearned" model and
trying to pull the data back out. Then it proposes an unlearning method that
resists those attacks better.

Base paper: MCU (arXiv:2605.11685) — https://github.com/sustech-nlp/MCU
Benchmark: TOFU (fictitious-author QA pairs, has a clean forget/retain split)

## Pipeline

1. **Target model** — a model already fine-tuned to "know" the TOFU authors
   (`locuslab/tofu_ft_phi-1.5`, 1.3B params — small enough for a free Colab T4).
2. **Unlearn** — run a baseline method (Gradient Ascent, Gradient Difference,
   NPO) on the forget set. `src/unlearn/`
3. **Audit** — attack the unlearned model 4 ways and measure how much forgotten
   data leaks back. `src/attacks/`
   - Relearning attack (RTT): fine-tune on 80% of the forget set, test recovery
     on the held-out 20%. Fast recovery = knowledge was hidden, not erased.
   - Membership inference: compare loss/perplexity on forget-set examples vs
     genuinely unseen data.
   - Rephrasing extraction: ask the same forgotten fact with paraphrased
     prompts, check if the model still answers correctly.
   - Representation probing: train a linear probe on hidden states to see if
     the "forgotten" entity is still linearly decodable internally.
4. **Improve** — modify the unlearning objective (adds a representation-level
   regularizer) so the audit suite finds less to recover. `src/eval/`,
   `src/unlearn/improved.py`
5. **Dashboard** — before/after leak rate per method per attack. `dashboard/`

## Where things run

This repo is built and versioned here. Actual fine-tuning / unlearning runs
happen on Google Colab (free T4) — see `notebooks/`. Each notebook clones this
repo, installs `requirements.txt`, and calls into `src/`.

## Repo layout

```
src/
  data_utils.py       load TOFU forget/retain splits
  unlearn/             grad_ascent.py, grad_diff.py, npo.py, improved.py
  attacks/             relearning.py, membership_inference.py, rephrase.py, probe.py
  eval/                metrics.py (forget quality, model utility), run_eval.py
notebooks/             Colab notebooks, one per pipeline stage
dashboard/              React app reading results/*.json
results/                metric outputs (json/csv)
```

## Running it

1. Open `notebooks/01_setup_and_baseline.ipynb` in Colab (T4 GPU), run it top to
   bottom. It saves the target model to your Google Drive.
2. Open `notebooks/02_unlearn_and_audit.ipynb` in Colab, run it top to bottom.
   It unlearns with all 4 methods, runs the full audit suite on each, and
   writes `results/*.json`.
3. Pull the results back into this repo (the notebook's last cell pushes them
   via git — or copy `results/*.json` down manually), then:
   ```
   python dashboard/build_data.py
   ```
4. Open `dashboard/index.html` directly in a browser (no server needed).

## Status

- [x] Repo scaffolded
- [x] Baseline load + sanity check (notebook 01)
- [x] Unlearning methods implemented (grad_ascent, grad_diff, npo)
- [x] Audit attacks implemented (RTT, MIA, rephrase, probe)
- [x] Improved method (RC-NPO)
- [x] Dashboard (`dashboard/index.html`, reads `dashboard/data.js`)
- [ ] Run the real Colab pipeline and replace placeholder numbers in `dashboard/data.js`
- [ ] Write up results / report
