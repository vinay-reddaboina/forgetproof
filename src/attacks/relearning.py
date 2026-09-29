"""
RTT (Relearn Then Test) attack — the strongest, most standard audit for
unlearning: if a model can be brought back to knowing a fact after only a
LITTLE fine-tuning on a small slice of the forget set, it never really
forgot; the information was suppressed, not erased.

Protocol:
1. Split the forget set 80/20 (data_utils.held_out_split).
2. Fine-tune the (already unlearned) model on the 80% "relearn" portion —
   a handful of gradient steps, deliberately small, not a full retrain.
3. Test on the 20% held-out portion, which the relearning step never saw.
4. If accuracy/loss on the held-out portion recovers close to the original
   (pre-unlearning) target model's level, the attack succeeded — the model
   generalized back to knowledge it was supposed to have lost, meaning
   unlearning failed to erase the underlying representation.
"""

import copy
import torch
import torch.nn.functional as F
import bitsandbytes as bnb
from torch.utils.data import DataLoader
from src.data_utils import format_qa, held_out_split
from src.eval.metrics import sequence_loss


def relearning_attack(unlearned_model, tokenizer, forget_ds, device="cuda",
                       relearn_epochs=2, lr=2e-5, batch_size=4, holdout_frac=0.2,
                       return_model=False, seed=42):
    """
    Returns:
        dict with pre-attack and post-attack mean loss on the held-out slice,
        plus a recovery score in [0, 1] (higher = attack recovered more).

    return_model=True also returns the relearned (attacked) model and the
    exact holdout_ds split used, as ("model", "holdout_ds") keys -- needed by
    src/eval/pca_diagnostics.recovery_ratio_per_pc, which measures recovery
    on the SAME held-out examples this attack was scored on, at the
    representation level rather than just the output loss.
    """
    relearn_ds, holdout_ds = held_out_split(forget_ds, holdout_frac=holdout_frac, seed=seed)

    attack_model = copy.deepcopy(unlearned_model).to(device)
    attack_model.train()

    pre_losses = [
        sequence_loss(attack_model, tokenizer, ex["question"], ex["answer"], device)
        for ex in holdout_ds
    ]
    pre_loss = sum(pre_losses) / len(pre_losses)

    tokenized = relearn_ds.map(lambda ex: format_qa(ex, tokenizer), batched=False)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    loader = DataLoader(tokenized, batch_size=batch_size, shuffle=True)
    # This attack runs after every unlearning method, on top of whatever
    # else is still on the GPU (the just-unlearned model, possibly
    # unlearned_models kept for diagnostics) -- 8-bit AdamW keeps its own
    # deepcopy + optimizer state from tipping a T4 over budget too.
    optim = bnb.optim.AdamW8bit(attack_model.parameters(), lr=lr)

    for epoch in range(relearn_epochs):
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            # Computed manually with a float32 upcast + gradient clipping,
            # rather than trusting the model's native-fp16 out.loss with no
            # clipping -- this is the RELEARNING step's own training loss,
            # separate from (and upstream of) sequence_loss's eval-time fix.
            # grad_ascent's output model is already severely destabilized
            # (its own forget-loss climbs into the hundreds by design), and
            # fine-tuning that starting point with an unclipped fp16 loss is
            # exactly what was overflowing to nan/inf here, poisoning the
            # model's weights permanently -- no downstream eval-side upcast
            # can rescue a forward pass through nan weights. This was
            # observed as post_attack_holdout_loss: NaN for grad_ascent on
            # two separate real Kaggle runs, the second one AFTER the
            # sequence_loss fix alone, confirming that fix wasn't sufficient
            # on its own.
            out = attack_model(input_ids=ids, attention_mask=mask)
            logits = out.logits[:, :-1, :].float()
            shift_labels = ids[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), shift_labels.reshape(-1))
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(attack_model.parameters(), max_norm=1.0)
            optim.step()

    attack_model.eval()
    post_losses = [
        sequence_loss(attack_model, tokenizer, ex["question"], ex["answer"], device)
        for ex in holdout_ds
    ]
    post_loss = sum(post_losses) / len(post_losses)

    # recovery score: how much the held-out loss dropped, normalized by the
    # pre-attack loss (0 = no recovery, 1 = loss went to ~0 i.e. full recall)
    recovery = max(0.0, min(1.0, (pre_loss - post_loss) / max(pre_loss, 1e-6)))

    result = {
        "pre_attack_holdout_loss": pre_loss,
        "post_attack_holdout_loss": post_loss,
        "recovery_score": recovery,
        "n_holdout": len(holdout_ds),
        "n_relearn": len(relearn_ds),
    }
    if return_model:
        result["model"] = attack_model
        result["holdout_ds"] = holdout_ds
    return result
