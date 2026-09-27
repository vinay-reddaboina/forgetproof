"""
Membership inference attack (MIA).

If unlearning truly erased a fact, the model's loss on that forget-set
example should look statistically the same as its loss on data it never
saw at all (the TOFU 'real_authors' set works as that unseen baseline —
different content, same QA format). If forget-set loss is still
systematically lower than the unseen baseline, the model is still
distinguishing "I was trained on this" from "I wasn't" — i.e. it still
carries a memorization signal for supposedly-forgotten data.

Implementation: simple loss-threshold MIA (Yeom et al. style) — the
standard, cheap version used for exactly this kind of audit rather than a
full shadow-model attack, appropriate for a course project's compute budget.
"""

from src.eval.metrics import sequence_loss


def membership_inference_attack(model, tokenizer, forget_ds, unseen_ds, device="cuda"):
    """
    Returns:
        forget_losses, unseen_losses: raw per-example losses
        gap: unseen_mean_loss - forget_mean_loss
             (>0 means forget-set is still easier for the model = still
              "remembered"; ~0 means indistinguishable = looks forgotten)
        leak_score: gap normalized by unseen_mean_loss, clipped to [0, 1]
    """
    forget_losses = [
        sequence_loss(model, tokenizer, ex["question"], ex["answer"], device)
        for ex in forget_ds
    ]
    unseen_losses = [
        sequence_loss(model, tokenizer, ex["question"], ex["answer"], device)
        for ex in unseen_ds
    ]

    forget_mean = sum(forget_losses) / len(forget_losses)
    unseen_mean = sum(unseen_losses) / len(unseen_losses)

    gap = unseen_mean - forget_mean
    leak_score = max(0.0, min(1.0, gap / max(unseen_mean, 1e-6)))

    return {
        "forget_mean_loss": forget_mean,
        "unseen_mean_loss": unseen_mean,
        "gap": gap,
        "leak_score": leak_score,
        "forget_losses": forget_losses,
        "unseen_losses": unseen_losses,
    }
