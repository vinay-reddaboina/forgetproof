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
from torch.utils.data import DataLoader
from src.data_utils import format_qa, held_out_split
from src.eval.metrics import sequence_loss


def relearning_attack(unlearned_model, tokenizer, forget_ds, device="cuda",
                       relearn_epochs=2, lr=2e-5, batch_size=4, holdout_frac=0.2):
    """
    Returns:
        dict with pre-attack and post-attack mean loss on the held-out slice,
        plus a recovery score in [0, 1] (higher = attack recovered more).
    """
    relearn_ds, holdout_ds = held_out_split(forget_ds, holdout_frac=holdout_frac)

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
    optim = torch.optim.AdamW(attack_model.parameters(), lr=lr)

    for epoch in range(relearn_epochs):
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = attack_model(input_ids=ids, attention_mask=mask, labels=ids)
            optim.zero_grad()
            out.loss.backward()
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

    return {
        "pre_attack_holdout_loss": pre_loss,
        "post_attack_holdout_loss": post_loss,
        "recovery_score": recovery,
        "n_holdout": len(holdout_ds),
        "n_relearn": len(relearn_ds),
    }
