"""
Loads the TOFU benchmark splits.

TOFU (locuslab/TOFU on HuggingFace) is a set of QA pairs about 200 fictitious
authors. A fixed subset of authors is the "forget set" (data that should be
unlearned); everyone else is the "retain set" (data that should stay intact).
The dataset ships pre-split by forget percentage: forget01/05/10 (1%/5%/10%
of authors) each paired with the matching retain09/95/90 split.
"""

from datasets import load_dataset


TOFU_SPLIT_PAIRS = {
    "forget01": "retain99",
    "forget05": "retain95",
    "forget10": "retain90",
}


def load_tofu(forget_split: str = "forget10"):
    """Returns (forget_ds, retain_ds) for the given forget split name."""
    if forget_split not in TOFU_SPLIT_PAIRS:
        raise ValueError(f"forget_split must be one of {list(TOFU_SPLIT_PAIRS)}")
    retain_split = TOFU_SPLIT_PAIRS[forget_split]
    forget_ds = load_dataset("locuslab/TOFU", forget_split)["train"]
    retain_ds = load_dataset("locuslab/TOFU", retain_split)["train"]
    return forget_ds, retain_ds


def load_tofu_real_authors():
    """
    'real_authors' split: QA about real, non-fictitious authors. Used as the
    'genuinely unseen / not memorized' comparison set for membership inference
    and to check the model didn't just get generally worse (utility check).
    """
    return load_dataset("locuslab/TOFU", "real_authors")["train"]


def format_qa(example, tokenizer, max_length=256):
    """TOFU examples have 'question' and 'answer' fields. Format as a single
    causal-LM training string: question + answer, with the answer masked out
    of the loss target only where needed by the caller."""
    text = f"Question: {example['question']}\nAnswer: {example['answer']}"
    enc = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")
    return enc


def group_by_author_block(forget_ds, block_size=20, max_authors=10, questions_per_author=10):
    """
    TOFU's QA pairs have NO explicit author/id column, but each split is built
    from consecutive, unshuffled blocks of `block_size` questions per
    fictitious author (verified against the HF dataset: forget10's first 20
    rows are all about one author, rows 20-39 the next, etc). This groups the
    unshuffled forget set back into per-author question lists for the
    representation-probe attack.

    IMPORTANT: only call this on an unshuffled split (don't call after
    .shuffle()) or the block boundaries won't line up with real authors.
    """
    n_authors = min(max_authors, len(forget_ds) // block_size)
    groups = {}
    for i in range(n_authors):
        block = forget_ds.select(range(i * block_size, i * block_size + block_size))
        author_label = f"author_{i}"
        groups[author_label] = [ex["question"] for ex in block][:questions_per_author]
    return groups


def held_out_split(forget_ds, holdout_frac=0.2, seed=42):
    """Splits the forget set into an 80% 'relearn' portion (fine-tune on this)
    and a 20% 'holdout' portion (test recovery on this) — for the RTT
    relearning attack."""
    shuffled = forget_ds.shuffle(seed=seed)
    n = len(shuffled)
    n_holdout = max(1, int(n * holdout_frac))
    holdout = shuffled.select(range(n_holdout))
    relearn = shuffled.select(range(n_holdout, n))
    return relearn, holdout
