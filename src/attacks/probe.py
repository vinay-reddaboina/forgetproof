"""
Representation-level probing attack.

Output-level checks (loss, generation) only see what the model is willing to
SAY. A model can be trained to refuse/deflect on a topic while its internal
hidden states still cleanly encode the "forgotten" entity — the knowledge
never left the weights, only the output head learned to route around it.

This probe trains a small linear classifier on the model's hidden states
(mean-pooled over the question tokens, from a chosen layer) to predict which
forget-set entity a question is about. If the probe can still classify
entities well from an "unlearned" model's internals, the representations
are still there — the audit's clearest evidence that unlearning only
suppressed output, not the underlying knowledge.
"""

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


@torch.no_grad()
def extract_hidden_states(model, tokenizer, questions, layer=-1, device="cuda"):
    """Mean-pooled hidden state at `layer` for each question. Returns a
    (N, hidden_dim) tensor."""
    model.eval()
    feats = []
    for q in questions:
        ids = tokenizer(q, return_tensors="pt").input_ids.to(device)
        out = model(ids, output_hidden_states=True)
        h = out.hidden_states[layer][0]  # (seq_len, hidden_dim)
        feats.append(h.mean(dim=0).cpu())
    return torch.stack(feats)


def representation_probe_attack(model, tokenizer, forget_examples_by_entity,
                                 layer=-1, device="cuda", test_size=0.3, seed=42):
    """
    forget_examples_by_entity: dict {entity_name: [question, question, ...]}
    covering the forget-set entities you want to check are still linearly
    decodable.

    Returns:
        probe_accuracy: held-out accuracy of a linear probe trained to tell
        entities apart from hidden states (chance level ~= 1/num_entities;
        well above chance = the model still internally distinguishes
        "forgotten" entities from each other, i.e. still represents them)
    """
    questions, labels = [], []
    for entity, qs in forget_examples_by_entity.items():
        for q in qs:
            questions.append(q)
            labels.append(entity)

    feats = extract_hidden_states(model, tokenizer, questions, layer=layer, device=device)
    X = feats.numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    probe = LogisticRegression(max_iter=1000)
    probe.fit(X_train, y_train)
    preds = probe.predict(X_test)
    acc = accuracy_score(y_test, preds)

    n_classes = len(set(labels))
    chance = 1.0 / n_classes

    return {
        "probe_accuracy": acc,
        "chance_accuracy": chance,
        "above_chance": max(0.0, acc - chance),
        "n_entities": n_classes,
        "n_examples": len(questions),
        "layer": layer,
    }
