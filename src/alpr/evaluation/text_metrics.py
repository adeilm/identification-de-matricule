"""Text recognition metrics: CER, character accuracy, exact full-plate accuracy."""

from __future__ import annotations


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (insertions, deletions, substitutions all cost 1)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def text_metrics(predictions: list[str], targets: list[str]) -> dict[str, float]:
    """``cer`` = total edits / total target characters; ``exact`` = share of plates read fully right."""
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets differ in length")
    if not targets:
        return {"n": 0, "cer": float("nan"), "char_accuracy": float("nan"), "exact": float("nan")}
    edits = sum(edit_distance(p, t) for p, t in zip(predictions, targets))
    chars = sum(len(t) for t in targets)
    cer = edits / max(1, chars)
    return {
        "n": len(targets),
        "cer": cer,
        "char_accuracy": max(0.0, 1.0 - cer),
        "exact": sum(p == t for p, t in zip(predictions, targets)) / len(targets),
    }
