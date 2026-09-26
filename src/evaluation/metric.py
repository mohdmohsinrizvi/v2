from __future__ import annotations

from typing import Dict, Iterable, List

BETA = 0.5
B2 = BETA * BETA  # 0.25


def f05(precision: float, recall: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return (1 + B2) * precision * recall / (B2 * precision + recall)


def entity_f05(pred: Iterable[str], truth: Iterable[str]) -> float:
    p = set(pred)
    t = set(truth)
    if not t:
        return 1.0 if not p else 0.0
    if not p:
        return 0.0
    tp = len(p & t)
    precision = tp / len(p)
    recall = tp / len(t)
    return f05(precision, recall)


def macro_f05(preds: Dict[str, List[str]], truths: Dict[str, List[str]],
              entities: Iterable[str] | None = None) -> float:
    keys = list(entities) if entities is not None else sorted(set(preds) | set(truths))
    if not keys:
        return 0.0
    total = 0.0
    for k in keys:
        total += entity_f05(preds.get(k, []), truths.get(k, []))
    return total / len(keys)


def pair_metrics(preds: Dict[str, List[str]], truths: Dict[str, List[str]]) -> dict:
    tp = fp = fn = 0
    for k, t in truths.items():
        p = set(preds.get(k, []))
        ts = set(t)
        tp += len(p & ts)
        fp += len(p - ts)
        fn += len(ts - p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f05": f05(precision, recall)}


def truth_to_dict(gt_rows) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for row in gt_rows:
        sid = row[0] if not isinstance(row, dict) else row["source1_entity_id"]
        raw = row[1] if not isinstance(row, dict) else row["matched_entity_ids"]
        out[sid] = [x for x in (raw or "").split(",") if x]
    return out
