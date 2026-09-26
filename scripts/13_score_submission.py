#!/usr/bin/env python3
"""Stage 13 — score a submission with the OFFICIAL metric (train/dev only).

Computes entity-level Macro F0.5 exactly as ``student_resource`` does:
every Source-1 entity of the split is one unit, a correct empty prediction on a
singleton scores 1.0, and any prediction on a singleton scores 0.0.

Never run this against the test set — there is no ground truth for it.

Usage:
    .venv/bin/python scripts/13_score_submission.py --split dev \
        --matching submissions_dev/matching_results.tsv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.metric import macro_f05, pair_metrics
from src.pipeline import paths as P
from src.pipeline.gt import ground_truth_pairs, s1_index, s1_ids, target_ids


def read_matching(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        if header[:2] != ["source1_entity_id", "matched_entity_ids"]:
            sys.exit(f"unexpected header in {path}: {header}")
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            s1, _, rest = line.partition("\t")
            out[s1] = [x for x in rest.split(",") if x]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "dev"], required=True)
    ap.add_argument("--matching", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    matching = Path(args.matching)
    preds = read_matching(matching)

    ids = s1_ids(args.split)
    sid_to_eid = dict(zip(ids["sid"].to_list(), ids["entity_id"].to_list()))
    tids = target_ids(args.split)
    tid_to_eid = dict(zip(tids["tid"].to_list(), tids["entity_id"].to_list()))

    gt = ground_truth_pairs(args.split)
    truths: dict[str, list[str]] = {e: [] for e in ids["entity_id"].to_list()}
    for sid, tid in zip(gt["sid"].to_list(), gt["tid"].to_list()):
        truths[sid_to_eid[sid]].append(tid_to_eid[tid])

    entities = sorted(truths)
    preds_r = {k: preds.get(k, []) for k in entities}
    macro = macro_f05(preds_r, truths, entities)
    pair = pair_metrics(preds_r, truths)

    singletons = [k for k in entities if not truths[k]]
    single_correct = sum(1 for k in singletons if not preds_r[k])

    report = {
        "split": args.split,
        "matching": str(matching),
        "macro_f05": round(macro, 6),
        "pair_precision": round(pair["precision"], 6),
        "pair_recall": round(pair["recall"], 6),
        "pair_f05": round(pair["f05"], 6),
        "tp": pair["tp"], "fp": pair["fp"], "fn": pair["fn"],
        "s1_entities": len(entities),
        "singletons": len(singletons),
        "singletons_correctly_empty": single_correct,
        "s1_predicted_empty": sum(1 for k in entities if not preds_r[k]),
        "s1_predicted_nonempty": sum(1 for k in entities if preds_r[k]),
    }
    out = Path(args.out) if args.out else P.REPORTS / f"score_{args.split}.json"
    P.write_json(out, report)
    print(report)


if __name__ == "__main__":
    main()
