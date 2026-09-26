#!/usr/bin/env python3
"""Stage 06 — candidate recall measurement (the gate before matcher training).

Reports, against the official ground truth:

* pair candidate recall        (target >= 0.97)
* at-least-one S1 coverage
* all-match S1 coverage        (fraction of S1 whose EVERY true match is retrieved)
* per-retrieval-method recall
* tfidf-rank bucket recall

Run shard-by-shard: the train candidate set is ~214M pairs and cannot be
concatenated in 8 GiB. Shards are disjoint in ``sid``, so every count below is
a simple sum over shards and matches the eager formulation exactly.

Usage:
    .venv/bin/python scripts/06_candidate_recall.py --split train
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import paths as P
from src.pipeline.blocking import BOOL_METHODS
from src.pipeline.gt import candidate_shards, ground_truth_pairs

TARGET_PAIR_RECALL = 0.97
RANK_KS = (1, 5, 10, 20, 50)
CAND_COLS = ["sid", "tid"] + BOOL_METHODS + ["tfidf_rank"]


def row_count(path: Path) -> int:
    """Row count from parquet metadata only — never materialises the frame."""
    return int(pl.scan_parquet(path).select(pl.len()).collect().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "dev"], default="train")
    args = ap.parse_args()

    t0 = time.time()
    gt_pairs = ground_truth_pairs(args.split)
    n_pos = gt_pairs.height

    n_found = 0
    cand_total = 0
    methods = {m: 0 for m in BOOL_METHODS}
    rank_cov = {f"top{k}": 0 for k in RANK_KS}
    hit_parts: list[pl.DataFrame] = []

    files = candidate_shards(args.split)
    for i, f in enumerate(files):
        cand = pl.read_parquet(f, columns=CAND_COLS)
        cand_total += cand.height
        joined = cand.join(gt_pairs, on=["sid", "tid"], how="inner")
        del cand
        if joined.height:
            n_found += joined.unique().height
            for m in BOOL_METHODS:
                methods[m] += int(joined.filter(pl.col(m)).height)
            for k in RANK_KS:
                rank_cov[f"top{k}"] += int(
                    joined.filter(
                        (pl.col("tfidf_rank") > 0) & (pl.col("tfidf_rank") <= k)
                    ).height
                )
            hit_parts.append(
                joined.unique().group_by("sid").agg(pl.len().alias("n_hit"))
            )
        if (i + 1) % 5 == 0 or i + 1 == len(files):
            print(f"  shard {i + 1}/{len(files)}: found={n_found} "
                  f"rss={__import__('psutil').Process().memory_info().rss // 2**20}MB",
                  flush=True)
        del joined

    per_s1_truth = gt_pairs.group_by("sid").agg(pl.len().alias("n_true"))
    gt_s1 = per_s1_truth.height
    per_s1_found = pl.concat(hit_parts, how="vertical") if hit_parts else None
    if per_s1_found is None:
        per_s1 = per_s1_truth.with_columns(pl.lit(0).alias("n_hit"))
    else:
        per_s1 = per_s1_truth.join(per_s1_found, on="sid", how="left").with_columns(
            pl.col("n_hit").fill_null(0)
        )
    del per_s1_truth, per_s1_found, hit_parts

    s1_total = row_count(P.split_dir("processed", args.split) / "s1_ids.parquet")

    report = {
        "split": args.split,
        "s1_total": s1_total,
        "s1_with_truth": gt_s1,
        "positive_pairs": n_pos,
        "positive_pairs_retrieved": n_found,
        "pair_candidate_recall": round(n_found / max(n_pos, 1), 4),
        "at_least_one_s1_coverage": round(
            per_s1.filter(pl.col("n_hit") > 0).height / max(gt_s1, 1), 4
        ),
        "all_match_s1_coverage": round(
            per_s1.filter(pl.col("n_hit") == pl.col("n_true")).height / max(gt_s1, 1), 4
        ),
        "mean_truth_matches_per_s1": round(n_pos / max(gt_s1, 1), 3),
        "candidate_pairs": cand_total,
        "candidates_per_s1": round(cand_total / max(s1_total, 1), 2),
        "per_method_pair_recall": {m: round(v / max(n_pos, 1), 4)
                                   for m, v in methods.items()},
        "tfidf_rank_pair_recall": {k: round(v / max(n_pos, 1), 4)
                                   for k, v in rank_cov.items()},
        "gate_target": TARGET_PAIR_RECALL,
        "gate_pass": bool(n_found / max(n_pos, 1) >= TARGET_PAIR_RECALL),
        "elapsed_sec": round(time.time() - t0, 1),
    }

    P.ensure(P.REPORTS)
    out = P.REPORTS / f"candidate_recall_{args.split}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if args.split == "train" and not report["gate_pass"]:
        print("\n*** GATE FAILED: pair candidate recall below "
              f"{TARGET_PAIR_RECALL}. Analyse retrieval misses before training. ***")
        sys.exit(2)


if __name__ == "__main__":
    main()
