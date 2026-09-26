#!/usr/bin/env python3
"""Stage 11 — threshold optimisation for entity-level Macro F0.5.

The official metric is Macro F0.5 = (1.25·P·R)/(0.25·P+R), averaged over ALL
Source-1 entities of the validation fold, so:

* a singleton scored correctly (predicted empty, truth empty) earns 1.0,
* any prediction at all for a singleton costs 0.0,
* an unmatched prediction for a matched entity drags precision down.

A fixed 0.5 threshold is therefore almost never optimal. This script sweeps a
threshold grid on the validation fold only and records the full curve.

Usage:
    .venv/bin/python scripts/11_threshold_search.py --split dev
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.metric import f05
from src.pipeline import paths as P
from src.pipeline import split as SPLIT
from src.pipeline.gt import ground_truth_pairs, s1_index, s1_ids, target_ids

CFG = {"thresh_gt": 1.0}


def grid() -> np.ndarray:
    g = np.concatenate(
        [
            np.arange(0.10, 0.50, 0.05),
            np.arange(0.50, 0.90, 0.02),
            np.arange(0.90, 1.0001, 0.01),
            np.array([0.995, 0.999, 0.9999]),
        ]
    )
    return np.unique(np.round(g, 4))


def load_scores(split: str) -> pl.DataFrame:
    files = P.shard_files(P.split_dir("predictions", split))
    if not files:
        sys.exit("no scores; run scripts/10 first")
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "dev"], required=True)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    s1 = s1_index(args.split)
    flags = SPLIT.assign(s1["entity_id"], args.val_frac).to_numpy()
    val_sids = np.flatnonzero(flags)

    scores = load_scores(args.split)
    scores = scores.filter(pl.col("sid").is_in(val_sids.tolist()))

    gt = ground_truth_pairs(args.split).filter(pl.col("sid").is_in(val_sids.tolist()))
    n_gt, n_scored = gt.height, gt.join(scores, on=["sid", "tid"], how="inner").height
    coverage = n_scored / n_gt if n_gt else 1.0
    print(f"[11] val-fold gt in scored candidates: {n_scored}/{n_gt} = {coverage:.4f}",
          flush=True)
    if n_gt and coverage < 0.97:
        sys.exit(
            f"val-fold candidate coverage {coverage:.4f} < 0.97: the scored "
            f"candidates do not cover this validation fold, so any threshold "
            f"picked here is meaningless. Most often scripts/08 --val-only was "
            f"run with a different --val-frac than {args.val_frac}."
        )
    truth_by_sid: dict[int, set[int]] = {}
    for sid, tid in zip(gt["sid"].to_list(), gt["tid"].to_list()):
        truth_by_sid.setdefault(sid, set()).add(tid)

    df = scores.join(gt.with_columns(pl.lit(True).alias("__t")), on=["sid", "tid"], how="left")
    df = df.with_columns(pl.col("__t").fill_null(False))
    df = df.sort("sid", "score", descending=[False, True])

    # sid is Int32 while every key below is an Int64 scalar/index, so numpy
    # re-casts the whole 73.6M-element array on EACH searchsorted call: 183 ms
    # measured on the production instance x 438886 calls = 22 hours. Cast once.
    sid_a = df["sid"].to_numpy().astype(np.int64)
    score_a = df["score"].to_numpy().astype(np.float64)
    hit_a = df["__t"].to_numpy()
    starts = np.searchsorted(sid_a, np.arange(len(val_sids) + 1), side="left")
    # starts[i] corresponds to val_sids[i] only if sids are contiguous 0..n-1
    offsets = {s: (int(np.searchsorted(sid_a, s, "left")),
                   int(np.searchsorted(sid_a, s, "right"))) for s in val_sids}

    n_s1 = len(val_sids)
    thr_grid = grid()
    macro = np.zeros(len(thr_grid))
    avg_pred = np.zeros(len(thr_grid))
    n_pred_nonzero = np.zeros(len(thr_grid))

    scores_by_sid = {}
    for s in val_sids:
        lo, hi = offsets[s]
        scores_by_sid[s] = (score_a[lo:hi], hit_a[lo:hi])

    for j, thr in enumerate(thr_grid):
        total = 0.0
        npred = 0
        for s in val_sids:
            ss, hh = scores_by_sid[s]
            t = truth_by_sid.get(s)
            # ss is descending: count of ss >= thr via negated ascending lookup
            keep = int(np.searchsorted(-ss, -thr, side="right"))
            if not t:
                total += 1.0 if keep == 0 else 0.0
                npred += keep
                continue
            if keep == 0:
                npred += 0
                continue
            tp = int(hh[:keep].sum())
            p = tp / keep
            r = tp / len(t)
            total += f05(p, r)
            npred += keep
        macro[j] = total / n_s1
        avg_pred[j] = npred / n_s1
        n_pred_nonzero[j] = npred > 0

    best = int(np.argmax(macro))
    # tie-break toward the HIGHER threshold (F0.5 rewards precision; fewer false
    # positives is the safer operating point among equal Macro scores)
    tied = np.flatnonzero(np.isclose(macro, macro[best], atol=1e-9))
    best = int(tied[-1])

    report = {
        "split": args.split,
        "val_frac": args.val_frac,
        "val_s1": int(n_s1),
        "val_s1_with_truth": int(len(truth_by_sid)),
        "val_s1_singletons": int(n_s1 - len(truth_by_sid)),
        "scored_pairs": int(df.height),
        "best_threshold": float(thr_grid[best]),
        "best_macro_f05": float(macro[best]),
        "avg_predictions_per_s1_at_best": float(avg_pred[best]),
        "curve": [
            {"threshold": float(t), "macro_f05": float(m),
             "avg_pred_per_s1": float(a)}
            for t, m, a in zip(thr_grid, macro, avg_pred)
        ],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out = Path(args.out) if args.out else P.REPORTS / f"threshold_{args.split}.json"
    P.write_json(out, report)

    print(f"best threshold = {report['best_threshold']}  "
          f"Macro F0.5 = {report['best_macro_f05']:.4f}  "
          f"(avg {report['avg_predictions_per_s1_at_best']:.2f} preds/S1)")
    print(f"singletons in fold: {report['val_s1_singletons']}")
    top = sorted(
        zip(thr_grid, macro), key=lambda kv: -kv[1]
    )[:8]
    print("top candidates:", [(round(float(t), 3), round(float(m), 4)) for t, m in top])
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
