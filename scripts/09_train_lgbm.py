#!/usr/bin/env python3
"""Stage 09 — train the LightGBM pair scorer.

Train on the ``is_val == False`` fold only; early-stop on the ``is_val`` fold.
The model is saved with the exact feature list and TF-IDF K used to build its
features, so scoring cannot silently use a mismatched schema.

Usage:
    .venv/bin/python scripts/09_train_lgbm.py --split dev --rounds 300
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import paths as P
from src.pipeline.features import FEATURE_NAMES


def pair_files(split: str, tfidf_k: int = 50) -> list[Path]:
    files = P.shard_files(P.split_dir("features", split) / "pairs")
    if files:
        P.check_feature_k(P.split_dir("features", split) / "pairs", tfidf_k,
                          "scripts/09_train_lgbm.py")
    if not files:
        sys.exit("no pair features; run scripts/08 --mode pairs first")
    return files


def fold_counts(files: list[Path]) -> tuple[int, int]:
    """(train_rows, val_rows) from the is_val column alone.

    Read first so the feature matrices can be preallocated: concatenating every
    shard into one frame peaks at 2x the corpus (5.4 GB -> ~11 GB on train) and
    is what an eager load_pairs() would cost here.
    """
    n_tr = n_va = 0
    for f in files:
        v = pl.read_parquet(f, columns=["is_val"])["is_val"].to_numpy()
        n_va += int(v.sum())
        n_tr += int(v.size) - int(v.sum())
    return n_tr, n_va


def load_xy(files: list[Path], n_tr: int, n_va: int):
    """Second pass: fill preallocated float32/int32 matrices, one shard at a time."""
    n_feat = len(FEATURE_NAMES)
    x_tr = np.empty((n_tr, n_feat), dtype=np.float32)
    y_tr = np.empty(n_tr, dtype=np.int32)
    x_va = np.empty((n_va, n_feat), dtype=np.float32)
    y_va = np.empty(n_va, dtype=np.int32)
    p_tr = p_va = 0
    for f in files:
        df = pl.read_parquet(f, columns=FEATURE_NAMES + ["label", "is_val"])
        v = df["is_val"].to_numpy()
        x = df.select(FEATURE_NAMES).cast(pl.Float32).to_numpy()
        y = df["label"].to_numpy().astype(np.int32, copy=False)
        n_va_here = int(v.sum())
        if n_va_here:
            x_va[p_va:p_va + n_va_here] = x[v]
            y_va[p_va:p_va + n_va_here] = y[v]
            p_va += n_va_here
        n_tr_here = x.shape[0] - n_va_here
        if n_tr_here:
            x_tr[p_tr:p_tr + n_tr_here] = x[~v]
            y_tr[p_tr:p_tr + n_tr_here] = y[~v]
            p_tr += n_tr_here
        del df, x, y, v
    assert p_tr == n_tr and p_va == n_va, f"row mismatch {p_tr}/{n_tr} {p_va}/{n_va}"
    return x_tr, y_tr, x_va, y_va


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "dev"], required=True)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--early-stopping", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--min-data-in-leaf", type=int, default=50)
    ap.add_argument("--tfidf-k", type=int, default=50)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    t0 = time.time()
    files = pair_files(args.split, args.tfidf_k)
    n_tr, n_va = fold_counts(files)
    x_tr, y_tr, x_va, y_va = load_xy(files, n_tr, n_va)
    n_train_rows = n_tr
    print(f"[09] {args.split}: train={n_tr} val={n_va} "
          f"pos_rate_train={y_tr.mean():.4f}", flush=True)
    if n_tr == 0 or n_va == 0:
        sys.exit("empty train or validation fold — check --val-frac")

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting": "gbdt",
        "num_leaves": args.num_leaves,
        "learning_rate": args.lr,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "min_data_in_leaf": args.min_data_in_leaf,
        "lambda_l2": 1.0,
        "seed": 42,
        "num_threads": args.threads,
        "verbose": -1,
    }
    # free_raw_data lets LightGBM drop the 4.9 GB training matrix once it has
    # binned it; we drop our own reference at the same time.
    dtrain = lgb.Dataset(x_tr, label=y_tr, feature_name=FEATURE_NAMES,
                         free_raw_data=True)
    dval = lgb.Dataset(x_va, label=y_va, reference=dtrain,
                       feature_name=FEATURE_NAMES, free_raw_data=True)
    del x_tr, y_tr

    t1 = time.time()
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=args.rounds,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(args.early_stopping, verbose=False),
                   lgb.log_evaluation(period=50)],
    )
    elapsed = time.time() - t1
    best_iter = booster.best_iteration or booster.current_iteration()
    p_va = booster.predict(x_va, num_iteration=best_iter)

    pos_val = int(y_va.sum())
    p = float(p_va.mean())
    r = float(y_va[p_va >= 0.5].sum() / max(pos_val, 1))
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_va, p_va))
    except Exception:
        auc = float("nan")

    model_dir = P.MODELS / args.split
    P.ensure(model_dir)
    model_path = model_dir / "lgbm.txt"
    booster.save_model(str(model_path), num_iteration=best_iter)

    meta = {
        "split": args.split,
        "model": str(model_path),
        "features": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "tfidf_k": args.tfidf_k,
        "params": params,
        "rounds_requested": args.rounds,
        "best_iteration": int(best_iter),
        "train_rows": int(n_train_rows),
        "val_rows": int(x_va.shape[0]),
        "val_pos_rate": float(y_va.mean()),
        "val_auc": auc,
        "val_p_at_0_5": p,
        "val_r_at_0_5": r,
        "fit_seconds": round(elapsed, 1),
        "total_seconds": round(time.time() - t0, 1),
        "importance": sorted(
            zip(FEATURE_NAMES, [round(float(v), 5) for v in booster.feature_importance("gain")]),
            key=lambda kv: -kv[1],
        )[:20],
    }
    P.write_json(model_dir / "meta.json", meta)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("params", "features")},
                     indent=2))
    print(f"[09] saved {model_path}")


if __name__ == "__main__":
    main()
