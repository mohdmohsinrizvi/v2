#!/usr/bin/env python3
"""Stage 10 — score candidate pairs with the trained model (sharded + resumable).

Output: ``artifacts/predictions/{split}/score_XXXXX.parquet`` with
``sid, tid, score``. Entity IDs are resolved only at submission time.

Usage:
    .venv/bin/python scripts/10_score_candidates.py --split dev
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


def shard_no(p: Path) -> int:
    """Shard number encoded in ``shard_00042.parquet``."""
    return int(p.stem.rsplit("_", 1)[-1])


def _rows_match(dst: Path, src: Path) -> bool:
    """A prediction shard is only valid for the exact feature shard it was
    produced from; features rebuilt with another fold or k must be rescored."""
    sm, dm = P.marker(src), P.marker(dst)
    if not (sm.exists() and dm.exists()):
        return False
    return P.read_json(sm).get("rows") == P.read_json(dm).get("rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "dev"], required=True)
    ap.add_argument("--model-split", default=None,
                    help="which model to use (default: same split)")
    ap.add_argument("--batch", type=int, default=500_000)
    ap.add_argument("--worker-index", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    args = ap.parse_args()
    if not (0 <= args.worker_index < args.num_workers):
        sys.exit("--worker-index must be in [0, --num-workers)")

    model_split = args.model_split or args.split
    meta = P.read_json(P.MODELS / model_split / "meta.json")
    feats = meta["features"]
    if feats != FEATURE_NAMES:
        sys.exit("model feature list does not match src/pipeline/features.py")
    model_path = P.MODELS / model_split / "lgbm.txt"
    if not model_path.exists():
        sys.exit(f"missing model {model_path}")
    booster = lgb.Booster(model_file=str(model_path))
    best_iter = int(meta.get("best_iteration") or 0) or None
    k = int(meta.get("tfidf_k", 50))

    src_dir = P.split_dir("features", args.split) / "candidates"
    P.check_feature_k(src_dir, k, "scripts/10_score_candidates.py")
    # Output names are taken from the source shard NUMBER, not a position in
    # this machine's local file list: with two machines each holding a
    # disjoint half of the features, enumerate() would restart at 0 on both
    # and the merged predictions would collide.
    src_files = [(shard_no(p), p) for p in P.shard_files(src_dir)]
    src_files = [(i, p) for i, p in src_files
                 if i % args.num_workers == args.worker_index]
    src_files.sort()
    if not src_files:
        sys.exit("no candidate features; run scripts/08 --mode candidates first")

    out_dir = P.split_dir("predictions", args.split)
    P.ensure(out_dir)
    t0 = time.time()
    total = 0
    for i, src in src_files:
        dst = P.shard_path(out_dir, i)
        if P.is_done(dst) and _rows_match(dst, src):
            continue
        t1 = time.time()
        df = pl.read_parquet(src)
        parts = []
        for off in range(0, df.height, args.batch):
            b = df.slice(off, args.batch)
            x = b.select(feats).to_numpy().astype(np.float32)
            # SCORE_WORKERS=4 processes each default to 2 OpenMP threads:
            # 8 spinning threads on 2 vCPUs. Measured on the instance that
            # caps the whole stage at ~31k rows/s, which would push the 582M
            # test pairs past the 15h watchdog. One thread per worker removes
            # the contention and lets the OS schedule 4 clean processes.
            proba = booster.predict(x, num_iteration=best_iter, num_threads=1)
            parts.append(
                pl.DataFrame(
                    {
                        "sid": b["sid"].to_numpy(),
                        "tid": b["tid"].to_numpy(),
                        "score": proba.astype(np.float32),
                    }
                )
            )
        out = pl.concat(parts, how="vertical")
        out.write_parquet(dst)
        P.mark_done(dst, {"rows": out.height, "src": src.name})
        total += out.height
        print(f"  shard {i + 1}/{len(src_files)}: {out.height} scores "
              f"({time.time() - t1:.1f}s)", flush=True)
        del df, parts, out

    summary = out_dir / ("summary.json" if args.num_workers == 1
                      else f"summary_w{args.worker_index}.json")
    P.write_json(
        summary,
        {
            "split": args.split,
            "model_split": model_split,
            "scores": total,
            "tfidf_k": k,
            "shards": len(src_files),
            "elapsed_sec": round(time.time() - t0, 1),
        },
    )
    print(f"[10] done: {total} scores in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
