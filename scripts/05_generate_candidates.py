#!/usr/bin/env python3
"""Stage 05 — candidate generation (chunked + resumable).

Writes ``artifacts/candidates/{split}/shard_XXXXX.parquet`` with one row per
candidate pair and its retrieval provenance, plus ``manifest.json``.

Each shard is independently skippable: if shard 70 fails, shards 1-69 and 71+
are not recomputed (``<file>.SUCCESS`` markers).

Usage:
    .venv/bin/python scripts/05_generate_candidates.py --split train [--chunk 100000]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import paths as P
from src.pipeline.blocking import (
    BOOL_METHODS,
    BlockConfig,
    candidate_chunk,
    merge_rank,
)
from src.pipeline.tfidf import CountryRetriever, TfidfConfig

BLOCK_COLS = ["sid", "country", "name_norm", "name_ss", "name_tok", "addr_tok", "street_num"]


def tfidf_pairs(tf: pl.DataFrame) -> pl.DataFrame:
    if tf.height == 0:
        return tf
    exprs = [
        pl.lit(False).alias("e_name"),
        pl.lit(False).alias("e_ss"),
        pl.lit(False).alias("rare"),
        pl.lit(False).alias("addr"),
        pl.lit(True).alias("tfidf"),
        pl.lit(0, dtype=pl.Int16).alias("n_rare_shared"),
    ]
    return tf.with_columns(exprs).select(
        ["sid", "tid", "n_rare_shared"] + BOOL_METHODS + ["tfidf_rank", "tfidf_score"]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "dev"], required=True)
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--limit-shards", type=int, default=0)
    ap.add_argument("--worker-index", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    args = ap.parse_args()
    if not (0 <= args.worker_index < args.num_workers):
        sys.exit("--worker-index must be in [0, --num-workers)")

    base = P.split_dir("processed", args.split)
    out_dir = P.split_dir("candidates", args.split)
    P.ensure(out_dir)

    idx_dir = P.ARTIFACTS / "indices" / args.split
    if not (idx_dir / "stats.joblib").exists():
        sys.exit(f"missing indices for {args.split}; run scripts/04_build_indices.py first")
    stats = joblib.load(idx_dir / "stats.joblib")
    cfg_job = joblib.load(idx_dir / "config.joblib")
    block_cfg = BlockConfig(**cfg_job["block"])
    tf_cfg = TfidfConfig(**cfg_job["tfidf"])

    t0 = time.time()
    s1 = (
        pl.read_parquet(base / "s1.parquet")
        .with_row_index("sid")
        .with_columns(pl.col("sid").cast(pl.Int32))
        .select(BLOCK_COLS)
    )
    print(f"[05] {args.split}: {s1.height} S1 rows", flush=True)

    retriever = CountryRetriever.load(idx_dir / "tfidf", tf_cfg)
    print(f"[05] tfidf indexes: {sorted(retriever.indexes)}", flush=True)

    n_shards = (s1.height + args.chunk - 1) // args.chunk
    if args.limit_shards:
        n_shards = min(n_shards, args.limit_shards)

    manifest = {"split": args.split, "chunk": args.chunk, "shards": []}
    for i in range(n_shards):
        # shard number == loop index, so two machines running with disjoint
        # --worker-index values produce disjoint, non-colliding shard files.
        if i % args.num_workers != args.worker_index:
            continue
        path = P.shard_path(out_dir, i)
        if P.is_done(path):
            manifest["shards"].append({"i": i, "file": path.name, "status": "cached"})
            continue
        t1 = time.time()
        q = s1.slice(i * args.chunk, args.chunk)
        cand = candidate_chunk(q, stats, block_cfg)
        tf = retriever.query(q)
        pairs = merge_rank(cand, tfidf_pairs(tf))
        pairs = pairs.select(
            ["sid", "tid", "n_rare_shared"] + BOOL_METHODS + ["tfidf_rank", "tfidf_score"]
        )
        pairs.write_parquet(path)
        P.mark_done(path, {"rows": pairs.height, "s1_rows": q.height})
        manifest["shards"].append(
            {
                "i": i,
                "file": path.name,
                "status": "written",
                "rows": pairs.height,
                "s1_rows": q.height,
                "sec": round(time.time() - t1, 1),
            }
        )
        P.save_manifest(out_dir, manifest)
        import psutil
        rss = psutil.Process().memory_info().rss / (1024 * 1024)
        print(
            f"  shard {i + 1}/{n_shards}: {pairs.height} pairs "
            f"from {q.height} S1 ({time.time() - t1:.1f}s, rss={rss:.0f}MB)",
            flush=True,
        )

    P.save_manifest(out_dir, manifest)
    total = 0
    for f in P.shard_files(out_dir):
        total += pl.scan_parquet(f).select(pl.len()).collect().item()
    print(f"[05] done: {total} candidate pairs across {n_shards} shards "
          f"(worker {args.worker_index}/{args.num_workers}) "
          f"in {time.time() - t0:.0f}s "
          f"({total / max(s1.height, 1):.1f} pairs per S1)")


if __name__ == "__main__":
    main()
