#!/usr/bin/env python3
"""Stage 08 — pairwise feature generation (sharded + resumable).

Two modes, both feeding the same model:

  --mode pairs       labelled training pairs     -> features/{split}/pairs/
  --mode candidates  ALL candidate pairs         -> features/{split}/cand/

Shard-level resumability means a crash at shard 40/100 costs only that shard.

Usage:
    .venv/bin/python scripts/08_generate_features.py --split dev --mode pairs
    .venv/bin/python scripts/08_generate_features.py --split dev --mode candidates
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import paths as P
from src.pipeline import split as SPLIT
from src.pipeline.features import FEATURE_NAMES, pair_features
from src.pipeline.gt import s1_index

S1_FEAT_COLS = [
    "name_norm", "name_ss", "name_tok", "name_num", "name_nchar",
    "addr_norm", "addr_tok", "addr_num", "addr_nchar",
    "street_num", "unit_num", "bldg_num", "postal", "has_addr",
]


def shard_no(p: Path) -> int:
    """Shard number encoded in ``shard_00042.parquet``."""
    return int(p.stem.rsplit("_", 1)[-1])


def load_side(split: str, n: str) -> pl.DataFrame:
    # column projection: the full frame is 18 columns on a 10M-row corpus
    df = pl.read_parquet(P.split_dir("processed", split) / f"s{n}.parquet",
                         columns=S1_FEAT_COLS)
    if n == "1":
        return (
            df.with_row_index("sid")
            .with_columns(pl.col("sid").cast(pl.Int32))
            .select(["sid"] + S1_FEAT_COLS)
        )
    return df.select(S1_FEAT_COLS)


def load_target(split: str) -> pl.DataFrame:
    """Target frame (tid + feature columns) for S2 then S3.

    Built lazily with the streaming engine: an eager concat of a 10M-row frame
    keeps the inputs and the output alive together, which is what pushed the old
    stage 08 past 8 GiB. ``tid`` is the row position after S2 then S3, the same
    layout as t_ids.parquet, so no separate id frame either.
    """
    base = P.split_dir("processed", split)
    return (
        pl.concat(
            [pl.scan_parquet(base / "s2.parquet").select(S1_FEAT_COLS),
             pl.scan_parquet(base / "s3.parquet").select(S1_FEAT_COLS)],
            how="vertical",
        )
        .with_row_index("tid")
        .with_columns(pl.col("tid").cast(pl.Int32))
        .collect(engine="streaming")
    )


def out_base(split: str, mode: str) -> Path:
    d = P.split_dir("features", split) / mode
    P.ensure(d)
    return d


def val_sid_series(split: str, val_frac: float) -> pl.Series:
    """Validation-fold Source-1 indices, as a Series for fast is_in joins."""
    flags = SPLIT.assign(s1_index(split)["entity_id"], val_frac).to_numpy()
    return pl.Series("sid", np.flatnonzero(flags), dtype=pl.Int32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "dev"], required=True)
    ap.add_argument("--mode", choices=["pairs", "candidates"], required=True)
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--tfidf-k", type=int, default=50)
    ap.add_argument("--val-frac", type=float, default=0.10,
                    help="must match scripts/07/11; used only by --val-only")
    ap.add_argument("--val-only", action="store_true",
                    help="candidates mode: keep only the validation-fold Source-1 rows "
                         "(threshold search does not need the train fold scored)")
    ap.add_argument("--worker-index", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument(
        "--shard-parity", choices=["all", "0", "1"], default="all",
        help="only consume candidate shards whose shard number is even/odd; "
             "lets two machines split the test candidates without transferring "
             "them (each keeps a disjoint half).",
    )
    args = ap.parse_args()
    if not (0 <= args.worker_index < args.num_workers):
        sys.exit("--worker-index must be in [0, --num-workers)")

    # Every call to pair_features rebuilds the hash tables for the 2.2M-row
    # Source-1 frame and the 10.3M-row target frame, a ~15 s fixed cost. At the
    # orchestrator's --chunk 200000 that is only ~11k rows/s; measured on the
    # production instance, chunk=1000000 runs at 34k rows/s. The test split is
    # ~580M candidate pairs, so the difference is ~10 h of wall clock. Output
    # shard granularity is not worth that (stages 10/11 stream the files), so
    # never process in blocks smaller than this.
    args.chunk = max(args.chunk, 1_000_000)
    val_sids = (val_sid_series(args.split, args.val_frac) if args.val_only else None)

    t0 = time.time()
    base = out_base(args.split, args.mode)
    s1 = load_side(args.split, "1")
    target = load_target(args.split)
    print(f"[08] s1={s1.height} target={target.height}", flush=True)

    if args.mode == "pairs":
        work = pl.read_parquet(P.split_dir("train_pairs", args.split) / "pairs.parquet")
        print(f"[08] pairs={work.height}", flush=True)
        n_chunks = (work.height + args.chunk - 1) // args.chunk
        assigned = [i for i in range(n_chunks)
                    if i % args.num_workers == args.worker_index]
        shard_files: list[Path] = [P.shard_path(base, i) for i in assigned]
        for i in assigned:
            path = P.shard_path(base, i)
            if P.is_done(path):
                continue
            chunk = work.slice(i * args.chunk, args.chunk)
            t1 = time.time()
            feats = pair_features(chunk, s1, target, args.tfidf_k)
            feats.write_parquet(path)
            P.mark_done(path, {"rows": feats.height, "tfidf_k": args.tfidf_k})
            print(f"  pair shard {i + 1}/{n_chunks}: {feats.height} rows "
                  f"({time.time() - t1:.1f}s)", flush=True)
        del work
    else:
        all_files = P.shard_files(P.split_dir("candidates", args.split))
        if not all_files:
            sys.exit("no candidate shards; run scripts/05 first")
        # Output names are i*STRIDE+piece, so `i` MUST be the shard number
        # parsed from the input filename, not a position in this machine's
        # local file list. Two machines each holding a disjoint half of the
        # candidates would otherwise both start at 0 and write colliding
        # shard names.
        cand_files = [(shard_no(p), p) for p in all_files]
        if args.shard_parity != "all":
            keep = int(args.shard_parity)
            cand_files = [(i, p) for i, p in cand_files if i % 2 == keep]
        cand_files = [(i, p) for i, p in cand_files
                      if i % args.num_workers == args.worker_index]
        cand_files.sort()
        # One input shard is ~33.6M candidate rows; pair_features joins them
        # against the 2.2M-row Source-1 frame and the 10.3M-row target frame,
        # which would peak at ~20 GB. Slice the shard into --chunk pieces and
        # write each piece as its own output shard (stages 10/11 only glob
        # shard_*.parquet). STRIDE keeps (input, piece) -> output index a
        # pure function, so restarts land on the same paths.
        STRIDE = 128
        shard_files: list[Path] = []
        for i, src in cand_files:
            t1 = time.time()
            cand = pl.read_parquet(src)
            raw_rows = cand.height
            if args.val_only:
                before = cand.height
                cand = cand.filter(pl.col("sid").is_in(val_sids))
                print(f"    val-only: {before} -> {cand.height}", flush=True)
            n_pieces = (raw_rows + args.chunk - 1) // args.chunk
            written = 0
            rows = 0
            for c in range(n_pieces):
                piece = cand.slice(c * args.chunk, args.chunk)
                if piece.height == 0:
                    continue
                path = P.shard_path(base, i * STRIDE + c)
                shard_files.append(path)
                if P.is_done(path):
                    del piece
                    continue
                t2 = time.time()
                feats = pair_features(piece, s1, target, args.tfidf_k)
                feats.write_parquet(path)
                P.mark_done(path, {"rows": feats.height, "src": src.name,
                                   "piece": c, "tfidf_k": args.tfidf_k,
                                   "val_frac": args.val_frac if args.val_only
                                   else None})
                written += 1
                rows += feats.height
                print(f"  cand shard {i + 1}/{len(cand_files)} piece "
                      f"{c + 1}/{n_pieces}: {feats.height} rows "
                      f"({time.time() - t2:.1f}s)", flush=True)
                del feats
            print(f"  cand shard {i + 1}/{len(cand_files)}: {rows} rows in "
                  f"{written} pieces ({time.time() - t1:.1f}s)", flush=True)
            del cand

    total = 0
    for f in shard_files:
        if f.exists():
            total += int(pl.scan_parquet(f).select(pl.len()).collect()
                         .item())
    summary = base / ("summary.json" if args.num_workers == 1
                      else f"summary_w{args.worker_index}.json")
    P.write_json(
        summary,
        {
            "split": args.split,
            "mode": args.mode,
            "rows": total,
            "features": len(FEATURE_NAMES),
            "tfidf_k": args.tfidf_k,
            "shards": len(shard_files),
            "elapsed_sec": round(time.time() - t0, 1),
        },
    )
    print(f"[08] done: {total} rows / {len(FEATURE_NAMES)} features "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
