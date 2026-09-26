#!/usr/bin/env python3
"""Stage 04 — build and cache the blocking index for one split.

Produces:
    artifacts/indices/{split}/stats.joblib       join-key group tables
    artifacts/indices/{split}/tfidf/             one TF-IDF index per country
    artifacts/processed/{split}/s1_ids.parquet   sid  -> entity_id
    artifacts/processed/{split}/t_ids.parquet    tid  -> entity_id

The work is split into three phases so the caller can run each in its own
process (``--phase ids|stats|tfidf``), returning memory to the OS between them.
That matters on the 8 GiB host: reading the full 18-column corpus once, then
concatenating it, peaks well past the limit. Every phase here reads only the
columns it consumes, so the parquet projection pushdown does the trimming.

Usage:
    .venv/bin/python scripts/04_build_indices.py --split train --phase ids
    .venv/bin/python scripts/04_build_indices.py --split train --phase stats
    .venv/bin/python scripts/04_build_indices.py --split train --phase tfidf
    .venv/bin/python scripts/04_build_indices.py --split dev --phase all
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
from src.pipeline.blocking import BlockConfig, build_target_stats
from src.pipeline.tfidf import CountryRetriever, TfidfConfig

STATS_COLS = ["country", "name_norm", "name_ss", "street_num", "name_tok"]
TF_COLS = ["country", "name_tok", "addr_tok"]

# The default df cutoffs (rare<=10, addr<=50, tfidf max_df<=1500) were tuned on
# the 272k-row dev target, where they all sit at a fixed *fraction* of the
# corpus. On the 10.3M-row train corpus the same absolute numbers are ~38x more
# selective: 'rare' blocks almost vanish (train recall 0.0227 vs dev 0.3119),
# the same for 'addr' (0.0357 vs 0.304) and TF-IDF (0.7018 vs 0.9872), which is
# what failed the 0.97 gate. Scale them by corpus size instead.
DF_REF_ROWS = 272_474


def target_rows(split: str) -> int:
    """S2+S3 row count from parquet metadata only."""
    base = P.split_dir("processed", split)
    return sum(
        int(pl.scan_parquet(base / f"{s}.parquet").select(pl.len()).collect().item())
        for s in ("s2", "s3")
    )


def rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / (1024 * 1024)


def base(split: str) -> Path:
    return P.split_dir("processed", split)


def read_cols(split: str, src: str, cols: list[str]) -> pl.DataFrame:
    return pl.read_parquet(base(split) / f"{src}.parquet", columns=cols)


def as_target(split: str, cols: list[str]) -> pl.DataFrame:
    """S2 then S3 with the stable global `tid` (same layout as blocking.make_target)."""
    s2 = read_cols(split, "s2", cols)
    s3 = read_cols(split, "s3", cols)
    return pl.concat(
        [s2.with_row_index("tid", offset=0), s3.with_row_index("tid", offset=s2.height)],
        how="vertical",
    )


def phase_ids(split: str) -> None:
    b = base(split)
    P.ensure(b)
    s1_path = b / "s1_ids.parquet"
    if not s1_path.exists():
        s1 = (
            read_cols(split, "s1", ["entity_id"])
            .with_row_index("sid")
            .with_columns(pl.col("sid").cast(pl.Int32))
        )
        s1.select(["sid", "entity_id"]).write_parquet(s1_path)
        del s1
        print(f"[04/ids] s1_ids <- {s1_path.name}", flush=True)
    else:
        print("[04/ids] s1_ids: cached", flush=True)

    t_path = b / "t_ids.parquet"
    if not t_path.exists():
        tgt = as_target(split, ["entity_id"])
        tgt.select(["tid", "entity_id"]).write_parquet(t_path)
        del tgt
        print(f"[04/ids] t_ids <- {t_path.name}", flush=True)
    else:
        print("[04/ids] t_ids: cached", flush=True)


def phase_stats(split: str, cfg: BlockConfig, rebuild: bool) -> None:
    idx_dir = P.ARTIFACTS / "indices" / split
    P.ensure(idx_dir)
    stats_path = idx_dir / "stats.joblib"
    if stats_path.exists() and not rebuild:
        print(f"[04/stats] cached ({stats_path.stat().st_size // 1024} KB)", flush=True)
        return

    t1 = time.time()
    tgt = as_target(split, STATS_COLS)
    print(f"[04/stats] corpus loaded: {tgt.height} rows rss={rss_mb():.0f}MB", flush=True)
    stats = build_target_stats(tgt, cfg)
    del tgt
    joblib.dump(stats, stats_path, compress=3)
    print(f"[04/stats] built in {time.time() - t1:.0f}s rss={rss_mb():.0f}MB -> "
          f"{ {k: v.height for k, v in stats.items()} }", flush=True)


def phase_tfidf(split: str, cfg: TfidfConfig, rebuild: bool) -> None:
    idx_dir = P.ARTIFACTS / "indices" / split
    tf_dir = idx_dir / "tfidf"
    P.ensure(idx_dir)
    if (tf_dir / "countries.joblib").exists() and not rebuild:
        print("[04/tfidf] cached", flush=True)
        return

    t1 = time.time()
    tgt = as_target(split, TF_COLS)
    print(f"[04/tfidf] corpus loaded: {tgt.height} rows rss={rss_mb():.0f}MB", flush=True)
    retriever = CountryRetriever(cfg)
    info = retriever.fit(tgt)
    del tgt
    retriever.save(tf_dir)
    print(f"[04/tfidf] fit in {time.time() - t1:.0f}s rss={rss_mb():.0f}MB -> "
          f"{ {c: (i['raw_vocab'], i['kept_vocab']) for c, i in info.items()} } "
          f"(raw,kept)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "dev"], required=True)
    ap.add_argument("--phase", choices=["all", "ids", "stats", "tfidf"], default="all")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--rare-max-df", type=int, default=10)
    ap.add_argument("--addr-max-df", type=int, default=50)
    ap.add_argument("--max-df", type=int, default=1500)
    ap.add_argument("--min-df", type=int, default=3,
                    help="kept absolute: scaling it up would prune the rare "
                         "tokens that make unique business names retrievable")
    ap.add_argument("--df-ref-rows", type=int, default=DF_REF_ROWS)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    P.ensure(P.ARTIFACTS / "indices")
    n_target = target_rows(args.split)
    scale = n_target / args.df_ref_rows
    block_cfg = BlockConfig(
        rare_max_df=max(2, round(args.rare_max_df * scale)),
        addr_max_df=max(2, round(args.addr_max_df * scale)),
        tfidf_k=args.k,
    )
    tf_cfg = TfidfConfig(min_df=args.min_df,
                         max_df=max(args.min_df, round(args.max_df * scale)),
                         k=args.k)
    print(f"[04] split={args.split} target_rows={n_target} df_scale={scale:.2f} "
          f"-> rare_max_df={block_cfg.rare_max_df} "
          f"addr_max_df={block_cfg.addr_max_df} "
          f"tfidf min_df={tf_cfg.min_df} max_df={tf_cfg.max_df} k={tf_cfg.k}",
          flush=True)
    idx_dir = P.ARTIFACTS / "indices" / args.split
    t0 = time.time()

    if args.phase in ("all", "ids"):
        phase_ids(args.split)
    if args.phase in ("all", "stats"):
        phase_stats(args.split, block_cfg, args.rebuild)
    if args.phase in ("all", "tfidf"):
        phase_tfidf(args.split, tf_cfg, args.rebuild)

    if args.phase in ("all", "tfidf"):
        joblib.dump(
            {"block": vars(block_cfg), "tfidf": vars(tf_cfg), "split": args.split},
            idx_dir / "config.joblib",
        )
    print(f"[04] phase={args.phase} done in {time.time() - t0:.0f}s "
          f"rss={rss_mb():.0f}MB", flush=True)


if __name__ == "__main__":
    main()
