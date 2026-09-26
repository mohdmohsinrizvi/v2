#!/usr/bin/env python3
"""Stage 02 — raw TSV -> normalized Parquet (chunked, resumable).

Usage:
    .venv/bin/python scripts/02_normalize.py --split train
    .venv/bin/python scripts/02_normalize.py --split test
    .venv/bin/python scripts/02_normalize.py --split all

Reads ``student_resource/dataset/{train,test}/source{n}.tsv`` and writes
``artifacts/processed/{split}/s{n}.parquet`` plus ``meta.json``.

Original data is only ever opened for reading.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import paths as P
from src.pipeline.normalize import normalize_df
from src.utils.io import DATA_DIR, SOURCE_COLS

CHUNK = 250_000


def scan_source(src: Path) -> pl.LazyFrame:
    """Raw TSV, or already-materialized Parquet (dev sample)."""
    if src.suffix == ".parquet":
        return pl.scan_parquet(src)
    # polars >= 1.x removed `columns=` from scan_csv; select() still pushes the
    # projection down to the CSV reader, so the scan stays streamed.
    return pl.scan_csv(
        src,
        separator="\t",
        infer_schema_length=10000,
        schema_overrides={"entity_id": pl.Utf8},
        ignore_errors=False,
    ).select(SOURCE_COLS)


def normalize_file(src: Path, dst: Path, chunk: int = CHUNK) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if P.is_done(dst):
        meta = P.read_json(Path(str(dst) + ".meta.json"))
        print(f"  [skip] {dst.name} already done ({meta.get('rows')} rows)")
        return meta

    t0 = time.time()
    total = 0
    shard_dir = dst.parent / (dst.stem + "_parts")
    shard_dir.mkdir(parents=True, exist_ok=True)
    parts = sorted(shard_dir.glob("part_*.parquet"))
    offset = len(parts) * chunk
    for part in parts:
        total += pl.scan_parquet(part).select(pl.len()).collect().item()

    while True:
        seg = scan_source(src).slice(offset, chunk).collect()
        if seg.height == 0:
            break
        out = normalize_df(seg)
        part = shard_dir / f"part_{offset // chunk:05d}.parquet"
        out.write_parquet(part)
        total += out.height
        offset += out.height
        print(f"    {src.name}: {offset} rows ({time.time() - t0:.0f}s)", flush=True)
        if out.height < chunk:
            break

    # merge parts lazily (streaming) so peak memory stays per-chunk
    part_files = sorted(shard_dir.glob("part_*.parquet"))
    lazy = pl.concat([pl.scan_parquet(p) for p in part_files])
    lazy.sink_parquet(dst)
    columns = pl.scan_parquet(dst).collect_schema().names()
    for p in part_files:
        p.unlink()
    shard_dir.rmdir()

    meta = {
        "source": str(src),
        "rows": total,
        "bytes_in": src.stat().st_size,
        "elapsed_sec": round(time.time() - t0, 1),
        "columns": columns,
    }
    P.write_json(Path(str(dst) + ".meta.json"), meta)
    P.mark_done(dst, meta)
    print(f"  [done] {dst.name}: {total} rows in {meta['elapsed_sec']}s")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "dev", "all"], default="all")
    ap.add_argument("--chunk", type=int, default=CHUNK)
    args = ap.parse_args()

    splits = ["train", "test"] if args.split == "all" else [args.split]
    out: dict = {}
    for split in splits:
        base = P.split_dir("processed", split)
        P.ensure(base)
        for n in (1, 2, 3):
            if split == "dev":
                src = P.ARTIFACTS / "dev" / f"dev_s{n}.parquet"
            else:
                src = DATA_DIR / split / f"{split}_source{n}.tsv"
            dst = base / f"s{n}.parquet"
            print(f"[normalize] {src.name} -> {dst}", flush=True)
            out[f"{split}/s{n}"] = normalize_file(src, dst, args.chunk)
    P.ensure(P.ARTIFACTS / "processed")
    P.write_json(P.ARTIFACTS / "processed" / f"{args.split}_normalize.json", out)
    print("[normalize] complete")


if __name__ == "__main__":
    main()
