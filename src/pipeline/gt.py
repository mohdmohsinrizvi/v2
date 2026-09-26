"""Ground-truth pair construction shared by measurement, training and inference."""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.pipeline import paths as P
from src.utils.io import read_ground_truth


def id_to_index(split: str) -> pl.DataFrame:
    """Map target ``entity_id`` -> contiguous ``tid`` (S2 rows first, then S3)."""
    base = P.split_dir("processed", split)
    # projection pushdown: this used to read all 18 columns of a 10M-row frame
    s2 = pl.read_parquet(base / "s2.parquet", columns=["entity_id"])
    s3 = pl.read_parquet(base / "s3.parquet", columns=["entity_id"])
    n2 = s2.height
    return pl.concat(
        [
            s2.with_row_index("tid").rename({"entity_id": "tid_eid"}),
            s3.with_row_index("tid", offset=n2).rename({"entity_id": "tid_eid"}),
        ],
        how="vertical",
    ).with_columns(pl.col("tid").cast(pl.Int32))


def s1_index(split: str) -> pl.DataFrame:
    base = P.split_dir("processed", split)
    return (
        pl.read_parquet(base / "s1.parquet", columns=["entity_id"])
        .with_row_index("sid")
        .with_columns(pl.col("sid").cast(pl.Int32))
    )


def ground_truth_pairs(split: str) -> pl.DataFrame:
    """All official positive links restricted to rows present in this split."""
    base = P.split_dir("processed", split)
    s1 = s1_index(split)
    gt = read_ground_truth()
    exploded = (
        gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("__m"))
        .explode("__m")
        .filter(pl.col("__m") != "")  # empty string, not an id — the singleton trap
        .rename({"source1_entity_id": "entity_id", "__m": "tid_eid"})
    )
    return (
        exploded.join(id_to_index(split), on="tid_eid", how="inner")
        .join(s1, on="entity_id", how="inner")
        .select(["sid", "tid"])
        .unique(maintain_order=True)
    )


def candidate_shards(split: str) -> list[Path]:
    files = P.shard_files(P.split_dir("candidates", split))
    if not files:
        raise FileNotFoundError(f"no candidate shards for {split}; run scripts/05 first")
    return files


def iter_candidate_shards(split: str, columns: list[str] | None = None):
    """Yield one candidate shard at a time.

    The full train candidate set is ~214M pairs; concatenating it does not fit
    in 8 GiB. Shards are contiguous and disjoint in ``sid``, so any per-S1
    aggregation can be accumulated shard by shard and is bit-identical to the
    eager version.
    """
    for f in candidate_shards(split):
        yield pl.read_parquet(f, columns=columns)


def load_candidates(split: str) -> pl.DataFrame:
    return pl.concat(list(iter_candidate_shards(split)), how="diagonal_relaxed")


def s1_ids(split: str) -> pl.DataFrame:
    return pl.read_parquet(P.split_dir("processed", split) / "s1_ids.parquet")


def target_ids(split: str) -> pl.DataFrame:
    return pl.read_parquet(P.split_dir("processed", split) / "t_ids.parquet")
