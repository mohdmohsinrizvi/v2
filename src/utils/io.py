from __future__ import annotations

import os
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("AMZ_DATA_DIR", PROJECT_ROOT.parent / "student_resource" / "dataset"))
ARTIFACTS = PROJECT_ROOT / "artifacts"
REPORTS = PROJECT_ROOT / "reports"
SUBMISSIONS = PROJECT_ROOT / "submissions"

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_source(path: str | Path) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", infer_schema_length=10000,
                       columns=SOURCE_COLS, schema_overrides={"entity_id": pl.Utf8})


def read_train_source(name: str) -> pl.DataFrame:
    return read_source(DATA_DIR / "train" / f"train_source{name}.tsv")


def read_test_source(name: str) -> pl.DataFrame:
    return read_source(DATA_DIR / "test" / f"test_source{name}.tsv")


def read_ground_truth() -> pl.DataFrame:
    df = pl.read_csv(DATA_DIR / "train" / "train_ground_truth.tsv", separator="\t",
                     schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8})
    return df.with_columns(pl.col("matched_entity_ids").fill_null("").alias("matched_entity_ids"))


def ensure_dirs() -> None:
    for d in (ARTIFACTS, REPORTS, SUBMISSIONS):
        d.mkdir(parents=True, exist_ok=True)


def write_parquet(df: pl.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


def read_parquet(path: str | Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def shard_marker(path: str | Path) -> Path:
    return Path(str(path) + ".SUCCESS")


def is_done(path: str | Path) -> bool:
    return shard_marker(path).exists()


def mark_done(path: str | Path) -> None:
    shard_marker(path).write_text("ok")


def rss_mb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / (1024 * 1024)


def disk_free_gb(path: str | Path = ".") -> float:
    return os.statvfs(path).f_bavail * os.statvfs(path).f_frsize / (1024 ** 3)
