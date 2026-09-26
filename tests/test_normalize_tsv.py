"""Regression tests for the raw-TSV normalisation path.

The dev pipeline exercises the parquet branch only, which is how polars 1.44
dropping `scan_csv(columns=...)` reached production: stage 02 died instantly
on the instance while every local test stayed green.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SPEC = importlib.util.spec_from_file_location(
    "norm02", Path(__file__).resolve().parents[1] / "scripts" / "02_normalize.py"
)
norm02 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(norm02)

HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
ROWS = [
    "E1\tACME GMBH\tHauptstrasse 1, 10115 Berlin\tDE",
    "E2\tACME GmbH\tHauptstrasse 1, Berlin\tDE",
    "E3\tBeta LLC\t221B Baker Street, London\tGB",
]
COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def write_tsv(path: Path, extra_column: bool = False) -> None:
    header = HEADER
    if extra_column:
        header = header.rstrip("\n") + "\tignored_extra\n"
    lines = [row + ("\tX" if extra_column else "") for row in ROWS]
    path.write_text(header + "\n".join(lines) + "\n")


def test_scan_source_reads_raw_tsv(tmp_path: Path) -> None:
    src = tmp_path / "train_source1.tsv"
    write_tsv(src)

    lf = norm02.scan_source(src)
    df = lf.collect()

    assert df.columns == COLUMNS
    assert df.height == len(ROWS)
    assert df["entity_id"].to_list() == ["E1", "E2", "E3"]
    assert df.schema["entity_id"] == pl.Utf8


def test_scan_source_projects_away_extra_columns(tmp_path: Path) -> None:
    src = tmp_path / "train_source1.tsv"
    write_tsv(src, extra_column=True)

    df = norm02.scan_source(src).collect()

    assert df.columns == COLUMNS
    assert df.height == len(ROWS)


def test_scan_source_parquet_branch(tmp_path: Path) -> None:
    src = tmp_path / "dev_s1.parquet"
    pl.DataFrame({"entity_id": ["E1"], "business_name": ["ACME"]}).write_parquet(src)

    assert norm02.scan_source(src).collect().columns == ["entity_id", "business_name"]


def test_normalize_file_end_to_end(tmp_path: Path) -> None:
    src = tmp_path / "train_source1.tsv"
    write_tsv(src)
    dst = tmp_path / "out" / "s1.parquet"

    meta = norm02.normalize_file(src, dst, chunk=2)

    assert meta["rows"] == len(ROWS)
    out = pl.read_parquet(dst)
    assert out.columns[:4] == COLUMNS
    assert out.height == len(ROWS)
    # resumable marker written, staging shards cleaned up
    assert (tmp_path / "out" / "s1.parquet.SUCCESS").exists()
    assert not (tmp_path / "out" / "s1_parts").exists()
