"""Deterministic Source-1 entity split for train/validation.

The official metric is entity-level Macro F0.5, and predictions for one S1 are
jointly decided, so the split unit MUST be the Source-1 row. Never split pairs.

Hash of ``entity_id`` (not row position) so the assignment is stable across runs
and across re-generated candidate shards.
"""
from __future__ import annotations

import hashlib

import polars as pl

SPLIT_COL = "is_val"


def _bucket(entity_id: str) -> int:
    return int.from_bytes(hashlib.md5(entity_id.encode("utf-8")).digest()[:4], "big") % 10000


def assign(s1_entity_ids: pl.Series, val_frac: float = 0.10) -> pl.Series:
    """True where the Source-1 row belongs to the validation fold."""
    threshold = int(round(val_frac * 10000))
    return pl.Series(
        name=SPLIT_COL,
        values=[(_bucket(e) < threshold) for e in s1_entity_ids.to_list()],
        dtype=pl.Boolean,
    )


def attach(is_val: pl.Series) -> pl.DataFrame:
    return pl.DataFrame({"sid": pl.arange(0, len(is_val), eager=True).cast(pl.Int32)}).with_columns(
        is_val.alias(SPLIT_COL)
    )


def summary(s1: pl.DataFrame, val_frac: float) -> dict:
    flags = assign(s1["entity_id"], val_frac)
    n_val = int(flags.sum())
    n = len(flags)
    return {
        "s1_total": n,
        "s1_val": n_val,
        "s1_train": n - n_val,
        "val_frac_requested": val_frac,
        "val_frac_actual": round(n_val / n, 4) if n else 0.0,
    }
