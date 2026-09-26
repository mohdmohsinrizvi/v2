"""Candidate generation (blocking) for entity resolution.

Design constraints enforced here:

* **Never** brute-force S1 x (S2+S3).
* Country is a safe blocking key (0/7,638,365 ground-truth links cross countries —
  see ``reports/data_audit.md``), so every method joins on ``country``.
* Work is chunked over Source-1 rows so peak memory stays bounded regardless of
  corpus size; each chunk yields one Parquet shard (resumable).

Methods (boolean columns, one per retrieval signal):

| column    | bit meaning                      |
|-----------|----------------------------------|
| `e_name`  | exact normalized business name   |
| `e_ss`    | exact suffix-stripped name       |
| `rare`    | rare-name-token inverted index   |
| `addr`    | important address-number match   |
| `tfidf`   | word TF-IDF top-K retrieval      |

Pairs carry `tfidf_rank` / `tfidf_score` when produced by TF-IDF.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

BOOL_METHODS = ["e_name", "e_ss", "rare", "addr", "tfidf"]
PAIR_COLS = ["sid", "tid"]


@dataclass
class BlockConfig:
    rare_max_df: int = 10
    rare_tokens_per_q: int = 6
    addr_max_df: int = 50
    max_cluster: int = 500
    tfidf_k: int = 50
    empty_token: str = ""


def make_target(s2: pl.DataFrame, s3: pl.DataFrame) -> pl.DataFrame:
    """Concatenate S2/S3 and assign a stable global target id `tid`.

    `tid` is the positional index: `[0, len(s2))` for S2, then S3 offset by
    `len(s2)`. It must be reproducible from the normalized files alone.
    """
    a = s2.with_row_index("tid", offset=0)
    b = s3.with_row_index("tid", offset=s2.height)
    return pl.concat([a, b], how="vertical")


def _token_frame(df: pl.DataFrame, key: str, id_col: str) -> pl.DataFrame:
    return (
        df.select([id_col, "country", key])
        .with_columns(pl.col(key).str.split(" ").alias("__tk"))
        .explode("__tk")
        .filter(pl.col("__tk") != "")
        .rename({key: "tok_raw", "__tk": "tok"})
    )


def build_target_stats(t: pl.DataFrame, cfg: BlockConfig) -> dict[str, pl.DataFrame]:
    """Pre-aggregate target-side lookup tables (computed once per corpus)."""
    stats: dict[str, pl.DataFrame] = {}

    stats["name"] = (
        t.filter(pl.col("name_norm") != "")
        .group_by(["country", "name_norm"])
        .agg(pl.col("tid").alias("tids"))
        .with_columns(pl.col("tids").list.head(cfg.max_cluster))
    )
    stats["ss"] = (
        t.filter(pl.col("name_ss") != "")
        .group_by(["country", "name_ss"])
        .agg(pl.col("tid").alias("tids"))
        .with_columns(pl.col("tids").list.head(cfg.max_cluster))
    )
    stats["addr"] = (
        t.filter((pl.col("street_num") != "") & (pl.col("street_num").str.len_chars() <= 6))
        .group_by(["country", "street_num"])
        .agg(pl.col("tid").alias("tids"), pl.len().alias("df"))
        .filter(pl.col("df") <= cfg.addr_max_df)
        .with_columns(pl.col("tids").list.head(cfg.max_cluster))
    )
    stats["tok"] = (
        _token_frame(t, "name_tok", "tid")
        .group_by(["country", "tok"])
        .agg(pl.col("tid").unique().alias("tids"), pl.len().alias("df"))
        .filter((pl.col("df") >= 2) & (pl.col("df") <= cfg.rare_max_df))
    )
    return stats


def _empty_pairs() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "sid": pl.Int32,
            "tid": pl.Int32,
            "e_name": pl.Boolean,
            "e_ss": pl.Boolean,
            "rare": pl.Boolean,
            "addr": pl.Boolean,
            "tfidf": pl.Boolean,
            "tfidf_rank": pl.Int16,
            "tfidf_score": pl.Float32,
            "n_rare_shared": pl.Int16,
        }
    )


def _pairs_from(
    joined: pl.DataFrame, method: str, has_rare_count: bool = False
) -> pl.DataFrame:
    """Turn a (sid, tid, ...) join result into the canonical pair schema."""
    exprs = [
        pl.col("sid").cast(pl.Int32),
        pl.col("tid").cast(pl.Int32),
    ]
    if has_rare_count:
        exprs.append(pl.col("n_rare_shared").cast(pl.Int16))
    else:
        exprs.append(pl.lit(0, dtype=pl.Int16).alias("n_rare_shared"))
    for m in BOOL_METHODS:
        exprs.append(pl.lit(m == method).alias(m))
    exprs.append(pl.lit(None, dtype=pl.Int16).alias("tfidf_rank"))
    exprs.append(pl.lit(None, dtype=pl.Float32).alias("tfidf_score"))
    return joined.select(exprs)


def candidate_chunk(
    q: pl.DataFrame, stats: dict[str, pl.DataFrame], cfg: BlockConfig
) -> pl.DataFrame:
    """Generate candidate pairs for one chunk of Source-1 rows."""
    out: list[pl.DataFrame] = []

    named = q.filter(pl.col("name_norm") != "").select(["sid", "country", "name_norm"])
    if named.height:
        j = named.join(stats["name"], on=["country", "name_norm"], how="inner")
        if j.height:
            out.append(_pairs_from(j.explode("tids").rename({"tids": "tid"}), "e_name"))

    ssq = q.filter(pl.col("name_ss") != "").select(["sid", "country", "name_ss"])
    if ssq.height:
        j = ssq.join(stats["ss"], on=["country", "name_ss"], how="inner")
        if j.height:
            out.append(_pairs_from(j.explode("tids").rename({"tids": "tid"}), "e_ss"))

    addrq = q.filter(
        (pl.col("street_num") != "") & (pl.col("street_num").str.len_chars() <= 6)
    ).select(["sid", "country", "street_num"])
    if addrq.height:
        j = addrq.join(stats["addr"], on=["country", "street_num"], how="inner")
        if j.height:
            out.append(_pairs_from(j.explode("tids").rename({"tids": "tid"}), "addr"))

    tk = (
        q.select(["sid", "country", "name_tok"])
        .with_columns(pl.col("name_tok").str.split(" ").alias("__tk"))
        .explode("__tk")
        .filter(pl.col("__tk") != "")
        .rename({"__tk": "tok"})
        .join(stats["tok"], on=["country", "tok"], how="inner")
    )
    if tk.height:
        tk = tk.sort(["sid", "df"]).group_by("sid").head(cfg.rare_tokens_per_q)
        j = tk.explode("tids").rename({"tids": "tid"})
        j = j.group_by(["sid", "tid"]).agg(pl.len().alias("n_rare_shared"))
        out.append(_pairs_from(j, "rare", has_rare_count=True))

    if not out:
        return _empty_pairs()

    merged = pl.concat(out, how="diagonal_relaxed")
    return (
        merged.group_by(["sid", "tid"])
        .agg(
            pl.any("e_name"),
            pl.any("e_ss"),
            pl.any("rare"),
            pl.any("addr"),
            pl.any("tfidf"),
            pl.col("tfidf_rank").drop_nulls().min().alias("tfidf_rank"),
            pl.col("tfidf_score").drop_nulls().min().alias("tfidf_score"),
            pl.col("n_rare_shared").max().alias("n_rare_shared"),
        )
        .with_columns(
            pl.col("tfidf_rank").cast(pl.Int16),
            pl.col("tfidf_score").cast(pl.Float32),
            pl.col("n_rare_shared").cast(pl.Int16),
            pl.col("sid").cast(pl.Int32),
            pl.col("tid").cast(pl.Int32),
        )
    )


def merge_rank(
    base: pl.DataFrame, extra: pl.DataFrame
) -> pl.DataFrame:
    """Merge a TF-IDF pair frame into an existing candidate frame."""
    if extra.height == 0:
        return base
    return (
        pl.concat([base, extra], how="diagonal_relaxed")
        .group_by(["sid", "tid"])
        .agg(
            pl.any("e_name"),
            pl.any("e_ss"),
            pl.any("rare"),
            pl.any("addr"),
            pl.any("tfidf"),
            pl.col("tfidf_rank").drop_nulls().min().alias("tfidf_rank"),
            pl.col("tfidf_score").drop_nulls().min().alias("tfidf_score"),
            pl.col("n_rare_shared").max().alias("n_rare_shared"),
        )
        .with_columns(
            pl.col("tfidf_rank").cast(pl.Int16),
            pl.col("tfidf_score").cast(pl.Float32),
            pl.col("n_rare_shared").cast(pl.Int16),
            pl.col("sid").cast(pl.Int32),
            pl.col("tid").cast(pl.Int32),
        )
    )
