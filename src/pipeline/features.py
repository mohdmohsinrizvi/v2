"""Pairwise similarity features for candidate (S1, S2/S3) rows.

Vectorized Polars expressions do the cheap work; one batched Python UDF does the
RapidFuzz string similarities and token-set overlaps (those cannot be expressed
as Polars expressions). Nothing here ever densifies a large matrix.
"""
from __future__ import annotations

import polars as pl
from rapidfuzz import fuzz

FUZZ_FIELDS = [
    ("n_fuzz_ratio", pl.Float32),
    ("n_fuzz_sort", pl.Float32),
    ("n_fuzz_set", pl.Float32),
    ("n_jaccard", pl.Float32),
    ("n_common_tokens", pl.Float32),
    ("a_fuzz_ratio", pl.Float32),
    ("a_jaccard", pl.Float32),
    ("n_num_jaccard", pl.Float32),
    ("a_num_jaccard", pl.Float32),
]

FUZZ_STRUCT = pl.Struct(dict(FUZZ_FIELDS))

INPUT_FIELDS = [
    "qa_name_norm",
    "tb_name_norm",
    "qa_name_tok",
    "tb_name_tok",
    "qa_addr_norm",
    "tb_addr_norm",
    "qa_name_num",
    "tb_name_num",
    "qa_addr_num",
    "tb_addr_num",
]

SCALAR_FEATURES = [
    "n_exact",
    "n_exact_ss",
    "n_len_diff",
    "n_nchar_ratio",
    "a_street_match",
    "a_street_conflict",
    "a_unit_match",
    "a_unit_conflict",
    "a_postal_match",
    "a_postal_conflict",
    "a_missing_either",
    "a_missing_both",
    "x_exact_name_addr",
    "x_exact_name_street",
    "x_rare_street",
    "r_tfidf_rank_norm",
    "r_tfidf_score",
    "r_n_methods",
    "n_rare_shared",
    "q_name_nchar",
    "t_name_nchar",
    "q_addr_nchar",
    "t_addr_nchar",
]

FUZZ_NAMES = [f for f, _ in FUZZ_FIELDS]
FEATURE_NAMES = SCALAR_FEATURES + FUZZ_NAMES


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    u = len(a) + len(b) - len(a & b)
    return (len(a & b) / u) if u else 0.0


def _score_batch(s: pl.Series) -> pl.Series:
    df = s.struct.unnest()
    cols = {c: df[c].to_list() for c in INPUT_FIELDS}
    out = {name: [] for name, _ in FUZZ_FIELDS}
    for i in range(df.height):
        na = cols["qa_name_norm"][i] or ""
        nb = cols["tb_name_norm"][i] or ""
        ta = cols["qa_name_tok"][i] or ""
        tb = cols["tb_name_tok"][i] or ""
        aa = cols["qa_addr_norm"][i] or ""
        ab = cols["tb_addr_norm"][i] or ""
        nna = cols["qa_name_num"][i] or ""
        nnb = cols["tb_name_num"][i] or ""
        ana = cols["qa_addr_num"][i] or ""
        anb = cols["tb_addr_num"][i] or ""

        sa = set(ta.split()) if ta else set()
        sb = set(tb.split()) if tb else set()

        out["n_fuzz_ratio"].append(fuzz.ratio(na, nb) / 100.0)
        out["n_fuzz_sort"].append(fuzz.token_sort_ratio(na, nb) / 100.0)
        out["n_fuzz_set"].append(fuzz.token_set_ratio(na, nb) / 100.0)
        out["n_jaccard"].append(_jaccard(sa, sb))
        out["n_common_tokens"].append(float(len(sa & sb)))
        out["a_fuzz_ratio"].append(fuzz.ratio(aa, ab) / 100.0 if (aa or ab) else 0.0)
        out["a_jaccard"].append(
            _jaccard(set(aa.split()) if aa else set(), set(ab.split()) if ab else set())
            if (aa or ab)
            else 0.0
        )
        out["n_num_jaccard"].append(
            _jaccard(set(nna.split(",")) - {""}, set(nnb.split(",")) - {""})
        )
        out["a_num_jaccard"].append(
            _jaccard(set(ana.split(",")) - {""}, set(anb.split(",")) - {""})
            if (ana or anb)
            else 0.0
        )

    return pl.DataFrame(out, schema=FUZZ_FIELDS).select(
        pl.struct([f for f, _ in FUZZ_FIELDS])
    ).to_series()


def pair_features(
    cand: pl.DataFrame,
    q: pl.DataFrame,
    t: pl.DataFrame,
    tfidf_k: int = 50,
) -> pl.DataFrame:
    """Attach all features to a candidate frame.

    ``cand``: ``sid, tid`` + retrieval-method flags.
    ``q``: normalized Source-1 rows keyed by ``sid``.
    ``t``: normalized target rows keyed by ``tid``.
    """
    qcols = [
        "name_norm", "name_ss", "name_tok", "name_num", "name_nchar",
        "addr_norm", "addr_tok", "addr_num", "addr_nchar",
        "street_num", "unit_num", "bldg_num", "postal", "has_addr",
    ]
    qq = q.select(["sid"] + qcols).rename({c: f"qa_{c}" for c in qcols})
    tt = t.select(["tid"] + qcols).rename({c: f"tb_{c}" for c in qcols})

    df = cand.join(qq, on="sid", how="inner").join(tt, on="tid", how="inner")

    both = lambda c: (pl.col(f"qa_{c}") != "") & (pl.col(f"tb_{c}") != "")
    either_missing = (pl.col("qa_addr_norm") == "") | (pl.col("tb_addr_norm") == "")
    both_missing = (pl.col("qa_addr_norm") == "") & (pl.col("tb_addr_norm") == "")

    scalar = [
        (pl.col("qa_name_norm") == pl.col("tb_name_norm")).cast(pl.Float32).alias("n_exact"),
        (pl.col("qa_name_ss") == pl.col("tb_name_ss")).cast(pl.Float32).alias("n_exact_ss"),
        (
            (pl.col("qa_name_nchar") - pl.col("tb_name_nchar")).abs()
            / pl.max_horizontal(pl.col("qa_name_nchar"), pl.col("tb_name_nchar")).clip(1, None)
        ).cast(pl.Float32).alias("n_len_diff"),
        (
            pl.min_horizontal(pl.col("qa_name_nchar"), pl.col("tb_name_nchar"))
            / pl.max_horizontal(pl.col("qa_name_nchar"), pl.col("tb_name_nchar")).clip(1, None)
        ).cast(pl.Float32).alias("n_nchar_ratio"),
        (both("street_num") & (pl.col("qa_street_num") == pl.col("tb_street_num")))
        .cast(pl.Float32).alias("a_street_match"),
        (both("street_num") & (pl.col("qa_street_num") != pl.col("tb_street_num")))
        .cast(pl.Float32).alias("a_street_conflict"),
        (both("unit_num") & (pl.col("qa_unit_num") == pl.col("tb_unit_num")))
        .cast(pl.Float32).alias("a_unit_match"),
        (both("unit_num") & (pl.col("qa_unit_num") != pl.col("tb_unit_num")))
        .cast(pl.Float32).alias("a_unit_conflict"),
        (both("postal") & (pl.col("qa_postal") == pl.col("tb_postal")))
        .cast(pl.Float32).alias("a_postal_match"),
        (both("postal") & (pl.col("qa_postal") != pl.col("tb_postal")))
        .cast(pl.Float32).alias("a_postal_conflict"),
        either_missing.cast(pl.Float32).alias("a_missing_either"),
        both_missing.cast(pl.Float32).alias("a_missing_both"),
        (
            (pl.col("qa_name_norm") == pl.col("tb_name_norm"))
            & (pl.col("a_num_jaccard") > 0)
        ).cast(pl.Float32).alias("x_exact_name_addr"),
        (
            (pl.col("qa_name_norm") == pl.col("tb_name_norm"))
            & (pl.col("qa_street_num") != "")
            & (pl.col("qa_street_num") == pl.col("tb_street_num"))
        ).cast(pl.Float32).alias("x_exact_name_street"),
        (
            (pl.col("n_rare_shared") > 0)
            & (pl.col("qa_street_num") != "")
            & (pl.col("qa_street_num") == pl.col("tb_street_num"))
        ).cast(pl.Float32).alias("x_rare_street"),
        (
            pl.when(pl.col("tfidf_rank") > 0)
            .then((tfidf_k + 1 - pl.col("tfidf_rank").cast(pl.Float32)) / (tfidf_k + 1))
            .otherwise(0.0)
        ).alias("r_tfidf_rank_norm"),
        pl.col("tfidf_score").fill_null(0.0).cast(pl.Float32).alias("r_tfidf_score"),
        (
            pl.col("e_name").cast(pl.Int8)
            + pl.col("e_ss").cast(pl.Int8)
            + pl.col("rare").cast(pl.Int8)
            + pl.col("addr").cast(pl.Int8)
            + pl.col("tfidf").cast(pl.Int8)
        ).cast(pl.Float32).alias("r_n_methods"),
        pl.col("qa_name_nchar").cast(pl.Float32).alias("q_name_nchar"),
        pl.col("tb_name_nchar").cast(pl.Float32).alias("t_name_nchar"),
        pl.col("qa_addr_nchar").cast(pl.Float32).alias("q_addr_nchar"),
        pl.col("tb_addr_nchar").cast(pl.Float32).alias("t_addr_nchar"),
    ]

    fuzz_expr = pl.struct(INPUT_FIELDS).map_batches(
        _score_batch, return_dtype=FUZZ_STRUCT
    ).alias("__f")
    df = df.with_columns(fuzz_expr).unnest("__f")
    for name, dtype in FUZZ_FIELDS:
        df = df.with_columns(pl.col(name).cast(dtype))

    df = df.with_columns(scalar)

    keep = [c for c in cand.columns if c in df.columns]
    return df.select(list(dict.fromkeys(keep + FEATURE_NAMES)))
