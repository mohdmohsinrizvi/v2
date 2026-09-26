#!/usr/bin/env python3
"""Stage 07 — build labelled training pairs (positives + sampled hard negatives).

Positives   : every official ground-truth link for this split (retrieved or not —
              training should see the true distribution, not only what survives
              the retriever).
Negatives   : candidate pairs that are NOT ground truth. They are hard negatives
              by construction: TF-IDF / rare-token / address retrieval already
              judged them plausible. Sampled deterministically per Source-1 row at
              ``per_pos`` negatives per true match (capped by what exists).

Candidates are consumed one shard at a time (train is ~214M pairs). Because
shards are disjoint in ``sid``, the per-S1 negative quota is computed on the
same rows the eager version would have seen, so the sample is identical.

Output:
    artifacts/train_pairs/{split}/pairs.parquet   sid, tid, label, is_val

Usage:
    .venv/bin/python scripts/07_build_train_pairs.py --split dev --per-pos 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import paths as P
from src.pipeline import split as SPLIT
from src.pipeline.blocking import BOOL_METHODS
from src.pipeline.gt import candidate_shards, ground_truth_pairs, s1_index

# (value, dtype) — dtypes must match the candidate shard schema so that the
# vertical concat of positives and negatives stays strict.
PROV_DEFAULTS = {
    "e_name": (False, pl.Boolean), "e_ss": (False, pl.Boolean),
    "rare": (False, pl.Boolean), "addr": (False, pl.Boolean),
    "tfidf": (False, pl.Boolean),
    "n_rare_shared": (0, pl.Int16), "tfidf_rank": (0, pl.Int16),
    "tfidf_score": (0.0, pl.Float32),
}


def sample_negatives(neg: pl.DataFrame, n_pos_per_sid: pl.DataFrame,
                     per_pos: int, seed: int) -> pl.DataFrame:
    """Deterministic per-S1 negative quota — same maths as the eager version."""
    if neg.height == 0:
        return neg
    neg = neg.join(n_pos_per_sid, on="sid", how="left").with_columns(
        pl.col("n_pos").fill_null(0)
    )
    rnd = (
        pl.col("sid").cast(pl.UInt64) * pl.lit(2_654_435_761, dtype=pl.UInt64)
        + pl.col("tid").cast(pl.UInt64) * pl.lit(40_503, dtype=pl.UInt64)
        + pl.lit(seed, dtype=pl.UInt64)
    ) % 1_000_000_007
    neg = neg.with_columns(rnd.alias("__rnd")).sort("sid", "__rnd")
    neg = neg.with_columns(pl.int_range(pl.len()).over("sid").alias("__r"))
    avail = neg.select(["sid", "__r"]).with_columns(pl.len().over("sid").alias("__n"))
    neg = neg.join(avail, on=["sid", "__r"], how="left")
    quota = (
        (pl.col("n_pos") * per_pos)
        .clip(upper_bound=pl.col("__n"))
        .fill_null(0)
        .cast(pl.Int64)
    )
    return neg.filter(pl.col("__r") < quota).drop(["__rnd", "__r", "__n", "n_pos"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "dev"], required=True)
    ap.add_argument("--per-pos", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.10)
    args = ap.parse_args()

    t0 = time.time()
    gt = ground_truth_pairs(args.split)
    n_pos_per_sid = gt.group_by("sid").agg(pl.len().alias("n_pos"))
    prov = ["n_rare_shared"] + BOOL_METHODS + ["tfidf_rank", "tfidf_score"]
    print(f"[07] positives={gt.height}", flush=True)

    pos_parts: list[pl.DataFrame] = []
    neg_parts: list[pl.DataFrame] = []
    n_pos_in_cand = 0
    n_cand = 0

    files = candidate_shards(args.split)
    for i, f in enumerate(files):
        cand = pl.read_parquet(f, columns=["sid", "tid"] + prov)
        n_cand += cand.height

        matched = cand.join(gt, on=["sid", "tid"], how="inner")
        if matched.height:
            matched = matched.unique()
            n_pos_in_cand += matched.height
            pos_parts.append(matched)

        neg = cand.join(gt, on=["sid", "tid"], how="anti")
        del cand
        neg = sample_negatives(neg, n_pos_per_sid, args.per_pos, args.seed)
        if neg.height:
            neg_parts.append(neg.select(["sid", "tid"] + prov))

        if (i + 1) % 5 == 0 or i + 1 == len(files):
            print(f"  shard {i + 1}/{len(files)}: pos={n_pos_in_cand} "
                  f"neg={sum(x.height for x in neg_parts)}", flush=True)
        del matched, neg

    print(f"[07] candidates={n_cand}", flush=True)
    print(f"[07] positives that survived retrieval: {n_pos_in_cand} "
          f"({n_pos_in_cand / max(gt.height, 1):.4f})", flush=True)

    matched_all = pl.concat(pos_parts, how="vertical") if pos_parts else None
    pos_parts.clear()

    # every ground-truth link is a positive, retrieved or not
    if matched_all is not None:
        pos = gt.join(matched_all.select(["sid", "tid"] + prov),
                      on=["sid", "tid"], how="left")
        del matched_all
    else:
        pos = gt
    for c, (v, dt) in PROV_DEFAULTS.items():
        if c not in pos.columns:
            pos = pos.with_columns(pl.lit(v, dtype=dt).alias(c))
    pos = pos.with_columns(
        [pl.col(c).fill_null(v) for c, (v, _) in PROV_DEFAULTS.items()]
    ).with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))

    if neg_parts:
        neg = pl.concat(neg_parts, how="vertical")
        neg = neg.select(["sid", "tid"] + prov).with_columns(
            pl.lit(0, dtype=pl.Int8).alias("label")
        )
    else:
        neg = pos.filter(pl.lit(False))  # empty, but same schema
    neg_parts.clear()

    pairs = pl.concat([pos, neg], how="vertical")
    n_pos_out, n_neg_out = pos.height, neg.height
    del pos, neg
    pairs = pairs.unique(subset=["sid", "tid"], keep="first")

    flags = SPLIT.attach(SPLIT.assign(s1_index(args.split)["entity_id"], args.val_frac))
    pairs = pairs.join(flags, on="sid", how="left").with_columns(
        pl.col("is_val").fill_null(False)
    )

    out_dir = P.split_dir("train_pairs", args.split)
    P.ensure(out_dir)
    out = out_dir / "pairs.parquet"
    pairs.write_parquet(out)

    summary = {
        "split": args.split,
        "positives": int(n_pos_out),
        "negatives": int(n_neg_out),
        "total": int(pairs.height),
        "pos_rate": round(n_pos_out / max(pairs.height, 1), 5),
        "positives_in_candidates": int(n_pos_in_cand),
        "per_pos": args.per_pos,
        "seed": args.seed,
        **SPLIT.summary(s1_index(args.split), args.val_frac),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    P.write_json(out_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
