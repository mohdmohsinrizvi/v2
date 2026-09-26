#!/usr/bin/env python3
"""Build a small, representative development sample (STAGE 4).

Outputs under artifacts/dev/:
  dev_s1.parquet, dev_s2.parquet, dev_s3.parquet, dev_gt.parquet, dev_meta.json

Sample contains: positives, singletons, multi-match entities, all train countries.
Corpus contains: every ground-truth match of the sampled S1 (so positives are
recoverable) + a large random negative pool from S2/S3.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.io import ARTIFACTS, ensure_dirs, read_ground_truth, read_train_source, write_parquet

SEED = 42
N_MATCHED = 6000
N_SINGLETON = 2000
N_MULTI = 2000
N_RANDOM_NEG_S2 = 120000
N_RANDOM_NEG_S3 = 120000


def main() -> None:
    ensure_dirs()
    t0 = time.time()
    print("[dev-sample] loading ground truth + source1", flush=True)
    gt = read_ground_truth()
    s1 = read_train_source(1)

    gt = gt.with_columns(
        pl.col("matched_entity_ids").str.split(",").alias("matches"),
    ).with_columns(
        pl.col("matches").list.eval(pl.element().filter(pl.element() != "")).alias("matches"),
    ).with_columns(
        pl.col("matches").list.len().alias("n_matches"),
    )

    matched = gt.filter(pl.col("n_matches") > 0)
    singleton = gt.filter(pl.col("n_matches") == 0)
    multi = gt.filter(pl.col("n_matches") >= 5)

    part_a = matched.sample(n=N_MATCHED, seed=SEED, with_replacement=False)
    part_b = singleton.sample(n=N_SINGLETON, seed=SEED, with_replacement=False)
    part_c = multi.sample(n=N_MULTI, seed=SEED, with_replacement=False)
    sample_gt = pl.concat([part_a, part_b, part_c]).unique(subset=["source1_entity_id"])

    sample_ids = sample_gt["source1_entity_id"].to_list()
    s1_dev = s1.filter(pl.col("entity_id").is_in(sample_ids))

    pos_ids = {i for row in sample_gt["matches"].to_list() for i in row}
    pos_s2 = {i for i in pos_ids if i.startswith("S2-")}
    pos_s3 = {i for i in pos_ids if i.startswith("S3-")}
    print(f"[dev-sample] s1={len(sample_ids)} positives={len(pos_ids)} "
          f"(S2 {len(pos_s2)} / S3 {len(pos_s3)})", flush=True)

    def build_source(name: int, n_random: int, pos: set[str]) -> pl.DataFrame:
        df = read_train_source(name)
        rand = df.sample(n=n_random, seed=SEED, with_replacement=False)
        keep = df.filter(pl.col("entity_id").is_in(list(pos))) if pos else df.head(0)
        out = pl.concat([keep, rand]).unique(subset=["entity_id"])
        print(f"[dev-sample] S{name}: kept={len(out)} (positives {len(keep)} + random {n_random})", flush=True)
        return out

    s2_dev = build_source(2, N_RANDOM_NEG_S2, pos_s2)
    s3_dev = build_source(3, N_RANDOM_NEG_S3, pos_s3)

    have = set(s2_dev["entity_id"].to_list()) | set(s3_dev["entity_id"].to_list())
    missing_pos = sorted(pos_ids - have)
    if missing_pos:
        print(f"[dev-sample] WARNING missing {len(missing_pos)} positive corpus rows "
              f"(sampled out of random pool) - fetching explicitly", flush=True)
        for name, ids in ((2, [i for i in missing_pos if i.startswith("S2-")]),
                          (3, [i for i in missing_pos if i.startswith("S3-")])):
            if ids:
                extra = read_train_source(name).filter(pl.col("entity_id").is_in(ids))
                if name == 2:
                    s2_dev = pl.concat([s2_dev, extra]).unique(subset=["entity_id"])
                else:
                    s3_dev = pl.concat([s3_dev, extra]).unique(subset=["entity_id"])

    write_parquet(s1_dev, ARTIFACTS / "dev" / "dev_s1.parquet")
    write_parquet(s2_dev, ARTIFACTS / "dev" / "dev_s2.parquet")
    write_parquet(s3_dev, ARTIFACTS / "dev" / "dev_s3.parquet")
    write_parquet(sample_gt.drop("matches").with_columns(
        pl.col("matched_entity_ids")), ARTIFACTS / "dev" / "dev_gt.parquet")

    dist = (sample_gt.group_by("n_matches").len().sort("n_matches")
            .with_columns((pl.col("n_matches") == 0).alias("singleton")).head(20))
    meta = {
        "seed": SEED, "s1_rows": len(s1_dev), "s2_rows": len(s2_dev), "s3_rows": len(s3_dev),
        "singletons_in_sample": int(sample_gt.filter(pl.col("n_matches") == 0).height),
        "matched_in_sample": int(sample_gt.filter(pl.col("n_matches") > 0).height),
        "positive_links_in_sample": int(sum(len(x) for x in sample_gt["matches"].to_list())),
        "n_matches_distribution": {str(r["n_matches"]): r["len"] for r in dist.to_dicts()},
        "countries": {r["country"]: r["len"] for r in s1_dev.group_by("country").len().to_dicts()},
        "elapsed_sec": round(time.time() - t0, 1),
        "note": "corpus-level recall measured on this sample is a MECHANICS check, not the full-data recall (Stage 7).",
    }
    (ARTIFACTS / "dev" / "dev_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
