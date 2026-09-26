# Data Audit (Read-Only, STAGE 0 reconnaissance)

Date: 2026-09-25 · Location: `/home/mohsin/Desktop/amazon/student_resource/dataset/`
Method: header reads + streaming `awk` scans (no file modified, no file rewritten).

## Files, sizes, row counts

| File | Bytes | Data rows |
|------|------:|----------:|
| train/train_source1.tsv | 210,069,713 | 2,206,821 |
| train/train_source2.tsv | 489,301,488 | 5,034,616 |
| train/train_source3.tsv | 503,705,637 | 5,285,603 |
| train/train_ground_truth.tsv | 127,015,583 | 2,206,821 |
| test/test_source1.tsv | 175,022,086 | 1,732,544 |
| test/test_source2.tsv | 509,456,422 | 4,887,273 |
| test/test_source3.tsv | 506,002,772 | 5,082,316 |
| **Total dataset** | **2,520,603,693 B (2.52 GB)** | — |

**Measured total is 2.52 GB — NOT ~10 GB as assumed in the task brief.** All size
planning below uses the measured value.

## Schema (all source files)

`entity_id` (str, `S1-`/`S2-`/`S3-` prefix) · `business_name` (str) · `business_address`
(str) · `country` (str label). Tab-separated, UTF-8, header present.

Ground truth: `source1_entity_id` · `matched_entity_ids` (comma-separated, may be empty).

## Integrity checks

| Check | train S1 | train S2 | train S3 | test S1 |
|-------|---------:|---------:|---------:|--------:|
| empty `business_name` | 0 | 0 | 0 | 0 |
| empty `business_address` | 0 | **168,967** (3.36%) | **175,916** (3.33%) | 0 |
| empty `country` | 0 | 0 | 0 | 0 |
| duplicate `entity_id` | 0 | 0 | 0 | 0 |

## Countries

| Set | US | India | France |
|-----|---:|------:|-------:|
| train S1 | 1,323,633 | 883,188 | 0 |
| test S1 | 663,106 | 809,986 | **259,452** |

France exists **only** in test → no hard-coded country logic, no country one-hots.

## Ground-truth structure

| Metric | Value |
|--------|------:|
| S1 entities with a ground-truth row | 2,206,821 |
| Matched S1 | 2,083,574 (94.42%) |
| Singletons (empty list) | **123,247 (5.58%)** |
| Total positive links | 7,638,365 |
| Mean links per matched S1 | 3.461 |
| Distribution (#links → #S1) | 1→119,157 · 2→375,212 · 3→530,841 · 4→484,115 · 5→321,957 · 6→164,868 · 7→63,968 · 8→18,680 · 9→4,205 · 10→534 · 11→37 |

Max observed fan-out = 11 → **never return Top-1 only.**

## Sample rows

```
S1-925783039  Orelee's Barbershop        1795 Westchester Drive, High Point, NC   US
S1-755362802  Prabhav Business Center    797, Lake Town Block A, Kolkata, ...     India
S1-156285671  << Team Ecole              175 Boulevard du Président ... Bordeaux  France   (test)
S1-8698171... wilfordhancock.com         Mack Rd, Haltom City, Texas              US       (train S3)
```

Note: names contain leading junk (`<< `), accents, Devanagari (train S2 row 2),
domain-style names, legal suffixes; addresses contain reordered components and
component-missing rows. Normalisation must be Unicode-safe and lossless on raw text.

## Submission shape derived from test

- `matching_results.tsv` = 1,732,544 data rows + 1 header.
- `candidate_pairs.tsv` = same row count; empty list allowed.
- At ~50 candidates/row × ~12 chars → candidate file alone is roughly **0.9–1.2 GB**.
  **Disk budget warning:** only **16 GB free** on `/`. Intermediate Parquet + candidates +
  features must be sharded and cleaned; do not duplicate raw data.

## Not yet verified (deferred to STAGE 2, still read-only)

- Same-country consistency of ground-truth links (planned: id→country maps, streamed).
- Duplicate/near-duplicate names inside a single source.
- Distribution of postal/PIN tokens per country.
- Devanagari / non-Latin token share (visible already in train S2).

## Local compute available for this audit

7.6 GB RAM (2.4 GB currently available) · 4 vCPU · 16 GB free disk · Python 3.12.3
Missing packages: `polars`, `pyarrow`, `lightgbm`, `rapidfuzz`, `boto3`, `tqdm`
(installable from PyPI; network verified working). Present: numpy 2.4.3, scipy 1.17.1,
scikit-learn 1.8.0, pandas 3.0.1, joblib 1.5.3, psutil 5.9.8.

---

## STAGE 2 ADDENDUM — deferred checks completed (2026-09-25 15:14)

Method: streamed Polars joins over `entity_id`/`country` only, plus normalization
pass over train S1. Runtime 155 s locally. Raw files unmodified.
Artifact: `artifacts/audit_gt_country.json`.

### 1. Country consistency of ground-truth links

| Check | Value |
|---|---:|
| Exploded link rows (incl. singleton `""` artifacts) | 7,761,612 |
| — of which bogus `""` from 123,247 singletons | 123,247 |
| **Real positive links** | **7,638,365** ✅ (matches the distribution-derived total) |
| Links resolving to Source 2 | 3,693,619 |
| Links resolving to Source 3 | 3,944,746 |
| **S1/S2 country mismatch** | **0** |
| **S1/S3 country mismatch** | **0** |
| Links whose target ID was not found | 0 (bogus `""` only) |

> **Design consequence: country is a SAFE blocking key.** Every one of the 7.6M real
> ground-truth links joins two records with identical `country`. Blocking on
> `country` therefore costs **zero** recall on train and cuts every index by ~3×
> (US / India / France partitions). Still no country-specific *rules* — France is
> test-only, so `country` is used purely as a grouping key, never as logic input.

### 2. Name collision profile (train S1, normalized)

| Representation | S1 rows sharing it with ≥1 other S1 |
|---|---:|
| `name_norm` (exact normalized) | 684,655 (31.0%) |
| `name_suffix_stripped` | 942,758 (42.7%) |

Largest `name_norm` clusters: `primary care group` 253 · `ear nose and throat group` 251 ·
`pediatric group` 222 · `womens health group` 220 · `physical therapy group` 218.

> **Design consequence:** exact-name lookup alone cannot be trusted as a *matcher* —
> clusters of 200+ exist — so exact-name matches are **candidate generators only**, and
> cluster size must be capped per query to stop a generic name from flooding the
> candidate set (and to protect the precision-heavy F0.5).

### 3. Singleton link artifact (pipeline bug caught by this audit)

`matched_entity_ids == ""` splits to `[""]`, not `[]`. Any pipeline that splits without
filtering empties will create 123,247 phantom positives in train and 1 phantom
"prediction" per singleton at inference. **`src/utils/io.py` and every consumer must
drop empty strings after `str.split`.**

### 4. Row counts re-verified this phase

`train_source1` 2,206,822 lines · `train_ground_truth` 2,206,822 lines ·
`test_source1` 1,732,545 lines (headers included) — identical to STAGE 0 figures.
