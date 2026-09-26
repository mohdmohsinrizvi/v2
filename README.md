# v2 — Amazon ML Challenge 2026 (Entity Resolution)

**Repository:** `git@github.com:mohdmohsinrizvi/v2.git` (private) · **Branch:** `main`
**Language:** Python 3 · **Stack:** Polars, LightGBM, RapidFuzz, scikit-learn, boto3
**Status:** Test-set submission produced and validated (2026-09-26 06:22 UTC)

This repository contains the complete end-to-end pipeline that takes the raw
competition TSV files and produces a submission-ready `matching_results.tsv`
plus `candidate_pairs.tsv`. It also contains every report, plan, cost ledger and
regression test accumulated while building it.

---

## Table of contents

1. [What the challenge is](#1-what-the-challenge-is)
2. [The metric](#2-the-metric)
3. [Dataset](#3-dataset)
4. [Repository layout](#4-repository-layout)
5. [Pipeline stages](#5-pipeline-stages)
6. [Driver scripts](#6-driver-scripts)
7. [Library modules (`src/`)](#7-library-modules-src)
8. [Retrieval / blocking design](#8-retrieval--blocking-design)
9. [Feature set](#9-feature-set)
10. [Model and threshold](#10-model-and-threshold)
11. [Configuration files](#11-configuration-files)
12. [Tests](#12-tests)
13. [Environment and setup](#13-environment-and-setup)
14. [How to run it](#14-how-to-run-it)
15. [Measured results](#15-measured-results)
16. [Final test-set submission](#16-final-test-set-submission)
17. [Two-instance execution design](#17-two-instance-execution-design)
18. [AWS infrastructure and cost](#18-aws-infrastructure-and-cost)
19. [S3 layout and status markers](#19-s3-layout-and-status-markers)
20. [Resumability and safety guards](#20-resumability-and-safety-guards)
21. [Bugs found and fixed](#21-bugs-found-and-fixed)
22. [Reports index](#22-reports-index)
23. [Git / team workflow](#23-git--team-workflow)
24. [Open questions and TODO](#24-open-questions-and-todo)

---

## 1. What the challenge is

Given three entity tables, link every **Source-1** business to the businesses in
**Source-2** and **Source-3** that describe the *same real-world business*.

| Table | Role | Train rows | Test rows |
|---|---|---:|---:|
| Source-1 (`S1-…`) | entities to resolve (query side) | 2,206,821 | 1,732,544 |
| Source-2 (`S2-…`) | candidate match targets | 5,034,616 | 4,887,273 |
| Source-3 (`S3-…`) | candidate match targets | 5,285,603 | 5,082,316 |
| Ground truth | `source1_entity_id → matched_entity_ids` | 2,206,821 | *(hidden)* |

Schema for all sources (tab-separated UTF-8, header present):

```
entity_id    business_name    business_address    country
```

Ground truth schema:

```
source1_entity_id    matched_entity_ids        # comma-separated, may be empty
```

Key domain facts established in `reports/data_audit.md`:

* Names contain legal suffixes, accents, Devanagari, leading junk (`<< `),
  domain-style names and reordered address components — hence heavy normalization.
* Addresses are frequently empty on the target side (**3.36 %** of S2,
  **3.33 %** of S3).
* **No ground-truth link crosses countries** (0 of 7,638,365), so `country` is a
  safe blocking key.
* **France appears only in the test set** (259,452 test S1 rows) — so nothing may
  be hard-coded per country and no country one-hots may be used.
* Max observed fan-out is **11** matches per S1 → the system must never return
  Top-1 only.
* 5.58 % of train S1 are **singletons** (no match); predicting anything for them
  costs full marks.

---

## 2. The metric

**Entity-level Macro F0.5**, averaged over **all** Source-1 entities:

```
F0.5 = (1 + 0.25) · P · R / (0.25 · P + R)
```

Per entity:

| Truth | Prediction | Score |
|---|---|---:|
| empty (singleton) | empty | **1.0** |
| empty (singleton) | anything | **0.0** |
| non-empty | empty | 0.0 |
| non-empty | partial | `1.25·P·R / (0.25·P + R)` for that entity |

Because each S1 is one unit, and every prediction for an S1 is decided jointly:

* a fixed threshold of 0.5 is almost never optimal — hence stage 11 sweeps it;
* the split unit for train/validation **must** be the Source-1 row, never the pair
  (`src/pipeline/split.py` hashes `entity_id`, 10 % to validation, stable across runs);
* precision is weighted 4× harder than recall (β = 0.5), but singletons dominate
  the average, so over-prediction is punished from two directions.

Implementation: `src/evaluation/metric.py` — `f05`, `entity_f05`, `macro_f05`,
`pair_metrics`.

---

## 3. Dataset

Measured total: **2,520,603,693 bytes (2.52 GB)** — not the ~10 GB assumed in the
task brief. Full audit in `reports/data_audit.md`.

| File | Bytes | Rows |
|---|---:|---:|
| `train/train_source1.tsv` | 210,069,713 | 2,206,821 |
| `train/train_source2.tsv` | 489,301,488 | 5,034,616 |
| `train/train_source3.tsv` | 503,705,637 | 5,285,603 |
| `train/train_ground_truth.tsv` | 127,015,583 | 2,206,821 |
| `test/test_source1.tsv` | 175,022,086 | 1,732,544 |
| `test/test_source2.tsv` | 509,456,422 | 4,887,273 |
| `test/test_source3.tsv` | 506,002,772 | 5,082,316 |

Ground-truth structure (train):

| Metric | Value |
|---|---:|
| Matched S1 | 2,083,574 (94.42 %) |
| Singletons | 123,247 (5.58 %) |
| Total positive links | 7,638,365 |
| Mean links per matched S1 | 3.461 |
| Link-count distribution | 1→119,157 · 2→375,212 · 3→530,841 · 4→484,115 · 5→321,957 · 6→164,868 · 7→63,968 · 8→18,680 · 9→4,205 · 10→534 · 11→37 |

Country distribution:

| Set | US | India | France |
|---|---:|---:|---:|
| train S1 | 1,323,633 | 883,188 | 0 |
| test S1 | 663,106 | 809,986 | 259,452 |

Integrity: zero empty names, zero empty countries, zero duplicate `entity_id`,
zero duplicate ids across tables.

The dataset is **not in this repository** (`.gitignore` excludes `*.tsv`). It
lives in the sibling folder `../student_resource/dataset/` (2.4 GB) and on S3.

---

## 4. Repository layout

```
.
├── .gitignore                 # excludes data, artifacts, submissions, .venv, secrets
├── README.md                  # this file
├── requirements.txt           # pinned dependencies
├── config/                    # AWS + EC2 configuration, no secrets
│   ├── aws.yaml               # bucket, region, tags, budget, cleanup rules
│   ├── ec2_instance_policy.json   # least-privilege S3 policy for EC2 role
│   ├── ec2_trust_policy.json      # trust policy for the instance profile
│   ├── iam_opencode_policy.json   # policy used by the operator identity
│   ├── userdata.sh            # EC2 user data: full single-instance run
│   ├── userdata_h.sh          # EC2 user data: helper (odd-shard) instance
│   └── userdata_test.sh       # EC2 user data: short smoke test
├── logs/                      # small captured stage logs (committed)
├── reports/                   # all documentation, audits, plans, results
├── scripts/                   # pipeline stages 02→14 + drivers 92 / 99
├── src/
│   ├── evaluation/metric.py   # F0.5 metric
│   ├── normalization/text.py  # character/name/address normalization rules
│   ├── pipeline/              # paths, normalize, split, blocking, tfidf,
│   │                          # features, gt
│   └── utils/io.py            # dataset paths + TSV readers
├── tests/                     # 6 test files, 39 tests
├── artifacts/                 # ALL generated data (git-ignored)
├── submissions/               # final TSVs (git-ignored)
├── submissions_dev/           # dev-run TSVs (git-ignored except summary json)
└── .venv/                     # virtualenv (git-ignored)
```

**Not in the repository, by design:** `student_resource/` (official kit +
dataset), `artifacts/` (133 MB → GB of parquet), `submissions/` (6.2 GB),
`.venv/`, and anything matching `.env`, `*.pem`, `credentials`.

**Committed:** 61 files, largest is `reports/aws_cost_ledger.md` at 25 KB.

### Artifact layout (generated at runtime)

```
artifacts/
├── processed/{split}/s1.parquet s2.parquet s3.parquet meta.json
├── processed/{split}/s1_ids.parquet t_ids.parquet
├── indices/{split}/stats.joblib
├── indices/{split}/tfidf/                  # one TF-IDF index per country
├── candidates/{split}/shard_XXXXX.parquet + manifest.json
├── dev/                                    # dev sample (stage 03)
├── train_pairs/{split}/pairs.parquet
├── features/{split}/pairs/                 # labelled training features
├── features/{split}/candidates/            # all candidate features
├── predictions/{split}/shard_XXXXX.parquet
├── model/{split}/model.txt + meta.json + threshold*.json
└── helper/                                 # helper-instance handshake files
```

Every shard file has a sibling `<file>.SUCCESS` JSON marker recording what it
was built from (`src`), its row count, `tfidf_k`, and a timestamp.

---

## 5. Pipeline stages

Each stage is a standalone script, resumable, and safe to re-run. Convention:

```bash
.venv/bin/python scripts/NN_name.py --split <train|dev|test> [options]
```

| # | Script | Purpose |
|---|---|---|
| 02 | `02_normalize.py` | Raw TSV → normalized Parquet, chunked (250k rows) and resumable. Produces `artifacts/processed/{split}/s{n}.parquet` + `meta.json`. Never writes to the source data. |
| 03 | `03_dev_sample.py` | Builds the development sample: 6,000 matched + 2,000 singleton + 2,000 multi-match S1 across all train countries, with a corpus of every ground-truth match of the sampled S1 plus 120k random negatives each from S2/S3. Seed 42. |
| 04 | `04_build_indices.py` | Blocking indices. Three phases (`--phase ids\|stats\|tfidf\|all`) run as **separate processes** so each returns memory to the OS — a single process peaks past 8 GiB. Produces `stats.joblib`, per-country TF-IDF indices, and `sid/tid` id maps. |
| 05 | `05_generate_candidates.py` | Candidate generation, chunked over Source-1 rows (`--chunk`, default 100,000). Writes `candidates/{split}/shard_XXXXX.parquet` with one row per candidate pair **plus retrieval provenance** (which method produced it, `tfidf_rank`, `tfidf_score`). Each shard independently skippable. |
| 06 | `06_candidate_recall.py` | **The gate before training.** Measures pair candidate recall, at-least-one S1 coverage, all-match S1 coverage, per-method recall and TF-IDF-rank bucket recall, shard by shard. Target ≥ 0.97. |
| 07 | `07_build_train_pairs.py` | Labelled training pairs. Positives = every official ground-truth link (retrieved or not). Negatives = candidate pairs that are *not* ground truth — hard by construction — sampled deterministically at `--per-pos` per true match. Output `train_pairs/{split}/pairs.parquet`. |
| 08 | `08_generate_features.py` | Pairwise features, two modes: `--mode pairs` (labelled) and `--mode candidates` (every candidate). Sharded + resumable. Supports `--shard-parity {all,0,1}` so two machines can split the work, and `--tfidf-k`. |
| 09 | `09_train_lgbm.py` | Trains LightGBM on the `is_val == False` fold, early-stops on `is_val`. Saves the model **with the exact feature list and TF-IDF K** so scoring cannot silently use a mismatched schema. |
| 10 | `10_score_candidates.py` | Scores candidate shards → `predictions/{split}/shard_XXXXX.parquet` with `sid, tid, score`. Entity IDs are resolved only at submission time. Supports `--worker-index/--num-workers` fan-out. |
| 11 | `11_threshold_search.py` | Sweeps a threshold grid on the validation fold, records the full curve and the best threshold. |
| 12 | `12_build_submission.py` | Builds `matching_results.tsv` and `candidate_pairs.tsv` with the exact official headers. **Every S1 gets a row**; an empty second column means "no match". Merges the helper instance's scores when running the two-box configuration. |
| 13 | `13_score_submission.py` | Scores a submission with the official metric. **Train/dev only** — there is no test ground truth. |
| 14 | `14_validate_submission.py` | Wraps the official `student_resource/utils/validate_submission.py`. Needs the sibling `student_resource/` folder, so run it on a machine that has it. |

### Conventions every stage follows

* `--split {train,dev,test}` selects the data slice.
* Output shards are `shard_XXXXX.parquet` via `paths.shard_path()`.
* A shard is "done" only when `<file>.SUCCESS` exists (`paths.mark_done()`).
* Names are derived from the **source shard number in the filename**, never from
  `enumerate()` of the local file list — that is what makes two machines able to
  work on disjoint halves without collisions.
* All work is chunked so peak memory stays bounded on an 8 GiB host.

---

## 6. Driver scripts

### `scripts/99_run_full.sh` — single-instance full run

Runs the whole sequence and writes an S3 status marker after every stage:

```
STARTED → 02 train norm → 04 train idx → 05 train cand → 06 recall gate
→ 07 train pairs → 08 pair feats → 08 val-cand feats → 09 model
→ 10 train score → 11 threshold
→ 02 test norm → 04 test idx → 05 test cand → 08 test feats
→ 10 test score → 12 submission
→ upload artifacts/model/predictions/threshold/submissions → ALL_DONE
```

Environment knobs:

| Var | Default | Meaning |
|---|---|---|
| `REPO` | `/opt/amz/code` | repository root on the instance |
| `AMZ_DATA_DIR` | `/opt/amz/data` | where the dataset lives |
| `WORKERS` | 4 | generic parallelism |
| `FEAT_WORKERS` | 1 | feature stages hold S1 + full target per worker → 1 |
| `SCORE_WORKERS` | 4 | scoring holds one shard + the booster → safe to fan out |
| `BUCKET` | `amazon-ml-challenge-2026-357112746764` | status/artifact bucket |

### `scripts/92_helper.sh` — second-instance helper

Runs the **odd** half of the test work while instance A keeps the even half:

* pulls `processed/test` and `indices/test` from S3 (produced by A);
* runs `05` on odd candidate shards, `08` on odd feature shards, `10` on odd
  features;
* asserts coverage at each handoff (e.g. H5b requires the prediction count to
  equal the feature count, otherwise it writes `H_FAIL` and exits non-zero);
* writes `H_PREDS_DONE.txt` to S3 so A's stage 12 knows the merge can proceed.

---

## 7. Library modules (`src/`)

| Module | Responsibility |
|---|---|
| `src/utils/io.py` | `DATA_DIR` (respects `AMZ_DATA_DIR`, else `../student_resource/dataset`), `SOURCE_COLS`, TSV readers, artifact writers. |
| `src/normalization/text.py` | Character-level rules: Unicode/Devanagari handling, legal-suffix list (`pvt`, `ltd`, `llc`, `sarl`, `gmbh`, …) and expansions, address part splitting, digit extraction. |
| `src/pipeline/normalize.py` | Row-level normalization producing the canonical schema: `name_norm`, `name_ss`, `name_tok`, `name_num`, `name_nchar`, `addr_norm`, `addr_tok`, … |
| `src/pipeline/paths.py` | Canonical artifact layout, `shard_path`, `marker`/`is_done`/`mark_done`, JSON helpers, `manifest`, `shard_files`, and `check_feature_k()` (refuses to score features built with a different TF-IDF K). |
| `src/pipeline/split.py` | Deterministic train/validation split by hashing `entity_id` (MD5 → 0..9999, 10 % to validation). Unit is the S1 row. |
| `src/pipeline/blocking.py` | `BlockConfig`, target-side pre-aggregation, `candidate_chunk()`, `merge_rank()`. Never brute-forces S1 × (S2+S3). |
| `src/pipeline/tfidf.py` | Sparse word TF-IDF top-K cosine retrieval, country-partitioned, name+address text. |
| `src/pipeline/features.py` | 32 pairwise features; Polars expressions for the cheap part, one batched Python UDF for RapidFuzz similarities. Never densifies a matrix. |
| `src/pipeline/gt.py` | Ground-truth pair construction shared by measurement, training and inference (`id_to_index`, `s1_index`, `s1_ids`, `target_ids`, `ground_truth_pairs`). |
| `src/evaluation/metric.py` | `f05`, `entity_f05`, `macro_f05`, `pair_metrics`. |

---

## 8. Retrieval / blocking design

Five boolean retrieval signals, one column per method (from
`src/pipeline/blocking.py`):

| Column | Meaning |
|---|---|
| `e_name` | exact normalized business name |
| `e_ss` | exact suffix-stripped name |
| `rare` | rare-name-token inverted index |
| `addr` | important address-number match |
| `tfidf` | word TF-IDF top-K retrieval (also emits `tfidf_rank`, `tfidf_score`) |

`BlockConfig` defaults: `rare_max_df=10`, `rare_tokens_per_q=6`,
`addr_max_df=50`, `max_cluster=500`, `tfidf_k=50`.

Two design choices dominate measured recall (see `reports/data_audit.md`):

1. **Country blocking** — each country gets its own index and its own K slots.
   Without it, foreign rows steal the K slots and pair recall drops from ~0.98
   to ~0.85.
2. **Name + address text together** — indexing `name_tok + addr_tok` lifts pair
   recall well above name-only indexing.

Why not `sklearn.NearestNeighbors`: its brute-force path materializes
`chunk × n_target` floats, which explodes at ~10M targets. Here both matrices are
L2-normalized **sparse**, so `Q @ T.T` *is* the cosine similarity and stays
sparse; top-K is read from each row's non-zeros. Vocabulary is restricted to
`min_df ≤ df ≤ max_df` on the target side so posting lists stay bounded.

---

## 9. Feature set

32 features total (`src/pipeline/features.py`).

**Scalar / retrieval features (23)**

```
n_exact  n_exact_ss  n_len_diff  n_nchar_ratio
a_street_match  a_street_conflict  a_unit_match  a_unit_conflict
a_postal_match  a_postal_conflict  a_missing_either  a_missing_both
x_exact_name_addr  x_exact_name_street  x_rare_street
r_tfidf_rank_norm  r_tfidf_score  r_n_methods  n_rare_shared
q_name_nchar  t_name_nchar  q_addr_nchar  t_addr_nchar
```

**RapidFuzz / set-similarity features (9)**

```
n_fuzz_ratio  n_fuzz_sort  n_fuzz_set  n_jaccard  n_common_tokens
a_fuzz_ratio  a_jaccard  n_num_jaccard  a_num_jaccard
```

Prefix convention: `n_` = name, `a_` = address, `r_` = retrieval, `x_` = cross,
`q_`/`t_` = query side / target side.

`r_tfidf_rank_norm` is normalized by `tfidf_k`; `paths.check_feature_k()`
refuses to score features whose recorded K differs from the model's.

---

## 10. Model and threshold

* **Model:** LightGBM (`lightgbm==4.7.0`), trained in stage 09 with early
  stopping on the validation fold.
* **Saved with:** the exact feature list and the TF-IDF K used to build the
  features, so stage 10 cannot score against a mismatched schema.
* **Threshold:** stage 11 sweeps a grid on the validation fold only.

Dev-fold sweep (`reports/threshold_dev.json`, 110,619 scored pairs,
1,055 validation S1):

| Threshold | Macro F0.5 | Avg predictions / S1 |
|---:|---:|---:|
| 0.50 | 0.9170 | 3.50 |
| 0.70 | 0.9370 | 3.42 |
| 0.90 | 0.9583 | 3.32 |
| 0.95 | 0.9647 | 3.26 |
| 0.96 | 0.9675 | 3.25 |
| 0.97 | 0.9697 | 3.22 |
| **0.98** | **0.9707** ← best | 3.19 |
| 0.99 | 0.9697 | 3.12 |
| 1.00 | 0.1991 | 0.00 |

The curve is monotone up to 0.98 and then collapses — the score is dominated by
singletons, which only a near-1.0 threshold keeps correctly empty.

---

## 11. Configuration files

| File | Contents |
|---|---|
| `config/aws.yaml` | Project name, account id, region (`us-east-1`), S3 bucket + prefix layout + encryption, resource tags, MCP endpoint, budget notes, cleanup rules. **No secrets** — credentials live only in the AWS credential chain. |
| `config/ec2_instance_policy.json` | Least-privilege: `s3:ListBucket` on the bucket and `GetObject`/`PutObject` on `bucket/*`. Nothing else. |
| `config/ec2_trust_policy.json` | Trust policy for the EC2 instance profile. |
| `config/iam_opencode_policy.json` | Policy attached to the operator identity. |
| `config/userdata.sh` | One-shot full run: swap file, env, code pull, venv, `99_run_full.sh`, then an `EXIT` trap that uploads the console log, logs, reports and `USERDATA_EXIT.txt` before self-shutdown. |
| `config/userdata_h.sh` | Helper-instance variant: same bootstrap, then `92_helper.sh`. |
| `config/userdata_test.sh` | Short smoke test used to validate the bootstrap before committing to a full run. |

Credentials and key files are excluded by `.gitignore`: `.env`, `*.pem`, `.aws/`,
`credentials`, `config/credentials*`.

---

## 12. Tests

39 tests across 6 files — run with:

```bash
.venv/bin/python -m pytest -q     # 39 passed in ~12 s
```

| File | What it protects |
|---|---|
| `test_metric.py` | F0.5 formula, per-entity and macro behaviour, singleton rules. |
| `test_normalization.py` | Name/address normalization rules (suffixes, accents, Devanagari, junk prefixes). |
| `test_normalize_tsv.py` | The raw-TSV branch of stage 02. Written after Polars 1.44 dropped `scan_csv(columns=…)`, which killed stage 02 on the instance while every local test stayed green. |
| `test_tfidf.py` | TF-IDF retriever behaviour and save/load parity — catches memory optimizations that silently change retrieval. |
| `test_feature_k_guard.py` | Guards added after the train recall-gate failure (0.8238 < 0.97): a features shard built with one `tfidf_k` must not be scored as if built with another. |
| `test_submission_grouping.py` | Stage 12 must emit exactly one line per S1. Score files are per *piece*, and a piece is a slice of rows not of sids, so grouping the full piece list per candidate shard before `group_by` is mandatory; also asserts the loud `sid < ptr` overlap guard. |

---

## 13. Environment and setup

**Dependencies** (`requirements.txt`, pinned):

```
boto3==1.43.102   joblib==1.6.0    lightgbm==4.7.0   numpy==2.5.3
pandas==3.0.6     polars==1.44.2   psutil==7.2.2     pyarrow==25.0.1
PyYAML==6.0.3     RapidFuzz==3.14.6 scikit-learn==1.9.1 scipy==1.18.1
tqdm==4.70.1
```

**Local setup:**

```bash
git clone git@github.com:mohdmohsinrizvi/v2.git
cd v2
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**Dataset:** place the official kit next to the repo, or point at it:

```
../student_resource/dataset/{train,test}/source{1,2,3}.tsv
```

Override with `AMZ_DATA_DIR=/path/to/dataset` (used by `src/utils/io.py` and all
driver scripts).

**AWS credentials:** `aws configure`, `aws login`, or any standard chain. They are
never stored in this repository.

---

## 14. How to run it

### Local development loop (dev sample)

```bash
export AMZ_DATA_DIR=../student_resource/dataset
PY=.venv/bin/python

$PY scripts/02_normalize.py --split train          # or --split dev
$PY scripts/03_dev_sample.py                       # build artifacts/dev/
$PY scripts/04_build_indices.py --split dev --phase all
$PY scripts/05_generate_candidates.py --split dev
$PY scripts/06_candidate_recall.py --split dev     # gate: >= 0.97
$PY scripts/07_build_train_pairs.py --split dev --per-pos 5
$PY scripts/08_generate_features.py --split dev --mode pairs
$PY scripts/08_generate_features.py --split dev --mode candidates
$PY scripts/09_train_lgbm.py --split dev --rounds 300
$PY scripts/10_score_candidates.py --split dev
$PY scripts/11_threshold_search.py --split dev
$PY scripts/12_build_submission.py --split dev
$PY scripts/13_score_submission.py --split dev \
      --matching submissions_dev/matching_results.tsv
$PY -m pytest -q
```

### Full test run

```bash
bash scripts/99_run_full.sh                 # single instance
# or, two instances:
bash scripts/99_run_full.sh                 # A: even shards + stage 12
bash scripts/92_helper.sh                   # H: odd shards
```

### Validation before submitting

```bash
# needs ../student_resource/ to exist
.venv/bin/python scripts/14_validate_submission.py --check-ids
```

Exit code 0 = safe to submit.

---

## 15. Measured results

### Candidate recall gate (`reports/candidate_recall_dev.json`)

| Metric | Value |
|---|---:|
| Pair candidate recall | **0.9912** (gate 0.97, **PASS**) |
| At-least-one S1 coverage | 0.9986 |
| All-match S1 coverage | 0.9722 |
| Candidate pairs | 1,042,919 (104.3 / S1) |
| Positive pairs retrieved | 32,946 / 33,239 |

Per-method pair recall: `tfidf` 0.9901 · `e_ss` 0.4851 · `rare` 0.3119 ·
`addr` 0.3040 · `e_name` 0.2563.

TF-IDF rank buckets: top1 0.238 · top5 0.895 · top10 0.9735 · top20 0.9812 ·
top50 0.9872.

### Dev end-to-end (`reports/experiments.csv`, EXP-001)

| Field | Value |
|---|---|
| candidate_recall | 0.9887 |
| all_match_coverage | 0.9646 |
| **validation_macro_f0_5** | **0.96879** |
| precision | 0.993261 |
| recall | 0.94004 |
| runtime | 148 s (dev sample: 9,995 S1 / 272,474 targets) |
| peak RAM | 468 MB RSS during candidate generation |

### Dev submission score (`reports/score_dev.json`)

| Field | Value |
|---|---:|
| macro_f05 | **0.970247** |
| pair_precision | 0.987038 |
| pair_recall | 0.957610 |
| pair_f05 | 0.981009 |
| tp / fp / fn | 31,830 / 418 / 1,409 |
| s1_entities | 9,995 (2,000 singletons) |
| singletons correctly empty | 1,875 / 2,000 |
| s1 predicted empty | 1,904 |

---

## 16. Final test-set submission

Produced **2026-09-26 06:22:21 UTC**, uploaded to
`s3://amazon-ml-challenge-2026-357112746764/run/artifacts/submissions/` and
mirrored locally under `submissions/`.

| File | Bytes |
|---|---:|
| `matching_results.tsv` | 94,621,216 |
| `candidate_pairs.tsv` | 6,502,597,351 |
| `submission_summary.json` | 273 |

`submission_summary.json`:

```json
{
  "split": "test",
  "threshold": 0.96,
  "s1_rows": 1732544,
  "s1_with_matches": 1627051,
  "s1_with_candidates": 1732544,
  "match_ids": 5600660,
  "candidate_ids": 502768853,
  "avg_matches_per_s1": 3.233,
  "avg_candidates_per_s1": 290.191,
  "elapsed_sec": 4687.1
}
```

* 93.9 % of test S1 received at least one match; 105,493 rows are legitimately
  empty.
* Threshold **0.96** comes from `threshold_train.json` (stage 11 on the train
  split). Stage 12 falls back to it automatically when `threshold_test.json`
  does not exist — the test split has no ground truth to tune against.

**Validation performed (both passed):**

1. Official `student_resource/utils/validate_submission.py --check-ids` on the
   instance → `rc=0`. 1,732,544 rows (105,493 empty / 1,627,051 non-empty); all
   match ids valid against the 9,969,589 test S2/S3 ids. Re-run locally → also
   `rc=0` in 2 m 28 s.
2. The official validator's *candidate* cross-check loads the whole 6.1 GiB file
   and is **OOM-killed (rc 137)** on an 8 GiB box, so a streaming subset check
   was used instead: keep only the 1.6 M matched S1 rows and walk
   `candidate_pairs.tsv` once. Result: **PASS** — all 1,627,051 matched rows have
   every id present in their candidate list, across all 1,732,544 candidate rows.

Submission format (UTF-8, tab-separated, exact official headers, every S1 has a
row):

```
source1_entity_id	matched_entity_ids
S1-714132312	S3-625880872,S3-867809779
S1-106407869	S2-705547832,S3-585937637,S3-613056593
```

Final package name required by the rules: `<team_name>_submission.zip`
containing `output/`.

---

## 17. Two-instance execution design

A single `m7i-flex.large` (2 vCPU, 8 GiB) cannot finish in one watchdog window —
candidates 1.9 h + features 1.9 h + scoring 3.7 h ≈ 7.5 h serialized. So the work
is split across two identical free-tier boxes that **never write the same file**:

| | Instance A | Instance H (helper) |
|---|---|---|
| ID | `i-0ff1c1a828c342047` | `i-0dafc89f92d67679a` |
| Role | even candidate shards + stages 05/08/10 + stage 12 | odd candidate shards + odd features |
| Script | `99_run_full.sh` | `92_helper.sh` |

**How the split is made collision-free**

* `05_generate_candidates.py` takes `--worker-index/--num-workers` and splits by
  shard number.
* `08_generate_features.py` takes `--shard-parity {all,0,1}`; output names are
  derived from the **source shard number** in the filename.
* `10_score_candidates.py` names predictions after the feature shard number.
* Feature shards are named `i*STRIDE + piece` with `STRIDE = 128`, so A's and H's
  feature name spaces are disjoint by construction.

**Marker chain (used by stage 12 to map scores back to candidate shards)**

```
score file  --.SUCCESS.src-->  feature file  --.SUCCESS.src-->  candidate shard
```

`12_build_submission.py` resolves a score file to its candidate shard purely via
markers (`group_score_shards()`, `_candidate_index()`), which is what makes
mis-named files recoverable without recomputing anything.

**Handshake:** H writes `run/status/H_PREDS_DONE.txt`; A's `merge_helper_predictions()`
polls for up to 3 hours and then `aws s3 sync`s the helper's scores in. The wait is
gated on `artifacts/helper.expected` so a normal single-instance run never blocks.

**Known operational fact:** EC2 **user data does not re-run on Stop/Start**. After
stopping, the boxes must be driven manually over SSM (`aws ssm send-command`).

---

## 18. AWS infrastructure and cost

* **Account** `357112746764`, **region** `us-east-1`.
* **Bucket** `amazon-ml-challenge-2026-357112746764` — SSE-S3 (AES256),
  block-public-access on, created by this project.
* **Instances** `m7i-flex.large` (2 vCPU / 8 GiB), 100 GB gp3 each, SSM-only (no
  key pair), AMI `ami-0fef201115eefe936`, profile `amz-ml-2026-ec2`,
  SG `sg-0a2aebb3cc26d46c7`, subnet `subnet-0bfb1b1e8c68d57fa`.
* **Access model:** the operator identity was the account root, so mutations go
  through the AWS MCP server / CLI with an explicit policy
  (`config/iam_opencode_policy.json`); EC2 only ever gets S3 list/read/write on
  the project bucket.
* **Credits:** $140 total ($100 AWS Free Tier + $20 budget + $20 EC2), all
  unexpired as of 2026-09-26. Project consumption ≈ **$1.30** (< 1 %).
* **Run 8** (the two-box test run, 04:54–06:44 UTC) cost ≈ **$0.35**.
* **Cleanup:** all three instances were **terminated** and all EBS volumes
  deleted (0 instances, 0 volumes remain). Only S3 storage accrues (~$0.50/mo).

Full per-resource history, prices and the credit balance:
`reports/aws_cost_ledger.md`. Budget ceiling: `reports/cost_gate.md`.

---

## 19. S3 layout and status markers

```
s3://amazon-ml-challenge-2026-357112746764/
├── raw/          immutable competition data (never delete/overwrite)
├── processed/    normalized parquet
├── indices/      blocking indices
├── candidates/   candidate shards
├── features/     feature shards
├── models/       trained LightGBM
├── submissions/  submissions
├── temp/         only prefix eligible for lifecycle expiry
└── run/
    ├── code.zip          the exact code deployed to the instances
    ├── status/*.txt      one marker per stage (see below)
    ├── logs/             captured stage logs
    ├── reports/          reports uploaded at the end of a run
    └── artifacts/
        ├── predictions/  scored shards + .SUCCESS markers
        ├── submissions/  matching_results.tsv, candidate_pairs.tsv, summary
        └── helper/       helper handshake files
```

Status markers written by `99_run_full.sh`:
`STARTED`, `S02_TRAIN_NORM`, `S04_TRAIN_IDX`, `S05_TRAIN_CAND`,
`S06_TRAIN_RECALL`, `S07_TRAIN_PAIRS`, `S08_PAIR_FEATS`, `S08_VAL_CAND_FEATS`,
`S09_MODEL`, `S10_TRAIN_SCORE`, `S11_THRESHOLD`, `S02_TEST_NORM`,
`S04_TEST_IDX`, `S05_TEST_CAND`, `S08_TEST_FEATS`, `S10_TEST_SCORE`,
`S12_SUBMISSION`, `ALL_DONE`.

Helper markers: `H_START`, `H_INPUTS`, `H_SYNC`, `H05`, `H08`, `H10`,
`H_PREDS_DONE`, `HELPER_RC`, `HELPER_EXIT`, `H_FAIL`.

User-data markers: `UD_00_START` … `UD_04B_ARTIFACTS`, `USERDATA_EXIT.txt`
(contains `userdata_exit`, `finished_utc`, `disk_free_gb`, `uptime`).

Because every stage uploads a marker, progress is observable with **no inbound
SSH** — only `aws s3 ls` / `aws ssm send-command`.

> Timestamp quirk: `aws s3 ls` prints times in UTC+5:30 in this environment;
> subtract 5.5 h to get UTC.

---

## 20. Resumability and safety guards

| Mechanism | Where | What it prevents |
|---|---|---|
| `<file>.SUCCESS` markers | all stages | re-doing completed shards after a crash |
| Chunked processing (`--chunk`) | 02, 05 | peak memory growing with corpus size |
| Separate phases in 04 | `04_build_indices.py` | one process peaking past 8 GiB and being OOM-killed |
| `check_feature_k()` | `paths.py` | scoring features built with a different TF-IDF K |
| Recall gate in 06 | before training | training a matcher on an inadequate candidate set |
| Coverage assertion (H5b) | `92_helper.sh` | silently shipping a half-scored test set |
| `local_cov` vs scored-count guard | `12_build_submission.py` | writing a submission with missing candidate shards |
| `set(g_sm) != set(g_sc)` guard | `12_build_submission.py` | mismatched score/feature/candidate shard sets |
| `sid < ptr` overlap guard | `12_build_submission.py` | duplicate S1 rows leaking into the output |
| Marker-chain mapping | `12_build_submission.py` | incorrect score→candidate association |
| Threshold fallback | `12_build_submission.py` | hard failure when `threshold_test.json` is absent |
| `EXIT` trap in user data | `config/userdata.sh` | losing the console log and failing marker on shutdown |

Hard rule enforced throughout: **the original dataset is only ever opened for
reading.**

---

## 21. Bugs found and fixed

All four were latent until the two-instance split exposed them. They are the most
valuable part of this documentation for anyone extending the pipeline.

1. **Row blow-up in stage 12.** Score files are written one per *piece* of a
   candidate shard, and a piece is a slice of **rows**, not of **sids** — candidate
   rows are ordered by retrieval method, so one piece already spans much of the
   shard. `group_by` therefore ran once per piece and wrote **48,756,072 rows**
   against an expected 1,732,544, then `sys.exit`-ed. Fixed by concatenating the
   full piece list per candidate shard *before* `group_by`, resolving each score
   file's candidate shard through the marker chain, and adding the `sid < ptr`
   overlap guard. Covered by `tests/test_submission_grouping.py`.

2. **The helper scored only half the test set.** Stage H5 ran
   `--worker-index 0 --num-workers 2`, which covers **even** feature indices only:
   125 of 246 files, and every odd-index piece (~120 M pairs) had no score at
   all. Fixed to run workers 1 and 3 of `--num-workers 4` concurrently, plus a
   coverage assertion that the prediction count equals the feature count.

3. **Instance A's scores were named by position, not by feature.** The copy of
   `10` running on A predates the source-shard naming fix, so its 267 scores
   landed as `shard_00000..shard_00266`. When the helper's correctly named half
   was merged in, 16 position-named files were **overwritten**. Repaired without
   recomputing: 222 files were renamed from their `src` markers (in two phases, to
   avoid stepping on files not yet moved), then the 15 genuinely missing features
   were rescored in ~11 minutes. Final coverage 267/267.

4. **`threshold_test.json` never exists.** The test split has no ground truth, so
   stage 12 hard-failed with `no threshold file …/threshold_test.json`. It now
   falls back to `threshold_train.json`.

Related earlier fixes that are already in the code: polars 1.44 dropping
`scan_csv(columns=…)` (regression test added), the `tfidf_k` mismatch guard added
after the train recall gate failed at 0.8238, and the score/candidate index
intersection bug that silently dropped all but one candidate shard.

---

## 22. Reports index

Everything in `reports/` is committed and is the authoritative context for the
code:

| File | What it holds |
|---|---|
| `data_audit.md` | STAGE 0 reconnaissance: file sizes, schema, integrity checks, countries, ground-truth structure, sample rows. |
| `competition_rules.md` | Extracted official rules, metric definition, submission package format, leaderboard notes. §4 has the official F0.5 definition. |
| `execution_plan.md` | The staged plan (STAGE 0–16) this pipeline implements. |
| `preflight.md` | Pre-launch checks. |
| `cost_gate.md` | Budget ceilings and the approval gate for any spend > $2. |
| `aws_cost_ledger.md` | Every billable resource, prices, per-run cost, credit balance, and the Run 8 outcome. |
| `aws_resources.json` | Read-only account inventory taken *before* anything was created. |
| `experiments.csv` | One row per experiment: recall, coverage, macro F0.5, precision, recall, runtime, peak RAM, cost, keep/drop. |
| `candidate_recall_dev.json` | Recall gate numbers + per-method and rank-bucket recall. |
| `threshold_dev.json` | Full threshold sweep curve. |
| `score_dev.json` | Dev submission score breakdown. |
| `local_data_manifest.json` | Local dataset file inventory with checksums. |

---

## 23. Git / team workflow

```bash
git clone git@github.com:mohdmohsinrizvi/v2.git
cd v2
```

* Branch: **`main`**. History starts at `75212f5` ("Add Amazon ML Challenge 2026
  pipeline").
* The repository is **private**. Do not make it public — it contains the full
  solution, the cost ledger and AWS account details, and publishing a competition
  solution invites copying.
* Teammates need to be added under **Settings → Collaborators and teams**; repo
  access is not automatic just because the URL is shared.
* Before pushing: `.venv/bin/python -m pytest -q` must show **39 passed**.
* Never commit: datasets (`*.tsv`), parquet, models, `artifacts/`, `submissions/`,
  `.env`, `*.pem`, `credentials`. `.gitignore` enforces all of these.
* Keep `reports/` updated — the cost ledger in particular must be appended to
  whenever anything billable is created.

---

## 24. Open questions and TODO

* **Team size / registration rules — UNKNOWN, must verify** (noted at
  `reports/competition_rules.md:102`). The final package name is
  `<team_name>_submission.zip`, so a team name is required.
* **Threshold transfer risk.** The test submission uses 0.96, tuned on the train
  split; the dev fold preferred 0.98. The gap is a known, accepted risk.
* **Memory ceiling.** Everything is tuned for 8 GiB. Feature generation still
  needs `FEAT_WORKERS=1`; scoring can fan out to 4.
* **Candidate volume.** The test candidate set is 502,768,853 pairs (290 / S1) —
  the dominant cost of the whole run. Reducing it while holding recall ≥ 0.97 is
  the clearest available optimisation.
* **Official candidate cross-check** could not run through
  `student_resource/utils/validate_submission.py` on an 8 GiB box (OOM). The
  streaming replacement used for validation is not yet checked into this repo.
* **France unseen in training** — 259,452 test S1 rows come from a country that
  does not exist in train. Recall on that slice is unmeasured.

---

*Documentation generated for the `v2` repository. Everything above is derived
from the committed code and reports; where a number was measured, the source file
is named inline.*
