# Preflight — READ-ONLY Phase 0

Date: 2026-09-25 · Agent: opencode · **No AWS resource created or modified by this phase.**
All AWS calls below are `Describe*`/`List*`/`Get*` only.

---

## 1. Local environment

| # | Check | Result |
|---|-------|--------|
| 1 | Working directory | `/home/mohsin/Desktop/amazon` |
| 2 | Project files | `amazon_ml_2026/` — `src/`, `scripts/`, `config/`, `reports/`, `tests/`, `submissions/`, `artifacts/`, `.venv/`, `requirements.txt`, `.gitignore` |
| 3 | Dataset location | `student_resource/dataset/{train,test}/*.tsv` (single copy, untouched) |
| 4 | Dataset total size | **2,520,603,693 B = 2.52 GB** (measured; the brief's "~10 GB" is wrong) |
| 5 | Free local disk | **9.4 GB** of 117 GB (`/`, 92% used) — **tight** |
| 6 | Local RAM | 7.6 GB total · ~2.1 GB available · 2.0 GB swap (1.5 GB used) |
| 7 | CPU | 4 vCPU · Intel Pentium N4200 @ 1.10 GHz (low-IPC, laptop class) |
| 8 | Python | 3.12.3 system · 3.12.3 in `.venv` |
| 9 | AWS CLI | aws-cli/2.37.1 (Python 3.14.6, linux x86_64) |
| 10 | AWS caller identity | `arn:aws:iam::357112746764:root` (account `357112746764`) — **root principal** |
| 11 | AWS MCP | **WORKS** (proxy v1.7.0) |
| 12 | AWS region | `us-east-1` (CLI default) |
| 13 | Existing EC2 | **0 instances in all 17 enabled regions** |
| 14 | Existing S3 (project) | `amazon-ml-challenge-2026-357112746764` — 11 objects, 2,520,812,871 B |
| 15 | Existing SageMaker | 0 endpoints · 0 notebooks · 0 training jobs · 0 apps · 1 idle Domain (no apps) |
| 16 | Running billable resources | **NONE** (0 EC2, 0 EBS, 0 SageMaker apps) |

### Python environment (`.venv` is complete)

`polars 1.44.2` · `pyarrow 25.0.1` · `numpy 2.5.3` · `scipy 1.18.1` · `scikit-learn 1.9.1` ·
`rapidfuzz 3.14.6` · `lightgbm 4.7.0` · `joblib 1.6.0` · `tqdm 4.70.1` · `psutil 7.2.2` ·
`pandas 3.0.6` · `boto3 1.43.102` · `PyYAML 6.0.3`

System Python lacks most of these — **always run `.venv/bin/python`**.
`pytest`: **19/19 tests pass** (`tests/test_metric.py`, `tests/test_normalization.py`).

### MCP connectivity note

`aws-mcp` works, but `call_boto3` requires **PascalCase** operation names
(`ListBuckets` ✅ vs `list_buckets` ❌) and **keyword-only** args
(`service_name=`, `operation_name=`, `params=`, `region_name=`).
Lowercase/snake_case names return `OperationNotFoundError` before reaching AWS.

---

## 2. AWS inventory (read-only, 2026-09-25)

| Service | Operation | Result |
|---|---|---|
| STS | `GetCallerIdentity` | `arn:aws:iam::357112746764:root` |
| EC2 | `DescribeRegions` (enabled) | 17 regions |
| EC2 | `DescribeInstances` ×17 regions | **0** (pending/running/stopping/stopped/shutting-down) |
| EC2 | `DescribeVolumes` ×17 regions | **0** |
| S3 | `ListBuckets` | 2 buckets |
| S3 | `ListObjectsV2` (project bucket) | 11 objects, 2,520,812,871 B |
| S3 | `GetPublicAccessBlock` (project) | all four flags `true` |
| S3 | `GetBucketPolicy` (project) | `NoSuchBucketPolicy` (no public policy) |
| S3 | `GetBucketAcl` (project) | owner `CanonicalUser` only — no public grants |
| SageMaker | `ListEndpoints/NotebookInstances/TrainingJobs/Apps` | all empty |
| SageMaker | `ListDomains` | 1 pre-existing `QuickSetupDomain-20260925T125597`, `InService`, **0 apps** |
| Budgets | `DescribeBudgets` | 1 pre-existing `My Zero-Spend Budget` ($1.00/mo) |
| Cost Explorer | `GetCostAndUsage` | `DataUnavailableException` — CE not ingested yet |

### S3 buckets

| Bucket | Owner | Status |
|---|---|---|
| `amazon-ml-challenge-2026-357112746764` | **this project** | private, encrypted (SSE-S3), tagged, raw/ uploaded byte-exact |
| `sagemaker-us-east-1-357112746764` | SageMaker service | **leave alone** |

Verified: Block Public Access all-`true`, no bucket policy, ACL owner-only,
no object is public. `raw/` = 2,520,812,871 B vs local 2,520,603,693 B
(difference = 4 meta files intentionally uploaded alongside).

---

## 3. Local disk breakdown (why only 9.4 GB is free)

| Path | Size |
|---|---|
| `student_resource/dataset` | 2.4 GB |
| `amazon_ml_2026/.venv` | 935 MB |
| `amazon_ml_2026/artifacts` | 11 MB |
| rest of `amazon_ml_2026` | ~1 MB |
| **Free on `/`** | **9.4 GB** |

**Implication:** full-data intermediates (normalized ~2.5 GB → candidates ~1–3 GB →
features ~5–15 GB) **cannot fit locally**. Local work must stay on the dev sample;
full-scale stages belong on cloud disk.

---

## 4. Project code status

| Stage | Status |
|---|---|
| `src/utils/io.py` | ✅ path/sharding/parquet helpers |
| `src/normalization/text.py` | ✅ name/address normalization (Unicode-safe, Devanagari-aware, suffix + address expansion) |
| `src/evaluation/metric.py` | ✅ entity-level Macro F0.5, verified against official 0.714 example |
| `scripts/10_dev_sample.py` | ✅ ran — 9,995 S1 (2,000 singletons) / 135,625 S2 / 136,849 S3, 11 MB |
| `tests/` | ✅ 19 passing |
| Candidates / features / model / inference scripts | ❌ **not yet written** |
| `reports/data_audit.md` | ✅ present (row counts re-verified this phase) |
| `reports/execution_plan.md` | ✅ present |
| `reports/aws_cost_ledger.md` | ✅ present — actual known spend **$0.00** |
| `reports/experiments.csv` | ✅ header only, no fabricated rows |

---

## 5. Verification of the metric against official material

`student_resource/README.md` L190–209 states
`F_0.5 = (1.25·P·R) / (0.25·P + R)`, macro-averaged over **all** S1 entities,
singletons included (correct empty = 1.0, any prediction on a true singleton = 0.0).
Official worked example: pred `[S2-47,S2-193,S3-812]` vs truth `[S2-47,S3-812]` → **0.714**.
`src/evaluation/metric.py` reproduces 0.714 within 1e-3 (unit test passing).

---

## 6. Preflight summary

```
LOCAL DATASET:        2.52 GB (2,520,603,693 B), 8 TSV files, single copy, unmodified
                      train S1 2,206,821 · S2 5,034,616 · S3 5,285,603 · GT 2,206,821
                      test  S1 1,732,544 · S2 4,887,273 · S3 5,082,316
TOTAL SIZE:           2.52 GB  (NOT ~10 GB)
RAM:                  7.6 GB total / ~2.1 GB available  → too small for full data
CPU:                  4 vCPU Pentium N4200 @ 1.10 GHz  → slow; local run = dev sample only
FREE DISK:            9.4 GB  → too small for full intermediates
AWS ACCOUNT:          357112746764 (principal = account root)
AWS REGION:           us-east-1
AWS MCP:              ✅ WORKS (PascalCase ops, keyword args)
EXISTING BILLABLE:    NONE running. 0 EC2 · 0 EBS · 0 SageMaker apps.
                      S3: 2.52 GB project data (~$0.06/mo storage), 1 idle SageMaker
                      Domain ($0 while no apps), 1 free $1 budget.
                      Actual known spend to date: $0.00.
RISKS:
  R1 9.4 GB free disk — full-data intermediates must live in the cloud, not locally.
  R2 7.6 GB RAM / 4 weak cores — full pipeline cannot run locally; benchmark → pick instance.
  R3 Caller is ACCOUNT ROOT — no IAM user/role exists yet. A project-scoped IAM role
     is required before any EC2 launch (least privilege, bucket-scoped).
  R4 Cost Explorer has no data yet → spend must be tracked by our own ledger +
     budget alerts, not by CE queries.
  R5 Credit balance/expiry UNVERIFIED — must be confirmed in Console → Billing → Credits.
  R6 France (259,452 test S1) never appears in train → no country-specific logic allowed.
  R7 Root principal could accidentally create expensive resources → all mutations gated
     behind the >$2 approval rule in this document's operating contract.
```

**Phase 0 status: COMPLETE. No AWS mutation performed in this phase.**
Next: Phase 1 data audit refresh + Phase 2/3 local code and dev-sample pipeline.
