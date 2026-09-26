# Execution Plan — Amazon ML Challenge 2026 Baseline V1

Status: **STAGES 0–5 COMPLETE LOCALLY (2026-09-25 16:45) · DEV PIPELINE GREEN END-TO-END · AWAITING APPROVAL FOR THE CLOUD COST GATE (`reports/cost_gate.md`).**
Date: 2026-09-25 · Prepared by opencode agent · No AWS resource created, no local raw data modified.

**Update after `aws login`:** MCP handshake OK (proxy v1.7.0), live tool catalog = 8 tools
(`aws___search_documentation`, `aws___read_documentation`, `aws___retrieve_skill`,
`aws___list_regions`, `aws___get_regional_availability`, `aws___run_script`,
`aws___get_tasks`, `aws___get_presigned_url`). Identity re-confirmed: account
`357112746764`, ARN `arn:aws:iam::357112746764:root` → **root, so mutations stay blocked**
until an IAM identity exists. Read-only inventory: see `reports/aws_resources.json`
(0 EC2 · 0 EBS · 0 SageMaker apps · 1 S3 bucket · 1 pre-existing $1 budget · $0 spend).

---

## 1. Current environment status

| Item | Value |
|------|-------|
| Working dir | `/home/mohsin/Desktop/amazon` |
| Git repo | **No** (`git status` → not a repository) |
| OS / CPU / RAM | Linux Mint 22 · 4 vCPU · 7.6 GB RAM (2.4 GB currently available) · 2 GB swap |
| Free disk | **16 GB** of 117 GB (87% used) |
| Python | 3.12.3 (`/usr/bin/python3`), pip 24.0, network to PyPI verified OK |
| Installed ML libs | numpy 2.4.3, scipy 1.17.1, scikit-learn 1.8.0, pandas 3.0.1, joblib, psutil |
| Missing (needed) | `polars`, `pyarrow`, `lightgbm`, `rapidfuzz`, `boto3`, `tqdm` |
| Dataset | `student_resource/dataset/` — **2.52 GB total**, single copy, no duplicates found |
| Other local files | `student_resource/README.md` (official rules), `Documentation_template.md`, `utils/validate_submission.py`, `day 1.pdf` (strategy notes), `promts.odt` (prompt playbook) |

Data audit: see `reports/data_audit.md`. Rules: see `reports/competition_rules.md`.
File manifest: see `reports/local_data_manifest.json`.

## 2. AWS MCP status

```
$ opencode mcp list
● ✗ aws-mcp  failed
   MCP error -32602: Invalid request parameters
   uvx mcp-proxy-for-aws@latest https://aws-mcp.us-east-1.api.aws/mcp --metadata INSTALL_SOURCE=aws-cli
```

**Root cause identified (not guessed).** I reproduced the handshake manually and captured
the proxy's stderr:

```
botocore.exceptions.LoginRefreshRequired: Your session has expired or credentials have
changed. Please reauthenticate using 'aws login'.
botocore.errorfactory.AccessDeniedException: CreateOAuth2Token ... The refresh token has expired.
```

Consequences:
- `initialize` returns `{"code":-32602,"message":"Invalid request parameters"}` — the proxy
  cannot SigV4-sign the request without credentials, so OpenCode sees it as a bad-params error.
  **Live tool catalog therefore cannot be enumerated yet** (returned `tools: []`).
- The config itself is structurally fine (`~/.config/opencode/opencode.json`, local
  `mcp-proxy-for-aws v1.7.0`, endpoint `https://aws-mcp.us-east-1.api.aws/mcp`). **No config
  edit is needed or permitted before approval.** `aws` CLI 2.37.1 is installed and uses the
  new `aws login` session mechanism.
- `aws sts get-caller-identity` → `ERROR: Your session has expired. Please reauthenticate using 'aws login'.`

**Two blockers to AWS work:**
1. Re-authenticate: **you** must run `aws login` (browser flow). I will not handle credentials.
2. The stored session is **root**: `login_session = arn:aws:iam::357112746764:root`.
   Per the standing rule, **no AWS mutations may run under root.** After login we must
   switch to a least-privilege IAM identity (see §6, first mutation).

## 3. AWS account / Free Tier / credit status

| Item | Status |
|------|--------|
| Account ID (from config ARN) | `357112746764` (root principal — identity not re-confirmed this session) |
| Caller ARN / principal type | root (from `~/.aws/config`), **not** re-verified live — session expired |
| Default region | `us-east-1` |
| Promotional credit balance | **UNKNOWN — MUST VERIFY** (billing API needs live credentials) |
| Credit expiry | **UNKNOWN — MUST VERIFY** → Console: *Billing and Cost Management → Credits* |
| Free vs Paid plan | **UNKNOWN — MUST VERIFY** → Console: *Billing → AWS Free Tier* |
| Account creation date | **UNKNOWN — MUST VERIFY** → Console: *Billing → Credits* shows expiry (credits expire 12 months from account creation) |
| SageMaker Free Tier eligibility | **UNKNOWN — MUST VERIFY** → Console: *SageMaker → Free Tier usage* / *Billing → Free Tier* |
| Existing billable resources | **UNKNOWN — MUST VERIFY** (cannot list without credentials). Expected findings: S3 buckets, EBS, SageMaker apps, EC2 — will be enumerated immediately after re-auth |

Official Free Tier facts (verified from AWS docs/pricing pages today, account applicability
still unknown):

- **New accounts (created on/after 2025-07-15):** Free **plan** for 6 months + **$100 sign-up
  credit** and up to **$100 more** earned credits; Free Tier credits expire 12 months from
  account creation; the free plan cannot be combined with other promotional credits
  (*AWS Free Tier FAQs*, docs.aws.amazon.com). Older accounts: classic 12-month Free Tier.
- **SageMaker AI:** 250 h/month of `ml.t3.medium` notebook/Studio usage for the **first 2
  months** from first SageMaker resource creation (*SageMaker pricing page*).
- **S3:** new customers get **5 GB Standard + 20k GET + 2k PUT + 100 GB egress per month for
  12 months** (*S3 FAQ*) — 2.52 GB fits inside it.
- **EC2:** 750 h/month `t3.micro`/`t2.micro` (12-month) for older accounts; free-plan accounts
  are limited to `t3.micro/small`, `t4g.micro/small`, `c7i-flex.large`, `m7i-flex.large`;
  separate **T4g free trial: 750 h/month `t4g.small`, all accounts, until 2026-12-31**
  (*EC2 free-tier docs*, *EC2 FAQ*).
- The user's stated **$100 promotional credit** cannot be confirmed programmatically until
  re-authentication — treat as *claimed, unverified*.

## 4. Existing billable resources

**UNKNOWN — MUST VERIFY.** First read-only action after re-auth will be:
`sts get-caller-identity`, `s3 list-buckets`, `ec2 describe-instances`,
`ec2 describe-volumes`, `sagemaker list-...`, `budgets describe-budgets` — all read-only,
output written to `reports/aws_resources.json`. Nothing is deleted or modified at that point.

## 5. Recommended workload region

**`us-east-1` (N. Virginia)** — single workload region for S3 + any compute.

Reasons: it is the account's configured default region; all pricing below was fetched from
the official Price List API for that region; no cross-region transfer cost; SageMaker and EC2
availability is maximal. (The MCP endpoint also lives in `us-east-1`, but region choice is
driven by workload, not by the MCP endpoint.)

## 6. Recommended compute architecture

**Local-first.** The dataset is 2.52 GB and the pipeline is CPU-only (TF-IDF + sparse
retrieval + LightGBM). The laptop can do everything if stages are sharded and memory-bounded.

| Stage | Recommended compute | Why | Cost |
|-------|--------------------|-----|------|
| Coding, tests, dev sample (5–20k S1) | **Local (this machine)** | free, no upload needed, instant iteration | $0 |
| Full normalisation + indices | **Local**, Polars/PyArrow, per-country, chunked, checkpointed | 2.5 GB fits; I/O bound | $0 |
| Candidate retrieval (word TF-IDF) | **Local first**, blocked sparse matmul (query shards × corpus shards), memmap-backed | avoids dense conversion; RAM-safe | $0 |
| Pair features + hard negatives | **Local**, shard-per-country, parquet out | CPU bound, checkpointable | $0 |
| LightGBM training | **Local** (4 cores, early stopping) | ~7.6 M candidate pairs × ~35 features fits easily | $0 |
| Full test inference | **Local**, streamed by S1 shard | same as above | $0 |
| **Escalation only if local benchmarks fail** | **EC2 `m6i.xlarge`/`m7i.xlarge` on-demand, hours not days, auto-terminate** | 4 vCPU/16 GB → RAM problem solved | ~$0.19/h |

Free Tier / optional:
- **SageMaker Studio notebook `ml.t3.medium`** is free for 250 h/month × 2 months, but adds
  EBS + idle-charge risk and no benefit over local for this workload → **not recommended for V1.**
- **EC2 `t4g.small`** free trial (750 h/month, all accounts, to 2026-12-31) is too small
  (2 vCPU / 2 GB) for retrieval, but is fine for a tiny smoke test.
- **GPU: forbidden for V1** (per standing rule) — not needed for TF-IDF/LightGBM.

Explicitly **not** creating: SageMaker endpoints, NAT Gateway, ELB, EMR, Redshift,
OpenSearch, EKS/ECS, RDS, GPU instances, distributed training.

### Honest risks of local-first

- **RAM is the binding constraint**: 2.4 GB currently free. Closing other apps is required
  before full-scale runs, or retrieval must be strictly shard-streamed.
- **Disk**: 16 GB free; `candidate_pairs.tsv` alone ≈ 1 GB, plus normalised parquet,
  indices, feature shards. Rule: one shard at a time for features; prune intermediates;
  never duplicate `raw/`.
- **Wall-clock**: full retrieval could take hours on 4 cores → if measured > ~4 h per
  stage, escalate to a single on-demand EC2 box (cost-gated, see §7).

## 7. Cost plan (official prices, `us-east-1`, fetched 2026-09-25)

Price sources (official AWS Price List bulk API):
- S3 Standard us-east-1: **$0.023/GB-month** first 50 TB (offer version 2026-09-18)
- EC2 On-Demand Linux, effective 2026-09-01: `t3.medium` **$0.0416/h**, `t4g.large`
  **$0.0672/h**, `t3.large` **$0.0832/h**, `m6i.large` **$0.096/h**, `m7i.large`
  **$0.1008/h**, `m6i.xlarge` **$0.192/h**, `c7i.large` **$0.08925/h**
- SageMaker Studio JupyterLab (effective 2026-09-01): `ml.t3.medium` **$0.05/h**,
  `ml.t3.large` **$0.10/h**, `ml.t3.xlarge` **$0.20/h**, `ml.m5.xlarge` **$0.23/h**

| Line item | Low | Base | Worst case |
|-----------|----:|-----:|-----------:|
| S3 storage (raw 2.52 GB + ~8 GB artifacts, 1 month) | $0.00 (free-tier 5 GB) | $0.26 | $0.30 |
| S3 requests/egress (one upload, small downloads) | $0.01 | $0.05 | $0.20 |
| Development (local) | $0 | $0 | $0 |
| Preprocessing / candidate generation (local) | $0 | $0 | $1.50 (8 h × `m6i.xlarge` if escalated) |
| Feature generation (local) | $0 | $0 | $1.50 (8 h escalated) |
| LightGBM training (local) | $0 | $0 | $0.40 |
| Inference (local) | $0 | $0 | $0.80 |
| AWS Budgets (first budget free) | $0 | $0 | $0 |
| Misc / retries | $0 | $0.05 | $0.50 |
| **TOTAL** | **≈ $0.01** | **≈ $0.36** | **≈ $5.20** |

Residual risk multiplier: if *several* stages need escalation and we repeat runs, multiply
the compute lines by 2–3 → **absolute worst case ≈ $15**, still far below the $30–40
baseline ceiling and the $100 credit. **No single operation in this plan exceeds $2**, so no
individual approval gate is expected to trip — any surprise > $2 operation will still be
paused for approval.

Cost posture: this is a **$0–$1 baseline** architecture; the credit is preserved for V2
experiments if they prove necessary.

## 8. Risks

1. **Root identity** — mutations blocked until a least-privilege IAM identity exists. (High)
2. **Expired AWS session** — everything AWS is blocked until you run `aws login`. (High, 2 min to fix)
3. **RAM (2.4 GB free)** — full retrieval can OOM. Mitigation: shard-streamed sparse matmul,
   per-country partitioning, `MEMORY PREFLIGHT` before every full run, and kill-switch. (High)
4. **Disk (16 GB free)** — candidate/feature files can fill `/`. Mitigation: shard, prune,
   keep S3 as the durable store, monitor `df` in every stage script. (Medium)
5. **Candidate recall shortfall** — matcher cannot recover missed links. Mitigation: measure
   recall at Stage 7 with an explicit stop condition before training. (Medium)
6. **France unseen country** — no hard-coded geography; validate recall *per country*. (Medium)
7. **Disk-vs-S3 egress** — never pull 2.5 GB back repeatedly; only small artifacts return. (Low)
8. **Rule drift** — submission limits/deadline/team rules are UNKNOWN; verify on the Portal
   before the first real submission. (Medium)
9. **Billing delay** — budget alerts are a guardrail, not a hard cap (never claimed otherwise). (Low)

## 9. Stage gates (revised to match reality)

```
STAGE 0  Reconnaissance ........................ DONE  reports/preflight.md
STAGE 1  Identity + guardrails + budget ........ DONE (read-only); 1 pre-existing $1 budget
STAGE 2  Local data audit ...................... DONE  reports/data_audit.md
STAGE 3  Project skeleton + local env .......... DONE  19/19 tests pass
STAGE 4  S3 bucket (private) + raw upload ...... DONE  bucket amazon-ml-challenge-2026-357112746764,
                                                        2,520,812,871 bytes verified byte-exact
STAGE 5  Dev sample end-to-end locally ......... DONE  9,995 S1 / 272,474 target, seed 42
STAGE 5b Candidate retrieval V1 recall gate .... DONE  0.9887 pair recall  (>= 0.97 gate PASS)
                                                        reports/candidate_recall_dev.json
STAGE 6  Normalization (full, sharded) ......... SCRIPT READY  scripts/02_normalize.py --split {train,test}
                                                        measured 12,469 rows/s
STAGE 7  Candidate retrieval (full) ............ SCRIPT READY  scripts/04 + 05 (+ resumable shards)
STAGE 8  Hard negatives + leakage-safe split ... DONE    scripts/07 (hash-of-entity_id S1 split)
STAGE 9  Pair features (sharded, checkpointed) . DONE (dev) / SCRIPT READY  scripts/08, 25,253 pairs/s
STAGE 10 LightGBM training .................... DONE (dev) AUC 0.9997, 209 rounds  scripts/09
STAGE 11 Entity-level Macro F0.5 validation ... DONE     0.9688 val fold / 0.9725 full dev
STAGE 12 Threshold search ..................... DONE     best 0.97  reports/threshold_dev.json
STAGE 13 Full training pipeline reproducibility  SCRIPT READY (same commands, --split train)
STAGE 14 Full test inference (streamed) ........ SCRIPT READY  scripts/10 + 12
STAGE 15 Submission validation ................. DONE (dev) official validator PASS, --check-ids
STAGE 16 V2 experiments ........................ PENDING
STAGE 17 Cleanup / archive / cost postmortem ... PENDING
```

**BLOCKED AT:** the full run needs ~17 GB of disk and ~6–8 GB of RAM; this machine has
9.7 GB free and 7.6 GB total. Cost sheet for the single temporary CPU instance is in
`reports/cost_gate.md` — expected **$1.52**, worst case **$4.73**, running cost to date
**$0.00 compute** (+ $0.06 S3 storage).

Note the re-ordering vs. the original brief: **local skeleton + dev sample come before S3
upload**, because nothing AWS-side is needed to prove the pipeline, and uploading 2.5 GB
before the pipeline is proven would spend money/storage for no decision value. S3 is created
at Stage 4 purely as durable backup + artifact store.

## 10. Exact proposed first mutation (single, minimal, reversible)

> **COST GATE**
> **Resource:** 1 S3 bucket (private) — *only after identity work*; plus IAM identity switch
> · **Region:** `us-east-1` · **Type:** S3 Standard, BPA on, SSE-S3, no public policy
> · **Duration:** project lifetime (≤ 1 month) · **Storage:** 2.52 GB raw + ~8 GB artifacts
> · **Official price source:** AWS Price List API, S3 us-east-1 (2026-09-18): $0.023/GB-month
> · **Estimated worst-case cost:** **$0.26/month** (≈ $0 if the 5 GB free-tier applies)
> · **Free Tier:** 5 GB/12 months for new customers — eligibility UNKNOWN, must verify
> · **Promotional credit:** covers this trivially if active
> · **Remaining budget after:** ≥ $99 of the stated $100
> · **Terminates:** `aws s3 rb s3://<bucket> --force` after emptying, or keep raw as archive
> · **Cleanup command:** recorded in `reports/aws_resources.json` at creation time

Sequence for **STAGE 1** (read-only first, then one mutation):

1. **You:** run `aws login` (browser). No credentials ever pass through this chat.
2. **Agent (read-only):** `sts get-caller-identity` → confirm account/ARN/region; if root,
   **stop** and guide creation of an IAM user/role with the minimum policy
   (STS + S3 on the one bucket + `iam:PassRole` restricted to one future SageMaker role +
   budgets read/write + cloudwatch logs) — **no AdministratorAccess**.
3. **Agent (read-only):** inventory existing S3/EC2/EBS/SageMaker/budgets → write
   `reports/aws_resources.json`; create **one** AWS Budget with alerts at
   $10 / $25 / $40 / $60 / $75 (first budget = free).
4. **Agent (mutation, ≤ $2):** create **one** deterministic bucket
   `amazon-ml-2026-<non-sensitive-unique>` with Block Public Access, SSE-S3, no public
   policy, project tags → record in manifest + cost ledger.
5. **Agent:** local `aws s3 sync` of `raw/` (2.52 GB, one multipart upload), then verify
   object count/bytes/permissions. **No compute is created in Stage 1.**

---

**Stage 1 note:** the S3 bucket and raw upload described above were already created and
verified (single mutation, $0.06 of storage). The **next** approval needed is the cloud
compute gate — see **`reports/cost_gate.md`** for the full resource table. No EC2
instance, volume, role or security group exists yet.
