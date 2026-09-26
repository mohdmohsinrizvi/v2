# AWS Cost Ledger — Amazon ML Challenge 2026

Rules: every planned/executed billable resource gets a row before launch. Budget ceilings:
Baseline V1 ≤ $30–40 · Baseline+V2 ≤ $60–70 · reserve ≥ $30 of the stated $100 credit.
Single operations > $2 (or un-estimatable) require explicit human approval.
Billing data is delayed — "Actual Known Cost" is only filled from the Billing console.

| Time | Resource | Purpose | Runtime | Estimated Cost | Actual Known Cost | Status |
|------|----------|---------|---------|----------------|-------------------|--------|
| 2026-09-25 13:50 | (pre-existing) Budget `My Zero-Spend Budget` $1/mo | Not created by us | n/a | $0 (free, 1st/2nd budget free) | $0.00 (reported ActualSpend) | inventoried, untouched |
| 2026-09-25 13:50 | (pre-existing) SageMaker domain `QuickSetupDomain-20260925T125597` | Not created by us; 0 apps running | n/a | $0 while idle | $0.00 | inventoried, untouched |
| 2026-09-25 13:50 | (pre-existing) S3 bucket `sagemaker-us-east-1-357112746764` | Not created by us | n/a | storage only (size not measured) | $0.00 | inventoried, untouched |
| 2026-09-25 14:13 | **S3 bucket `amazon-ml-challenge-2026-357112746764`** | Project data + artifacts store | ≤1 month | **$0.26/mo** (2.52 GB × $0.023 + ~$0.02 requests) | $0.00 (billing delayed) | **CREATED + VERIFIED** |
| 2026-09-25 14:13–14:20 | One-time raw upload (11 objects, 2,520,812,871 B) | Durable copy of competition data | ~7 min | ~$0.02 (PUT/multipart requests) | $0.00 (billing delayed) | **DONE, byte-exact verified** |

Live account inventory (2026-09-25, read-only via aws-mcp): 17 regions scanned →
**0 EC2 instances, 0 EBS volumes, 0 SageMaker notebooks/endpoints/training/processing/studio apps,
1 S3 bucket, 1 budget ($1 limit, spend $0.00), no project resources.**
Verified spend to date: **$0.00**.

## Cumulative

- Estimated to date: **$0.06** (S3 Standard storage of 2,520,812,871 bytes at $0.023/GB-month; PUT requests < $0.01)
- Actual known to date: **$0.00 compute** — Cost Explorer returns no data for this account (never activated), so EC2/EBS/SageMaker usage cannot be read from the API; inventory confirms 0 instances, 0 volumes, 0 endpoints in all 17 regions
- Budget remaining (per user-stated $100 credit): **credit balance UNVERIFIED — check Console → Billing → Credits**
- Next proposed spend: see `reports/cost_gate.md` — expected **$1.52**, worst case **$4.73** (currently BLOCKED, awaiting approval)

## Price reference (official AWS Price List API, us-east-1, fetched 2026-09-25)

| Item | Price |
|------|-------|
| S3 Standard storage | $0.023 / GB-month (first 50 TB), offer 2026-09-18 |
| S3 PUT requests | (see S3 offer file; GET/LIST similar order of magnitude) — confirm before heavy request loops |
| EC2 t3.medium / t4g.large / t3.large (Linux On-Demand) | $0.0416 / $0.0672 / $0.0832 per hour |
| EC2 **c7g.4xlarge** (16 vCPU / 32 GiB, Graviton3, us-east-1, Linux On-Demand) | **$0.58 / hour** — cross-checked: c6g.4xlarge $0.5440 is listed at -6% vs c7g |
| EC2 m7g.4xlarge (16 vCPU / 64 GiB, us-east-1) | $0.6528 / hour (aws-pricing.com + Vantage agree) |
| EC2 c7i.4xlarge (16 vCPU / 32 GiB, us-east-1) | $0.7140 / hour |
| EBS gp3 | $0.08 / GB-month, per-second prorated |
| S3 → EC2 same-region inbound data transfer | $0.00 |
| EC2 m6i.large / m7i.large / m6i.xlarge | $0.096 / $0.1008 / $0.192 per hour |
| SageMaker Studio JupyterLab ml.t3.medium / ml.t3.xlarge | $0.05 / $0.20 per hour |
| AWS Budgets | first 2 budgets free ($0.02/day each after) |

Free Tier reference: S3 5 GB/mo + 20k GET + 2k PUT (12 months, new customers);
EC2 750 h/mo t3.micro-class (account-age dependent) + T4g trial 750 h/mo t4g.small to 2026-12-31;
SageMaker 250 h/mo ml.t3.medium for first 2 months. **Account-specific eligibility:
UNKNOWN — MUST VERIFY in Console → Billing → Free Tier.**

## 2026-09-25 17:30–17:50 — cost-gate execution attempt (stage 1)

Created, all free ($0.00 each):

| Time | Resource | Purpose | Estimated Cost | Status |
|------|----------|---------|----------------|--------|
| 17:31 | IAM role `amz-ml-2026-ec2` + inline policy `project-bucket-rw` (S3 List/Get/Put on the one project bucket) + `AmazonSSMManagedInstanceCore` + instance profile | least-privilege EC2 role, SSM for shell access without inbound ports | $0 | **CREATED, verified** |
| 17:32 | Security group `sg-0a2aebb3cc26d46c7` (default VPC `vpc-01854b0ece6df7a8b`) | egress-only, **ingress = []** verified | $0 | **CREATED, verified** |
| 17:33 | AWS Budget `amz-ml-2026-cap`, $20/month, account-wide (no tag filter) | hard cost cap | $0 (budgets 1–2 free) | **CREATED, verified** |
| 17:43 | S3 object `run/code.zip` (56,701 B) | pipeline code for the instance | <$0.001 | **UPLOADED** |

No EC2 instance was launched. No instance-hours accrued.

### BLOCKER — account will not launch the approved instance type

`aws ec2 run-instances --instance-type c7g.4xlarge ...` →

> `InvalidParameterCombination: The specified instance type is not eligible for Free Tier.`

Confirmed by bisection (2026-09-25 17:47):

- minimal launch (`--image-id` + `--instance-type c7g.4xlarge` + subnet + SG only) → **same error**, so it is not caused by our tags/BDM/user-data/instance-profile.
- `--dry-run` on the identical c7g.4xlarge request → `DryRunOperation: Request would have succeeded`, so IAM/params are fine; the free-tier gate only runs on a real launch.
- `describe-instance-types --filters Name=free-tier-eligible,Values=true` → **8 eligible types only**: `t8i.micro`, `t8i.small`, `t3.micro`, `t3.small`, `t4g.micro`, `t4g.small`, `c7i-flex.large`, `m7i-flex.large`.
- Largest eligible type: **`m7i-flex.large` — 2 vCPU / 8 GiB** ($0.09576/h on-demand, us-east-1).
- `freetier get-free-tier-usage` returns `[]` (no free-tier usage rows readable via API).

Conclusion: this account can only launch free-tier-eligible instance types. The approved
plan (1× c7g.4xlarge, $0.58/h) **cannot be executed as written**. Awaiting user decision —
see revised options in `reports/cost_gate.md`.

## 2026-09-25 17:53 — APPROVED REVISED PLAN, instance launched

User approved Option A in `reports/cost_gate.md` addendum (1× m7i-flex.large instead of the
blocked c7g.4xlarge).

| Time (UTC) | Resource | Purpose | Runtime | Estimated Cost | Actual | Status |
|---|---|---|---|---|---|---|
| 2026-09-25 12:23:56 | **EC2 `i-08da82e3da2393e28`** m7i-flex.large, us-east-1a, AL2023 x86_64 `ami-0fef201115eefe936`, IMDSv2 required, no key pair, `instance-initiated-shutdown-behavior=terminate`, tags Project/ManagedBy/StopCondition | Baseline V1 full pipeline | expected ~9 h, watchdog **15 h** | **$0.86** (9 h × $0.09576) | $0.00 (billing delayed) | **LAUNCHING** |
| 2026-09-25 12:23:56 | EBS root `/dev/xvda` 100 GB gp3, encrypted, `DeleteOnTermination=true` | code + data + artifacts (~25 GB needed) | same lifetime | **$0.09** prorated (worst case $1.33/mo) | $0.00 | **CREATED** |

- IAM role/profile, egress-only SG, $20 budget, S3 `run/code.zip`: created earlier, **$0**.
- Guards in force: zero ingress (verified `IpPermissions: []`), least-privilege role,
  `timeout 15h`, self-terminating UserData (`trap` → `shutdown -h now`), $20 account budget.
- Cost sheet revised total: **expected $0.86 + $0.09 EBS ≈ $0.95 · worst case ≈ $1.53**,
  i.e. **below** the originally approved $1.45 / $4.64.

## 2026-09-25 14:52 UTC — runs 2-5 (iteration loop), run 6 launched

| Time (UTC) | Resource | Purpose | Runtime | Estimated Cost | Actual | Status |
|---|---|---|---|---|---|---|
| 2026-09-25 | EC2 `i-0f916be312cce57ac` m7i-flex.large | run 2 | ~12 min | <= $0.02 | $0.00 (billing delayed) | **TERMINATED** — stage 04 OOM (rc 137) |
| 2026-09-25 | EC2 `i-09d8a357899a69de8` m7i-flex.large | run 3 | ~15 min | <= $0.03 | $0.00 (billing delayed) | **TERMINATED** — stage 04 OOM after swap enabled |
| 2026-09-25 | EC2 `i-07f0c5b367f1ab0de` m7i-flex.large | run 4 | ~20 min | <= $0.03 | $0.00 (billing delayed) | **TERMINATED** — stage 06 OOM (214M pairs) |
| 2026-09-25 13:54:25 | EC2 `i-0b44434f83bc7f92c` m7i-flex.large | run 5 | ~40 min | <= $0.07 | $0.00 (billing delayed) | **TERMINATED** — recall gate 0.8238 < 0.97 (rc 2) |
| 2026-09-25 14:52:36 | **EC2 `i-03cada07212692561`** m7i-flex.large, us-east-1a, `ami-0fef201115eefe936`, IMDSv2, no key pair, self-terminate, tags Name=`amz-ml-2026-baseline-v1-r6` | run 6 — scaled df thresholds + k=100 | expected ~7 h, watchdog 15 h | **$0.67** (7 h x $0.09576), worst case **$1.44** (15 h) | $0.00 (billing delayed) | **RUNNING** |
| 2026-09-25 14:52:36 | EBS root `/dev/xvda` 100 GB gp3, encrypted, delete-on-terminate | code + data + artifacts | same lifetime | **$0.09** prorated | $0.00 | **CREATED** |

**Cumulative estimated cost of all runs so far: < $0.30** of the approved
$1.45 worst-case (run 6 inclusive: $0.76 worst case if it hits the 15 h cap).
S3 storage of the raw dataset upload (2.52 GB) and artifacts is billed monthly
at standard rates and is not part of the per-run EC2 figure.

### Run 6 changes vs run 5 (why the recall gate should now pass)

Run 5 failed the stage-06 gate because every df cutoff in
`TfidfConfig`/`BlockConfig` is an **absolute document count tuned on the
272,474-row dev target**, and the train target is 10,320,219 rows — 38x
larger, so the same numbers were ~38x more selective:

| method | dev recall | train recall (run 5) |
|---|---|---|
| e_name | 0.2563 | 0.2564 |
| e_ss | 0.4851 | 0.4667 |
| rare | 0.3119 | **0.0227** |
| addr | 0.3040 | **0.0357** |
| tfidf | 0.9872 | **0.7018** |

Fixes in `code.zip` (verified locally, 35 tests pass):

1. `scripts/04_build_indices.py` now scales `rare_max_df`, `addr_max_df` and
   the TF-IDF `max_df` by `target_rows / 272_474` (so dev is a no-op,
   scale=1.00), keeps `min_df=3` absolute, and logs the scale factor.
2. retrieval `k` raised 50 -> 100 everywhere (`04`, `08 --tfidf-k`,
   `09 --tfidf-k`, `99_run_full.sh`).
3. **New hard guard**: `paths.check_feature_k` — `r_tfidf_rank_norm` is
   normalised by `tfidf_k`, so features built with one k can no longer be
   scored with another.
4. **New hard guard**: `scripts/11_threshold_search.py` now refuses to pick a
   threshold when the scored candidates cover < 0.97 of the validation fold's
   ground truth. This is what caught the local regression where
   `08 --val-only` hard-coded the fold at 0.10 while `07`/`11` ran at 0.2 —
   only 52.3% of the fold was scored and Macro F0.5 collapsed to 0.6089
   despite AUC 0.99979.
5. `08` gained an explicit `--val-frac`; `10` only reuses a cached prediction
   shard when its row count matches the feature shard.

Local dev evidence after the fixes (35 tests pass):

- pair candidate recall **0.9912** (gate 0.97) at 104.3 pairs/S1
- LightGBM AUC **0.99979**, best threshold 0.99, val Macro F0.5 **0.9701**
- full-dev submission Macro F0.5 **0.9702** (0.9701 @ thr 0.99)

---

## Run 6 — live status (instance `i-03cada07212692561`, launched 2026-09-25T14:52:36Z)

### Full-corpus recall gate PASSED

`pair_candidate_recall = 0.9723` vs gate `0.97` (`gate_pass: true`, stage 4 of 16,
log line 256). The probe taken from the first five shards (0.9724) was
essentially exact — no sampling drift.

| per-method | value |
|---|---|
| e_name | 0.2564 |
| e_ss | 0.4667 |
| rare | 0.3756 |
| addr | 0.2482 |
| tfidf | 0.9666 |
| tfidf top5 / top10 / top20 / top50 | 0.8334 / 0.9213 / 0.9432 / 0.959 |

Stage 5 (`07_build_train_pairs`): 7,638,365 positives, 38,191,639 negatives,
**45,830,004 labelled pairs**, pos_rate 0.1667, val fold 219,443 / 2,206,821
S1 (`val_frac_actual` 0.0994), elapsed 1685 s.

### Hot-patches applied to the running instance (SSM, files re-read per stage)

`99_run_full.sh` cannot be edited while bash is executing it, but every stage
spawns a fresh Python process, so scripts were patched in place:

1. **`08 --chunk` floor at 1,000,000.** `pair_features` rebuilds the hash tables
   for the 2.2M-row Source-1 and 10.3M-row target frames on every call, a fixed
   ~15 s cost. At the orchestrator's `--chunk 200000` that capped throughput at
   ~11k rows/s; at 1M it measures **93k rows/s** (10.7 s/1M on the instance).
   Test is ~582M candidate pairs, so this alone is the difference between
   ~14 h and ~1.8 h of feature generation.
2. **`08` candidates mode now slices by `--chunk`.** Latent OOM: the mode
   ignored `--chunk` and joined a whole 33.6M-row candidate shard against the
   10.3M-row target — a ~21 GB peak that would have killed the test split.
   Output is now written as one file per piece at `index = shard*128 + piece`
   (pure function of the input, so restarts are idempotent).
3. **`09` logs evaluation every 50 rounds** so training progress is visible in
   the stage log instead of a silent 30+ minutes.

All three are in the re-uploaded `s3://.../run/code.zip` (52 entries, 66,877 B),
so the next run starts with them. 35 local tests still pass.

### Throughput measured on the instance (2 vCPU)

| stage | measured |
|---|---|
| 04 train indices (ids/stats/tfidf) | 1 s / 70 s / 103 s |
| 05 candidates (23 shards) | ~550 s per 100k-S1 shard -> ~3.5 h |
| 06 recall gate (full corpus) | 77 s |
| 07 train pairs (45.8M) | 1685 s |
| 08 features, pairs (45.8M) | **503 s** |
| 08 features, val-only candidates (73.7M) | ~38 s/shard -> ~15 min |

---

## Run 6 — outcome (completed 2026-09-25 22:42 UTC)

Full-corpus **recall gate passed**: `pair_candidate_recall: 0.9723 >= 0.97`
(`gate_pass: true`).  Final stage timings (UTC, instance start 14:52:49):

| stage | start -> done | elapsed |
|---|---|---|
| 02 normalize train | 14:52:49 -> 14:57:13 | 3m53s |
| 04 indices train | -> 15:00:23 | 3m10s |
| 05 candidates train | -> **18:22:22** | **3h22m** |
| 06 recall gate | -> 18:23:41 | 1m19s |
| 07 train pairs (45.8M) | -> 18:51:50 | 28m09s |
| 08 features, pairs (45.8M) | -> 19:00:15 | 8m25s |
| 08 features, val candidates (73.6M) | -> 19:14:03 | 13m48s (89k rows/s) |
| 09 LightGBM (41.3M/4.6M, 600 iters) | -> 19:49:13 | 35m10s (val_auc 0.999710) |
| 10 score val candidates (73.6M) | -> 20:38:01 | 48m48s (25.1k rows/s) |
| 11 threshold search | 20:38 -> **stuck** | killed 22:42 |

### Stage 11 bug (root cause, fixed)

`scripts/11_threshold_search.py` calls `np.searchsorted(sid_a, s, ...)` where
`sid_a` is **Int32** (73.6M rows) and `s` is an **Int64** scalar.  numpy
promotes the *array* to int64 on every call — a full 73.6M-element cast:

```
int32 array vs int64 key : 183.1 ms per call
int64 array vs int64 key :   0.005 ms per call
```

Line 96 does `2 * 219,443 = 438,886` such calls -> **22 h of pure casting**.
Fix: one `sid_a.astype(np.int64)` up front (line 93).  Re-run completed in
**4.5 minutes** instead of 2 h+ of stall.

Threshold report: `best_threshold = 0.96`, `best_macro_f05 = 0.9363`,
avg 3.115 preds/S1, coverage 0.9724, 12,250 singletons in the fold.
(Full-corpus recall is 0.9723 vs 0.9912 on dev, which is what pulls Macro F0.5
from the dev value 0.9701 down to 0.9363 on this fold.)

### Free-tier instance restriction (blocks the 8-vCPU plan)

`run-instances` for `m7i.2xlarge` failed with
`InvalidParameterCombination: The specified instance type is not eligible for
Free Tier`.  Free-tier-eligible types in us-east-1 are **all 2 vCPU**:
`c7i-flex.large, m7i-flex.large, t3.micro/small, t4g.micro/small,
t8i.micro/small`.  EC2 quota is 8 vCPU, but the account only accepts
free-tier-eligible types, so the fastest single instance is `m7i-flex.large`
(2 vCPU / 8 GiB).  Parallel speed-up is only possible by running **more**
2-vCPU instances, not a bigger one.

### Artifacts harvested from run 6 (uploaded before shutdown)

- `run/artifacts/model/train/{lgbm.txt,meta.json}` — 4.2 MB, 600 trees
- `run/reports/*` — incl. `threshold_train.json`, `candidate_recall_train.json`
- `run/code.zip` rebuilt (54 files, rooted at `code/`) with the stage-11 fix,
  the stage-08/09/10 hot-patches, and `PIPE_MODE=test` support in `99_run_full.sh`

## Run 7 — test-only restart (instance i-0ff1c1a828c342047)

`m7i-flex.large` us-east-1, 100 GB gp3, shutdown behaviour `stop`,
watchdog 12 h, `PIPE_MODE=test` (stages 1-10 restored from S3),
`FEAT_WORKERS=1 SCORE_WORKERS=2` (2 threads on 2 cores; 4 workers x default
threads measured only 25.1k rows/s).

Started **23:02:46 UTC**.  Setup (uv + venv + 2.5 GB data + artifacts) took
**34 s**.  Remaining: normalize test, indices test, candidates test (~2.6 h),
features test 582M (~1.9 h), score test (~3.7 h at 44k rows/s), submission.
Projected finish **~07:40 UTC**.  Cost: ~9 h x $0.09576/h = **~$0.86** +
EBS ~$0.06.

### Run 7 progress (23:02-23:45 UTC)

| stage | started | finished |
|---|---|---|
| setup (uv/venv/data/artifacts) | 23:02:46 | 23:03:20 (34 s) |
| [02] normalize test | 23:03:30 | 23:19:20 (~16 min, 5.08M + 9.97M rows) |
| [04] indices test (ids/stats/tfidf) | 23:19:30 | 23:24:25 (2.9 min) |
| [05] candidates test (18 shards) | 23:24:30 | est. 01:16 (~6.2 min/shard) |

## Run 8 - two-instance split (helper `i-0dafc89f92d67679a`)

A single `m7i-flex.large` cannot finish in one watchdog window: candidates
1.9 h + features 1.9 h + scoring 3.7 h = 7.5 h of serialized work on 2 vCPU,
and the submission could only start after it all.  **Go as fast as money
allows** -> run a second identical free-tier box on a disjoint half of the
work.

**Design.**  A keeps even candidate shards, the helper keeps odd ones.  The
two halves never write the same file:

- `08_generate_features.py` now takes `--shard-parity {all,0,1}`; A runs with
  `default="0"` (patched in place on the instance), the helper owns only odd
  shards so it stays on `all`.
- Output names in `08` (candidates mode) and `10` are now derived from the
  **source shard number** in the filename instead of `enumerate()` of the
  local file list.  Without this both machines would start at index 0 and
  write colliding `shard_00000.parquet` names.
- `05_generate_candidates.py` gained `--worker-index/--num-workers` so the
  helper builds shards 1,3,...,17 locally (no candidate transfer: building is
  faster than shipping 15 GB).
- `12_build_submission.py` gained `merge_helper_predictions()`: it polls
  `run/status/H_PREDS_DONE.txt` for up to 3 h and then `aws s3 sync`s the
  helper's scores.  Gated on `artifacts/helper.expected` so a normal
  single-instance run never blocks.

**Latent bug found while doing this.**  The stage-08 STRIDE=128 hot-patch
made score shards `i*128+piece` while candidate shards stayed `0..N-1`, but
`12` joined the two dicts on their index and used the intersection.  With
N=18 that kept only score index 0 and **dropped every other shard** from
`matching_results.tsv`.  The two outputs are independent, so `12` now feeds
each its full set (35 unit tests still pass).

**Helper launch - first attempt failed.**  `i-01c616d102e74a345` died in 65 s
with `Client.InstanceInitiatedShutdown`: the AMI root volume is only 8 GB, the
8 GB swapfile filled it (`dd: No space left on device`), and the next step
(`mkdir /root/.local`) failed ENOSPC, so the EXIT trap shut the box down.
Fixed two ways - userdata now only creates swap when `/` has >= 16 GB free,
and the replacement instance is launched with an explicit 100 GB gp3 volume
like A.  Also fixed the `set -e`-unsafe `fallocate || dd` chain.

**Helper `i-0dafc89f92d67679a`** - `m7i-flex.large`, us-east-1, 100 GB gp3,
AMI `ami-0fef201115eefe936`, same subnet/SG/instance-profile as A, no key
pair (SSM only), tag `Role=helper-odd-shards`, 8 h `timeout` in userdata.

Launched 23:39 UTC.  Through uv + venv + model + code in **20 s**
(`HU_00_START` -> `HU_04_MODEL`), then `H_INPUTS`/`H_SYNC` pulled A's 1.5 GB
`processed/test` + 725 MB `indices/test` from S3 in ~10 s (same-region) and
started `05` on the odd half.

Both boxes are now running `05` in parallel (A: all 18 shards, H: 9 odd
shards).  A then takes even features/scores, H odd; H is the long pole at
~04:00 UTC and A's `12` waits for `H_PREDS_DONE` before writing the
submission.  Expected submission ~04:15-04:30 UTC vs ~07:40 single-box.

**Cost.**  Two `m7i-flex.large` for ~4.5 h = 9 instance-hours x $0.09576 =
**~$0.86** (unchanged from the single-box estimate - we buy wall-clock, not
compute), plus ~0.2 GB-month of extra gp3 on the helper (~$0.01).

## Credit balance (read via `billing:GetCredits`, 2026-09-26 00:05 UTC)

`cost_gate.md` recorded that Cost Explorer returned nothing for this account
("appears never to have been activated") so the balance could not be read from
the API.  It can now - the `Billing` API `GetCredits` returns all three
promotions, each still at **full** remaining amount:

| creditId | description | initial | remaining | expires |
|---|---|---|---|---|
| 10067756434 | AWS Free Tier | $100.00 | **$100.00** | 2027-09-24 |
| 10067788511 | Explore AWS: set up a cost budget | $20.00 | **$20.00** | 2027-09-24 |
| 10067820254 | Explore AWS: launch an EC2 instance | $20.00 | **$20.00** | 2027-09-24 |
| | **total** | **$140.00** | **$140.00** | |

All three `creditStatus=ENABLED`, `applicationType` applies during billing.
The EC2-launch credit was granted at **23:03:09 UTC**, i.e. seconds after run 7
came up, and the budget credit the previous morning - so the Explore series is
being earned as we use the account.

Cost Explorer for 2026-09-01..09-26 reports **$0.00** Unblended and Net
(`Estimated: true`) across EC2, S3, VPC, Glue, KMS, SQS, SNS - the run-6/7
charges have not posted yet, they land with the usual ~24 h billing lag.
Projected consumption of this project: **~$1.30** of the $140, i.e. **~1%**.

## Run 8 — outcome: submission produced and validated (2026-09-26 06:22 UTC)

Both boxes were restarted 04:54 UTC.  User data does **not** re-run on
Stop/Start, so the rest of the run was driven manually over SSM
(`/tmp/opencode/drive_a.sh`, `/tmp/opencode/drive_h.sh`, base64 one-liners).

| time (UTC) | event |
|---|---|
| 04:54 | both instances started (launched 2026-09-26T04:54:44Z) |
| 05:05 | A stage 12 relaunched after adding the `threshold_train.json` fallback |
| 06:17 | helper wrote `H_PREDS_DONE.txt` (246/246 odd-shard scores) |
| **06:22:21** | **stage 12 wrote the submission** (`S12_SUBMISSION.txt`) |
| 06:23:30 | `SUBMISSION_DONE.txt`, artifacts uploaded to `run/artifacts/submissions/` |
| 06:44:52 | both instances **stopped** (billing for compute halted) |

**Artifacts** — `s3://…/run/artifacts/submissions/`
`matching_results.tsv` 94,621,216 B · `candidate_pairs.tsv` 6,502,597,351 B ·
`submission_summary.json` 273 B.

**Validation (both passed).**

- Official `student_resource/utils/validate_submission.py --check-ids` on
  `matching_results.tsv`: **PASS**, `VALIDATOR_RC=0` — 1,732,544 rows
  (105,493 empty / 1,627,051 non-empty), all match ids valid against the
  9,969,589 test S2/S3 ids.
- The candidate cross-check could not run through the official validator: it
  loads the whole 6.1 GiB candidate file and was **OOM-killed (rc 137)** on the
  8 GiB box.  Replaced with a streaming subset check
  (`check_subset.py`, keeps only the 1.6M matched S1 rows) — **PASS**:
  all 1,627,051 matched rows have every id present in their candidate list,
  across all 1,732,544 candidate rows.

`submission_summary.json`: threshold **0.96**, s1_rows 1,732,544,
s1_with_matches 1,627,051 (93.9%), match_ids 5,600,660 (avg 3.233/S1),
candidate_ids 502,768,853 (avg 290.2/S1), stage 12 elapsed 4,687 s.

**Bugs found and fixed in this run** (all had been latent until the two-box
split exposed them):

1. **Row blow-up in `12`** — score files are per *piece*, and pieces slice
   *rows* not sids, so one sid appeared in many pieces; `group_by` ran once per
   piece and wrote 48,756,072 rows against 1,732,544.  Fixed by grouping the
   full piece list per candidate shard before `group_by`, with a loud
   `sid < ptr` overlap guard; marker-chain lookup (`score -> feature ->
   candidate`) replaces arithmetic.  39 tests pass (`test_submission_grouping.py`).
2. **Helper scored only half the test set** — H5 ran
   `--worker-index 0 --num-workers 2`, which covers even feature indices only:
   125 of 246 files, and every odd-index piece (~120M pairs) had no score.
   Fixed to workers 1 and 3 of `--num-workers 4` plus a coverage assertion.
3. **A's scores were named by position, not by feature** — A's instance copy
   of `10` predates the source-shard naming fix, so its 267 scores landed as
   `shard_00000..shard_00266`.  When the helper's correctly named half was
   merged in, 16 position-named files were overwritten.  Repair renamed 222
   files from their `src` markers (two-phase, to avoid stepping on unrenamed
   files) and rescored the 15 genuinely missing features in ~11 min; final
   coverage 267/267.
4. `12` looked only for `threshold_test.json` on the test split — falls back to
   `threshold_train.json`.

**Run 8 cost.**  Both `m7i-flex.large` ran 04:54–06:44 UTC (1.83 h each =
3.67 instance-hours ≈ **$0.35**) on top of the 23:39–04:24 window already
estimated; total project compute is still far under the $140 credit
(`reports/cost_gate.md` ceiling $30–40).  Boxes are **stopped**, not
terminated: 2 × 100 GB gp3 continues to accrue ~$0.53/day until deleted.
