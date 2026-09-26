# Cost Gate — approval required before any paid or mutating AWS operation

Prepared: 2026-09-25 · Account `357112746764` · Region `us-east-1`
Project: Amazon ML Challenge 2026 — Baseline V1

**Nothing below this line has been created. No EC2 instance, volume, role,
security group or key pair exists yet. Every row needs an explicit go-ahead.**

## Why compute is needed at all

The local machine cannot finish the run:

| constraint | available | required (measured) |
|---|---|---|
| RAM | 7.6 GB total (~3.9 GB free now) | ~6–8 GB just to hold the 10M-row normalized target + TF-IDF index, plus ~2 GB per feature worker |
| free disk | **9.7 GB** | ~17 GB of artifacts (see below) + 2.5 GB raw |
| vCPU | 4 (Pentium N4200) | 16 for the ~2 h feature/scoring stage |

Stages 02–14 are proven end-to-end on the local dev sample (9,995 S1 / 272,474
target): candidate recall **0.9887**, validation Macro F0.5 **0.9688**,
full-dev Macro F0.5 **0.9725**, official validator **PASS**. The same code,
run with `--split train` and `--split test`, needs the full corpus.

## Required resources

| RESOURCE | REGION | INSTANCE-SERVICE | EXPECTED RUNTIME | ESTIMATED COST | WORST-CASE COST | WHY NEEDED | HOW IT WILL BE STOPPED | CURRENT PROJECT COST ESTIMATE |
|---|---|---|---|---|---|---|---|---|
| EC2 instance, **c7g.4xlarge** (16 vCPU, 32 GiB, Graviton3, Linux, on-demand) | us-east-1 | `c7g.4xlarge`, on-demand | 2.5 h expected, from the benchmark below | **$1.45** (2.5 h × $0.58) | **$4.64** (8 h × $0.58) | Full-corpus normalize → blocking → TF-IDF → features → LightGBM → threshold → submission. 32 GiB is the memory floor for the 10M-row target; 16 vCPU is what makes the ~160M feature rows finish in minutes instead of hours. | `aws ec2 terminate-instances` as the **last** command of the run, immediately after artifacts are verified and pushed to S3. No stop-but-leave-running: a stopped instance still bills the attached volume. A 8-hour wall-clock watchdog (local `timeout` + shutdown-on-complete launch flag) guarantees termination even if this session dies. | **$0.00 compute to date** (0 instances, 0 volumes, 0 endpoints in all 17 regions) |
| EBS **gp3 volume, 100 GB**, attached to that instance | us-east-1 | `CreateVolume` type `gp3`, 100 GiB | lifetime of the instance | **$0.01** (prorated, per-second billing) | **$0.09** (8 h) | ~17 GB of artifacts + 2.5 GB raw + working headroom; 100 GB is the smallest round number with real margin, so a mid-run space failure does not cost the whole job. | `DeleteVolume` runs in the same terminate script (termination deletes it only if `DeleteOnTermination` is true, which we set explicitly at launch). | $0.00 |
| S3 — existing project bucket `amazon-ml-challenge-2026-357112746764` | us-east-1 | S3 Standard, `raw/` (already uploaded) | n/a (persistent) | **$0.06**/month storage for the 2.5 GB already there + ~$0.01 PUT requests | **$0.10** | Staging the dataset for the instance (already done); pushing final artifacts and the submission back. | Not stopped — it is the project's dataset copy and costs cents. `raw/` is private, BPA enforced, owner-only ACL. | **$0.06** (measured: 2,520,812,871 bytes uploaded) |
| S3 → EC2 data transfer (same region) | us-east-1 | `GetObject` from the instance | one 2.5 GB pull | **$0.00** | **$0.00** | Instance must read the dataset. | n/a | $0.00 |
| IAM: one instance role + one S3 read/write policy (scoped to `arn:aws:s3:::amazon-ml-challenge-2026-357112746764/*`), attached as instance profile; plus `AmazonSSMManagedInstanceCore` | us-east-1 | `CreateRole` / `CreatePolicy` / `CreateInstanceProfile` | n/a (persistent, no hourly cost) | **$0.00** | **$0.00** | Least-privilege credentials for the instance: exactly one bucket, and SSM so we never open an inbound SSH port. | Role and policy are **kept** (they are free and re-usable) unless you want them removed; a removal command is included in the run script. | $0.00 |
| SSM Session Manager session (no key pair, no inbound rules, no bastion) | us-east-1 | SSM, free feature | 2.5 h | **$0.00** | **$0.00** | Execute commands on the instance without an SSH listener, security-group hole or key material. | Session ends when the instance terminates. | $0.00 |
| Security group, egress-only, **no inbound rules** | us-east-1 | `CreateSecurityGroup` | n/a | **$0.00** | **$0.00** | Default egress to S3/SSM/SSM endpoints; explicitly zero ingress. | Deleted with the instance teardown. | $0.00 |
| **TOTAL for one full run** | | | **2.5 h** | **≈ $1.52** | **≈ $4.73** | | | |

### Deliberate exclusions (per your constraint list)

GPU · SageMaker endpoints/notebooks/training jobs · NAT gateway · EMR ·
OpenSearch · RDS · Redshift · EKS · any always-running service · any cluster of
more than **one** instance · Spot (rejection risk and interruption restarts are
not worth 60% on a $1.50 job) · Lambda/Batch/Fargate (extra moving parts for no
gain) · CloudWatch agent/alarms (we log to local files and push to S3).

**Budget headroom:** worst case $4.73 of ~$100 of credit, plus $0.06 already
spent on S3. Even a full 8-hour run repeated **four times** stays under $20.

## Estimated runtime — measured, not guessed

Benchmarks run on this machine (4 vCPU Pentium N4200, 7.6 GB RAM), 2026-09-25,
`logs/bench_scale2.log`. The c7g.4xlarge has 4× the cores and far faster ones,
so every figure below is a **conservative upper bound** for the cloud run
(also assumes 4 parallel shard workers for stages 08 and 10, which the
shard-resumable design already supports).

| stage | measured locally | projected on c7g.4xlarge |
|---|---|---|
| 02 normalize (23.4M rows, train + test) | 12,469 rows/s | **16 min** |
| 04 TF-IDF fit (10M target docs) | 29.8 s / 1M docs | **6 min** (60 s/1M × 10, upper bound) |
| 04 target stats | 5.8 s / 1M rows | **1 min** |
| 05 TF-IDF query (1.73M S1 × 10M target) | 73 s / 200k S1 × 1M target | **25–50 min** (the one cost that grows faster than linearly) |
| 05 join blocks (1.73M S1) | 3.0 s / 200k S1 | **1 min** |
| 06 recall measurement | 4 s (dev) | **2 min** |
| 07 train pairs (≈40M labelled) | 4.6 s (199k) | **2 min** |
| 08 features (≈160M rows: 40M train pairs + 120M test candidates) | 25,253 pairs/s (500k in 19.8 s) | **21 min** with 4 shard workers |
| 09 LightGBM (40M × 32 features, 400 rounds) | 26.5 s (159k rows, dev) | **10 min** |
| 10 scoring (≈160M pairs) | 70,351 pairs/s (562,811 in 8 s) | **7 min** with 4 shard workers |
| 11 threshold grid | 9 s (dev) | **3 min** |
| 12 submission build | 1 s (dev) | **5 min** |
| 14 official validator | instant | **1 min** |
| overhead: boot, AMI pull, dataset download, pip deps, artifact push | — | **15 min** |
| **expected total** | | **≈ 2.0 h → budget 2.5 h** |
| **worst case** (cold start, slower query scaling, one shard-level retry, re-run of a failed stage) | | **8 h** |

## Disk budget (why 100 GB)

| artifact | dev measured | extrapolated |
|---|---|---|
| `artifacts/processed/*` (24.2M rows) | 35.6 MB / 564,938 rows | **3 GB** |
| `artifacts/indices/*` (TF-IDF per country + join stats) | 22 MB / 272k target | **2 GB** |
| `artifacts/candidates/test` (~120M pairs) | 10.1 B/row | **1.2 GB** |
| `artifacts/features/test` (~120M rows × 32 features) | 35.6 B/row | **4.3 GB** |
| `artifacts/features/train` (~40M rows) | 36.5 B/row | **1.5 GB** |
| `artifacts/predictions/*` | 9.3 B/row | **1.1 GB** |
| `artifacts/train_pairs/train` (~40M) | 10.6 B/row | **0.4 GB** |
| `submissions/*` (both TSVs, 1.73M S1) | 7.6 MB / 9,995 S1 | **1.3 GB** |
| raw dataset (copied to the volume) | 2.5 GB | **2.5 GB** |
| transient/working headroom | — | **3 GB** |
| **total** | | **≈ 17.3 GB → 100 GB volume with 5× margin** |

## Controls that keep this inside the gate

1. **Launch with a hard lifetime.** `--instance-initiated-shutdown-behavior
   terminate` plus an in-instance watchdog: `timeout 8h` wraps the whole
   pipeline, and the final step is `shutdown -h now`. Forgotten session = the
   box destroys itself, so the 8-hour worst case is a ceiling, not a risk.
2. **Terminate, never stop.** Stop would keep billing the 100 GB volume forever.
3. **Budget alert.** An AWS Budget at **$20** with an e-mail action, created
   before launch, so a surprise cannot become a $100 surprise.
4. **No inbound.** Zero ingress rules; SSM only. No key pair to leak.
5. **Least privilege.** The instance role can read/write exactly one bucket and
   nothing else. No `AdministratorAccess`, no `*`, no new managed policies
   beyond `AmazonSSMManagedInstanceCore`.
6. **Shard-resumable.** Stages 04–10 write `<file>.SUCCESS` markers, so a crash
   at shard 40/100 resumes at 41 instead of restarting a $1.45 job.
7. **Every launch is logged** in `reports/aws_cost_ledger.md` with instance ID,
   start time, terminate time and computed cost.

## Answers I do not have

* **Credit balance and expiry** — Cost Explorer returns nothing for this
  account (it appears never to have been activated), so the remaining credit
  cannot be read from the API. Confirm in **Console → Billing → Credits**
  before approving. If the balance is materially below $20, tell me and I will
  re-scale this sheet.
* **Whether a 16-vCPU Graviton instance is capacity-available in every AZ** —
  if `c7g.4xlarge` is unavailable in one AZ we fall back to `m7g.4xlarge`
  ($0.6528/h → worst case $5.22 instead of $4.64) or `c7g.2xlarge`
  ($0.29/h → worst case $2.32, but expect the runtime to double).

## Explicit approval requested

Create the following, run the pipeline once, then delete everything except the
IAM role/policy and the S3 bucket:

```
1 x IAM role "amz-ml-2026-ec2" + inline policy (S3 on the project bucket only)
1 x instance profile "amz-ml-2026-ec2"
1 x security group, egress-only, zero inbound
1 x EBS gp3 100 GB volume
1 x EC2 c7g.4xlarge in us-east-1, Amazon Linux 2023 (arm64)
1 x AWS Budget $20 with an e-mail alert
```

Expected cost **$1.52**, worst-case cost **$4.73**, running cost to date **$0.00**
(+ $0.06 S3 storage).

**Reply "approved" to create and run the above, or tell me what to change.**

---

## ADDENDUM 2026-09-25 17:50 — approved plan is not executable; revised options

The approved row `1× c7g.4xlarge (16 vCPU/32 GiB) @ $0.58/h` is **rejected by the account**:

```
InvalidParameterCombination: The specified instance type is not eligible for Free Tier.
```

This is an account-level gate, not a request error (verified by bisection — see
`reports/aws_cost_ledger.md`). Only these 8 instance types can be launched:

`t8i.micro` (1 GiB) · `t3.micro`/`t4g.micro` (1 GiB) · `t3.small`/`t4g.small`/`t8i.small` (2 GiB) ·
`c7i-flex.large` (2 vCPU / 4 GiB, $0.08479/h) · **`m7i-flex.large` (2 vCPU / 8 GiB, $0.09576/h)**

### Option A — run Baseline V1 on 1× m7i-flex.large (free-tier eligible, needs no account change)

| RESOURCE | REGION | INSTANCE-SERVICE | EXPECTED RUNTIME | ESTIMATED COST | WORST-CASE COST |
|---|---|---|---|---|---|
| 1× m7i-flex.large, 100 GB gp3 (encrypted, delete-on-terminate) | us-east-1 | EC2 on-demand | ~9 h (2 vCPU vs the 16 vCPU we planned) | **$0.86** (9 h × $0.09576) — may be $0 if free-tier hours cover it | **$1.44** (15 h guard) + $0.09 EBS |
| S3 `run/` artifacts + code | us-east-1 | S3 Standard | 1 month | $0.01 | $0.05 |

- **WHY NEEDED:** same as approved plan; the instance is simply smaller than planned.
- **HOW IT WILL BE STOPPED:** unchanged — `instance-initiated-shutdown-behavior terminate`,
  `timeout` watchdog, self-terminating UserData, $20 budget.
- **RISK:** 8 GiB RAM → feature/scoring stages must run **serially (1 worker)**, which is the
  main reason for the ~9 h estimate; Flex instances can also baseline-throttle under sustained
  load. `--chunk 100000` keeps each stage under ~2 GB. Watchdog must be raised from 7 h to ~15 h.
- **STILL CHEAPER than the approved plan** ($0.86 vs $1.45 expected).

### Option B — unblock the account, then run the approved plan unchanged

The rejection means AWS is limiting this account to free-tier instance types (typically: payment
method not verified/added, or an account in a restricted promotional state). Verify in
**Console → Billing → Credits and payment methods**. If c7g.4xlarge becomes launchable, the
originally approved sheet applies as written ($1.45 expected / $4.64 worst case) and I re-run
the launch exactly as approved.

### Option C — do not use AWS; deliver locally (not viable)

Local disk has 9.7 GB free and the artifacts need ~25 GB; RAM is 7.6 GB vs ~3 GB peak with a
single worker. Full run is not possible on this machine.

**Current spend: $0.00 compute · $0.06 S3 storage estimate · all resources created so far are $0.**
