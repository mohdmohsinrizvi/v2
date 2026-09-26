#!/usr/bin/env bash
# Full Baseline V1 run: train -> threshold -> test -> submission.
# Designed for one 16-vCPU Graviton instance; every stage uploads a status
# marker to S3 so progress is observable without inbound SSH.
set -euo pipefail

REPO="${REPO:-/opt/amz/code}"
export AMZ_DATA_DIR="${AMZ_DATA_DIR:-/opt/amz/data}"
PY="${PY:-$REPO/.venv/bin/python}"
WORKERS="${WORKERS:-4}"
# 8 GiB RAM: feature stages hold s1 + full target per worker -> 1 worker.
# scoring only holds one feature shard + the booster -> safe to fan out.
FEAT_WORKERS="${FEAT_WORKERS:-1}"
SCORE_WORKERS="${SCORE_WORKERS:-4}"
BUCKET="${BUCKET:-amazon-ml-challenge-2026-357112746764}"
STATUS_PREFIX="run/status"

cd "$REPO"
mkdir -p logs reports

status() {
  local name="$1"; shift
  printf '%s\n  time=%s\n  stage=%s\n  pid=%s\n  free_disk_gb=%s\n  uptime=%s\n' \
    "$*" "$(date -u +%FT%TZ)" "$name" "$$" \
    "$(df -BG --output=avail / | tail -1 | tr -dc 0-9)" "$(uptime -p 2>/dev/null || true)" \
    | aws s3 cp - "s3://$BUCKET/$STATUS_PREFIX/$name.txt" --only-show-errors || true
}

run() { echo ">>> $*"; "$@"; }

par() {
  # parallel shard workers: par <n-workers> <stage-name> <script> <extra args...>
  local n="$1"; shift
  local stage="$1"; shift
  local pids=() i
  for ((i = 0; i < n; i++)); do
    "$PY" "$@" --worker-index "$i" --num-workers "$n" > "logs/${stage}_w${i}.log" 2>&1 &
    pids+=($!)
  done
  local rc=0 pid
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  if [[ $rc -ne 0 ]]; then
    tail -n 40 logs/"${stage}"_w*.log >&2 || true
    return 1
  fi
}

TOTAL_START=$(date +%s)
status STARTED "Baseline V1 run start (mode=${PIPE_MODE:-full} feat_workers=$FEAT_WORKERS score_workers=$SCORE_WORKERS)"

if [[ "${PIPE_MODE:-full}" == "test" ]]; then
  # Restored-artifact run: stages 1-10 already produced the model and the
  # threshold on a previous instance; they were synced from S3 by userdata.
  echo "=== [1/16-10/16] skipped (PIPE_MODE=test) ==="
  THRESHOLD=$("$PY" -c 'import json;print(json.load(open("reports/threshold_train.json"))["best_threshold"])')
  echo "selected threshold (restored) = $THRESHOLD"
  status S11_THRESHOLD "threshold restored from artifacts (best=$THRESHOLD)"
else

echo "=== [1/16] normalize train ==="
run "$PY" scripts/02_normalize.py --split train
status S02_TRAIN_NORM "normalize train done"

# 04 is three separate processes on purpose: each reads only the columns it
# needs and exits, so the next phase starts from a clean RSS. Running them as
# one process peaks past 8 GiB and gets OOM-killed.
echo "=== [2/16] indices train (ids/stats/tfidf) ==="
run "$PY" scripts/04_build_indices.py --split train --phase ids --k 100
run "$PY" scripts/04_build_indices.py --split train --phase stats --k 100
run "$PY" scripts/04_build_indices.py --split train --phase tfidf --k 100
status S04_TRAIN_IDX "indices train done"

echo "=== [3/16] candidates train ==="
run "$PY" scripts/05_generate_candidates.py --split train --chunk 100000
status S05_TRAIN_CAND "candidates train done"

echo "=== [4/16] recall gate train (>=0.97) ==="
run "$PY" scripts/06_candidate_recall.py --split train
status S06_TRAIN_RECALL "recall gate measured"

echo "=== [5/16] train pairs ==="
run "$PY" scripts/07_build_train_pairs.py --split train --per-pos 5 --val-frac 0.10
status S07_TRAIN_PAIRS "train pairs done"

echo "=== [6/16] features: train pairs ==="
par "$FEAT_WORKERS" feat_pairs scripts/08_generate_features.py --split train --mode pairs --chunk 200000 --tfidf-k 100
status S08_PAIR_FEATS "pair features done"

echo "=== [7/16] features: train candidates (val fold only) ==="
par "$FEAT_WORKERS" feat_cand_tr scripts/08_generate_features.py --split train --mode candidates \
  --chunk 200000 --val-only --tfidf-k 100 --val-frac 0.10
status S08_VAL_CAND_FEATS "val-fold candidate features done"

echo "=== [8/16] train LightGBM ==="
run "$PY" scripts/09_train_lgbm.py --split train --rounds 600 --early-stopping 50 --tfidf-k 100
status S09_MODEL "model trained"

echo "=== [9/16] score train (val fold) candidates ==="
par "$SCORE_WORKERS" score_tr scripts/10_score_candidates.py --split train
status S10_TRAIN_SCORE "val-fold scores done"

echo "=== [10/16] threshold search ==="
run "$PY" scripts/11_threshold_search.py --split train --val-frac 0.10
THRESHOLD=$("$PY" -c 'import json;print(json.load(open("reports/threshold_train.json"))["best_threshold"])')
echo "selected threshold = $THRESHOLD"
status S11_THRESHOLD "threshold search done (best=$THRESHOLD)"

fi

echo "=== [11/16] normalize test ==="
run "$PY" scripts/02_normalize.py --split test
status S02_TEST_NORM "normalize test done"

echo "=== [12/16] indices test (ids/stats/tfidf) ==="
run "$PY" scripts/04_build_indices.py --split test --phase ids --k 100
run "$PY" scripts/04_build_indices.py --split test --phase stats --k 100
run "$PY" scripts/04_build_indices.py --split test --phase tfidf --k 100
status S04_TEST_IDX "indices test done"

echo "=== [13/16] candidates test ==="
run "$PY" scripts/05_generate_candidates.py --split test --chunk 100000
status S05_TEST_CAND "candidates test done"

echo "=== [14/16] features: test candidates ==="
par "$FEAT_WORKERS" feat_cand_te scripts/08_generate_features.py --split test --mode candidates \
  --chunk 200000 --tfidf-k 100
status S08_TEST_FEATS "test features done"

echo "=== [15/16] score test candidates ==="
par "$SCORE_WORKERS" score_te scripts/10_score_candidates.py --split test --model-split train
status S10_TEST_SCORE "test scores done"

echo "=== [16/16] build submission ==="
run "$PY" scripts/12_build_submission.py --split test --threshold "$THRESHOLD"
status S12_SUBMISSION "submission built"

ELAPSED=$(( $(date +%s) - TOTAL_START ))
{
  echo "stage=all_done"
  echo "elapsed_seconds=$ELAPSED"
  echo "elapsed_hours=$(awk -v s="$ELAPSED" 'BEGIN{printf "%.2f", s/3600}')"
  echo "workers_feat=$FEAT_WORKERS workers_score=$SCORE_WORKERS"
  echo "disk_avail_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)"
  echo "finished_utc=$(date -u +%FT%TZ)"
} | aws s3 cp - "s3://$BUCKET/$STATUS_PREFIX/PIPELINE_DONE.txt" --only-show-errors || true

# ship artifacts, model, thresholds and the submission
for p in reports submissions model predictions threshold; do
  if [[ -e "$REPO/$p" ]]; then
    aws s3 cp "$REPO/$p" "s3://$BUCKET/run/artifacts/$p" --recursive --only-show-errors || true
  fi
done
aws s3 cp "$REPO/submissions/matching_results.tsv" \
  "s3://$BUCKET/run/final/matching_results.tsv" --only-show-errors || true
aws s3 cp "$REPO/submissions/candidate_pairs.tsv" \
  "s3://$BUCKET/run/final/candidate_pairs.tsv" --only-show-errors || true

status ALL_DONE "Baseline V1 complete in ${ELAPSED}s"
