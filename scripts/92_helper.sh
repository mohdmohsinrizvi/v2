#!/usr/bin/env bash
# Helper half of the two-instance test run.
#
# Instance A runs scripts/99_run_full.sh (stages 11-16) and keeps the even
# candidate shards; this machine takes the odd ones. The two halves never
# touch the same file: scripts/05 splits by shard number, scripts/08 derives
# its output names from the source shard number, and scripts/10 names its
# predictions after the feature shard number.
#
# Inputs (processed/test + indices/test) are produced by A and pulled from S3;
# the scores this machine produces go back to S3 for A to merge.
set -euo pipefail

REPO="${REPO:-/opt/amz/code}"
export AMZ_DATA_DIR="${AMZ_DATA_DIR:-/opt/amz/data}"
PY="${PY:-$REPO/.venv/bin/python}"
BUCKET="${BUCKET:-amazon-ml-challenge-2026-357112746764}"
S3_OUT="s3://$BUCKET/run/artifacts"

cd "$REPO"
mkdir -p logs

hs() {  # hs <marker> <message>
  printf '%s\n  time=%s\n  host=%s\n  message=%s\n' \
    "$2" "$(date -u +%FT%TZ)" "$(hostname)" "$2" \
    | aws s3 cp - "s3://$BUCKET/run/status/$1.txt" --only-show-errors || true
}

hs H_START "helper started (odd shard half)"

echo "=== [H1] waiting for A's test indices ==="
for i in $(seq 1 3600); do
  aws s3 ls "s3://$BUCKET/run/status/INPUTS_READY.txt" >/dev/null 2>&1 && break
  sleep 5
done
aws s3 ls "s3://$BUCKET/run/status/INPUTS_READY.txt" >/dev/null 2>&1 \
  || { hs H_FAIL "inputs never arrived"; exit 1; }
hs H_INPUTS "inputs marker seen; downloading"

echo "=== [H2] download processed + indices ==="
aws s3 sync "$S3_OUT/processed/test/" "$REPO/artifacts/processed/test/" --only-show-errors
aws s3 sync "$S3_OUT/indices/test/"   "$REPO/artifacts/indices/test/"   --only-show-errors
du -sh "$REPO/artifacts/processed/test" "$REPO/artifacts/indices/test"
hs H_SYNC "processed+indices downloaded"

echo "=== [H3] candidates test (odd shards) ==="
"$PY" scripts/05_generate_candidates.py --split test --chunk 100000 \
  --worker-index 1 --num-workers 2 2>&1 | tee logs/h05.log
hs H05 "odd candidate shards done"

echo "=== [H4] features test ==="
"$PY" scripts/08_generate_features.py --split test --mode candidates \
  --chunk 200000 --tfidf-k 100 \
  --worker-index 0 --num-workers 1 2>&1 | tee logs/h08.log
hs H08 "helper feature shards done"

echo "=== [H5] score test ==="
# scripts/10 partitions the feature list by i % num_workers, and this box
# holds only the ODD candidate shards, whose features are named i*STRIDE+piece
# so both even and odd piece indices occur here. The first run used
# --worker-index 0 --num-workers 2, which took only the even piece indices and
# silently left every odd-index piece unscored: 125 of 246 shards, i.e. half
# of the odd half of the candidate set. num_workers=4 with workers 1 and 3
# covers exactly the odd indices and splits them across both vCPUs (workers 0
# and 2 of that layout are the even pieces -- already scored, and scripts/10
# skips them by row-count match).
pids=()
for wi in 1 3; do
  "$PY" scripts/10_score_candidates.py --split test --model-split train \
    --worker-index "$wi" --num-workers 4 > "logs/h10_w${wi}.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
if [ "$rc" -ne 0 ]; then
  tail -n 40 logs/h10_w*.log >&2 || true
  hs H_FAIL "scoring worker failed rc=$rc"
  exit 1
fi

echo "=== [H5b] coverage check ==="
nf=$(ls "$REPO/artifacts/features/test/candidates/"*.parquet 2>/dev/null | wc -l)
np=$(ls "$REPO/artifacts/predictions/test/"*.parquet 2>/dev/null | wc -l)
echo "feature shards=$nf prediction shards=$np"
if [ "$np" -ne "$nf" ]; then
  hs H_FAIL "predictions $np != features $nf"
  exit 1
fi
hs H10 "helper predictions done ($np shards)"

echo "=== [H6] upload predictions ==="
aws s3 sync "$REPO/artifacts/predictions/test/" "$S3_OUT/predictions/test/" --only-show-errors
hs H_PREDS_DONE "helper predictions uploaded"

echo "=== helper complete ==="
