#!/bin/bash
# UserData: Amazon ML Challenge 2026 Baseline V1 — one-shot, terminates itself.
exec > >(tee /var/log/amz-pipeline.log) 2>&1
set -x

export PATH="/root/.local/bin:$PATH"
BUCKET="amazon-ml-challenge-2026-357112746764"
REPO="/opt/amz/code"
DATA="/opt/amz/data"
export AMZ_DATA_DIR="$DATA"

on_exit() {
  rc=$?
  # capture the whole console log before the instance disappears
  if [ -f /var/log/amz-pipeline.log ]; then
    aws s3 cp /var/log/amz-pipeline.log "s3://$BUCKET/run/status/instance-log.txt" --only-show-errors || true
  fi
  if [ -d "$REPO/logs" ]; then
    aws s3 cp "$REPO/logs" "s3://$BUCKET/run/logs/" --recursive --only-show-errors || true
  fi
  if [ -f "$REPO/reports/candidate_recall_train.json" ]; then
    aws s3 cp "$REPO/reports" "s3://$BUCKET/run/reports/" --recursive --only-show-errors || true
  fi
  {
    echo "userdata_exit=$rc"
    echo "finished_utc=$(date -u +%FT%TZ)"
    echo "disk_free_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)"
    echo "uptime=$(uptime -p)"
  } | aws s3 cp - "s3://$BUCKET/run/status/USERDATA_EXIT.txt" || true
  /usr/sbin/shutdown -h now || true
}
trap on_exit EXIT

mark() {
  echo "$1 utc=$(date -u +%FT%TZ)" | aws s3 cp - "s3://$BUCKET/run/status/$2.txt" || true
}

mark "userdata started" UD_00_START

# 8 GiB instance, no swap by default: a transient spike would OOM-kill the
# pipeline (stage 02->04 already did once). 4 GiB of EBS-backed swap costs
# nothing beyond the volume we are already paying for and only pages under
# memory pressure.
if ! swapon --show --noheadings 2>/dev/null | grep -q .; then
  fallocate -l 8G /swapfile 2>/dev/null || dd if=/dev/zero of=/swapfile bs=1M count=8192 status=none
  chmod 600 /swapfile
  mkswap /swapfile >/dev/null 2>&1
  swapon /swapfile && echo "swap enabled: $(swapon --show --noheadings)" || echo "swap enable failed (continuing)"
fi
# glibc arena fragmentation adds up across polars/sklearn workers
export MALLOC_ARENA_MAX=2

# --- 1. toolchain -------------------------------------------------------
curl -LsSf https://astral.sh/uv/install.sh | sh || { mark "uv install FAILED" UD_01_FAIL; exit 1; }
export PATH="/root/.local/bin:$PATH"
uv --version || { mark "uv missing" UD_01_FAIL; exit 1; }
mark "uv ready" UD_01_UV

# --- 2. code ------------------------------------------------------------
mkdir -p /opt/amz
aws s3 cp "s3://$BUCKET/run/code.zip" /tmp/code.zip --only-show-errors \
  || { mark "code download FAILED" UD_02_FAIL; exit 1; }
python3 -m zipfile -e /tmp/code.zip /opt/amz || { mark "code extract FAILED" UD_02_FAIL; exit 1; }
chmod +x "$REPO/scripts/"*.sh
mark "code extracted" UD_02_CODE

# --- 3. python environment ---------------------------------------------
cd "$REPO" || exit 1
uv venv --python 3.12 .venv || { mark "venv FAILED" UD_03_FAIL; exit 1; }
uv pip install --python "$REPO/.venv/bin/python" -r "$REPO/requirements.txt" \
  || { mark "deps FAILED" UD_03_FAIL; exit 1; }
"$REPO/.venv/bin/python" -c "import polars,lightgbm,sklearn,rapidfuzz,pyarrow;print('deps ok')" \
  || { mark "deps import FAILED" UD_03_FAIL; exit 1; }
mark "venv ready" UD_03_VENV

# --- 4. dataset ---------------------------------------------------------
mkdir -p "$DATA"
aws s3 sync "s3://$BUCKET/raw/" "$DATA/" --only-show-errors \
  || { mark "data sync FAILED" UD_04_FAIL; exit 1; }
du -sh "$DATA" | tee /dev/stderr
ls -la "$DATA/train" "$DATA/test"
mark "data synced" UD_04_DATA

# --- 5. pipeline --------------------------------------------------------
cd "$REPO" || exit 1
# 2 vCPU / 8 GiB: one feature worker (holds s1 + full target), four scoring workers
export FEAT_WORKERS=1
export SCORE_WORKERS=4
timeout --signal=TERM --kill-after=300s 15h bash "$REPO/scripts/99_run_full.sh"
rc=$?
echo "pipeline rc=$rc"
echo "$rc" | aws s3 cp - "s3://$BUCKET/run/status/PIPELINE_RC.txt" || true
[[ $rc -eq 0 ]] && mark "pipeline SUCCEEDED rc=0" UD_05_DONE || mark "pipeline FAILED rc=$rc" UD_05_FAIL
exit $rc
