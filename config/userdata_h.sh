#!/bin/bash
# UserData: helper worker for the two-instance test run (odd shards).
# Instance A runs the normal test pipeline; this box only waits for A's test
# indices, builds the odd candidate shards, features and scores, then uploads.
exec > >(tee /var/log/amz-pipeline.log) 2>&1
set -x

export PATH="/root/.local/bin:$PATH"
BUCKET="amazon-ml-challenge-2026-357112746764"
REPO="/opt/amz/code"
DATA="/opt/amz/data"
export AMZ_DATA_DIR="$DATA"

on_exit() {
  rc=$?
  if [ -f /var/log/amz-pipeline.log ]; then
    aws s3 cp /var/log/amz-pipeline.log "s3://$BUCKET/run/status/helper-log.txt" --only-show-errors || true
  fi
  if [ -d "$REPO/logs" ]; then
    aws s3 cp "$REPO/logs" "s3://$BUCKET/run/logs/helper/" --recursive --only-show-errors || true
  fi
  {
    echo "helper_userdata_exit=$rc"
    echo "finished_utc=$(date -u +%FT%TZ)"
    echo "disk_free_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)"
    echo "uptime=$(uptime -p)"
  } | aws s3 cp - "s3://$BUCKET/run/status/HELPER_EXIT.txt" || true
  /usr/sbin/shutdown -h now || true
}
trap on_exit EXIT

mark() {
  echo "$1 utc=$(date -u +%FT%TZ)" | aws s3 cp - "s3://$BUCKET/run/status/$2.txt" || true
}

mark "helper userdata started" HU_00_START

# Only add swap when the root volume can spare it: an 8G swapfile on a full
# disk fills / and every later install step fails with ENOSPC.
avail_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
if [ "${avail_gb:-0}" -ge 16 ] && ! swapon --show --noheadings 2>/dev/null | grep -q .; then
  if fallocate -l 8G /swapfile 2>/dev/null; then
    chmod 600 /swapfile
    mkswap /swapfile >/dev/null 2>&1
    swapon /swapfile && echo "swap enabled" || echo "swap enable failed (continuing)"
  else
    echo "skipping swap (fallocate failed, avail=${avail_gb}G)"
    rm -f /swapfile || true
  fi
else
  echo "skipping swap (avail=${avail_gb}G)"
fi
export MALLOC_ARENA_MAX=2

# --- toolchain ----------------------------------------------------------
curl -LsSf https://astral.sh/uv/install.sh | sh || { mark "uv install FAILED" HU_01_FAIL; exit 1; }
export PATH="/root/.local/bin:$PATH"
uv --version || { mark "uv missing" HU_01_FAIL; exit 1; }

# --- code (zip is rooted at code/; tolerate a repo-root zip too) --------
mkdir -p /opt/amz
aws s3 cp "s3://$BUCKET/run/code.zip" /tmp/code.zip --only-show-errors \
  || { mark "code download FAILED" HU_02_FAIL; exit 1; }
rm -rf /tmp/x && mkdir -p /tmp/x /opt/amz/code
python3 -m zipfile -e /tmp/code.zip /tmp/x || { mark "code extract FAILED" HU_02_FAIL; exit 1; }
if [ -d /tmp/x/code ]; then cp -a /tmp/x/code/. /opt/amz/code/; else cp -a /tmp/x/. /opt/amz/code/; fi
chmod +x "$REPO/scripts/"*.sh
test -f "$REPO/scripts/92_helper.sh" || { mark "code layout FAILED" HU_02_FAIL; exit 1; }
mark "code extracted" HU_02_CODE

# --- python environment -------------------------------------------------
cd "$REPO" || exit 1
uv venv --python 3.12 .venv || { mark "venv FAILED" HU_03_FAIL; exit 1; }
uv pip install --python "$REPO/.venv/bin/python" -r "$REPO/requirements.txt" \
  || { mark "deps FAILED" HU_03_FAIL; exit 1; }
"$REPO/.venv/bin/python" -c "import polars,lightgbm,sklearn,rapidfuzz,pyarrow;print('deps ok')" \
  || { mark "deps import FAILED" HU_03_FAIL; exit 1; }
mark "venv ready" HU_03_VENV

# --- model (the only artifact needed before A's indices arrive) ---------
mkdir -p "$REPO/artifacts/model"
aws s3 sync "s3://$BUCKET/run/artifacts/model/" "$REPO/artifacts/model/" --only-show-errors \
  || { mark "model sync FAILED" HU_04_FAIL; exit 1; }
test -f "$REPO/artifacts/model/train/lgbm.txt" || { mark "model missing HU_04_FAIL" HU_04_FAIL; exit 1; }
mark "model restored" HU_04_MODEL

# --- helper pipeline ----------------------------------------------------
cd "$REPO" || exit 1
timeout --signal=TERM --kill-after=300s 8h bash "$REPO/scripts/92_helper.sh"
rc=$?
echo "helper rc=$rc"
echo "$rc" | aws s3 cp - "s3://$BUCKET/run/status/HELPER_RC.txt" || true
[[ $rc -eq 0 ]] && mark "helper SUCCEEDED rc=0" HU_05_DONE || mark "helper FAILED rc=$rc" HU_05_FAIL
exit $rc
