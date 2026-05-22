#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
HOST="${PRECISE_SDE_GENEVAL_HOST:-127.0.0.1}"
PORT="${PRECISE_SDE_GENEVAL_PORT:-18085}"
NUM_DEVICES="${PRECISE_SDE_GENEVAL_NUM_DEVICES:-1}"
DEFAULT_MODEL_DIR="${SCRIPT_DIR}/model/mask2former2"
DEFAULT_MMDET_DIR="${SCRIPT_DIR}/mmdetection"
BOOTSTRAP_STATE_FILE="${SCRIPT_DIR}/.bootstrap-state"
VENV_DIR="${SCRIPT_DIR}/.venv"
VENV_GUNICORN="${VENV_DIR}/bin/gunicorn"

export PRECISE_SDE_GENEVAL_MODEL_DIR="${PRECISE_SDE_GENEVAL_MODEL_DIR:-$DEFAULT_MODEL_DIR}"
export PRECISE_SDE_GENEVAL_MMDET_DIR="${PRECISE_SDE_GENEVAL_MMDET_DIR:-$DEFAULT_MMDET_DIR}"
export PRECISE_SDE_GENEVAL_HOST="$HOST"
export PRECISE_SDE_GENEVAL_PORT="$PORT"
export PRECISE_SDE_GENEVAL_NUM_DEVICES="$NUM_DEVICES"

bootstrap_error() {
  echo "[ERROR] $1" >&2
  echo "        Run bash precise_sde/rewards/servers/geneval/bootstrap.sh first." >&2
  exit 1
}

if [ ! -f "$BOOTSTRAP_STATE_FILE" ]; then
  bootstrap_error "GenEval bootstrap state missing at $BOOTSTRAP_STATE_FILE"
fi

mapfile -t BOOTSTRAP_STATE < "$BOOTSTRAP_STATE_FILE"
BOOTSTRAPPED_MODEL_DIR="${BOOTSTRAP_STATE[0]:-}"
BOOTSTRAPPED_MMDET_DIR="${BOOTSTRAP_STATE[1]:-}"

if [ -z "$BOOTSTRAPPED_MODEL_DIR" ] || [ -z "$BOOTSTRAPPED_MMDET_DIR" ]; then
  bootstrap_error "GenEval bootstrap state is incomplete at $BOOTSTRAP_STATE_FILE"
fi

if [ "$BOOTSTRAPPED_MODEL_DIR" != "$PRECISE_SDE_GENEVAL_MODEL_DIR" ]; then
  bootstrap_error "Current PRECISE_SDE_GENEVAL_MODEL_DIR (${PRECISE_SDE_GENEVAL_MODEL_DIR}) does not match bootstrapped path (${BOOTSTRAPPED_MODEL_DIR})"
fi

if [ "$BOOTSTRAPPED_MMDET_DIR" != "$PRECISE_SDE_GENEVAL_MMDET_DIR" ]; then
  bootstrap_error "Current PRECISE_SDE_GENEVAL_MMDET_DIR (${PRECISE_SDE_GENEVAL_MMDET_DIR}) does not match bootstrapped path (${BOOTSTRAPPED_MMDET_DIR})"
fi

if [ ! -x "$VENV_GUNICORN" ]; then
  echo "[ERROR] Expected bootstrapped Gunicorn binary at $VENV_GUNICORN" >&2
  echo "        Re-run bash precise_sde/rewards/servers/geneval/bootstrap.sh to rebuild the local .venv." >&2
  exit 1
fi

if [ -n "${PRECISE_SDE_GENEVAL_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$PRECISE_SDE_GENEVAL_VISIBLE_DEVICES"
fi

if [ ! -f "${PRECISE_SDE_GENEVAL_MODEL_DIR}/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth" ]; then
  echo "[ERROR] Mask2Former checkpoint missing under ${PRECISE_SDE_GENEVAL_MODEL_DIR}" >&2
  echo "        Run bash precise_sde/rewards/servers/geneval/bootstrap.sh first." >&2
  exit 1
fi

if [ ! -d "${PRECISE_SDE_GENEVAL_MMDET_DIR}/mmdet" ]; then
  echo "[ERROR] MMDetection checkout missing under ${PRECISE_SDE_GENEVAL_MMDET_DIR}" >&2
  echo "        Run bash precise_sde/rewards/servers/geneval/bootstrap.sh first." >&2
  exit 1
fi

cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${PRECISE_SDE_GENEVAL_MMDET_DIR}:${PYTHONPATH:-}"

echo "[INFO] Starting GenEval reward server on ${HOST}:${PORT}"
echo "[INFO] venv: ${VENV_DIR}"
echo "[INFO] MMDetection dir: ${PRECISE_SDE_GENEVAL_MMDET_DIR}"
echo "[INFO] Model dir: ${PRECISE_SDE_GENEVAL_MODEL_DIR}"
echo "[INFO] Workers / devices: ${NUM_DEVICES}"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

"$VENV_GUNICORN" \
  --chdir "${SCRIPT_DIR}" \
  --config "${SCRIPT_DIR}/gunicorn.conf.py" \
  "app:create_app()"
