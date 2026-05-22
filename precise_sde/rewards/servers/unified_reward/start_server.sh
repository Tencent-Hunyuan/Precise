#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"

CONDA_ENV="${PRECISE_SDE_UNIFIEDREWARD_CONDA_ENV:-vllm}"
HOST="${PRECISE_SDE_UNIFIEDREWARD_HOST:-0.0.0.0}"
PORT="${PRECISE_SDE_UNIFIEDREWARD_PORT:-8080}"
SERVED_MODEL_NAME="${PRECISE_SDE_UNIFIEDREWARD_MODEL_NAME:-UnifiedReward}"
GPU_MEMORY_UTILIZATION="${PRECISE_SDE_UNIFIEDREWARD_GPU_MEMORY_UTILIZATION:-0.95}"
TENSOR_PARALLEL_SIZE="${PRECISE_SDE_UNIFIEDREWARD_TENSOR_PARALLEL_SIZE:-8}"

activate_conda_env() {
  if [ -z "${CONDA_ENV}" ] || [ "${CONDA_DEFAULT_ENV:-}" = "${CONDA_ENV}" ]; then
    return
  fi

  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
    return
  fi

  for conda_sh in \
    "${CONDA_PREFIX:-}/etc/profile.d/conda.sh" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"
  do
    if [ -f "${conda_sh}" ]; then
      # shellcheck source=/dev/null
      source "${conda_sh}"
      conda activate "${CONDA_ENV}"
      return
    fi
  done

  echo "[WARN] Could not activate conda env '${CONDA_ENV}'. Continuing with current PATH." >&2
}

activate_conda_env

if ! command -v vllm >/dev/null 2>&1; then
  echo "[ERROR] vllm not found." >&2
  echo "        Create the server environment with:" >&2
  echo "        conda create -n ${CONDA_ENV:-vllm} python=3.12 -y" >&2
  echo "        conda activate ${CONDA_ENV:-vllm}" >&2
  echo "        pip install -r ${SCRIPT_DIR}/requirements.txt" >&2
  exit 1
fi

MODEL_PATH="$(python "${REPO_ROOT}/precise_sde/core/model_paths.py" model "UnifiedReward-2.0-qwen35-9b")"
MODEL_REVISION="$(python "${REPO_ROOT}/precise_sde/core/model_paths.py" revision "UnifiedReward-2.0-qwen35-9b")"

export VLLM_DISABLE_FLASHINFER_GDN_PREFILL="${VLLM_DISABLE_FLASHINFER_GDN_PREFILL:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [ -n "${PRECISE_SDE_UNIFIEDREWARD_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${PRECISE_SDE_UNIFIEDREWARD_VISIBLE_DEVICES}"
fi

if [[ "${MODEL_PATH}" = /* && ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "[ERROR] Checkpoint not found at ${MODEL_PATH}" >&2
  echo "        Populate that local mirror, or unset PRECISE_SDE_MODEL_ROOT to use" >&2
  echo "        the pinned Hugging Face checkpoint." >&2
  exit 1
fi

args=(
  serve "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --trust-remote-code
  --served-model-name "${SERVED_MODEL_NAME}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --enable-prefix-caching
  --default-chat-template-kwargs '{"enable_thinking": false}'
)
if [ -n "${MODEL_REVISION}" ]; then
  args+=(--revision "${MODEL_REVISION}")
  args+=(--tokenizer-revision "${MODEL_REVISION}")
  args+=(--code-revision "${MODEL_REVISION}")
fi

if [ -n "${PRECISE_SDE_UNIFIEDREWARD_MM_ENCODER_TP_MODE:-}" ]; then
  args+=(--mm-encoder-tp-mode "${PRECISE_SDE_UNIFIEDREWARD_MM_ENCODER_TP_MODE}")
fi

if [ -n "${PRECISE_SDE_UNIFIEDREWARD_MM_PROCESSOR_CACHE_TYPE:-}" ]; then
  args+=(--mm-processor-cache-type "${PRECISE_SDE_UNIFIEDREWARD_MM_PROCESSOR_CACHE_TYPE}")
fi

echo "[INFO] Starting UnifiedReward v2 server on ${HOST}:${PORT}"
echo "[INFO] Conda env: ${CONDA_DEFAULT_ENV:-current PATH}"
echo "[INFO] Model path: ${MODEL_PATH}"
echo "[INFO] Model revision: ${MODEL_REVISION:-none}"
echo "[INFO] Served model name: ${SERVED_MODEL_NAME}"
echo "[INFO] Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

exec vllm "${args[@]}"
