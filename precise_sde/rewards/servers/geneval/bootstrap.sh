#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MODEL_DIR="${SCRIPT_DIR}/model/mask2former2"
DEFAULT_MMDET_DIR="${SCRIPT_DIR}/mmdetection"
BOOTSTRAP_STATE_FILE="${SCRIPT_DIR}/.bootstrap-state"
MODEL_DIR="${PRECISE_SDE_GENEVAL_MODEL_DIR:-$DEFAULT_MODEL_DIR}"
MMDET_DIR="${PRECISE_SDE_GENEVAL_MMDET_DIR:-$DEFAULT_MMDET_DIR}"
MMDET_REVISION="${PRECISE_SDE_GENEVAL_MMDET_REVISION:-e9cae2d0787cd5c2fc6165a6061f92fa09e48fb1}"
PYTHON_VERSION="${PRECISE_SDE_GENEVAL_PYTHON:-3.10.16}"
MMCV_MAX_VERSION="${PRECISE_SDE_GENEVAL_MMCV_MAX_VERSION:-2.3.0}"
MMCV_VERSION="${PRECISE_SDE_GENEVAL_MMCV_VERSION:-1.7.2}"
MMCV_SOURCE_BUILD="${PRECISE_SDE_GENEVAL_MMCV_SOURCE_BUILD:-auto}"
CKPT_NAME="mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
CKPT_REPO="tsbpp/geneval_mask2former"
CKPT_REVISION="22b5a198cedf6b45e45165cf1c865d58de4a2832"
CKPT_SHA256="743b7d99015f1224c6d57fd4b14d04b15cc8ec72ae7ee7831e7c71d8873b7a54"

detect_cuda_arch_list() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | tr -d ' ' \
    | awk 'NF {print $1}' \
    | sort -u \
    | paste -sd';' -
}

should_build_mmcv_from_source() {
  case "$MMCV_SOURCE_BUILD" in
    1|true|TRUE|yes|YES)
      return 0
      ;;
    0|false|FALSE|no|NO)
      return 1
      ;;
    auto|AUTO)
      local detected_arches="$1"
      if [ -z "$detected_arches" ]; then
        return 1
      fi
      if printf '%s\n' "$detected_arches" | tr ';' '\n' | awk -F. '$1 >= 9 {found=1} END {exit found ? 0 : 1}'; then
        return 0
      fi
      return 1
      ;;
    *)
      echo "[ERROR] Unsupported PRECISE_SDE_GENEVAL_MMCV_SOURCE_BUILD value: $MMCV_SOURCE_BUILD" >&2
      echo "        Expected one of: auto, 0, 1" >&2
      exit 1
      ;;
  esac
}

rebuild_mmcv_from_source() {
  local detected_arches="$1"
  local arch_list="${PRECISE_SDE_GENEVAL_TORCH_CUDA_ARCH_LIST:-$detected_arches}"
  local venv_python="${SCRIPT_DIR}/.venv/bin/python"
  local max_jobs
  max_jobs="${PRECISE_SDE_GENEVAL_MAX_JOBS:-$(nproc)}"

  if [ -z "$arch_list" ]; then
    echo "[ERROR] Could not determine TORCH_CUDA_ARCH_LIST for MMCV source build." >&2
    echo "        Set PRECISE_SDE_GENEVAL_TORCH_CUDA_ARCH_LIST explicitly." >&2
    exit 1
  fi

  echo "[INFO] Rebuilding mmcv-full==$MMCV_VERSION from source for CUDA arch list: $arch_list"
  uv pip uninstall --python "$venv_python" mmcv-full
  TORCH_CUDA_ARCH_LIST="$arch_list" \
  MMCV_WITH_OPS=1 \
  FORCE_CUDA=1 \
  MAX_JOBS="$max_jobs" \
  uv pip install --python "$venv_python" --no-build-isolation --no-binary mmcv-full "mmcv-full==$MMCV_VERSION"
}

verify_checkpoint_sha256() {
  local checkpoint_path="$1"
  "${SCRIPT_DIR}/.venv/bin/python" - "$checkpoint_path" "$CKPT_SHA256" <<'PY'
import hashlib
import sys

path, expected = sys.argv[1:3]
digest = hashlib.sha256()
with open(path, "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)

actual = digest.hexdigest()
if actual != expected:
    raise SystemExit(f"{path} sha256 mismatch: expected {expected}, got {actual}")
PY
}

download_checkpoint() {
  local checkpoint_path="${MODEL_DIR}/${CKPT_NAME}"
  if [ -f "$checkpoint_path" ]; then
    verify_checkpoint_sha256 "$checkpoint_path"
    return
  fi

  "${SCRIPT_DIR}/.venv/bin/python" - "$CKPT_REPO" "$CKPT_REVISION" "$CKPT_NAME" "$MODEL_DIR" <<'PY'
import sys

from huggingface_hub import hf_hub_download

repo_id, revision, filename, local_dir = sys.argv[1:5]
hf_hub_download(
    repo_id=repo_id,
    revision=revision,
    filename=filename,
    local_dir=local_dir,
)
PY
  verify_checkpoint_sha256 "$checkpoint_path"
}

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv not found. Install uv before bootstrapping the GenEval server environment." >&2
  exit 1
fi

rm -f "$BOOTSTRAP_STATE_FILE"

mkdir -p "$MODEL_DIR"

if [ ! -d "$MMDET_DIR/.git" ]; then
  git clone https://github.com/open-mmlab/mmdetection.git "$MMDET_DIR"
fi

git -C "$MMDET_DIR" fetch --all --tags --quiet
git -C "$MMDET_DIR" checkout --detach "$MMDET_REVISION"
MMDET_ACTUAL_REVISION="$(git -C "$MMDET_DIR" rev-parse HEAD)"
if [ "$MMDET_ACTUAL_REVISION" != "$MMDET_REVISION" ]; then
  echo "[ERROR] MMDetection checkout mismatch: expected $MMDET_REVISION, got $MMDET_ACTUAL_REVISION" >&2
  exit 1
fi

MMDET_INIT="${MMDET_DIR}/mmdet/__init__.py"
if [ ! -f "$MMDET_INIT" ]; then
  echo "[ERROR] Expected MMDetection init file at $MMDET_INIT" >&2
  exit 1
fi

# Match the upstream reward-server setup by widening MMDetection's mmcv version
# gate for the 2.x checkout.
if ! grep -q "mmcv_maximum_version = '${MMCV_MAX_VERSION}'" "$MMDET_INIT"; then
  perl -0pi -e "s/mmcv_maximum_version = '([^']+)'/mmcv_maximum_version = '${MMCV_MAX_VERSION}'/" "$MMDET_INIT"
fi

uv sync --project "$SCRIPT_DIR" --python "$PYTHON_VERSION"

DETECTED_CUDA_ARCH_LIST="$(detect_cuda_arch_list)"
if should_build_mmcv_from_source "$DETECTED_CUDA_ARCH_LIST"; then
  rebuild_mmcv_from_source "$DETECTED_CUDA_ARCH_LIST"
elif [ -n "$DETECTED_CUDA_ARCH_LIST" ]; then
  echo "[INFO] Using prebuilt mmcv-full wheel for detected CUDA arch list: $DETECTED_CUDA_ARCH_LIST"
fi

download_checkpoint

{
  printf '%s\n' "$MODEL_DIR"
  printf '%s\n' "$MMDET_DIR"
  printf '%s\n' "$PYTHON_VERSION"
  printf '%s\n' "$MMDET_ACTUAL_REVISION"
} > "$BOOTSTRAP_STATE_FILE"

echo "[INFO] GenEval bootstrap complete."
echo "[INFO] uv environment: $SCRIPT_DIR/.venv"
echo "[INFO] MMDetection checkout: $MMDET_DIR@$MMDET_ACTUAL_REVISION"
echo "[INFO] Patched MMDetection mmcv_maximum_version to: $MMCV_MAX_VERSION"
if [ -n "${DETECTED_CUDA_ARCH_LIST:-}" ]; then
  echo "[INFO] Detected CUDA arch list: $DETECTED_CUDA_ARCH_LIST"
fi
echo "[INFO] mmcv-full source build mode: $MMCV_SOURCE_BUILD"
echo "[INFO] Mask2Former checkpoint: $MODEL_DIR/$CKPT_NAME"
echo "[INFO] Mask2Former revision: ${CKPT_REPO}@${CKPT_REVISION}"
echo "[INFO] Next step: bash precise_sde/rewards/servers/geneval/start_server.sh"
