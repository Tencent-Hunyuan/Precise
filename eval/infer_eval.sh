#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] uv not found. Install uv before running the eval runner." >&2
    exit 1
fi

repo_python() {
    uv run --project "$REPO_ROOT" python "$@"
}

resolve_model_path() {
    repo_python "${REPO_ROOT}/precise_sde/core/model_paths.py" resolve-model "$@"
}

resolve_model_revision() {
    repo_python "${REPO_ROOT}/precise_sde/core/model_paths.py" revision "$@"
}

reward_model_name() {
    printf '%s\n' "$1" |
        sed -E 's/^[[:space:]]*\{//; s/\}[[:space:]]*$//; s/"([^"]+)"[[:space:]]*:[^,}]+/\1/g; s/[[:space:]]*,[[:space:]]*/+/g'
}

usage() {
    cat <<'EOF'
Usage: eval/infer_eval.sh [--flux] [options]

Options:
  --model {flux}                      Model family to evaluate
  --pretrained-model PATH_OR_ID        Base model path/model id
  --pretrained-revision REVISION       Optional pinned Hugging Face revision for --pretrained-model
  --ckpt-base PATH                     Required checkpoint base to evaluate; repeat for multiple bases
  --eval-config CONFIG                STEP|SDE_TYPE|NOISE_LEVEL|NUM_STEPS|DATASET_TYPE|REWARD_FN; repeat to match --ckpt-base
  --num-gpus N                        Number of GPUs for torchrun
  --batch-size N                      Batch size per GPU
  --mixed-precision {fp16,bf16,no}     Inference autocast dtype
  --guidance-scale VALUE              Guidance scale
  --resolution N                      Output resolution
  --seed N                            Random seed
  --save-images                       Save generated PNGs under the eval output directory
  --dry-run                           Print the torchrun command without executing it
EOF
}

MODEL="flux"
DRY_RUN=0
PRETRAINED_MODEL=""
PRETRAINED_REVISION=""
CKPT_BASES=()
EVAL_CONFIGS=()
GUIDANCE_SCALE="1.0"
RESOLUTION="512"
NUM_GPUS="8"
BATCH_SIZE=""
SEED="42"
MIXED_PRECISION=""
SAVE_IMAGES="0"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --flux|--flux2|--flux2-klein|--flux2_klein)
            MODEL="flux"
            shift
            ;;
        --model)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] --model requires a value" >&2
                exit 1
            fi
            MODEL="$2"
            shift 2
            ;;
        --pretrained-model)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] --pretrained-model requires a value" >&2
                exit 1
            fi
            PRETRAINED_MODEL="$2"
            shift 2
            ;;
        --pretrained-revision)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] --pretrained-revision requires a value" >&2
                exit 1
            fi
            PRETRAINED_REVISION="$2"
            shift 2
            ;;
        --ckpt-base)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] --ckpt-base requires a value" >&2
                exit 1
            fi
            CKPT_BASES+=("$2")
            shift 2
            ;;
        --eval-config)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] --eval-config requires a value" >&2
                exit 1
            fi
            EVAL_CONFIGS+=("$2")
            shift 2
            ;;
        --num-gpus)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] --num-gpus requires a value" >&2
                exit 1
            fi
            NUM_GPUS="$2"
            shift 2
            ;;
        --batch-size)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] --batch-size requires a value" >&2
                exit 1
            fi
            BATCH_SIZE="$2"
            shift 2
            ;;
        --mixed-precision)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] --mixed-precision requires a value" >&2
                exit 1
            fi
            MIXED_PRECISION="$2"
            shift 2
            ;;
        --guidance-scale)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] --guidance-scale requires a value" >&2
                exit 1
            fi
            GUIDANCE_SCALE="$2"
            shift 2
            ;;
        --resolution)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] --resolution requires a value" >&2
                exit 1
            fi
            RESOLUTION="$2"
            shift 2
            ;;
        --seed)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] --seed requires a value" >&2
                exit 1
            fi
            SEED="$2"
            shift 2
            ;;
        --save-images)
            SAVE_IMAGES="1"
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

case "$MODEL" in
    flux|flux2|flux2-klein|flux2_klein)
        MODEL="flux"
        MODEL_DISPLAY_NAME="FLUX.2 Klein"
        DEFAULT_PRETRAINED_MODEL_REF="black-forest-labs/FLUX.2-klein-base-4B"
        DEFAULT_MIXED_PRECISION="bf16"
        DEFAULT_BATCH_SIZE="16"
        DEFAULT_EVAL_CONFIG='3000|precise|0|20|pickscore|{"clipscore": 1.0}'
        ;;
    *)
        echo "[ERROR] Unknown eval model: $MODEL (must be 'flux')" >&2
        exit 1
        ;;
esac

# Reward service defaults.
export PRECISE_SDE_GENEVAL_URL="${PRECISE_SDE_GENEVAL_URL:-http://127.0.0.1:18085}"
export PRECISE_SDE_REWARD_TRUST_ENV="${PRECISE_SDE_REWARD_TRUST_ENV:-0}"

# Ensure localhost reward calls bypass external proxies.
NO_PROXY="${NO_PROXY:-${no_proxy:-}}"
case ",$NO_PROXY," in
  *",127.0.0.1,"*) ;;
  *) NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1" ;;
esac
case ",$NO_PROXY," in
  *",localhost,"*) ;;
  *) NO_PROXY="${NO_PROXY:+$NO_PROXY,}localhost" ;;
esac
export NO_PROXY
export no_proxy="$NO_PROXY"

if [ -z "$PRETRAINED_MODEL" ]; then
    PRETRAINED_MODEL_REF="$DEFAULT_PRETRAINED_MODEL_REF"
else
    PRETRAINED_MODEL_REF="$PRETRAINED_MODEL"
fi
PRETRAINED_MODEL="$(resolve_model_path "$PRETRAINED_MODEL_REF")"
if [ -z "$PRETRAINED_REVISION" ]; then
    PRETRAINED_REVISION="$(resolve_model_revision "$PRETRAINED_MODEL_REF")"
fi
if [ ${#CKPT_BASES[@]} -eq 0 ]; then
    echo "[ERROR] At least one --ckpt-base is required." >&2
    exit 1
fi
if [ ${#EVAL_CONFIGS[@]} -eq 0 ]; then
    EVAL_CONFIGS=("$DEFAULT_EVAL_CONFIG")
fi
if [ -z "$BATCH_SIZE" ]; then
    BATCH_SIZE="$DEFAULT_BATCH_SIZE"
fi
if [ -z "$MIXED_PRECISION" ]; then
    MIXED_PRECISION="$DEFAULT_MIXED_PRECISION"
fi

if [ ${#CKPT_BASES[@]} -ne ${#EVAL_CONFIGS[@]} ]; then
    echo "ERROR: CKPT_BASES (${#CKPT_BASES[@]}) and EVAL_CONFIGS (${#EVAL_CONFIGS[@]}) must have the same length"
    exit 1
fi

for i in "${!CKPT_BASES[@]}"; do
    CKPT_BASE="${CKPT_BASES[$i]}"
    EVAL_CFG="${EVAL_CONFIGS[$i]}"

    # Parse the pipe-separated eval config (parameter expansion avoids newline from read/<<<)
    STEP="${EVAL_CFG%%|*}"
    rest="${EVAL_CFG#*|}"
    SDE_TYPE="${rest%%|*}"
    rest="${rest#*|}"
    NOISE_LEVEL="${rest%%|*}"
    rest="${rest#*|}"
    NUM_INFERENCE_STEPS="${rest%%|*}"
    rest="${rest#*|}"
    DATASET_TYPE="${rest%%|*}"
    REWARD_FN="${rest#*|}"

    case "$SDE_TYPE" in
        flow_grpo|dance_grpo|cps|precise|dance_precise) ;;
        *)
            echo "Unsupported SDE_TYPE in --eval-config: $SDE_TYPE (must be one of flow_grpo, dance_grpo, cps, precise, dance_precise)" >&2
            exit 1
            ;;
    esac

    # Resolve dataset path and prompt function from DATASET_TYPE
    if [ "$DATASET_TYPE" = "pickscore" ]; then
        DATASET="dataset/pickscore"
        PROMPT_FN="general_ocr"
    elif [ "$DATASET_TYPE" = "geneval" ]; then
        DATASET="dataset/geneval"
        PROMPT_FN="geneval"
    else
        echo "Unknown DATASET_TYPE: $DATASET_TYPE (must be 'pickscore' or 'geneval')"
        exit 1
    fi

    RM_NAME="$(reward_model_name "$REWARD_FN")"
    SUMMARY_DIR="${CKPT_BASE}/MODEL_${MODEL}_STEP_${STEP}_${SDE_TYPE}_noise_level=${NOISE_LEVEL}_rm=${RM_NAME}"

    # EVAL_SUBDIR must include RM_NAME so different reward configs use different output dirs (avoid reusing single-reward results as multi-reward)
    EVAL_SUBDIR="MODEL_${MODEL}_STEP_${STEP}_INFER_STEPS_${NUM_INFERENCE_STEPS}_SDE_TYPE_${SDE_TYPE}_NOISE_LEVEL_${NOISE_LEVEL}_DATASET_TYPE_${DATASET_TYPE}_RM_${RM_NAME}"

    echo ""
    echo "###############################################"
    echo "# CKPT_BASE:  $CKPT_BASE"
    echo "# EVAL_CFG:   $EVAL_CFG"
    echo "# STEP:       $STEP"
    echo "# NUM_INFERENCE_STEPS: $NUM_INFERENCE_STEPS"
    echo "# SUMMARY_DIR:$SUMMARY_DIR"
    echo "###############################################"
    echo ""

    # ---- Check if aggregation already done ----
    if [ -f "${SUMMARY_DIR}/eval_summary.json" ]; then
        echo "[SKIP] Aggregation already exists: ${SUMMARY_DIR}/eval_summary.json"
        continue
    fi

    CKPT_PATH="${CKPT_BASE}/checkpoint-${STEP}/lora"
    OUTPUT_DIR="${CKPT_BASE}/checkpoint-${STEP}/${EVAL_SUBDIR}"

    # Skip if this checkpoint has already been evaluated
    if [ -f "${OUTPUT_DIR}/eval_results.json" ]; then
        echo "[SKIP] Already evaluated: ${OUTPUT_DIR}/eval_results.json"
        continue
    fi

    echo "=========================================="
    echo " ${MODEL_DISPLAY_NAME} Inference + Evaluation (in-memory)"
    echo "  MODEL:              $MODEL"
    echo "  STEP:               $STEP"
    echo "  CKPT_PATH:          $CKPT_PATH"
    echo "  OUTPUT_DIR:         $OUTPUT_DIR"
    echo "  PRETRAINED_MODEL:   $PRETRAINED_MODEL"
    echo "  PRETRAINED_REVISION:${PRETRAINED_REVISION:-none}"
    echo "  DATASET_TYPE:       $DATASET_TYPE"
    echo "  DATASET:            $DATASET"
    echo "  PROMPT_FN:          $PROMPT_FN"
    echo "  REWARD_FN:          $REWARD_FN"
    echo "  SDE_TYPE:           $SDE_TYPE"
    echo "  NOISE_LEVEL:        $NOISE_LEVEL"
    echo "  NUM_INFERENCE_STEPS:$NUM_INFERENCE_STEPS"
    echo "  GUIDANCE_SCALE:     $GUIDANCE_SCALE"
    echo "  RESOLUTION:         $RESOLUTION"
    echo "  NUM_GPUS:           $NUM_GPUS"
    echo "  BATCH_SIZE:         $BATCH_SIZE"
    echo "  SAVE_IMAGES:        $SAVE_IMAGES"
    echo "=========================================="

    COMMAND=(
        uv run --project "$REPO_ROOT" torchrun --nproc_per_node="$NUM_GPUS" "${REPO_ROOT}/precise_sde/eval/infer_eval.py" \
        --model "$MODEL" \
        --ckpt_path "$CKPT_PATH" \
        --pretrained_model "$PRETRAINED_MODEL" \
        --dataset "$DATASET" \
        --prompt_fn "$PROMPT_FN" \
        --reward_fn "$REWARD_FN" \
        --sde_type "$SDE_TYPE" \
        --noise_level "$NOISE_LEVEL" \
        --num_inference_steps "$NUM_INFERENCE_STEPS" \
        --guidance_scale "$GUIDANCE_SCALE" \
        --resolution "$RESOLUTION" \
        --batch_size "$BATCH_SIZE" \
        --seed "$SEED" \
        --mixed_precision "$MIXED_PRECISION" \
        --output_dir "$OUTPUT_DIR"
    )
    if [ -n "$PRETRAINED_REVISION" ]; then
        COMMAND+=(--pretrained_revision "$PRETRAINED_REVISION")
    fi
    if [ "$SAVE_IMAGES" != "0" ]; then
        COMMAND+=("--save_images")
    fi

    if [ "$DRY_RUN" = "1" ]; then
        printf 'PYTHONPATH=. '
        printf '%q ' "${COMMAND[@]}"
        printf '\n'
        continue
    fi

    PYTHONPATH=. "${COMMAND[@]}"

done
