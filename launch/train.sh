#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

[ -f env.sh ] && source env.sh

usage() {
  cat <<'EOF'
Usage: bash launch/train.sh [options] [-- trainer_args...]

Experiment:
  --reward {mix,pickscore,geneval}
  --sde {precise,flow_grpo,cps,dance_precise,dance_grpo}
  --noise-level VALUE
  --step N               Set training and evaluation denoising steps

Runtime:
  --num-processes N
  --main-process-port PORT
  --accelerate-config PATH
  --dry-run              Print the resolved command without running training
  -h, --help
EOF
}

model="flux"
reward="mix"
sde_type="flow_grpo"
noise_level=""
step_count=""
dry_run=0
num_processes="${PRECISE_SDE_NUM_PROCESSES:-8}"
main_process_port="${PRECISE_SDE_MAIN_PROCESS_PORT:-${ACCELERATE_MAIN_PROCESS_PORT:-29502}}"
accelerate_config="${PRECISE_SDE_ACCELERATE_CONFIG:-accelerate_configs/multi_gpu.yaml}"
trainer_args=()

normalize_sde_type() {
  case "$1" in
    precise|flow_grpo|cps|dance_precise|dance_grpo) printf '%s\n' "$1" ;;
    *)
      echo "[ERROR] unsupported --sde value: $1" >&2
      exit 2
      ;;
  esac
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --flux|--flux2|--flux2-klein|--flux2_klein)
      model="flux"
      shift
      ;;
    --reward)
      reward="${2:?--reward requires a value}"
      shift 2
      ;;
    --reward=*)
      reward="${1#*=}"
      shift
      ;;
    --sde)
      sde_type="$(normalize_sde_type "${2:?--sde requires a value}")"
      shift 2
      ;;
    --sde=*)
      sde_type="$(normalize_sde_type "${1#*=}")"
      shift
      ;;
    --noise-level)
      noise_level="${2:?--noise-level requires a value}"
      shift 2
      ;;
    --noise-level=*)
      noise_level="${1#*=}"
      shift
      ;;
    --step)
      step_count="${2:?--step requires a value}"
      shift 2
      ;;
    --step=*)
      step_count="${1#*=}"
      shift
      ;;
    --num-processes)
      num_processes="${2:?--num-processes requires a value}"
      shift 2
      ;;
    --num-processes=*)
      num_processes="${1#*=}"
      shift
      ;;
    --main-process-port)
      main_process_port="${2:?--main-process-port requires a value}"
      shift 2
      ;;
    --main-process-port=*)
      main_process_port="${1#*=}"
      shift
      ;;
    --accelerate-config)
      accelerate_config="${2:?--accelerate-config requires a value}"
      shift 2
      ;;
    --accelerate-config=*)
      accelerate_config="${1#*=}"
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      trainer_args=("$@")
      break
      ;;
    *)
      echo "[ERROR] unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$reward" in
  mix|pickscore|geneval) ;;
  *)
    echo "[ERROR] unsupported --reward value: $reward" >&2
    exit 2
    ;;
esac

if [ "$dry_run" != "1" ] && ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv not found. Install uv before running this launcher." >&2
  exit 1
fi

export PRECISE_SDE_GENEVAL_URL="${PRECISE_SDE_GENEVAL_URL:-http://127.0.0.1:18085}"
export PRECISE_SDE_REWARD_TRUST_ENV="${PRECISE_SDE_REWARD_TRUST_ENV:-0}"

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

export PRECISE_SDE_LAUNCH_MODEL="$model"
export PRECISE_SDE_LAUNCH_REWARD="$reward"
export PRECISE_SDE_LAUNCH_SDE="$sde_type"
if [ -n "$noise_level" ]; then
  export PRECISE_SDE_LAUNCH_NOISE_LEVEL="$noise_level"
else
  unset PRECISE_SDE_LAUNCH_NOISE_LEVEL || true
fi
if [ -n "$step_count" ]; then
  export PRECISE_SDE_LAUNCH_STEP="$step_count"
else
  unset PRECISE_SDE_LAUNCH_STEP || true
fi

repo_python() {
  uv run --project "$REPO_ROOT" python "$@"
}

geneval_python() {
  local bootstrap_state="$REPO_ROOT/precise_sde/rewards/servers/geneval/.bootstrap-state"
  local venv_python="$REPO_ROOT/precise_sde/rewards/servers/geneval/.venv/bin/python"
  if [ ! -f "$bootstrap_state" ] || [ ! -x "$venv_python" ]; then
    return 1
  fi
  "$venv_python" "$@"
}

maybe_start_geneval() {
  if [ "$reward" != "geneval" ] || [ "${PRECISE_SDE_AUTO_START_GENEVAL:-1}" != "1" ]; then
    return 0
  fi

  export PRECISE_SDE_GENEVAL_NUM_DEVICES="${PRECISE_SDE_GENEVAL_NUM_DEVICES:-$num_processes}"
  local geneval_bind
  geneval_bind="$(repo_python - <<'PY'
import os
from urllib.parse import urlparse

url = os.environ["PRECISE_SDE_GENEVAL_URL"]
parsed = urlparse(url)
host = (parsed.hostname or "").strip().lower()
if host not in {"127.0.0.1", "localhost"}:
    raise SystemExit(1)
port = parsed.port or (443 if parsed.scheme == "https" else 80)
print(parsed.hostname or "127.0.0.1")
print(port)
PY
)" || geneval_bind=""
  if [ -z "$geneval_bind" ]; then
    echo "[INFO] PRECISE_SDE_GENEVAL_URL targets a non-local host; skipping local auto-start"
    return 0
  fi

  if geneval_python precise_sde/rewards/servers/geneval/check_server.py --timeout 5 --body-chars 120 >/dev/null 2>&1; then
    echo "[INFO] Reusing existing local GenEval server at $PRECISE_SDE_GENEVAL_URL"
    return 0
  fi

  export PRECISE_SDE_GENEVAL_HOST="$(printf '%s\n' "$geneval_bind" | sed -n '1p')"
  export PRECISE_SDE_GENEVAL_PORT="$(printf '%s\n' "$geneval_bind" | sed -n '2p')"
  local geneval_log_dir="logs/geneval"
  mkdir -p "$geneval_log_dir"
  local geneval_log_file="$geneval_log_dir/server_$(date +%Y%m%d_%H%M%S).log"
  echo "[INFO] Starting local GenEval server; log=$geneval_log_file"
  bash precise_sde/rewards/servers/geneval/start_server.sh >"$geneval_log_file" 2>&1 &
  PRECISE_SDE_GENEVAL_SERVER_PID=$!
  cleanup_geneval_server() {
    if [ -n "${PRECISE_SDE_GENEVAL_SERVER_PID:-}" ] && kill -0 "$PRECISE_SDE_GENEVAL_SERVER_PID" >/dev/null 2>&1; then
      echo "[INFO] Stopping local GenEval server (pid=$PRECISE_SDE_GENEVAL_SERVER_PID)"
      kill "$PRECISE_SDE_GENEVAL_SERVER_PID" >/dev/null 2>&1 || true
      wait "$PRECISE_SDE_GENEVAL_SERVER_PID" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup_geneval_server EXIT INT TERM

  for _ in $(seq 1 "${PRECISE_SDE_GENEVAL_START_RETRIES:-30}"); do
    if geneval_python precise_sde/rewards/servers/geneval/check_server.py --timeout 5 --body-chars 120 >/dev/null 2>&1; then
      echo "[INFO] Local GenEval server is healthy"
      return 0
    fi
    if ! kill -0 "$PRECISE_SDE_GENEVAL_SERVER_PID" >/dev/null 2>&1; then
      break
    fi
    sleep "${PRECISE_SDE_GENEVAL_START_SLEEP_SECS:-2}"
  done

  echo "[ERROR] Local GenEval server failed to become healthy" >&2
  tail -n 40 "$geneval_log_file" >&2 || true
  exit 1
}

trainer="train/train_flux2_klein.py"
config_name="flux_launch"

cmd=(
  uv run --project "$REPO_ROOT"
  accelerate launch
  --config_file "$accelerate_config"
  --num_processes "$num_processes"
  --main_process_port "$main_process_port"
  "$trainer"
  --config "config/config.py:${config_name}"
)
if [ "${#trainer_args[@]}" -gt 0 ]; then
  cmd+=("${trainer_args[@]}")
fi

variant="${model}_${reward}_${sde_type}"
timestamp="$(date +%Y%m%d_%H%M%S)"
export PRECISE_SDE_RUN_ID="${PRECISE_SDE_RUN_ID:-${timestamp}}"

if [ "$dry_run" = "1" ]; then
  echo "PRECISE_SDE_LAUNCH_MODEL=$PRECISE_SDE_LAUNCH_MODEL"
  echo "PRECISE_SDE_LAUNCH_REWARD=$PRECISE_SDE_LAUNCH_REWARD"
  echo "PRECISE_SDE_LAUNCH_SDE=$PRECISE_SDE_LAUNCH_SDE"
  echo "PRECISE_SDE_RUN_ID=$PRECISE_SDE_RUN_ID"
  if [ -n "${PRECISE_SDE_LAUNCH_NOISE_LEVEL:-}" ]; then
    echo "PRECISE_SDE_LAUNCH_NOISE_LEVEL=$PRECISE_SDE_LAUNCH_NOISE_LEVEL"
  fi
  if [ -n "${PRECISE_SDE_LAUNCH_STEP:-}" ]; then
    echo "PRECISE_SDE_LAUNCH_STEP=$PRECISE_SDE_LAUNCH_STEP"
  fi
  printf 'PYTHONPATH=. '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

maybe_start_geneval

echo "[INFO] PRECISE_SDE_RUN_ID=$PRECISE_SDE_RUN_ID"
echo "[INFO] PRECISE_SDE_GENEVAL_URL=$PRECISE_SDE_GENEVAL_URL"
echo "[INFO] PRECISE_SDE_REWARD_TRUST_ENV=$PRECISE_SDE_REWARD_TRUST_ENV"
echo "[INFO] NO_PROXY=$NO_PROXY"
echo "[INFO] model=$model reward=$reward sde=$sde_type noise_level=${noise_level:-default} step=${step_count:-default}"
echo "[INFO] main_process_port=$main_process_port"
PYTHONPATH=. "${cmd[@]}"
