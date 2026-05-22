# UnifiedReward v2 Server

This directory contains the thin server wrapper needed by Precise-SDE. It does
not vendor UnifiedReward itself; it launches the
`CodeGoat24/UnifiedReward-2.0-qwen35-9b` checkpoint through the
OpenAI-compatible `vllm serve` API.

Keep this environment separate from the main training environment. The training
stack only needs the OpenAI client, while the server needs a version-sensitive
vLLM runtime. The vLLM docs recommend a fresh environment and installing vLLM
from its wheel rather than mixing it into an existing PyTorch stack:
https://docs.vllm.ai/en/latest/getting_started/installation/gpu/

## Files

- `requirements.txt`: small pip requirements list for a fresh conda env
- `start_server.sh`: activates a conda env and starts `vllm serve`
- `test_api.py`: probes `/health`, `/v1/models`, and the reward prompts used by
  training
- `url_utils.py`: normalizes server root URLs and `/v1` URLs for clients

## Environment Setup

Create a fresh conda environment, then use pip inside that environment:

```bash
conda create -n vllm python=3.12 -y
conda activate vllm
pip install -r precise_sde/rewards/servers/unified_reward/requirements.txt
python -c "import vllm, torch, transformers; print(vllm.__version__, torch.__version__, transformers.__version__)"
```

Do not install PyTorch with conda first. Let the vLLM wheel resolve its matching
PyTorch/CUDA Python packages inside the fresh environment.

Start the server:

```bash
bash precise_sde/rewards/servers/unified_reward/start_server.sh
```

The script tries to activate `PRECISE_SDE_UNIFIEDREWARD_CONDA_ENV`, defaulting
to `vllm`. If you already activated the correct environment, the script can run
with the current `PATH`.

`start_server.sh` always resolves the checkpoint path through
`precise_sde/core/model_paths.py`. By default, `UnifiedReward-2.0-qwen35-9b`
resolves to the pinned Hugging Face checkpoint:

```text
CodeGoat24/UnifiedReward-2.0-qwen35-9b@f01548b009741e12ff9817ed91dba94701ed9579
```

Use `PRECISE_SDE_MODEL_ROOT=/path/to/models` for a shared local model mirror. In
that mode, `UnifiedReward-2.0-qwen35-9b` resolves to
`/path/to/models/UnifiedReward-2.0-qwen35-9b` and no Hugging Face revision is
passed to vLLM.

## Smoke Test

From the repo root:

```bash
uv run --project . python precise_sde/rewards/servers/unified_reward/test_api.py \
  --base-url http://127.0.0.1:8080 --tests 1,2,4
```

Test `4` is the pointwise ACS scoring request used by training.

## API Contract Used By Precise-SDE

The training client in `precise_sde/rewards/remote.py` talks to
`POST /v1/chat/completions`.

Training currently expects this behavior:

- Base URL source: `PRECISE_SDE_UNIFIEDREWARD_URLS` or
  `PRECISE_SDE_UNIFIEDREWARD_URL`
- URLs may be either server roots such as `http://127.0.0.1:8080` or
  OpenAI-compatible prefixes such as `http://127.0.0.1:8080/v1`
- Model name: `UnifiedReward` by default
- Request shape: one image sent as a `data:image/jpeg;base64,...` `image_url`
  block plus a text instruction asking for three scores
- Response text must contain:
  `Alignment Score (1-5): X`
  `Coherence Score (1-5): Y`
  `Style Score (1-5): Z`

If the response stops matching those three lines, the parser in
`precise_sde/rewards/remote.py` falls back to sentinel values.

## Runtime Configuration

Server-side env vars:

- `PRECISE_SDE_UNIFIEDREWARD_CONDA_ENV`: conda env to activate, default `vllm`
- `PRECISE_SDE_UNIFIEDREWARD_HOST`: bind host, default `0.0.0.0`
- `PRECISE_SDE_UNIFIEDREWARD_PORT`: bind port, default `8080`
- `PRECISE_SDE_UNIFIEDREWARD_MODEL_NAME`: served model name, default
  `UnifiedReward`
- `PRECISE_SDE_UNIFIEDREWARD_GPU_MEMORY_UTILIZATION`: vLLM setting, default
  `0.95`
- `PRECISE_SDE_UNIFIEDREWARD_TENSOR_PARALLEL_SIZE`: vLLM tensor parallel size,
  default `8`
- `PRECISE_SDE_UNIFIEDREWARD_VISIBLE_DEVICES`: optional `CUDA_VISIBLE_DEVICES`
  value
- `PRECISE_SDE_UNIFIEDREWARD_MM_ENCODER_TP_MODE`: optional vLLM multimodal TP
  mode
- `PRECISE_SDE_UNIFIEDREWARD_MM_PROCESSOR_CACHE_TYPE`: optional vLLM multimodal
  processor cache type

Client-side env vars:

- `PRECISE_SDE_UNIFIEDREWARD_URLS`: comma-separated reward server base URLs for
  training
- `PRECISE_SDE_UNIFIEDREWARD_URL`: single reward server base URL
- `PRECISE_SDE_UNIFIEDREWARD_CONCURRENCY`: per-node async request concurrency
- `PRECISE_SDE_UNIFIEDREWARD_BASE_URL`: server root URL or `/v1` API URL used
  by `test_api.py`
- `PRECISE_SDE_UNIFIEDREWARD_MODEL_NAME`: served model name expected by
  `test_api.py`, default `UnifiedReward`
