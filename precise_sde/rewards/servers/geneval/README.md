# GenEval Server

This directory vendors the thin GenEval server integration layer used by
Precise-SDE while keeping the heavyweight runtime dependencies in a separate
uv-managed project environment from the main training stack.

## Why Separate It

The GenEval server depends on MMDetection/MMCV and a Mask2Former checkpoint.
Those dependencies are intentionally not included in the main
`precise-sde` environment because they are large, version-sensitive, and not
required for every training run.

Use the dedicated project-local environment from `pyproject.toml` instead.
The helper scripts call that project through `uv` directly; there is no manual
activation step.

## Files

- `pyproject.toml`: dedicated UV dependency spec for the GenEval server
- `bootstrap.sh`: installs MMDetection and downloads the Mask2Former checkpoint
- `start_server.sh`: launches the Gunicorn-backed HTTP server
- `app.py`: Flask app for Precise-SDE's pickled request format
- `gen_eval.py`: GenEval scoring implementation adapted from upstream
- `gunicorn.conf.py`: worker/GPU assignment via env vars
- `object_names.txt`: COCO object labels used by GenEval

## Recommended Setup

```bash
bash precise_sde/rewards/servers/geneval/bootstrap.sh
bash precise_sde/rewards/servers/geneval/start_server.sh
uv run --project precise_sde/rewards/servers/geneval \
  python precise_sde/rewards/servers/geneval/check_server.py
```

The bootstrap script mirrors the upstream `reward-server` MMDetection setup by
patching `mmdet/__init__.py` so `mmcv_maximum_version = '2.3.0'`.

## Runtime Configuration

The helpers use env vars instead of hard-coded local path edits:

- `PRECISE_SDE_GENEVAL_HOST`: bind host, default `127.0.0.1`
- `PRECISE_SDE_GENEVAL_PORT`: bind port, default `18085`
- `PRECISE_SDE_GENEVAL_NUM_DEVICES`: Gunicorn workers / visible GPUs, default `1`
- `PRECISE_SDE_GENEVAL_VISIBLE_DEVICES`: optional explicit GPU list, e.g. `0,1,2,3`
- `PRECISE_SDE_GENEVAL_MODEL_DIR`: checkpoint directory
- `PRECISE_SDE_GENEVAL_MMDET_DIR`: local MMDetection checkout used by `bootstrap.sh`
- `PRECISE_SDE_GENEVAL_CONFIG`: explicit Mask2Former config path
- `PRECISE_SDE_GENEVAL_CKPT`: explicit Mask2Former checkpoint path

If you run multi-node training and want a node-local reward server, keep the
default host `127.0.0.1` and run one server per node. If you instead use a
shared remote reward machine, set `PRECISE_SDE_GENEVAL_HOST=0.0.0.0` here and
point `PRECISE_SDE_GENEVAL_URL` in the training environment to that host.

For GenEval training through `launch/train.sh --reward geneval`, the launcher
defaults the local GenEval server to `PRECISE_SDE_GENEVAL_NUM_DEVICES=8` unless
you override it explicitly.

On H-type GPUs, use the bootstrap-controlled MMCV source build path for
`ms_deformable_im2col_cuda` kernel errors:

```bash
PRECISE_SDE_GENEVAL_MMCV_SOURCE_BUILD=1 \
PRECISE_SDE_GENEVAL_TORCH_CUDA_ARCH_LIST="9.0" \
  bash precise_sde/rewards/servers/geneval/bootstrap.sh
```
