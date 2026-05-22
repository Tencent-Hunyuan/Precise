import os

from config.base import (
    default,
    geneval_eval_targets,
    flux2_klein_mix_eval_targets,
    flux2_klein_pickscore_eval_targets,
    mix_reward_fn,
    pickscore_reward_fn,
    with_run_id,
)
FLUX2_KLEIN_MODEL = "black-forest-labs/FLUX.2-klein-base-4B"
FLUX2_KLEIN_MODEL_REVISION = "a3b4f4849157f664bdbc776fd7453c2783562f4d"


def _base_config():
    config = default()
    config.pretrained.model = FLUX2_KLEIN_MODEL
    config.pretrained.revision = FLUX2_KLEIN_MODEL_REVISION
    config.pretrained.text_encoder_out_layers = (9, 18, 27)
    config.dataset = "pickscore"
    config.use_lora = True

    config.prompt_fn = "general_ocr"
    config.reward_fn = {"jpeg_compressibility": 1.0}
    config.eval_targets = []
    config.per_prompt_stat_tracking = True
    return config


def _apply_common_config(
    config,
    *,
    dataset_name,
    prompt_fn,
    sde_type,
    noise_level,
    run_name,
    reward_fn,
    eval_targets,
    num_steps,
):
    gpu_count = 8

    config.dataset = dataset_name
    config.pretrained.model = FLUX2_KLEIN_MODEL
    config.pretrained.revision = FLUX2_KLEIN_MODEL_REVISION

    config.sample.num_steps = num_steps
    config.sample.eval_num_steps = num_steps
    config.sample.guidance_scale = 1.0
    config.sample.eval_guidance_scale = 1.0
    config.sample.sde_type = sde_type
    config.sample.noise_level = noise_level

    config.run_name = with_run_id(run_name)
    config.run_project = "Precise-SDE"
    config.resolution = 512

    config.sample.train_batch_size = 2
    config.sample.num_image_per_prompt = 8
    config.sample.num_batches_per_epoch = int(
        32 / (gpu_count * config.sample.train_batch_size / config.sample.num_image_per_prompt)
    )
    assert config.sample.num_batches_per_epoch % 2 == 0, (
        "Please set config.sample.num_batches_per_epoch to an even number so "
        "config.train.gradient_accumulation_steps can stay aligned."
    )
    config.sample.test_batch_size = 8

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.0
    config.train.cfg = False
    config.train.ema = True

    config.sample.global_std = False
    config.sample.same_latent = False

    config.save_freq = 50
    config.eval_freq = 50
    config.save_dir = f"checkpoints/logs/{config.run_name}"
    config.reward_fn = reward_fn
    config.eval_targets = eval_targets
    for target in config.eval_targets:
        target.setdefault("sde_type", sde_type)

    config.prompt_fn = prompt_fn
    config.mixed_precision = "bf16"
    config.train.clip_range = 4e-6
    config.train.highclip_range = 4e-6
    config.train.learning_rate = 8e-5
    config.rationorm = True
    config.per_prompt_stat_tracking = True
    return config


def _launch_float(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _launch_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _default_noise_level(sde_type):
    if sde_type == "precise":
        return 1.5
    if sde_type.startswith("dance_"):
        return 0.3
    return 0.7


def flux2_klein_config(*, reward="mix", sde_type="flow_grpo", noise_level=None, num_steps=20):
    if reward not in {"mix", "pickscore", "geneval"}:
        raise ValueError(f"Unsupported FLUX reward: {reward!r}")

    if noise_level is None:
        noise_level = _default_noise_level(sde_type)

    if reward == "geneval":
        return _apply_common_config(
            _base_config(),
            dataset_name="geneval",
            prompt_fn="geneval",
            sde_type=sde_type,
            noise_level=noise_level,
            run_name=f"[flux2-klein][geneval]{sde_type}_noise_level={noise_level}_train{num_steps}_eval{num_steps}",
            reward_fn={"geneval": 1.0},
            eval_targets=geneval_eval_targets(noise_level),
            num_steps=num_steps,
        )

    if reward == "pickscore":
        return _apply_common_config(
            _base_config(),
            dataset_name="pickscore",
            prompt_fn="general_ocr",
            sde_type=sde_type,
            noise_level=noise_level,
            run_name=f"[flux2-klein][pickscore]{sde_type}_noise_level={noise_level}_train{num_steps}_eval{num_steps}",
            reward_fn=pickscore_reward_fn(),
            eval_targets=flux2_klein_pickscore_eval_targets("flux2_klein_pickscore", noise_level),
            num_steps=num_steps,
        )

    return _apply_common_config(
        _base_config(),
        dataset_name="pickscore",
        prompt_fn="general_ocr",
        sde_type=sde_type,
        noise_level=noise_level,
        run_name=f"[flux2-klein][mix]{sde_type}_noise_level={noise_level}_train{num_steps}_eval{num_steps}",
        reward_fn=mix_reward_fn(),
        eval_targets=flux2_klein_mix_eval_targets("flux2_klein_eval_multi_reward", noise_level),
        num_steps=num_steps,
    )


def flux_launch():
    sde_type = os.environ.get("PRECISE_SDE_LAUNCH_SDE", "flow_grpo")
    return flux2_klein_config(
        reward=os.environ.get("PRECISE_SDE_LAUNCH_REWARD", "mix"),
        sde_type=sde_type,
        noise_level=_launch_float("PRECISE_SDE_LAUNCH_NOISE_LEVEL", _default_noise_level(sde_type)),
        num_steps=_launch_int("PRECISE_SDE_LAUNCH_STEP", 20),
    )
