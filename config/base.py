import os

import ml_collections


def default():
    config = ml_collections.ConfigDict()

    config.run_name = ""
    config.debug = False
    config.seed = 42
    config.logdir = "logs"
    config.save_freq = 50
    config.eval_freq = 50
    config.num_checkpoint_limit = 5
    config.mixed_precision = "fp16"
    config.allow_tf32 = True
    config.use_lora = True
    config.dataset = ""
    config.resolution = 768

    config.pretrained = pretrained = ml_collections.ConfigDict()
    pretrained.model = ""
    pretrained.revision = None

    config.sample = sample = ml_collections.ConfigDict()
    sample.num_steps = 40
    sample.eval_num_steps = 40
    sample.guidance_scale = 1.0
    sample.eval_guidance_scale = 1.0
    sample.train_batch_size = 1
    sample.num_image_per_prompt = 1
    sample.test_batch_size = 1
    sample.num_batches_per_epoch = 2
    sample.global_std = True
    sample.noise_level = 0.7
    sample.same_latent = False
    sample.sde_type = "flow_grpo"

    config.train = train = ml_collections.ConfigDict()
    train.batch_size = 1
    train.use_8bit_adam = False
    train.learning_rate = 3e-4
    train.adam_beta1 = 0.9
    train.adam_beta2 = 0.999
    train.adam_weight_decay = 1e-4
    train.adam_epsilon = 1e-8
    train.gradient_accumulation_steps = 1
    train.max_grad_norm = 1.0
    train.num_inner_epochs = 1
    train.cfg = False
    train.adv_clip_max = 5
    train.clip_range = 1e-4
    train.timestep_fraction = 1.0
    train.beta = 0.0
    train.lora_path = None
    train.ema = False

    config.prompt_fn = "imagenet_animals"
    config.prompt_fn_kwargs = {}
    config.reward_fn = ml_collections.ConfigDict()
    config.normalize_rewards = True
    config.save_dir = ""
    config.per_prompt_stat_tracking = True

    return config


def mix_reward_fn():
    return {"pickscore": 1.0, "clipscore": 1.0, "hpsv2": 1.0}


def pickscore_reward_fn():
    return {"pickscore": 1.0}


def with_run_id(run_name):
    run_id = os.environ.get("PRECISE_SDE_RUN_ID")
    if not run_id:
        return run_name
    return f"{run_name}_{run_id}" if run_name else run_id


def geneval_eval_targets(noise_level):
    return [
        {
            "name": "geneval",
            "dataset": "geneval",
            "prompt_fn": "geneval",
            "reward_fn": {"geneval": 1.0},
            "test_batch_size": 7,
            "noise_level": noise_level,
        },
    ]


def mix_eval_targets(name, noise_level):
    return [
        {
            "name": name,
            "dataset": "pickscore",
            "prompt_fn": "general_ocr",
            "reward_fn": {
                "pickscore": 1.0,
                "imagereward": 1.0,
                "clipscore": 1.0,
                "hpsv2": 1.0,
            },
            "test_batch_size": 16,
            "noise_level": noise_level,
        },
    ]


def pickscore_eval_targets(name, noise_level):
    return [
        {
            "name": name,
            "dataset": "pickscore",
            "prompt_fn": "general_ocr",
            "reward_fn": pickscore_reward_fn(),
            "test_batch_size": 16,
            "noise_level": noise_level,
        },
    ]


def flux2_klein_mix_eval_targets(name, noise_level):
    targets = mix_eval_targets(name, noise_level)
    for target in targets:
        target["test_batch_size"] = 8
    return targets


def flux2_klein_pickscore_eval_targets(name, noise_level):
    targets = pickscore_eval_targets(name, noise_level)
    for target in targets:
        target["test_batch_size"] = 8
    return targets
