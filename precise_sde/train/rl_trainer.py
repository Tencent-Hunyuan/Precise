from collections import defaultdict
import contextlib
import os
import datetime
import shutil
from concurrent import futures
import time
import hashlib
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import precise_sde.rewards as reward_registry
from precise_sde.core.model_paths import model_revision, resolve_dataset_reference, resolve_model_reference
from precise_sde.core.stat_tracking import PerPromptStatTracker
from precise_sde.prompt_data import build_eval_dataloader as build_shared_eval_dataloader
from precise_sde.prompt_data import build_prompt_dataset
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
import random
from torch.utils.data import DataLoader, Sampler
from precise_sde.core.ema import EMAModuleWrapper

os.environ.setdefault("NCCL_TIMEOUT", "5400")
os.environ.setdefault("NCCL_IB_TIMEOUT", "230")
os.environ.setdefault("NCCL_SOCKET_TIMEOUT", "600")
os.environ["WANDB_HTTP_TIMEOUT"] = "300"
os.environ.setdefault("WANDB_INIT_TIMEOUT", "300")

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


logger = get_logger(__name__)


class CheckpointRestoreError(RuntimeError):
    pass


def _build_train_dataset(prompt_fn, dataset):
    dataset_obj = build_prompt_dataset(prompt_fn, dataset, split="train", include_index=False)
    return dataset_obj, dataset_obj.collate_fn


def _build_eval_dataloader(prompt_fn, dataset, batch_size):
    return build_shared_eval_dataloader(prompt_fn, dataset, batch_size)


def _normalize_eval_targets(config):
    raw_targets = getattr(config, "eval_targets", None)
    if raw_targets is None or len(raw_targets) == 0:
        raw_targets = [
            {
                "name": "train_pair",
                "dataset": resolve_dataset_reference(config.dataset),
                "prompt_fn": config.prompt_fn,
                "reward_fn": config.reward_fn,
                "test_batch_size": config.sample.test_batch_size,
            }
        ]
    elif isinstance(raw_targets, dict):
        raw_targets = [raw_targets]

    eval_targets = []
    for i, raw_target in enumerate(raw_targets):
        target_cfg = dict(raw_target)
        name = str(target_cfg.get("name", f"target_{i}"))
        dataset = resolve_dataset_reference(target_cfg.get("dataset"))
        prompt_fn = target_cfg.get("prompt_fn", config.prompt_fn)
        reward_cfg = target_cfg.get("reward_fn")
        if reward_cfg is None:
            raise ValueError(f"eval_targets[{i}].reward_fn is required.")
        reward_fn = {}
        for k, v in dict(reward_cfg).items():
            k = str(k)
            if isinstance(v, (int, float)):
                reward_fn[k] = float(v)
            elif hasattr(v, 'items'):
                reward_fn[k] = {str(kk): vv for kk, vv in dict(v).items()}
            else:
                reward_fn[k] = float(v)
        if not reward_fn:
            raise ValueError(f"eval_targets[{i}].reward_fn must be non-empty.")
        test_batch_size = int(target_cfg.get("test_batch_size", config.sample.test_batch_size))
        eval_num_steps = int(target_cfg.get("eval_num_steps", config.sample.eval_num_steps))
        guidance_scale = float(target_cfg.get("guidance_scale", config.sample.guidance_scale))
        sde_type = target_cfg.get("sde_type", config.sample.sde_type)
        noise_level = float(target_cfg.get("noise_level", 0.0))

        if not dataset:
            raise ValueError(f"eval_targets[{i}].dataset is required.")
        if "image_similarity" in reward_fn:
            raise ValueError("image_similarity is not supported in this trainer because ref_images are not provided.")
        if "geneval" in reward_fn and prompt_fn != "geneval":
            raise ValueError(f"eval target '{name}' uses geneval reward but prompt_fn is '{prompt_fn}'.")

        eval_targets.append(
            {
                "name": name,
                "dataset": dataset,
                "prompt_fn": prompt_fn,
                "reward_fn": reward_fn,
                "test_batch_size": test_batch_size,
                "eval_num_steps": eval_num_steps,
                "guidance_scale": guidance_scale,
                "sde_type": sde_type,
                "noise_level": noise_level,
            }
        )

    return eval_targets

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])

            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Calculate the proportion of unique prompts whose reward standard deviation is zero.

    Args:
        prompts: List of prompts.
        gathered_rewards: Dictionary containing rewards, must include the key 'ori_avg'.

    Returns:
        zero_std_ratio: Proportion of prompts with zero standard deviation.
        prompt_std_devs: Mean standard deviation across all unique prompts.
    """
    prompt_array = np.array(prompts)

    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array,
        return_inverse=True,
        return_counts=True
    )

    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)

    prompt_std_devs = np.array([np.std(group) for group in reward_groups])

    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)

    return zero_std_ratio, prompt_std_devs.mean()

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')
        seed = (base_seed + prompt_hash_int) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

def _repeat_conditioning(conditioning, batch_size):
    return {
        key: value.repeat(batch_size, *([1] * (value.ndim - 1)))
        for key, value in conditioning.items()
    }


def eval(adapter, test_dataloader, config, accelerator, global_step, reward_fn, executor, autocast, ema, transformer_trainable_parameters, eval_target):
    pipeline = adapter.pipeline
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    eval_name = eval_target["name"]

    # test_dataloader = itertools.islice(test_dataloader, 2)
    all_rewards = defaultdict(list)
    for test_batch in tqdm(
            test_dataloader,
            desc=f"Eval[{eval_name}]: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        conditioning = adapter.encode_prompts(prompts, device=accelerator.device, max_sequence_length=128)
        with autocast():
            with torch.no_grad():
                images, _, _ = adapter.eval_sample(conditioning, eval_target)
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)

    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
    last_batch_prompt_ids = adapter.prompt_ids(prompts, accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = adapter.decode_prompt_ids(last_batch_prompt_ids_gather)
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(40, len(last_batch_images_gather))
                # sample_indices = random.sample(range(len(images)), num_samples)
                sample_indices = range(num_samples)
                for idx, index in enumerate(sample_indices):
                    image = last_batch_images_gather[index]
                    pil = Image.fromarray(
                        (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
                sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
                sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
                for key, value in all_rewards.items():
                    print(key, value.shape)
                eval_images = [
                    wandb.Image(
                        os.path.join(tmpdir, f"{idx}.jpg"),
                        caption=f"{prompt:.1000} | " + " | ".join(
                            f"{k}: {(v * 26 if k == 'pickscore' else v):.2f}"
                            for k, v in reward.items()
                            if v != -10 and "avg" not in k
                        ),
                    )
                    for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                ]
                eval_log_payload = {
                    f"eval_images_{eval_name}": eval_images,
                    **{
                        f"eval_reward_{eval_name}/{key}": np.mean(value[value != -10]) * (26 if key == "pickscore" else 1)
                        for key, value in all_rewards.items()
                        if "avg" not in key
                    },
                }
                if eval_name == "pickscore":
                    eval_log_payload["eval_images"] = eval_images
                safe_wandb_log(eval_log_payload, step=global_step, context=f"eval {eval_name} images/metrics")
        except Exception as exc:
            _warn_best_effort_failure(f"eval {eval_name} images/metrics", exc)
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def _warn_best_effort_failure(context, exc):
    try:
        logger.warning("Skipping %s after persistence failure: %r", context, exc)
    except Exception:
        pass


def safe_wandb_log(payload, *, step=None, context="wandb.log"):
    try:
        if payload:
            wandb.log(payload, step=step)
        return True
    except Exception as exc:
        _warn_best_effort_failure(context, exc)
        return False


def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    if not accelerator.is_main_process:
        return

    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    save_root = os.path.join(checkpoint_dir, f"checkpoint-{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if os.path.exists(save_root):
        raise FileExistsError(save_root)

    tmp_root = tempfile.mkdtemp(prefix=f".checkpoint-{global_step}.", dir=checkpoint_dir)
    try:
        save_root_lora = os.path.join(tmp_root, "lora")
        os.makedirs(save_root_lora, exist_ok=True)

        restore_ema = bool(config.train.ema)
        try:
            if restore_ema:
                ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
            unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        finally:
            if restore_ema:
                try:
                    ema.copy_temp_to(transformer_trainable_parameters)
                except Exception as exc:
                    raise CheckpointRestoreError(
                        f"Failed to restore training weights after checkpoint save at step {global_step}"
                    ) from exc

        os.rename(tmp_root, save_root)
        tmp_root = None
    finally:
        if tmp_root is not None:
            shutil.rmtree(tmp_root, ignore_errors=True)


def safe_save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    try:
        save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)
        return True
    except CheckpointRestoreError:
        raise
    except Exception as exc:
        _warn_best_effort_failure(f"checkpoint save at step {global_step}", exc)
        return False


def run_training(config, adapter):
    os.environ["PRECISE_SDE_RUN_ID"]

    config.dataset = resolve_dataset_reference(config.dataset)
    pretrained_ref = config.pretrained.model
    resolved_revision = model_revision(pretrained_ref)
    config.pretrained.model = resolve_model_reference(pretrained_ref)
    if resolved_revision is None:
        config.pretrained.revision = None
    elif not getattr(config.pretrained, "revision", None):
        config.pretrained.revision = resolved_revision

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    from accelerate import InitProcessGroupKwargs
    init_pg_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=5400))

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
        kwargs_handlers=[init_pg_kwargs],
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.run_project,
            config=config.to_dict(),
            init_kwargs={
                "wandb": {
                    "name": config.run_name,
                    "settings": wandb.Settings(
                        init_timeout=300,
                    ),
                }
            },
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline = adapter.load_pipeline()
    transformer = adapter.prepare_models(accelerator, inference_dtype)
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # This ema setting affects the previous 20 × 8 = 160 steps on average.
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "bitsandbytes is required for 8-bit Adam. Sync the repo's uv environment so it is installed before rerunning."
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    train_reward_cfg = {}
    for k, v in dict(config.reward_fn).items():
        k = str(k)
        if isinstance(v, (int, float)):
            train_reward_cfg[k] = float(v)
        elif hasattr(v, 'items'):
            train_reward_cfg[k] = {str(kk): vv for kk, vv in dict(v).items()}
        else:
            train_reward_cfg[k] = float(v)
    if not train_reward_cfg:
        raise ValueError("config.reward_fn must be non-empty.")
    # if len(train_reward_cfg) != 1:
    #     raise ValueError(
    #         f"Training expects exactly one reward model in config.reward_fn, got: {list(train_reward_cfg.keys())}"
    #     )
    if "image_similarity" in train_reward_cfg:
        raise ValueError("image_similarity is not supported in this trainer because ref_images are not provided.")
    if "geneval" in train_reward_cfg and config.prompt_fn != "geneval":
        raise ValueError("geneval reward requires config.prompt_fn == 'geneval'.")
    logger.info(f"Train reward: {train_reward_cfg}")
    reward_fn = getattr(reward_registry, "multi_score")(accelerator.device, train_reward_cfg)

    train_dataset, train_collate_fn = _build_train_dataset(config.prompt_fn, config.dataset)
    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=int(config.seed),
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=1,
        collate_fn=train_collate_fn,
    )

    eval_targets_cfg = _normalize_eval_targets(config)
    eval_targets = []
    for target in eval_targets_cfg:
        logger.info(
            f"Eval target '{target['name']}': prompt_fn={target['prompt_fn']} dataset={target['dataset']} "
            f"reward={target['reward_fn']} eval_num_steps={target['eval_num_steps']} "
            f"guidance_scale={target['guidance_scale']} sde_type={target['sde_type']} noise_level={target['noise_level']}"
        )
        eval_dataloader = _build_eval_dataloader(
            prompt_fn=target["prompt_fn"],
            dataset=target["dataset"],
            batch_size=target["test_batch_size"],
        )
        eval_reward_fn = getattr(reward_registry, "multi_score")(accelerator.device, target["reward_fn"])
        eval_targets.append(
            {
                "config": target,
                "dataloader": eval_dataloader,
                "reward_fn": eval_reward_fn,
            }
        )

    negative_train_conditioning = None
    if config.train.cfg:
        empty_conditioning = adapter.encode_prompts([""], device=accelerator.device, max_sequence_length=128)
        negative_train_conditioning = _repeat_conditioning(empty_conditioning, config.train.batch_size)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    transformer, optimizer, train_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader)
    for target in eval_targets:
        target["dataloader"] = accelerator.prepare(target["dataloader"])

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    # assert config.sample.train_batch_size >= config.train.batch_size
    # assert config.sample.train_batch_size % config.train.batch_size == 0
    # assert samples_per_epoch % total_train_batch_size == 0

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    while True:
        #################### EVAL ####################
        pipeline.transformer.eval()
        if epoch % config.eval_freq == 0 and epoch > 0:
            for target in eval_targets:
                eval(
                    adapter,
                    target["dataloader"],
                    config,
                    accelerator,
                    global_step,
                    target["reward_fn"],
                    executor,
                    autocast,
                    ema,
                    transformer_trainable_parameters,
                    target["config"],
                )
        if epoch % config.save_freq == 0 and epoch > 0 and global_step <= 3000 and accelerator.is_main_process:
            safe_save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)

        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            conditioning = adapter.encode_prompts(prompts, device=accelerator.device, max_sequence_length=128)
            prompt_ids = adapter.prompt_ids(prompts, accelerator.device)

            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch*10000+i)
            else:
                generator = None
            with autocast():
                with torch.no_grad():
                    if config.rationorm:
                        images, latents, log_probs, prev_sample_mean = adapter.sample_with_logprob(
                            conditioning,
                            generator=generator,
                            return_prev_sample_mean=True,
                        )
                    else:
                        images, latents, log_probs = adapter.sample_with_logprob(
                            conditioning,
                            generator=generator,
                            return_prev_sample_mean=False,
                        )

            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 16, 96, 96)
            log_probs = torch.stack(log_probs, dim=1)  # shape after stack (batch_size, num_steps)
            if config.rationorm:
                prev_sample_mean = torch.stack(prev_sample_mean, dim=1)
            else:
                prev_sample_mean = log_probs

            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.train_batch_size, 1
            )  # (batch_size, num_steps)
            sample_extras = adapter.sample_extras(latents)

            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    **conditioning,
                    "timesteps": timesteps,
                    **sample_extras,
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "prev_sample_mean": prev_sample_mean,
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )

        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }

        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if epoch % 10 == 0 and accelerator.is_main_process:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    num_samples = min(15, len(images))
                    sample_indices = random.sample(range(len(images)), num_samples)

                    for idx, i in enumerate(sample_indices):
                        image = images[i]
                        pil = Image.fromarray(
                            (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        )
                        pil = pil.resize((config.resolution, config.resolution))
                        pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                    sampled_prompts = [prompts[i] for i in sample_indices]
                    sampled_rewards = [rewards['avg'][i] for i in sample_indices]

                    safe_wandb_log({
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                            )
                            for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                        ],
                    }, step=global_step, context="sample images")
            except Exception as exc:
                _warn_best_effort_failure("sample images", exc)
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        # The purpose of repeating `adv` along the timestep dimension here is to make it easier to introduce timestep-dependent advantages later, such as adding a KL reward.
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}

        if getattr(config, 'normalize_rewards', False):
            from precise_sde.rewards import _parse_reward_cfg
            _parsed_cfg = _parse_reward_cfg(train_reward_cfg)
            _SENT = -10.0
            new_avg = np.zeros_like(gathered_rewards['ori_avg'])
            any_sentinel = np.zeros(len(new_avg), dtype=bool)
            for rname, rcfg in _parsed_cfg.items():
                w = rcfg["weight"]
                scores = gathered_rewards.get(rname)
                if scores is None:
                    continue
                valid = scores != _SENT
                any_sentinel |= ~valid
                if valid.any():
                    mean = scores[valid].mean()
                    std = scores[valid].std() + 1e-8
                    normed = np.where(valid, (scores - mean) / std, 0.0)
                else:
                    normed = np.zeros_like(scores)
                new_avg += w * normed
            new_avg[any_sentinel] = _SENT
            gathered_rewards['ori_avg'] = new_avg
            gathered_rewards['avg'] = np.repeat(
                new_avg[:, np.newaxis], num_train_timesteps, axis=1
            )

        # log rewards and images
        if accelerator.is_main_process:
            safe_wandb_log(
                {
                    "epoch": epoch,
                    **{
                        f"reward_{key}": np.mean(value[value != -10]) * (26 if key == "pickscore" else 1)
                        for key, value in gathered_rewards.items()
                        if '_strict_accuracy' not in key and '_accuracy' not in key and "avg" not in key
                    },
                },
                step=global_step,
                context="reward metrics",
            )

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = adapter.decode_prompt_ids(prompt_ids)
            advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompts))
                print("len unique prompts", len(set(prompts)))

            group_size, trained_prompt_num = stat_tracker.get_stats()

            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)

            if accelerator.is_main_process:
                safe_wandb_log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                    },
                    step=global_step,
                    context="per-prompt stats",
                )
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]
        del samples["prompt_ids"]

        # Get the mask for samples where all advantages are zero across the time dimension
        mask = (samples["advantages"].abs().sum(dim=1) != 0)

        # If the number of True values in mask is not divisible by config.sample.num_batches_per_epoch,
        # randomly change some False values to True to make it divisible
        num_batches = config.sample.num_batches_per_epoch
        true_count = mask.sum()
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        if accelerator.is_main_process:
            safe_wandb_log(
                {
                    "actual_batch_size": mask.sum().item()//config.sample.num_batches_per_epoch,
                },
                step=global_step,
                context="actual batch size",
            )
        # Filter out samples where the entire time dimension of advantages is zero
        samples = {k: v[mask] for k, v in samples.items()}

        total_batch_size, num_timesteps = samples["timesteps"].shape
        # assert (
        #     total_batch_size
        #     == config.sample.train_batch_size * config.sample.num_batches_per_epoch
        # )
        assert num_timesteps == config.sample.num_steps
        nonempty_ranks = accelerator.reduce(
            torch.tensor(int(total_batch_size > 0), device=accelerator.device),
            reduction="sum",
        )
        if nonempty_ranks.item() < accelerator.num_processes:
            if accelerator.is_main_process:
                safe_wandb_log({"skipped_empty_rank_epoch": 1}, step=global_step, context="empty-rank skip")
            epoch += 1
            continue

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                conditioning = adapter.training_conditioning(sample, negative_train_conditioning)

                train_timesteps = [step_index  for step_index in range(num_train_timesteps)]
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t, jac = adapter.compute_log_prob(
                                transformer, sample, j, conditioning
                            )
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    with adapter.maybe_disable_adapter(transformer):
                                        _, _, prev_sample_mean_ref, _, _ = adapter.compute_log_prob(
                                            transformer, sample, j, conditioning
                                        )

                        # grpo logic
                        advantages = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        if config.rationorm:
                            sigma_t = std_dev_t.mean()
                            ratio_mean_bias = (prev_sample_mean - sample["prev_sample_mean"][:, j]).pow(2).mean(
                                dim=adapter.ratio_mean_bias_dims(log_prob, prev_sample_mean)
                            )
                            ratio_mean_bias = ratio_mean_bias / (2 * sigma_t ** 2)
                            ratio = torch.exp((log_prob - sample["log_probs"][:, j] + ratio_mean_bias) * sigma_t)
                        else:
                            ratio = torch.exp(log_prob - sample["log_probs"][:, j])

                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        if config.rationorm:
                            policy_loss = policy_loss / jac.mean()

                        if config.train.beta > 0:
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(
                                dim=tuple(range(1, prev_sample_mean.ndim))
                            )
                            kl_loss = kl_loss.mean()
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_gt_one"].append(
                            torch.mean(
                                (
                                    ratio - 1.0 > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_lt_one"].append(
                            torch.mean(
                                (
                                    1.0 - ratio > config.train.clip_range
                                ).float()
                            )
                        )
                        info["policy_loss"].append(policy_loss)
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)

                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        # assert (j == train_timesteps[-1]) and (
                        #     i + 1
                        # ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            safe_wandb_log(info, step=global_step, context="training metrics")
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)
            # make sure we did an optimization step at the end of the inner epoch
            # assert accelerator.sync_gradients

        epoch+=1
