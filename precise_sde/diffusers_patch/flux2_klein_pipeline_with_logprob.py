# Adapted from Diffusers' Flux2KleinPipeline.__call__ implementation.
# This version replaces scheduler stepping with `sde_step_with_logprob` and 
# returns intermediate latents plus per-step log probabilities for RL training.

import contextlib
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps

from .sde_with_logprob import sde_step_with_logprob


def _repeat_generator(generator, count):
    if isinstance(generator, list):
        if len(generator) != count:
            raise ValueError(
                f"Expected {count} generators for the effective batch, got {len(generator)}."
            )
        return generator
    return [generator] * count


def _model_dtype(model):
    dtype = getattr(model, "dtype", None)
    if dtype is not None:
        return dtype
    return next(model.parameters()).dtype


def _transformer_config(transformer):
    config = getattr(transformer, "config", None)
    if config is None and hasattr(transformer, "get_base_model"):
        config = getattr(transformer.get_base_model(), "config", None)
    if config is None:
        raise AttributeError("Unable to find transformer config")
    return config


def _cache_context(model, name):
    if hasattr(model, "cache_context"):
        return model.cache_context(name)
    return contextlib.nullcontext()


def _unpack_latents_with_ids(pipe, latents, latent_ids, height, width):
    unpack = pipe._unpack_latents_with_ids
    params = inspect.signature(unpack).parameters
    if "height" in params and "width" in params:
        return unpack(latents, latent_ids, height, width)
    return unpack(latents, latent_ids)


def default_flux2_scheduler_sigmas(num_inference_steps: int) -> List[float]:
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be at least 1.")
    return np.linspace(1.0, 1 / num_inference_steps, num_inference_steps).tolist()


def default_flux2_shifted_sigmas(image_seq_len: int, num_inference_steps: int) -> List[float]:
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be at least 1.")

    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
    linear = np.linspace(1.0, 0.0, num_inference_steps + 1)
    exp_mu = np.exp(mu)
    shifted = (exp_mu * linear) / (1.0 + (exp_mu - 1.0) * linear)
    return shifted[:-1].tolist()


@torch.no_grad()
def pipeline_with_logprob(
    self,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 1.0,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    text_ids: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    text_encoder_out_layers: tuple[int] = (9, 18, 27),
    noise_level: float = 0.7,
    sde_type: Optional[str] = "flow_grpo",
    return_prev_sample_mean: bool = False,
    eval_mode: bool = False,
):
    if guidance_scale != 1.0:
        raise ValueError("Precise-SDE FLUX.2 Klein training runs without CFG; guidance_scale must be 1.0.")

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    self.check_inputs(
        prompt,
        height,
        width,
        prompt_embeds=prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        guidance_scale=guidance_scale,
    )

    self._guidance_scale = guidance_scale
    if (
        joint_attention_kwargs is not None
        and attention_kwargs is not None
        and joint_attention_kwargs is not attention_kwargs
    ):
        raise ValueError("Pass only one of joint_attention_kwargs or attention_kwargs.")
    self._attention_kwargs = attention_kwargs if attention_kwargs is not None else joint_attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    if prompt_embeds is None or text_ids is None:
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )
    elif num_images_per_prompt != 1:
        original_batch, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(original_batch * num_images_per_prompt, seq_len, -1)
        text_ids = text_ids.repeat_interleave(num_images_per_prompt, dim=0)

    transformer_config = _transformer_config(self.transformer)
    num_channels_latents = transformer_config.in_channels // 4
    latents, latent_ids = self.prepare_latents(
        batch_size=batch_size * num_images_per_prompt,
        num_latents_channels=num_channels_latents,
        height=height,
        width=width,
        dtype=prompt_embeds.dtype,
        device=device,
        generator=generator,
        latents=latents,
    )
    latents = latents.float()

    image_seq_len = latents.shape[1]
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
    scheduler_sigmas = default_flux2_scheduler_sigmas(num_inference_steps) if sigmas is None else sigmas

    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=scheduler_sigmas,
        mu=mu,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    if hasattr(self.scheduler, "set_begin_index"):
        self.scheduler.set_begin_index(0)

    if not eval_mode:
        all_latents = [latents]
        all_log_probs = []
        all_prev_latents_mean = []

    generators = _repeat_generator(generator, latents.shape[0])
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            self._current_timestep = t
            transformer_dtype = _model_dtype(self.transformer)
            latent_model_input = latents.to(transformer_dtype)
            transformer_prompt_embeds = prompt_embeds.to(transformer_dtype)
            timestep = t.expand(latents.shape[0]).to(latent_model_input.dtype)

            with _cache_context(self.transformer, "cond"):
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=None,
                    encoder_hidden_states=transformer_prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    joint_attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]
            noise_pred = noise_pred[:, : latents.size(1), :]

            prev_latents = []
            log_probs = []
            prev_latents_mean = []
            for row, row_generator in enumerate(generators):
                prev_sample, log_prob, prev_sample_mean = sde_step_with_logprob(
                    self.scheduler,
                    noise_pred[row : row + 1].float(),
                    t.unsqueeze(0),
                    latents[row : row + 1].float(),
                    noise_level=noise_level,
                    generator=row_generator,
                    sde_type=sde_type,
                    compute_log_prob=not eval_mode,
                )
                prev_latents.append(prev_sample)
                if not eval_mode:
                    log_probs.append(log_prob)
                    prev_latents_mean.append(prev_sample_mean)
            latents = torch.cat(prev_latents, dim=0)

            if not eval_mode:
                all_latents.append(latents)
                all_log_probs.append(torch.cat(log_probs, dim=0))
                all_prev_latents_mean.append(torch.cat(prev_latents_mean, dim=0))

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

    self._current_timestep = None
    latent_height = 2 * (int(height) // (self.vae_scale_factor * 2))
    latent_width = 2 * (int(width) // (self.vae_scale_factor * 2))
    latents = _unpack_latents_with_ids(self, latents, latent_ids, latent_height // 2, latent_width // 2)
    latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    latents_bn_std = torch.sqrt(
        self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
    ).to(latents.device, latents.dtype)
    latents = latents * latents_bn_std + latents_bn_mean
    latents = self._unpatchify_latents(latents)

    if output_type == "latent":
        image = latents
    else:
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    self.maybe_free_model_hooks()

    if eval_mode:
        return image, None, None
    if return_prev_sample_mean:
        return image, all_latents, all_log_probs, all_prev_latents_mean
    return image, all_latents, all_log_probs
