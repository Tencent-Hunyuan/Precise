from __future__ import annotations

import contextlib
import torch
from peft import LoraConfig, PeftModel, get_peft_model

from precise_sde.diffusers_patch.sde_with_logprob import grpo_guard, sde_step_with_logprob


class BaseFlowModelAdapter:
    name = "base"

    def __init__(self, config):
        self.config = config
        self.pipeline = None
        self.transformer = None

    def load_pipeline(self):
        raise NotImplementedError

    def prepare_models(self, accelerator, inference_dtype):
        raise NotImplementedError

    def encode_prompts(self, prompts, device, max_sequence_length):
        raise NotImplementedError

    def prompt_ids(self, prompts, device):
        tokenized = self.pipeline.tokenizer(
            prompts,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )
        return tokenized.input_ids.to(device)

    def decode_prompt_ids(self, prompt_ids):
        return self.pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)

    def sample_with_logprob(self, conditioning, generator, return_prev_sample_mean):
        raise NotImplementedError

    def training_conditioning(self, sample, negative_conditioning):
        return sample

    def sample_extras(self, latents):
        return {}

    def compute_log_prob(self, transformer, sample, j, conditioning):
        raise NotImplementedError

    def maybe_disable_adapter(self, transformer):
        module = getattr(transformer, "module", transformer)
        if hasattr(module, "disable_adapter"):
            return module.disable_adapter()
        return contextlib.nullcontext()

    def _model_dtype(self, model):
        dtype = getattr(model, "dtype", None)
        if dtype is not None:
            return dtype
        return next(model.parameters()).dtype

    def _cache_context(self, model, name):
        if hasattr(model, "cache_context"):
            return model.cache_context(name)
        return contextlib.nullcontext()

    def _guard_terms(self, sample, j):
        if not self.config.rationorm:
            return None, None
        std_tensor, jac_tensor = grpo_guard(
            self.pipeline.scheduler,
            eta=self.config.sample.noise_level,
            device=sample["latents"].device,
            sde_type=self.config.sample.sde_type,
        )
        step_indices = [self.pipeline.scheduler.index_for_timestep(t) for t in sample["timesteps"][:, j]]
        view_shape = (-1,) + (1,) * (sample["latents"][:, j].ndim - 1)
        return std_tensor[step_indices].view(view_shape), jac_tensor[step_indices].view(view_shape)

    def ratio_mean_bias_dims(self, log_prob, prev_sample_mean):
        return tuple(range(1, log_prob.ndim))


class Flux2KleinAdapter(BaseFlowModelAdapter):
    name = "flux2_klein"

    def load_pipeline(self):
        from diffusers import Flux2KleinPipeline

        self.pipeline = Flux2KleinPipeline.from_pretrained(
            self.config.pretrained.model,
            revision=getattr(self.config.pretrained, "revision", None),
        )
        return self.pipeline

    def prepare_models(self, accelerator, inference_dtype):
        pipeline = self.pipeline
        if self.config.train.cfg or self.config.sample.guidance_scale != 1.0:
            raise ValueError("FLUX.2 Klein training in this repo must run without CFG/guidance.")
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.transformer.requires_grad_(not self.config.use_lora)
        pipeline.set_progress_bar_config(
            position=1,
            disable=not accelerator.is_local_main_process,
            leave=False,
            desc="Timestep",
            dynamic_ncols=True,
        )
        pipeline.vae.to(accelerator.device, dtype=torch.float32)
        pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
        pipeline.transformer.to(accelerator.device)

        if self.config.use_lora:
            target_modules = [
                "attn.to_k",
                "attn.to_q",
                "attn.to_v",
                "attn.to_out.0",
                "attn.to_qkv_mlp_proj",
                "attn.add_k_proj",
                "attn.add_q_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
                "ff.linear_in",
                "ff.linear_out",
                "ff_context.linear_in",
                "ff_context.linear_out",
            ]
            lora_config = LoraConfig(
                r=int(getattr(self.config.train, "lora_rank", 16)),
                lora_alpha=int(getattr(self.config.train, "lora_alpha", 16)),
                init_lora_weights="gaussian",
                target_modules=target_modules,
            )
            if self.config.train.lora_path:
                pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, self.config.train.lora_path)
                pipeline.transformer.set_adapter("default")
            else:
                pipeline.transformer = get_peft_model(pipeline.transformer, lora_config)

        self.transformer = pipeline.transformer
        return self.transformer

    def encode_prompts(self, prompts, device, max_sequence_length):
        out_layers = tuple(getattr(self.config.pretrained, "text_encoder_out_layers", (9, 18, 27)))
        with torch.no_grad():
            prompt_embeds, text_ids = self.pipeline.encode_prompt(
                prompt=prompts,
                device=device,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=out_layers,
            )
        return {"prompt_embeds": prompt_embeds.to(device), "text_ids": text_ids.to(device)}

    def sample_with_logprob(self, conditioning, generator, return_prev_sample_mean):
        from precise_sde.diffusers_patch.flux2_klein_pipeline_with_logprob import pipeline_with_logprob

        kwargs = dict(
            self=self.pipeline,
            prompt_embeds=conditioning["prompt_embeds"],
            text_ids=conditioning["text_ids"],
            num_inference_steps=self.config.sample.num_steps,
            guidance_scale=1.0,
            output_type="pt",
            height=self.config.resolution,
            width=self.config.resolution,
            noise_level=self.config.sample.noise_level,
            sde_type=self.config.sample.sde_type,
            generator=generator,
            return_prev_sample_mean=return_prev_sample_mean,
        )
        return pipeline_with_logprob(**kwargs)

    def eval_sample(self, conditioning, eval_target):
        from precise_sde.diffusers_patch.flux2_klein_pipeline_with_logprob import pipeline_with_logprob

        return pipeline_with_logprob(
            self.pipeline,
            prompt_embeds=conditioning["prompt_embeds"],
            text_ids=conditioning["text_ids"],
            num_inference_steps=eval_target["eval_num_steps"],
            guidance_scale=1.0,
            output_type="pt",
            height=self.config.resolution,
            width=self.config.resolution,
            noise_level=eval_target["noise_level"],
            sde_type=eval_target["sde_type"],
        )

    def sample_extras(self, latents):
        if latents.ndim == 3:
            batch_size, image_seq_len, _ = latents.shape
        elif latents.ndim == 4:
            batch_size, _, image_seq_len, _ = latents.shape
        else:
            raise ValueError(
                "FLUX.2 Klein training expects packed latents shaped "
                f"(batch, seq, channels) or (batch, steps + 1, seq, channels); got {tuple(latents.shape)}."
            )
        height = int(image_seq_len**0.5)
        if height * height != image_seq_len:
            raise ValueError(
                "FLUX.2 Klein training currently expects square packed latents; "
                f"got sequence length {image_seq_len}."
            )
        shape_only = torch.empty(batch_size, 1, height, height, device=latents.device)
        return {"latent_ids": self.pipeline._prepare_latent_ids(shape_only).to(latents.device)}

    def ratio_mean_bias_dims(self, log_prob, prev_sample_mean):
        return tuple(range(1, prev_sample_mean.ndim))

    def compute_log_prob(self, transformer, sample, j, conditioning):
        transformer_dtype = self._model_dtype(transformer)
        hidden_states = sample["latents"][:, j].to(transformer_dtype)
        timesteps = sample["timesteps"][:, j].to(hidden_states.dtype) / 1000
        with self._cache_context(transformer, "cond"):
            noise_pred = transformer(
                hidden_states=hidden_states,
                timestep=timesteps,
                guidance=None,
                encoder_hidden_states=conditioning["prompt_embeds"].to(transformer_dtype),
                txt_ids=conditioning["text_ids"],
                img_ids=sample["latent_ids"],
                return_dict=False,
            )[0]
        noise_pred = noise_pred[:, : hidden_states.size(1), :]
        prev_sample, log_prob, prev_sample_mean = sde_step_with_logprob(
            self.pipeline.scheduler,
            noise_pred.float(),
            sample["timesteps"][:, j],
            sample["latents"][:, j].float(),
            prev_sample=sample["next_latents"][:, j].float(),
            noise_level=self.config.sample.noise_level,
            sde_type=self.config.sample.sde_type,
        )
        std_dev_t, jac = self._guard_terms(sample, j)
        return prev_sample, log_prob, prev_sample_mean, std_dev_t, jac
