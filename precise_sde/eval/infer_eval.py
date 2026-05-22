"""
Integrated FLUX.2 Klein inference + evaluation script.

Generates images in memory and evaluates them with reward models in a single
distributed run. Optionally saves generated images to disk under the eval
output directory.

Usage:
    PYTHONPATH=. torchrun --nproc_per_node=8 precise_sde/eval/infer_eval.py \
        --ckpt_path checkpoints/logs/.../checkpoint-100/lora \
        --model flux \
        --pretrained_model black-forest-labs/FLUX.2-klein-base-4B \
        --dataset dataset/pickscore \
        --prompt_fn general_ocr \
        --reward_fn '{"pickscore": 1.0, "clipscore": 1.0}' \
        --sde_type cps \
        --noise_level 0.7
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
import contextlib
from collections import defaultdict

import torch
import torch.distributed as dist
import numpy as np
from peft import PeftModel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import precise_sde.rewards as reward_registry
from precise_sde.eval.image_saving import save_generated_images_as_pngs
from precise_sde.prompt_data import build_prompt_dataset, collate_examples
from precise_sde.sde import CANONICAL_SDE_TYPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_dataset(prompt_fn, dataset_path):
    return build_prompt_dataset(prompt_fn, dataset_path, split="test", include_index=True)


def _scores_to_list(values):
    """Convert reward scores (tensor/list/array) to a list of Python floats."""
    if isinstance(values, torch.Tensor):
        t = values.cpu().tolist()
        return [t] if isinstance(t, (int, float)) else t
    if isinstance(values, (list, np.ndarray)):
        return [v.item() if hasattr(v, "item") else float(v) for v in values]
    if isinstance(values, (int, float)):
        return [float(values)]
    return []


def save_generated_images(images, indices, output_dir):
    image_rows = (
        images.detach()
        .mul(255)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .tolist()
    )
    return save_generated_images_as_pngs(zip(indices, image_rows), output_dir)


class BaseEvalModelAdapter:
    display_name = "Model"
    max_sequence_length = 128

    def __init__(self, args, device, inference_dtype, rank):
        self.args = args
        self.device = device
        self.inference_dtype = inference_dtype
        self.rank = rank
        self.pipeline = None

    def load_pipeline(self):
        raise NotImplementedError

    def encode_prompts(self, prompts):
        raise NotImplementedError

    def offload_text_models(self):
        raise NotImplementedError

    def select_conditioning(self, conditioning_cache, indices):
        positions = [conditioning_cache["index_to_position"][int(index)] for index in indices]
        return {
            key: value[positions].to(self.device)
            for key, value in conditioning_cache["tensors"].items()
        }

    def precompute_conditioning(self, prompts, batch_size, indices=None):
        if self.rank == 0:
            print("Precomputing text embeddings for local eval shard...")
        indexed_prompts = [
            (int(index), prompts[int(index)])
            for index in (range(len(prompts)) if indices is None else indices)
        ]
        chunks = []
        precompute_bs = batch_size * 2
        for start in range(0, len(indexed_prompts), precompute_bs):
            chunk = [prompt for _, prompt in indexed_prompts[start:start + precompute_bs]]
            encoded = self.encode_prompts(chunk)
            chunks.append({key: value.cpu() for key, value in encoded.items()})
        tensors = {
            key: torch.cat([chunk[key] for chunk in chunks], dim=0)
            for key in chunks[0]
        }
        return {
            "index_to_position": {index: position for position, (index, _) in enumerate(indexed_prompts)},
            "tensors": tensors,
        }

    def generate(self, conditioning, actual_sde_type, actual_noise_level, generator):
        raise NotImplementedError

    def compile_transformer(self):
        if self.args.torch_compile:
            if self.rank == 0:
                print("Compiling transformer with torch.compile (first batch will be slower)...")
            self.pipeline.transformer = torch.compile(self.pipeline.transformer, mode="reduce-overhead")


class FluxEvalAdapter(BaseEvalModelAdapter):
    display_name = "FLUX.2 Klein"
    max_sequence_length = 128
    text_encoder_out_layers = (9, 18, 27)

    def load_pipeline(self):
        from diffusers import Flux2KleinPipeline

        if self.args.guidance_scale != 1.0:
            raise ValueError("FLUX.2 Klein eval must run with --guidance_scale 1.0.")

        pipeline = Flux2KleinPipeline.from_pretrained(
            self.args.pretrained_model,
            revision=self.args.pretrained_revision,
        )
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.transformer.requires_grad_(False)

        pipeline.vae.to(self.device, dtype=torch.float32)
        pipeline.text_encoder.to(self.device, dtype=self.inference_dtype)

        if self.args.ckpt_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, self.args.ckpt_path)
        pipeline.transformer.to(self.device, dtype=self.inference_dtype)
        pipeline.transformer.eval()
        self.pipeline = pipeline
        self.compile_transformer()
        self.pipeline.set_progress_bar_config(
            position=1, disable=(self.rank != 0), leave=False, desc="Timestep", dynamic_ncols=True,
        )
        return self.pipeline

    def encode_prompts(self, prompts):
        with torch.no_grad():
            prompt_embeds, text_ids = self.pipeline.encode_prompt(
                prompt=prompts,
                device=self.device,
                max_sequence_length=self.max_sequence_length,
                text_encoder_out_layers=self.text_encoder_out_layers,
            )
        return {"prompt_embeds": prompt_embeds.to(self.device), "text_ids": text_ids.to(self.device)}

    def offload_text_models(self):
        self.pipeline.text_encoder.cpu()
        self.pipeline.text_encoder = None
        self.pipeline.tokenizer = None

    def generate(self, conditioning, actual_sde_type, actual_noise_level, generator):
        from precise_sde.diffusers_patch.flux2_klein_pipeline_with_logprob import pipeline_with_logprob

        return pipeline_with_logprob(
            self.pipeline,
            prompt_embeds=conditioning["prompt_embeds"],
            text_ids=conditioning["text_ids"],
            num_inference_steps=self.args.num_inference_steps,
            guidance_scale=1.0,
            output_type="pt",
            height=self.args.resolution,
            width=self.args.resolution,
            noise_level=actual_noise_level,
            sde_type=actual_sde_type,
            eval_mode=True,
            generator=generator,
        )


def build_eval_model_adapter(args, device, inference_dtype, rank):
    if args.model == "flux":
        return FluxEvalAdapter(args, device, inference_dtype, rank)
    raise ValueError(f"Unsupported eval model: {args.model!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_arg_parser():
    parser = argparse.ArgumentParser(description="FLUX.2 Klein Inference + Evaluation")
    parser.add_argument("--model", type=str, default="flux", choices=["flux"],
                        help="Base model family to evaluate.")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Path to LoRA checkpoint directory. None = baseline model.")
    parser.add_argument("--pretrained_model", type=str, required=True,
                        help="Path or model id for the base model")
    parser.add_argument("--pretrained_revision", type=str, default=None,
                        help="Pinned Hugging Face revision for --pretrained_model")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset directory (contains test.txt or test_metadata.jsonl)")
    parser.add_argument("--prompt_fn", type=str, default="general_ocr",
                        choices=["general_ocr", "geneval"],
                        help="Prompt dataset type")
    parser.add_argument("--reward_fn", type=str, required=True,
                        help='Reward config as JSON string, e.g. \'{"pickscore": 1.0}\'')
    parser.add_argument("--sde_type", type=str, default=None,
                        choices=CANONICAL_SDE_TYPES,
                        help="Sampling mode. Must be set explicitly.")
    parser.add_argument("--noise_level", type=float, default=0.7,
                        help="Noise level for SDE sampling.")
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size per GPU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default=None,
                        choices=["fp16", "bf16", "no"])
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for eval_results.json. Default: auto-generated under ckpt parent.")
    parser.add_argument("--save_images", action="store_true", default=False,
                        help="Save generated images as PNG files under output_dir/images.")
    parser.add_argument("--torch_compile", action="store_true", default=False,
                        help="Use torch.compile on the transformer for faster denoising (first batch has compilation overhead)")
    return parser


def parse_args(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.sde_type is None:
        parser.error("--sde_type must be set explicitly.")
    return args


def main(args):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)

    if args.batch_size is None:
        args.batch_size = 16
    if args.mixed_precision is None:
        args.mixed_precision = "bf16"

    seed = args.seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    if rank == 0:
        for k, v in vars(args).items():
            print(f"  {k}: {v}")

    actual_sde_type = args.sde_type
    actual_noise_level = args.noise_level

    # Parse reward config (strip in case of trailing newline from shell)
    reward_config = json.loads(args.reward_fn.strip())
    if not reward_config:
        raise ValueError("--reward_fn must be a non-empty JSON dict")

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        ckpt_parent = os.path.dirname(os.path.dirname(args.ckpt_path)) if args.ckpt_path else "output"
        suffix = f"infer_eval_{args.sde_type}_noise{args.noise_level}"
        output_dir = os.path.join(ckpt_parent, suffix)
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load pipeline ----
    inference_dtype = torch.float16 if args.mixed_precision == "fp16" else (
        torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
    )

    adapter = build_eval_model_adapter(args, device, inference_dtype, rank)
    pipeline = adapter.load_pipeline()

    if args.mixed_precision == "no":
        def inference_autocast():
            return contextlib.nullcontext()
    else:
        def inference_autocast():
            return torch.amp.autocast(device_type="cuda", dtype=inference_dtype)

    # ---- Dataset (built early so we can precompute text embeddings) ----
    dataset = build_dataset(args.prompt_fn, args.dataset)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        collate_fn=collate_examples, num_workers=4,
    )

    # Precompute this rank's text embeddings so text encoders can be freed.
    all_prompts = [dataset[i]["prompt"] for i in range(len(dataset))]
    local_indices = list(iter(sampler))
    conditioning_cache = adapter.precompute_conditioning(all_prompts, args.batch_size, local_indices)
    adapter.offload_text_models()
    if rank == 0:
        print("Text encoders offloaded.")

    # ---- Load reward models (coexist with pipeline on GPU) ----
    reward_fn = reward_registry.multi_score(device, reward_config)
    if rank == 0:
        print("Reward models loaded. Starting unified generate+score loop...")

    # ======== Single-pass: Generate + Score (no CPU round-trip) ========
    local_per_item = []
    local_all_scores = defaultdict(list)

    for batch in tqdm(dataloader, desc=f"[Gen+Score] Rank {rank}", disable=(rank != 0)):
        prompts = [item["prompt"] for item in batch]
        metadatas = [item["metadata"] for item in batch]
        indices = [item["index"] for item in batch]
        conditioning = adapter.select_conditioning(conditioning_cache, indices)

        with torch.no_grad():
            with inference_autocast():
                images, _, _ = adapter.generate(conditioning, actual_sde_type, actual_noise_level, generator)

        image_paths = {}
        if args.save_images:
            image_paths = save_generated_images(images, indices, output_dir)

        scores, _ = reward_fn(images, prompts, metadatas, only_strict=False)
        del images

        for key, values in scores.items():
            local_all_scores[key].extend(_scores_to_list(values))

        for i in range(len(prompts)):
            item_score = {}
            for key, values in scores.items():
                vl = _scores_to_list(values)
                if i < len(vl):
                    item_score[key] = vl[i]
            result = {
                "index": indices[i],
                "prompt": prompts[i],
                "scores": item_score,
            }
            if args.save_images:
                result["image_path"] = image_paths[indices[i]]
            local_per_item.append(result)

    del pipeline, conditioning_cache

    # ---- Gather results across ranks ----
    all_per_item_list = [None] * world_size
    all_scores_list = [None] * world_size
    dist.all_gather_object(all_per_item_list, local_per_item)
    dist.all_gather_object(all_scores_list, dict(local_all_scores))

    if rank == 0:
        merged_per_item = []
        for rank_items in all_per_item_list:
            merged_per_item.extend(rank_items)
        merged_per_item.sort(key=lambda x: x["index"])

        merged_scores = defaultdict(list)
        for rank_scores in all_scores_list:
            for key, values in rank_scores.items():
                merged_scores[key].extend(values)

        summary = {}
        for key, values in merged_scores.items():
            if key == "avg":
                continue
            arr = np.array(values)
            valid = arr[arr != -10]
            if len(valid) > 0:
                summary[key] = {
                    "mean": float(np.mean(valid)),
                    "std": float(np.std(valid)),
                    "count": int(len(valid)),
                }

        agg_value = 0.0
        agg_weight_sum = 0.0
        for rm_name, weight in reward_config.items():
            if rm_name in summary:
                agg_value += summary[rm_name]["mean"] * weight
                agg_weight_sum += weight
        if agg_weight_sum > 0:
            summary["aggregated"] = {"value": agg_value / agg_weight_sum, "weight_sum": agg_weight_sum}

        display_scale = {"pickscore": 26}

        print("\n========== Evaluation Results ==========")
        for key, stats in summary.items():
            if key == "aggregated":
                print(f"  aggregated: value={stats['value']:.4f}")
                continue
            s = display_scale.get(key, 1)
            print(f"  {key}: mean={stats['mean'] * s:.4f}  std={stats['std'] * s:.4f}  count={stats['count']}")
        print("========================================\n")

        for key in summary:
            if key in display_scale:
                summary[key]["mean"] *= display_scale[key]
                summary[key]["std"] *= display_scale[key]

        output_file = os.path.join(output_dir, "eval_results.json")
        results = {
            "summary": summary,
            "reward_config": reward_config,
            "num_images": len(merged_per_item),
            "per_item": merged_per_item,
        }
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main(parse_args())
