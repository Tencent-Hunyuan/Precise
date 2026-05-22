import io

import numpy as np
import torch
from PIL import Image


def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        del prompts, metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew / 500, meta

    return _fn


def aesthetic_score():
    from precise_sde.rewards.scorers.aesthetic import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        del prompts, metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def clip_score(device):
    from precise_sde.rewards.scorers.clip import ClipScorer

    scorer = ClipScorer(device=device)

    def _fn(images, prompts, metadata):
        del metadata
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)
            images = torch.tensor(images, dtype=torch.uint8) / 255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def pickscore_score(device):
    from precise_sde.rewards.scorers.pickscore import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
            images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn


def imagereward_score(device):
    from precise_sde.rewards.scorers.imagereward import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
            images = [Image.fromarray(image) for image in images]
        scores = scorer(list(prompts), images)
        return scores, {}

    return _fn


def hps_v2(device):
    from precise_sde.core.model_paths import model_path
    from precise_sde.rewards.scorers.hpsv2 import HPSClipRewardModel

    scorer = HPSClipRewardModel(
        device=device,
        clip_ckpt_path=model_path("CLIP-ViT-H-14-laion2B-s32B-b79K", "open_clip_pytorch_model.bin"),
        hps_ckpt_path=model_path("hpsv2.1", "HPS_v2.1_compressed.pt"),
    )

    def _fn(images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
            images = [Image.fromarray(image) for image in images]
        return scorer(images=images, texts=prompts), {}

    return _fn
