import torch

from precise_sde.core.model_paths import model_path


def _patch_imagereward_transformers_imports():
    try:
        import transformers.modeling_utils as modeling_utils
        import transformers.pytorch_utils as pytorch_utils
    except ImportError:
        return

    for name in (
        "apply_chunking_to_forward",
        "find_pruneable_heads_and_indices",
        "prune_linear_layer",
    ):
        if not hasattr(modeling_utils, name) and hasattr(pytorch_utils, name):
            setattr(modeling_utils, name, getattr(pytorch_utils, name))


_patch_imagereward_transformers_imports()

import ImageReward as RM


class ImageRewardScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.model_path = model_path("ImageReward", "ImageReward.pt")
        self.med_config_path = model_path("ImageReward", "med_config.json")
        self.device = device
        self.dtype = dtype
        self.model = RM.load(self.model_path, device=device, med_config=self.med_config_path).eval().to(dtype=dtype)
        self.model.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, prompts, images):
        rewards = []
        for prompt, image in zip(prompts, images):
            _, reward = self.model.inference_rank(prompt, [image])
            rewards.append(reward)
        return rewards
