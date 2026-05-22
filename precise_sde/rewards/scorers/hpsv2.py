from typing import Union, List

import torch
from open_clip import create_model_and_transforms, get_tokenizer
from PIL import Image


class HPSClipRewardModel(object):
    def __init__(self, device, clip_ckpt_path, hps_ckpt_path, model_name="ViT-H-14"):
        self.device = device
        self.clip_ckpt_path = clip_ckpt_path
        self.hps_ckpt_path = hps_ckpt_path
        self.model_name = model_name
        self.reward_model, self.text_processor, self.img_processor = self.build_reward_model()

    def build_reward_model(self):
        model, _, img_preprocess_val = create_model_and_transforms(
            self.model_name,
            pretrained=self.clip_ckpt_path,
            precision="amp",
            device=self.device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            aug_cfg={},
            output_dict=True,
        )

        if hasattr(self.device, "index"):
            device_str = str(self.device)
        elif isinstance(self.device, int):
            device_str = f"cuda:{self.device}"
        else:
            device_str = str(self.device)

        checkpoint = torch.load(self.hps_ckpt_path, map_location=device_str, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        text_processor = get_tokenizer(self.model_name)
        reward_model = model.to(self.device)
        reward_model.eval()

        return reward_model, text_processor, img_preprocess_val

    @torch.no_grad()
    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image]],
        texts: Union[str, List[str]],
    ):
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(texts, str):
            texts = [texts]

        rewards = []
        for image, text in zip(images, texts):
            image = self.img_processor(image).unsqueeze(0).to(self.device, non_blocking=True)
            text = self.text_processor([text]).to(device=self.device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                outputs = self.reward_model(image, text)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T
                hps_score = torch.diagonal(logits_per_image)

                rewards.append(hps_score.float().cpu().item())

        return rewards
