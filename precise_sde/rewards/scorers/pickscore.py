from transformers import CLIPProcessor, CLIPModel
import torch

from precise_sde.core.model_paths import model_path as _model_path
from precise_sde.core.model_paths import model_revision as _model_revision

class PickScoreScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        processor_path = _model_path("CLIP-ViT-H-14-laion2B-s32B-b79K")
        pickscore_path = _model_path("PickScore_v1")
        processor_revision = _model_revision("CLIP-ViT-H-14-laion2B-s32B-b79K")
        pickscore_revision = _model_revision("PickScore_v1")
        self.device = device
        self.dtype = dtype
        self.processor = CLIPProcessor.from_pretrained(processor_path, revision=processor_revision)
        self.model = CLIPModel.from_pretrained(pickscore_path, revision=pickscore_revision).eval().to(device)
        self.model = self.model.to(dtype=dtype)

    @torch.no_grad()
    def __call__(self, prompt, images):
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}
        # Preprocess text
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}

        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * (text_embs @ image_embs.T)
        scores = scores.diag()
        scores = scores / 26
        return scores
