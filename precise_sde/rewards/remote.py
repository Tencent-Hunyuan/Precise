import asyncio
import base64
import itertools
import logging
import os
import pickle
import re
import threading
from collections import defaultdict
from io import BytesIO

import numpy as np
import torch
from openai import AsyncOpenAI
from PIL import Image
import requests
from requests.adapters import HTTPAdapter, Retry

from precise_sde.rewards.servers.unified_reward.url_utils import (
    normalize_unifiedreward_base_url,
)


def geneval_score(device):
    del device

    batch_size = 64
    url = os.environ.get("PRECISE_SDE_GENEVAL_URL", "http://127.0.0.1:18085")
    session = requests.Session()
    session.trust_env = os.environ.get("PRECISE_SDE_REWARD_TRUST_ENV", "0") == "1"
    retries = Retry(total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False)
    session.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadatas, only_strict):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / batch_size))

        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []

        for image_batch, metadata_batch in zip(images_batched, metadatas_batched):
            jpeg_images = []
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            data = {
                "images": jpeg_images,
                "meta_datas": list(metadata_batch),
                "only_strict": only_strict,
            }
            response = session.post(url, data=pickle.dumps(data), timeout=120)
            response_data = pickle.loads(response.content)

            all_scores += response_data["scores"]
            all_rewards += response_data["rewards"]
            all_strict_rewards += response_data["strict_rewards"]
            all_group_strict_rewards.append(response_data["group_strict_rewards"])
            all_group_rewards.append(response_data["group_rewards"])

        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)

        return (
            all_scores,
            all_rewards,
            all_strict_rewards,
            dict(all_group_rewards_dict),
            dict(all_group_strict_rewards_dict),
        )

    return _fn


def unifiedreward_score_v2(device, acs_weights=None):
    del device

    default_nodes = []

    env_urls = os.environ.get("PRECISE_SDE_UNIFIEDREWARD_URLS", "")
    if env_urls:
        nodes = [
            normalize_unifiedreward_base_url(url)
            for url in env_urls.split(",")
            if url.strip()
        ]
    else:
        single_url = os.environ.get("PRECISE_SDE_UNIFIEDREWARD_URL", "").strip()
        nodes = [normalize_unifiedreward_base_url(single_url)] if single_url else default_nodes
    assert nodes, "No UnifiedReward nodes configured. Set PRECISE_SDE_UNIFIEDREWARD_URL or PRECISE_SDE_UNIFIEDREWARD_URLS environment variable."

    max_concurrency = int(
        os.environ.get("PRECISE_SDE_UNIFIEDREWARD_CONCURRENCY", "16")
    )
    node_cycle = itertools.cycle(range(len(nodes)))
    node_cycle_lock = threading.Lock()
    sentinel = -10.0

    def _next_node():
        with node_cycle_lock:
            return next(node_cycle)

    def _encode_image(pil_img):
        img = pil_img.convert("RGB")
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _extract_acs_scores(text):
        raw = {}
        for key, pattern in [
            ("alignment", r"Alignment Score \(1-5\):\s*([\d.]+)"),
            ("coherence", r"Coherence Score \(1-5\):\s*([\d.]+)"),
            ("style", r"Style Score \(1-5\):\s*([\d.]+)"),
        ]:
            match = re.search(pattern, text)
            try:
                raw[key] = float(match.group(1)) if match else None
            except (AttributeError, ValueError):
                raw[key] = None

        if any(value is None for value in raw.values()):
            return {
                "alignment": sentinel,
                "coherence": sentinel,
                "style": sentinel,
                "avg": sentinel,
            }

        named = {key: max(0.0, min(value, 5.0)) / 5.0 for key, value in raw.items()}
        if acs_weights:
            total_weight = sum(acs_weights.get(key, 0.0) for key in ("alignment", "coherence", "style"))
            if total_weight > 0:
                named["avg"] = sum(
                    named[key] * acs_weights.get(key, 0.0)
                    for key in ("alignment", "coherence", "style")
                ) / total_weight
            else:
                named["avg"] = sum(named.values()) / 3.0
        else:
            named["avg"] = sum(named.values()) / 3.0
        return named

    async def _evaluate_batch(images, prompts):
        clients = [
            AsyncOpenAI(base_url=f"{node}/v1", api_key="EMPTY", timeout=300, max_retries=5)
            for node in nodes
        ]
        semaphores = [asyncio.Semaphore(max_concurrency) for _ in nodes]

        async def _evaluate_one(prompt, pil_img, node_idx):
            image_b64 = _encode_image(pil_img)
            problem = (
                "You are presented with a generated image and its associated text caption. "
                "Provide overall assessments along the following axes (each rated from 1 to 5):\n"
                "- Alignment Score: How well the image matches the caption.\n"
                "- Coherence Score: How logically consistent the image is.\n"
                "- Style Score: How aesthetically appealing the image looks.\n\n"
                "Output your evaluation using the format below:\n\n"
                "Alignment Score (1-5): X\n"
                "Coherence Score (1-5): Y\n"
                "Style Score (1-5): Z\n\n"
                f"Your task is provided as follows:\nText Caption: [{prompt}]"
            )
            async with semaphores[node_idx]:
                response = await clients[node_idx].chat.completions.create(
                    model="UnifiedReward",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                                },
                                {"type": "text", "text": problem},
                            ],
                        }
                    ],
                    max_tokens=256,
                    temperature=0,
                )
            return _extract_acs_scores(response.choices[0].message.content)

        tasks = [_evaluate_one(prompt, image, _next_node()) for prompt, image in zip(prompts, images)]
        try:
            return await asyncio.gather(*tasks)
        finally:
            for client in clients:
                await client.close()

    def _fn(images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)

        pil_images = [Image.fromarray(image).resize((512, 512)) for image in images]
        results = asyncio.run(_evaluate_batch(pil_images, prompts))
        failures = sum(1 for result in results if result["avg"] == sentinel)
        if failures:
            logging.warning(
                "unifiedreward_v2: %d / %d samples failed to parse and were marked with sentinel values",
                failures,
                len(results),
            )

        sub_scores = {
            "unifiedreward_v2/alignment": [result["alignment"] for result in results],
            "unifiedreward_v2/coherence": [result["coherence"] for result in results],
            "unifiedreward_v2/style": [result["style"] for result in results],
            "unifiedreward_v2/parse_fail": [failures] * len(results),
        }
        return [result["avg"] for result in results], sub_scores

    return _fn
