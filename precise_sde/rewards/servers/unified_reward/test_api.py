#!/usr/bin/env python3
"""
UnifiedReward-2.0 API test script.

Use this from a training node to verify the reward service inference endpoints.

Default server: http://127.0.0.1:8080
Default model: UnifiedReward

Examples:
    uv run --project . python precise_sde/rewards/servers/unified_reward/test_api.py
    uv run --project . python precise_sde/rewards/servers/unified_reward/test_api.py --base-url http://reward-server:8080
    uv run --project . python precise_sde/rewards/servers/unified_reward/test_api.py --image1 examples/a.jpg --image2 examples/b.jpg --prompt "a cat"
"""

import argparse
import base64
import json
import os
import sys
import time
from io import BytesIO

import requests
from PIL import Image, ImageDraw

from precise_sde.rewards.servers.unified_reward.url_utils import (
    normalize_unifiedreward_base_url,
)

DEFAULT_BASE_URL = os.environ.get(
    "PRECISE_SDE_UNIFIEDREWARD_BASE_URL",
    os.environ.get("PRECISE_SDE_UNIFIEDREWARD_URL", "http://127.0.0.1:8080"),
)
BASE_URL = normalize_unifiedreward_base_url(DEFAULT_BASE_URL)
MODEL_NAME = os.environ.get(
    "PRECISE_SDE_UNIFIEDREWARD_MODEL_NAME",
    os.environ.get("PRECISE_SDE_UNIFIEDREWARD_MODEL", "UnifiedReward"),
)
TIMEOUT = 120  # seconds


def encode_image(source) -> str:
    """Encode a file path or PIL image as a base64 JPEG string."""
    if isinstance(source, str):
        with Image.open(source) as img:
            img = img.convert("RGB")
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=95)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    else:
        img = source.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def make_image_message(b64: str) -> dict:
    """Build an image content block using a data URI."""
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
    }


def chat_completion(messages: list, max_tokens: int = 512) -> dict:
    """
    Call /v1/chat/completions.

    BASE_URL may be either the server root or an OpenAI-compatible /v1 prefix.
    Returns the full response JSON and raises for request failures.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    resp = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json=payload,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def extract_text(response_json: dict) -> str:
    return response_json["choices"][0]["message"]["content"]


def _make_test_image(color: tuple, label: str) -> Image.Image:
    img = Image.new("RGB", (320, 240), color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 280, 200], outline=(0, 0, 0), width=3)
    draw.text((10, 210), label, fill=(0, 0, 0))
    return img


def get_default_images():
    img1 = _make_test_image((210, 180, 140), "Image-1: a cat on a bench")
    img2 = _make_test_image((140, 180, 210), "Image-2: a dog in the park")
    return img1, img2


def test_health():
    """Test 1: health check."""
    print("\n" + "=" * 60)
    print("TEST 1: Health Check  GET /health")
    print("=" * 60)
    resp = requests.get(f"{BASE_URL}/health", timeout=10)
    print(f"  HTTP Status : {resp.status_code}")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    print("  ✅ PASS")


def test_model_list():
    """Test 2: model list."""
    print("\n" + "=" * 60)
    print("TEST 2: Model List  GET /v1/models")
    print("=" * 60)
    resp = requests.get(f"{BASE_URL}/v1/models", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    model_ids = [m["id"] for m in data["data"]]
    print(f"  Available models : {model_ids}")
    assert MODEL_NAME in model_ids, f"Model '{MODEL_NAME}' not found"
    print("  ✅ PASS")


def test_pointwise_text_only():
    """Test 3: lightweight text-only pointwise scoring."""
    print("\n" + "=" * 60)
    print("TEST 3: Pointwise Score — Text Only")
    print("=" * 60)
    messages = [{
        "role": "user",
        "content": "Rate the quality of this answer from 1 to 10: 'The capital of France is Paris.'"
    }]
    t0 = time.time()
    out = chat_completion(messages, max_tokens=64)
    elapsed = time.time() - t0
    text = extract_text(out)
    print(f"  Response  : {text!r}")
    print(f"  Latency   : {elapsed:.2f}s")
    print(f"  Tokens    : prompt={out['usage']['prompt_tokens']}, completion={out['usage']['completion_tokens']}")
    print("  ✅ PASS")


def test_image_generation_pointwise(img1_src, prompt: str):
    """
    Test 4: image-generation pointwise ACS scoring.

    Input: one image and one text caption.
    Output: Alignment, Coherence, and Style scores from 1 to 5.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Image Generation — Pointwise ACS Score (single image)")
    print("=" * 60)
    b64 = encode_image(img1_src)
    problem = (
        "You are presented with a generated image and its associated text caption. "
        "Your task is to analyze the image across multiple dimensions in relation to the caption. Specifically:\n"
        "Provide overall assessments for the image along the following axes (each rated from 1 to 5):\n"
        "- Alignment Score: How well the image matches the caption in terms of content.\n"
        "- Coherence Score: How logically consistent the image is (absence of visual glitches, object distortions, etc.).\n"
        "- Style Score: How aesthetically appealing the image looks, regardless of caption accuracy.\n\n"
        "Output your evaluation using the format below:\n\n"
        "Alignment Score (1-5): X\n"
        "Coherence Score (1-5): Y\n"
        "Style Score (1-5): Z\n\n"
        "Your task is provided as follows:\n"
        f"Text Caption: [{prompt}]"
    )
    messages = [{
        "role": "user",
        "content": [
            make_image_message(b64),
            {"type": "text", "text": problem},
        ]
    }]
    t0 = time.time()
    out = chat_completion(messages, max_tokens=128)
    elapsed = time.time() - t0
    text = extract_text(out)
    print(f"  Prompt    : {prompt!r}")
    print(f"  Response  :\n{text}")
    print(f"  Latency   : {elapsed:.2f}s")
    print("  ✅ PASS")
    return text


def test_image_generation_pairwise(img1_src, img2_src, prompt: str):
    """
    Test 5: image-generation pairwise scoring.

    Input: two images and one shared caption.
    Output: relative Alignment, Coherence, and Style scores for each image.
    """
    print("\n" + "=" * 60)
    print("TEST 5: Image Generation — Pairwise Score (two images)")
    print("=" * 60)
    b64_1 = encode_image(img1_src)
    b64_2 = encode_image(img2_src)
    problem = (
        "You are presented with two generated images (Image 1 and Image 2) along with a shared text caption. "
        "Your task is to comparatively evaluate the two images across three specific dimensions:\n\n"
        "- Alignment Score: How well each image matches the caption in terms of content.\n"
        "- Coherence Score: How logically consistent and visually coherent each image is.\n"
        "- Style Score: How aesthetically appealing each image is, regardless of caption accuracy.\n\n"
        "For each dimension, assign relative scores to Image 1 and Image 2 such that:\n"
        "- Each score is a float between 0 and 1 (inclusive).\n"
        "- The scores for Image 1 and Image 2 must sum to exactly 1.0 for each dimension.\n\n"
        "Please provide your evaluation in the format below:\n\n"
        "Alignment Score:\n"
        " Image 1: X\n"
        " Image 2: Y\n\n"
        "Coherence Score:\n"
        " Image 1: X\n"
        " Image 2: Y\n\n"
        "Style Score:\n"
        " Image 1: X\n"
        " Image 2: Y\n\n"
        "Your task is provided as follows:\n"
        f"Text Caption: [{prompt}]"
    )
    messages = [{
        "role": "user",
        "content": [
            make_image_message(b64_1),
            make_image_message(b64_2),
            {"type": "text", "text": problem},
        ]
    }]
    t0 = time.time()
    out = chat_completion(messages, max_tokens=256)
    elapsed = time.time() - t0
    text = extract_text(out)
    print(f"  Prompt    : {prompt!r}")
    print(f"  Response  :\n{text}")
    print(f"  Latency   : {elapsed:.2f}s")
    print("  ✅ PASS")
    return text


def test_image_generation_pair_rank(img1_src, img2_src, prompt: str):
    """
    Test 6: image-generation pair rank.

    Input: two images and one caption.
    Output: a natural-language preference, such as "Image 1 is better."
    """
    print("\n" + "=" * 60)
    print("TEST 6: Image Generation — Pair Rank (which is better)")
    print("=" * 60)
    b64_1 = encode_image(img1_src)
    b64_2 = encode_image(img2_src)
    problem = (
        f"You are given a text caption and two generated images based on that caption. "
        f"Your task is to evaluate and compare these images based on two key criteria:\n"
        f"1. Alignment with the Caption: Assess how well each image aligns with the provided caption.\n"
        f"2. Overall Image Quality: Examine visual quality including clarity, detail, color accuracy, and aesthetics.\n"
        f"Compare both images and select the one that better aligns with the caption while exhibiting superior quality.\n"
        f"Provide a clear conclusion such as \"Image 1 is better.\", \"Image 2 is better.\" or \"Both images are equally good.\"\n"
        f"Your task is provided as follows:\nText Caption: [{prompt}]"
    )
    messages = [{
        "role": "user",
        "content": [
            make_image_message(b64_1),
            make_image_message(b64_2),
            {"type": "text", "text": problem},
        ]
    }]
    t0 = time.time()
    out = chat_completion(messages, max_tokens=256)
    elapsed = time.time() - t0
    text = extract_text(out)
    print(f"  Prompt    : {prompt!r}")
    print(f"  Response  : {text!r}")
    print(f"  Latency   : {elapsed:.2f}s")
    print("  ✅ PASS")
    return text


def test_image_understanding_pointwise(img1_src):
    """
    Test 7: image-understanding pointwise scoring.

    Input: one image, one question, and one model answer.
    Output: one overall quality score from 0 to 100.
    """
    print("\n" + "=" * 60)
    print("TEST 7: Image Understanding — Pointwise Score (0-100)")
    print("=" * 60)
    b64 = encode_image(img1_src)
    question = "What is the main subject in this image?"
    response_text = "The main subject appears to be a rectangular frame or box drawn on a colored background."
    problem = (
        f"You are provided with an image and a question for this image. "
        f"Please review the corresponding response based on the following 5 factors:\n\n"
        f"1. Accuracy in Object Description\n"
        f"2. Accuracy in Depicting Relationships\n"
        f"3. Accuracy in Describing Attributes\n"
        f"4. Helpfulness\n"
        f"5. Ethical Considerations\n\n"
        f"From 0 to 100, how much do you rate for this response in terms of the correct and comprehensive description of the image? "
        f"Provide a few lines for explanation and the rate number at last after \"Final Score:\".\n\n"
        f"Question: {question}\n"
        f"Response: {response_text}\n"
    )
    messages = [{
        "role": "user",
        "content": [
            make_image_message(b64),
            {"type": "text", "text": problem},
        ]
    }]
    t0 = time.time()
    out = chat_completion(messages, max_tokens=256)
    elapsed = time.time() - t0
    text = extract_text(out)
    print(f"  Response  :\n{text}")
    print(f"  Latency   : {elapsed:.2f}s")
    print("  ✅ PASS")
    return text


def test_image_understanding_pairwise(img1_src):
    """
    Test 8: image-understanding pairwise rank.

    Input: one image, one question, and two candidate answers.
    Output: a preference between the two answers.
    """
    print("\n" + "=" * 60)
    print("TEST 8: Image Understanding — Pairwise Rank (two answers)")
    print("=" * 60)
    b64 = encode_image(img1_src)
    question = "What colors are dominant in this image?"
    answer1 = "The image is dominated by warm brownish tones."
    answer2 = "Blue is the main color in this image."
    problem = (
        "You are given an image and a question related to it. Evaluate two responses based on:\n"
        "1. Accuracy of Object Descriptions\n"
        "2. Relationship Between Objects\n"
        "3. Description of Attributes\n"
        "4. Helpfulness\n"
        "5. Ethical Concerns\n\n"
        "After evaluating, clearly state your decision such as 'Answer 1 is better' or 'Answer 2 is better.'\n\n"
        f"Question: {question}\n"
        f"Answer 1: {answer1}\n"
        f"Answer 2: {answer2}\n"
    )
    messages = [{
        "role": "user",
        "content": [
            make_image_message(b64),
            {"type": "text", "text": problem},
        ]
    }]
    t0 = time.time()
    out = chat_completion(messages, max_tokens=256)
    elapsed = time.time() - t0
    text = extract_text(out)
    print(f"  Response  : {text!r}")
    print(f"  Latency   : {elapsed:.2f}s")
    print("  ✅ PASS")
    return text


def main():
    global BASE_URL, MODEL_NAME

    parser = argparse.ArgumentParser(description="Test UnifiedReward API")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="UnifiedReward server root URL or /v1 API URL",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help=(
            "Served model name "
            "(defaults to PRECISE_SDE_UNIFIEDREWARD_MODEL_NAME, then "
            "PRECISE_SDE_UNIFIEDREWARD_MODEL, then UnifiedReward)"
        ),
    )
    parser.add_argument("--image1", default=None, help="Path to image 1 (optional, uses synthetic image if not set)")
    parser.add_argument("--image2", default=None, help="Path to image 2 (optional)")
    parser.add_argument("--prompt", default="a cat sitting on a bench in a park", help="Text caption/prompt")
    parser.add_argument("--tests", default="all", help="Comma-separated test IDs to run, e.g. 1,4,5 (default: all)")
    args = parser.parse_args()
    BASE_URL = normalize_unifiedreward_base_url(args.base_url)
    MODEL_NAME = args.model_name

    if args.image1:
        img1_src = args.image1
        img2_src = args.image2 or args.image1
        print(f"Using provided images: {img1_src}, {img2_src}")
    else:
        img1_pil, img2_pil = get_default_images()
        img1_src = img1_pil
        img2_src = img2_pil
        print("Using synthetic test images (no --image1 provided)")

    prompt = args.prompt
    run_all = args.tests == "all"
    selected = set(args.tests.split(",")) if not run_all else None

    def should_run(tid: str) -> bool:
        return run_all or tid in selected

    passed, failed = 0, 0
    results = {}

    print(f"\n{'='*60}")
    print(f"  UnifiedReward API Test Suite")
    print(f"  Server : {BASE_URL}")
    print(f"  Model  : {MODEL_NAME}")
    print(f"{'='*60}")

    tests = [
        ("1", test_health,                      []),
        ("2", test_model_list,                   []),
        ("3", test_pointwise_text_only,          []),
        ("4", test_image_generation_pointwise,   [img1_src, prompt]),
        ("5", test_image_generation_pairwise,    [img1_src, img2_src, prompt]),
        ("6", test_image_generation_pair_rank,   [img1_src, img2_src, prompt]),
        ("7", test_image_understanding_pointwise,[img1_src]),
        ("8", test_image_understanding_pairwise, [img1_src]),
    ]

    for tid, func, args_list in tests:
        if not should_run(tid):
            continue
        try:
            result = func(*args_list)
            results[tid] = result
            passed += 1
        except Exception as e:
            print(f"\n  ❌ FAIL: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*60}\n")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
