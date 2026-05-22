#!/usr/bin/env python3
"""Quick probe for the GenEval reward server used by precise_sde.rewards."""

import argparse
import json
import os
import pickle
import socket
import sys
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image


EXPECTED_KEYS = {
    "scores",
    "rewards",
    "strict_rewards",
    "group_rewards",
    "group_strict_rewards",
}

PROXY_ENV_KEYS = [
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
]


def load_first_metadata(metadata_file: Path) -> dict:
    if not metadata_file.exists():
        return {"prompt": "a photo of a cat"}

    with metadata_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                return obj
    return {"prompt": "a photo of a cat"}


def make_test_image() -> bytes:
    img = Image.new("RGB", (64, 64), color=(127, 127, 127))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def print_proxy_env(url: str) -> None:
    print("[INFO] Proxy-related environment variables:")
    has_any = False
    for key in PROXY_ENV_KEYS:
        if key in os.environ:
            has_any = True
            print(f"  - {key}={os.environ[key]!r}")
    if not has_any:
        print("  - <none set>")
    print(f"[INFO] requests environ proxies for URL: {requests.utils.get_environ_proxies(url)}")


def raw_http_probe(host: str, port: int, timeout: float) -> None:
    req = f"GET / HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n".encode("ascii")
    data = b""
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.sendall(req)
            while len(data) < 4096:
                chunk = sock.recv(4096 - len(data))
                if not chunk:
                    break
                data += chunk
    except OSError as e:
        print(f"[WARN] Raw HTTP probe failed: {e}")
        return
    preview = data.decode("latin1", errors="replace").replace("\r", "")
    preview = "\n".join(preview.splitlines()[:20])
    print("[INFO] Raw HTTP probe (first lines):")
    print(preview if preview else "<empty response>")


def summarize_response(response: requests.Response, body_chars: int) -> None:
    print(
        f"[INFO] HTTP {response.status_code}, content-type={response.headers.get('content-type', '<missing>')}, "
        f"bytes={len(response.content)}"
    )
    interesting_headers = [
        "server",
        "via",
        "x-cache",
        "x-cache-lookup",
        "x-squid-error",
        "content-type",
        "content-length",
    ]
    print("[INFO] Response headers (selected):")
    for key in interesting_headers:
        if key in response.headers:
            print(f"  - {key}: {response.headers[key]}")

    body_snippet = response.text[:body_chars].replace("\n", " ")
    if body_snippet:
        print(f"[INFO] Body snippet (first {body_chars} chars): {body_snippet!r}")
    lower = (response.text[:2000]).lower()
    header_blob = " ".join(f"{k}:{v}".lower() for k, v in response.headers.items())
    if "squid" in lower or "squid" in header_blob:
        print("[WARN] Response looks like it came from Squid (proxy/cache), not the GenEval app.")


def post_pickled(url: str, payload: dict, timeout: float, trust_env: bool) -> requests.Response:
    sess = requests.Session()
    sess.trust_env = trust_env
    return sess.post(url, data=pickle.dumps(payload), timeout=timeout)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check if GenEval reward server is healthy")
    parser.add_argument(
        "--url",
        default=os.environ.get("PRECISE_SDE_GENEVAL_URL", "http://127.0.0.1:18085"),
        help="GenEval reward URL",
    )
    parser.add_argument(
        "--metadata-file",
        default="dataset/geneval/test_metadata.jsonl",
        help="JSONL file to reuse one real metadata row",
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--only-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to request strict-only GenEval scoring",
    )
    parser.add_argument("--body-chars", type=int, default=300, help="Body preview length for diagnostics")
    args = parser.parse_args()

    parsed = urlparse(args.url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        print(f"[FAIL] Invalid URL: {args.url}")
        return 2

    try:
        with socket.create_connection((host, port), timeout=args.timeout):
            pass
        print(f"[OK] TCP reachable: {host}:{port}")
    except OSError as e:
        print(f"[FAIL] TCP not reachable: {host}:{port} ({e})")
        return 2

    metadata = load_first_metadata(Path(args.metadata_file))
    payload = {
        "images": [make_test_image()],
        "meta_datas": [metadata],
        "only_strict": args.only_strict,
    }

    print_proxy_env(args.url)
    raw_http_probe(host, port, args.timeout)

    train_trust_env = os.environ.get("PRECISE_SDE_REWARD_TRUST_ENV", "0") == "1"
    print(f"[INFO] Training path uses requests.Session(trust_env={train_trust_env})")

    try:
        response = post_pickled(args.url, payload, timeout=args.timeout, trust_env=train_trust_env)
    except requests.RequestException as e:
        print(f"[FAIL] HTTP request failed: {e}")
        return 2

    print(f"[INFO] requests.Session(trust_env={train_trust_env}) result:")
    summarize_response(response, args.body_chars)

    direct_response = None
    if response.status_code != 200:
        try:
            direct_response = post_pickled(
                args.url, payload, timeout=args.timeout, trust_env=not train_trust_env
            )
            print(f"[INFO] requests.Session(trust_env={not train_trust_env}) result:")
            summarize_response(direct_response, args.body_chars)
        except requests.RequestException as e:
            print(f"[WARN] Alternate (trust_env={not train_trust_env}) request failed: {e}")

    effective_response = response

    if response.status_code != 200 and direct_response is not None and direct_response.status_code == 200:
        print("[WARN] One path failed but the alternate path succeeded.")
        if train_trust_env:
            print(
                "[WARN] Training will fail unless you set NO_PROXY for localhost/127.0.0.1 "
                "or disable proxy usage."
            )
        effective_response = direct_response

    if response.status_code != 200:
        print(
            "[FAIL] Non-200 response via configured training path "
            f"(trust_env={train_trust_env})."
        )
        return 1

    try:
        obj = pickle.loads(effective_response.content)
    except Exception as e:  # noqa: BLE001
        head = effective_response.content[:120]
        print(f"[FAIL] Response is not pickle-decodable: {type(e).__name__}: {e}")
        print(f"[INFO] First bytes: {head!r}")
        return 1

    if not isinstance(obj, dict):
        print(f"[FAIL] Pickle decoded, but response type is {type(obj).__name__}, expected dict")
        return 1

    keys = set(obj.keys())
    missing = EXPECTED_KEYS - keys
    print(f"[INFO] Response keys: {sorted(keys)}")

    if missing:
        print(f"[FAIL] Missing expected keys: {sorted(missing)}")
        return 1

    print("[OK] GenEval reward server returned a valid pickled response with expected keys.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
