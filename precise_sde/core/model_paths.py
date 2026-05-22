import argparse
import os


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOCAL_DATASET_ROOT = os.path.join(REPO_ROOT, "dataset")
LOCAL_MODEL_ROOT = os.path.join(REPO_ROOT, "model")

_PINNED_HF_MODELS = {
    "black-forest-labs/FLUX.2-klein-base-4B": {
        "repo_id": "black-forest-labs/FLUX.2-klein-base-4B",
        "revision": "a3b4f4849157f664bdbc776fd7453c2783562f4d",
        "local_name": "FLUX.2-klein-base-4B",
    },
    "FLUX.2-klein-base-4B": {
        "repo_id": "black-forest-labs/FLUX.2-klein-base-4B",
        "revision": "a3b4f4849157f664bdbc776fd7453c2783562f4d",
        "local_name": "FLUX.2-klein-base-4B",
    },
    "clip-vit-large-patch14": {
        "repo_id": "openai/clip-vit-large-patch14",
        "revision": "32bd64288804d66eefd0ccbe215aa642df71cc41",
        "local_name": "clip-vit-large-patch14",
    },
    "openai/clip-vit-large-patch14": {
        "repo_id": "openai/clip-vit-large-patch14",
        "revision": "32bd64288804d66eefd0ccbe215aa642df71cc41",
        "local_name": "clip-vit-large-patch14",
    },
    "CLIP-ViT-H-14-laion2B-s32B-b79K": {
        "repo_id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "revision": "1c2b8495b28150b8a4922ee1c8edee224c284c0c",
        "local_name": "CLIP-ViT-H-14-laion2B-s32B-b79K",
    },
    "PickScore_v1": {
        "repo_id": "yuvalkirstain/PickScore_v1",
        "revision": "a4e4367c6dfa7288a00c550414478f865b875800",
        "local_name": "PickScore_v1",
    },
    "hpsv2.1": {
        "repo_id": "xswu/HPSv2",
        "revision": "697403c78157020a1ae59d23f111aa58ced35b0a",
        "local_name": "hpsv2.1",
    },
    "ImageReward": {
        "repo_id": "zai-org/ImageReward",
        "revision": "5736be03b2652728fb87788c9797b0570450ab72",
        "local_name": "ImageReward",
    },
    "UnifiedReward-2.0-qwen35-9b": {
        "repo_id": "CodeGoat24/UnifiedReward-2.0-qwen35-9b",
        "revision": "f01548b009741e12ff9817ed91dba94701ed9579",
        "local_name": "UnifiedReward-2.0-qwen35-9b",
    },
    "CodeGoat24/UnifiedReward-2.0-qwen35-9b": {
        "repo_id": "CodeGoat24/UnifiedReward-2.0-qwen35-9b",
        "revision": "f01548b009741e12ff9817ed91dba94701ed9579",
        "local_name": "UnifiedReward-2.0-qwen35-9b",
    },
}


def get_local_dataset_root():
    return os.environ.get("PRECISE_SDE_DATASET_ROOT") or LOCAL_DATASET_ROOT


def get_local_model_root():
    return os.environ.get("PRECISE_SDE_LOCAL_MODEL_ROOT") or LOCAL_MODEL_ROOT


def dataset_path(name, *sub_parts):
    """
    Get the absolute path for a named dataset in the local dataset directory.

    Examples:
        dataset_path("pickscore")
        dataset_path("geneval")
    """
    base = os.path.join(get_local_dataset_root(), name)
    return os.path.join(base, *sub_parts) if sub_parts else base


def resolve_dataset_reference(value):
    if not value:
        return value
    if os.path.isabs(value) or os.path.exists(value):
        return value
    return dataset_path(value)


def _pinned_model(name):
    return _PINNED_HF_MODELS.get(name)


def _local_model_name(name):
    pinned = _pinned_model(name)
    return pinned["local_name"] if pinned else name


def _model_root_override():
    return os.environ.get("PRECISE_SDE_MODEL_ROOT") or os.environ.get("PRECISE_SDE_LOCAL_MODEL_ROOT")


def model_revision(name):
    if not name or os.path.isabs(name) or os.path.exists(name) or _model_root_override():
        return None
    pinned = _pinned_model(name)
    return pinned["revision"] if pinned else None


def model_path(name, *sub_parts):
    """
    Get the default pinned Hugging Face model id, or a local override path.
    When sub_parts are provided for a pinned model, download that file from the
    pinned Hugging Face revision and return the cache path.

    Examples:
        model_path("clip-vit-large-patch14")
        model_path("hpsv2.1", "HPS_v2.1_compressed.pt")
        model_path("ImageReward", "ImageReward.pt")
    """
    model_root_override = _model_root_override()
    if model_root_override:
        base = os.path.join(model_root_override, _local_model_name(name))
        return os.path.join(base, *sub_parts) if sub_parts else base

    pinned = _pinned_model(name)
    if not pinned:
        base = os.path.join(get_local_model_root(), _local_model_name(name))
        return os.path.join(base, *sub_parts) if sub_parts else base

    if not sub_parts:
        return pinned["repo_id"]

    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=pinned["repo_id"],
        revision=pinned["revision"],
        filename=os.path.join(*sub_parts),
    )


def resolve_model_reference(value):
    if not value:
        return value
    if os.path.isabs(value) or os.path.exists(value):
        return value
    return model_path(value)


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Resolve centralized Precise-SDE model and dataset paths.")
    parser.add_argument(
        "kind",
        choices=("model", "dataset", "resolve-model", "resolve-dataset", "revision"),
        help="What kind of path lookup to perform.",
    )
    parser.add_argument("value", help="Model or dataset name/reference.")
    parser.add_argument("sub_parts", nargs="*", help="Optional path suffix components.")
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if args.kind == "model":
        print(model_path(args.value, *args.sub_parts))
    elif args.kind == "dataset":
        print(dataset_path(args.value, *args.sub_parts))
    elif args.kind == "resolve-model":
        if args.sub_parts:
            raise SystemExit("resolve-model does not accept extra path components")
        print(resolve_model_reference(args.value))
    elif args.kind == "revision":
        if args.sub_parts:
            raise SystemExit("revision does not accept extra path components")
        print(model_revision(args.value) or "")
    else:
        if args.sub_parts:
            raise SystemExit("resolve-dataset does not accept extra path components")
        print(resolve_dataset_reference(args.value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
