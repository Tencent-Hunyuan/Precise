from .builtin import (
    aesthetic_score,
    clip_score,
    hps_v2,
    imagereward_score,
    jpeg_compressibility,
    pickscore_score,
)
from .remote import geneval_score, unifiedreward_score_v2


def _parse_reward_cfg(score_dict):
    parsed = {}
    for name, cfg in score_dict.items():
        name = str(name)
        if isinstance(cfg, (int, float)):
            parsed[name] = {"weight": float(cfg)}
        elif hasattr(cfg, "items"):
            parsed_cfg = {str(key): value for key, value in dict(cfg).items()}
            parsed_cfg.setdefault("weight", 1.0)
            parsed_cfg["weight"] = float(parsed_cfg["weight"])
            parsed[name] = parsed_cfg
        else:
            parsed[name] = {"weight": float(cfg)}
    return parsed


def multi_score(device, score_dict):
    score_functions = {
        "aesthetic": aesthetic_score,
        "clipscore": clip_score,
        "geneval": geneval_score,
        "hpsv2": hps_v2,
        "imagereward": imagereward_score,
        "jpeg_compressibility": jpeg_compressibility,
        "pickscore": pickscore_score,
        "unifiedreward_v2": unifiedreward_score_v2,
    }

    parsed = _parse_reward_cfg(score_dict)
    score_fns = {}
    for score_name, cfg in parsed.items():
        fn = score_functions[score_name]
        extra_kwargs = {}
        if score_name == "unifiedreward_v2":
            acs_keys = {"alignment", "coherence", "style"}
            acs_weights = {key: float(value) for key, value in cfg.items() if key in acs_keys}
            if acs_weights:
                extra_kwargs["acs_weights"] = acs_weights
        if "device" in fn.__code__.co_varnames:
            score_fns[score_name] = fn(device, **extra_kwargs)
        else:
            score_fns[score_name] = fn(**extra_kwargs)

    def _fn(images, prompts, metadata, ref_images=None, only_strict=True):
        del ref_images
        total_scores = []
        score_details = {}
        sentinel = -10.0

        for score_name, cfg in parsed.items():
            weight = cfg["weight"]
            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](
                    images, prompts, metadata, only_strict
                )
                score_details["accuracy"] = rewards
                score_details["strict_accuracy"] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f"{key}_strict_accuracy"] = value
                for key, value in group_rewards.items():
                    score_details[f"{key}_accuracy"] = value
            else:
                scores, rewards = score_fns[score_name](images, prompts, metadata)
                if isinstance(rewards, dict):
                    for sub_key, sub_vals in rewards.items():
                        score_details[sub_key] = sub_vals

            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [
                    sentinel if (total == sentinel or weighted == weight * sentinel) else total + weighted
                    for total, weighted in zip(total_scores, weighted_scores)
                ]

        score_details["avg"] = total_scores
        return score_details, {}

    return _fn
