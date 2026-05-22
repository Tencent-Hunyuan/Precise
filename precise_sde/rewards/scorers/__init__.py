from importlib import import_module

__all__ = [
    "AestheticScorer",
    "ClipScorer",
    "HPSClipRewardModel",
    "ImageRewardScorer",
    "PickScoreScorer",
]

_SCORER_IMPORTS = {
    "AestheticScorer": ("precise_sde.rewards.scorers.aesthetic", "AestheticScorer"),
    "ClipScorer": ("precise_sde.rewards.scorers.clip", "ClipScorer"),
    "HPSClipRewardModel": ("precise_sde.rewards.scorers.hpsv2", "HPSClipRewardModel"),
    "ImageRewardScorer": ("precise_sde.rewards.scorers.imagereward", "ImageRewardScorer"),
    "PickScoreScorer": ("precise_sde.rewards.scorers.pickscore", "PickScoreScorer"),
}


def __getattr__(name):
    if name not in _SCORER_IMPORTS:
        raise AttributeError(name)
    module_name, attr_name = _SCORER_IMPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
