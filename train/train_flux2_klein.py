import os
from pathlib import Path

from absl import app, flags
from ml_collections import config_flags


FLAGS = flags.FLAGS
_DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "config" / "config.py")


def _define_config_flag():
    if "config" not in FLAGS:
        config_flags.DEFINE_config_file("config", _DEFAULT_CONFIG_PATH, "Training configuration.")


_define_config_flag()


def main(_):
    if os.environ.get("PRECISE_SDE_LAUNCH_MODEL") != "flux":
        raise SystemExit(
            "Direct trainer execution is disabled. Use "
            "`bash launch/train.sh --flux ...` instead."
        )

    from precise_sde.train.adapters import Flux2KleinAdapter
    from precise_sde.train.rl_trainer import run_training

    config = FLAGS.config
    return run_training(config, Flux2KleinAdapter(config))


if __name__ == "__main__":
    app.run(main)
