from config.base import default
from config.flux2_klein import flux2_klein_config, flux_launch


def get_config(name="default"):
    return globals()[name]()
