CANONICAL_SDE_TYPES = ("flow_grpo", "dance_grpo", "cps", "precise", "dance_precise")
CLI_SDE_TYPES = ("ode",) + CANONICAL_SDE_TYPES


def canonicalize_sde_type(sde_type):
    if sde_type not in CANONICAL_SDE_TYPES:
        raise ValueError(f"Unsupported sde_type: {sde_type}")
    return sde_type
