import math
from typing import Optional, Tuple, Union

import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from precise_sde.sde import canonicalize_sde_type


def _clamped_sqrt(value, *, min_value=0.0, max_value=None):
    if max_value is None:
        return torch.sqrt(value.clamp(min=min_value))
    return torch.sqrt(value.clamp(min=min_value, max=max_value))


def _get_step_indices(scheduler: FlowMatchEulerDiscreteScheduler, timesteps: Union[float, torch.FloatTensor]) -> torch.LongTensor:
    if isinstance(timesteps, torch.Tensor):
        return torch.tensor(
            [scheduler.index_for_timestep(t) for t in timesteps],
            device=timesteps.device,
            dtype=torch.long,
        )
    return torch.tensor(
        [scheduler.index_for_timestep(timesteps)],
        device=scheduler.sigmas.device,
        dtype=torch.long,
    )


def _get_step_sigmas(
    scheduler: FlowMatchEulerDiscreteScheduler,
    step_indices: torch.LongTensor,
    sample_ndim: int,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    sigma = scheduler.sigmas[step_indices].view(-1, *([1] * (sample_ndim - 1)))
    next_sigma = scheduler.sigmas[step_indices + 1].view(-1, *([1] * (sample_ndim - 1)))
    dt = next_sigma - sigma
    return sigma, next_sigma, dt


def compute_std_dev_t(self, eta, sde_type, device=None, sigmas=None):
    """
    Compute per-step noise standard deviation for a given SDE type.
    Returns a tensor of length len(sigmas) - 1.
    """
    sde_type = canonicalize_sde_type(sde_type)
    sigma_tensor = torch.as_tensor(sigmas if sigmas is not None else self.sigmas, dtype=torch.float32)
    if sigma_tensor.ndim != 1:
        sigma_tensor = sigma_tensor.flatten()
    sigma_tensor = sigma_tensor.to(device) if device is not None else sigma_tensor

    std_list = []
    for idx in range(sigma_tensor.numel() - 1):
        sigma = sigma_tensor[idx]
        next_sigma = sigma_tensor[idx + 1]
        dt = next_sigma - sigma

        if sde_type == "dance_grpo":
            std = eta * torch.sqrt(-dt)
        elif sde_type == "flow_grpo":
            sigma_max = sigma_tensor[1]
            denom_sigma = torch.where(sigma == 1, sigma_max, sigma)
            base_std = torch.sqrt(
                sigma / (1 - denom_sigma)
            ) * eta
            std = base_std * torch.sqrt(-dt)
        elif sde_type == "cps":
            std = next_sigma * math.sin(eta * math.pi / 2)
        elif sde_type == "precise":
            ratio = (next_sigma * (1 - sigma)) / (sigma * (1 - next_sigma))
            std = next_sigma * torch.sqrt(1 - (ratio ** (eta ** 2)))
        elif sde_type == "dance_precise":
            reciprocal_next_sigma = next_sigma.clamp(min=1e-6)
            reciprocal_sigma = sigma.clamp(min=1e-6)
            rho = torch.exp(-0.5 * eta**2 * ((1 / reciprocal_next_sigma) - (1 / reciprocal_sigma)))
            std = next_sigma * _clamped_sqrt(1 - rho**2, max_value=1.0)
        else:
            raise ValueError(f"Unsupported sde_type: {sde_type}")

        std_list.append(std)

    return torch.stack(std_list).clamp(min=1e-6)


def grpo_guard(self, eta, device, sde_type, sigmas=None):
    """
    Compute per-step |∂ prev_sample_mean / ∂ model_output| used to balance
    gradients during GRPO. Mirrors the closed-form coefficients from
    sde_step_with_logprob for each SDE variant.
    """
    sde_type = canonicalize_sde_type(sde_type)
    sigma_tensor = torch.as_tensor(sigmas if sigmas is not None else self.sigmas, dtype=torch.float32)
    sigma_tensor = sigma_tensor.to(device) if device is not None else sigma_tensor

    std_tensor = compute_std_dev_t(self, eta=eta, sde_type=sde_type, device=device, sigmas=sigmas)

    all_jac = []
    for idx in range(sigma_tensor.numel() - 1):
        sigma = sigma_tensor[idx]
        next_sigma = sigma_tensor[idx + 1]
        dt = next_sigma - sigma
        std_dev_t = std_tensor[idx]
        std_dev_t_sq = std_dev_t ** 2

        if sde_type in ("flow_grpo", "dance_grpo"):
            jac = dt - 0.5 * std_dev_t_sq * ((1 - sigma) / sigma)
        elif sde_type in ("cps", "precise", "dance_precise"):
            base = next_sigma ** 2 - std_dev_t_sq
            sqrt_term = _clamped_sqrt(base)
            jac = -sigma * (1 - next_sigma) + (1 - sigma) * sqrt_term
        else:
            raise ValueError(f"Unsupported sde_type: {sde_type}")

        all_jac.append(torch.abs(jac))

    return std_tensor, torch.stack(all_jac)


def sde_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    sde_type: Optional[str] = "flow_grpo",
    compute_log_prob: bool = True,
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
    process from the learned model outputs (most often the predicted velocity).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned flow model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
    """
    # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    sde_type = canonicalize_sde_type(sde_type)
    step_indices = _get_step_indices(self, timestep)
    sigma, next_sigma, dt = _get_step_sigmas(self, step_indices, sample.ndim)
    std_tensor = compute_std_dev_t(self, eta=noise_level, sde_type=sde_type, device=sample.device)
    std_dev_t = std_tensor[step_indices].view(-1, *([1] * (sample.ndim - 1)))

    pred_original_sample = sample - sigma * model_output

    if sde_type in ("flow_grpo", "dance_grpo"):
        score_estimate = -(sample - pred_original_sample * (1 - sigma)) / (sigma ** 2)
        prev_sample_mean = sample + dt * model_output + 0.5 * (std_dev_t ** 2) * score_estimate
    elif sde_type in ("cps", "precise", "dance_precise"):
        noise_estimate = sample + model_output * (1 - sigma)
        prev_sample_mean = pred_original_sample * (1 - next_sigma) + noise_estimate * torch.sqrt(
            (next_sigma ** 2 - std_dev_t ** 2).clamp(min=0)
        )
    else:
        raise ValueError(f"Unsupported sde_type: {sde_type}")

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * variance_noise

    if compute_log_prob:
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t ** 2))
            - torch.log(std_dev_t)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    else:
        log_prob = None

    return prev_sample, log_prob, prev_sample_mean
