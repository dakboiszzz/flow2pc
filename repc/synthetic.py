import math

import torch
from torch import nn

LOG_2PI = math.log(2.0 * math.pi)


class BranchingGaussianFlow(nn.Module):
    """Analytic vanilla-CFM path for a Gaussian-mixture endpoint.

    The base sample is standard normal and is independently paired with a
    Gaussian-mixture endpoint sample. At time ``t`` the component marginals are
    Gaussian with mean ``t * endpoint_mean`` and variance
    ``(1 - t)^2 + t^2 * endpoint_variance``. Marginalizing the conditional CFM
    targets gives the exact probability-flow velocity implemented below.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        endpoint_means: torch.Tensor,
        endpoint_scales: torch.Tensor,
    ):
        super().__init__()
        if endpoint_means.ndim != 2:
            raise ValueError("endpoint_means must have shape [components, dim]")
        if endpoint_scales.shape != endpoint_means.shape:
            raise ValueError("endpoint_scales must match endpoint_means")
        if weights.shape != endpoint_means.shape[:1]:
            raise ValueError("weights must have one value per component")
        if torch.any(weights <= 0.0):
            raise ValueError("all mixture weights must be positive")
        if torch.any(endpoint_scales <= 0.0):
            raise ValueError("all endpoint scales must be positive")

        normalized_weights = weights / weights.sum()
        self.register_buffer("log_weights", normalized_weights.log())
        self.register_buffer("endpoint_means", endpoint_means)
        self.register_buffer("endpoint_scales", endpoint_scales)
        self.num_components = len(weights)
        self.dim = endpoint_means.shape[1]

    @property
    def weights(self) -> torch.Tensor:
        return self.log_weights.exp()

    def marginal_parameters(
        self,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return component means and scales at scalar or batched times."""
        if time.ndim == 0:
            means = time * self.endpoint_means
            variances = (1.0 - time).square() + time.square() * (
                self.endpoint_scales.square()
            )
            return means, variances.sqrt()

        expanded_time = time.reshape(-1, 1, 1)
        means = expanded_time * self.endpoint_means[None]
        variances = (1.0 - expanded_time).square() + expanded_time.square() * (
            self.endpoint_scales.square()[None]
        )
        return means, variances.sqrt()

    def _component_log_prob(
        self,
        values: torch.Tensor,
        means: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        if means.ndim == 2:
            standardized = (values[:, None] - means[None]) / scales[None]
            log_determinant = scales.log().sum(dim=-1)[None]
        else:
            standardized = (values[:, None] - means) / scales
            log_determinant = scales.log().sum(dim=-1)
        return (
            -0.5 * (standardized.square().sum(dim=-1) + self.dim * LOG_2PI)
            - log_determinant
        )

    def log_prob(self, values: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        means, scales = self.marginal_parameters(time)
        component_log_prob = self._component_log_prob(values, means, scales)
        return torch.logsumexp(self.log_weights[None] + component_log_prob, dim=1)

    def velocity(self, values: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Evaluate the exact marginal CFM velocity E[x1 - x0 | x_t]."""
        means, scales = self.marginal_parameters(time)
        component_log_prob = self._component_log_prob(values, means, scales)
        responsibilities = torch.softmax(
            self.log_weights[None] + component_log_prob,
            dim=1,
        )

        if time.ndim == 0:
            variance = scales.square()
            scale_rate = (
                time * self.endpoint_scales.square() - (1.0 - time)
            ) / variance
            component_velocity = self.endpoint_means + scale_rate * (
                values[:, None] - means[None]
            )
        else:
            expanded_time = time.reshape(-1, 1, 1)
            variance = scales.square()
            scale_rate = (
                expanded_time * self.endpoint_scales.square()[None]
                - (1.0 - expanded_time)
            ) / variance
            component_velocity = self.endpoint_means[None] + scale_rate * (
                values[:, None] - means
            )
        return (responsibilities[:, :, None] * component_velocity).sum(dim=1)

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        time: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw independent samples from one scalar or per-example path time."""
        if time.ndim > 0 and time.numel() != num_samples:
            raise ValueError("batched time must contain one value per sample")
        components = torch.multinomial(
            self.weights,
            num_samples,
            replacement=True,
            generator=generator,
        )
        noise = torch.randn(
            num_samples,
            self.dim,
            device=self.endpoint_means.device,
            dtype=self.endpoint_means.dtype,
            generator=generator,
        )
        means, scales = self.marginal_parameters(time)
        if time.ndim == 0:
            return means[components] + scales[components] * noise
        indices = torch.arange(num_samples, device=components.device)
        return means[indices, components] + scales[indices, components] * noise


def standard_branching_flow(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> BranchingGaussianFlow:
    """Return the fixed six-branch source used by the controlled benchmark."""
    weights = torch.tensor(
        [0.10, 0.18, 0.14, 0.22, 0.16, 0.20],
        device=device,
        dtype=dtype,
    )
    endpoint_means = torch.tensor(
        [
            [-3.2, -2.1],
            [-3.0, 2.2],
            [-0.7, -3.2],
            [0.4, 3.1],
            [3.0, -2.0],
            [3.3, 2.0],
        ],
        device=device,
        dtype=dtype,
    )
    endpoint_scales = torch.tensor(
        [
            [0.34, 0.42],
            [0.40, 0.32],
            [0.30, 0.48],
            [0.46, 0.30],
            [0.36, 0.44],
            [0.42, 0.36],
        ],
        device=device,
        dtype=dtype,
    )
    return BranchingGaussianFlow(weights, endpoint_means, endpoint_scales)


def sample_path_times(
    count: int,
    device: torch.device,
    generator: torch.Generator,
    sampling: str,
    beta_alpha: float = 1.0,
    beta_beta: float = 0.8,
) -> torch.Tensor:
    """Sample uniform or endpoint-biased path times."""
    if sampling == "uniform":
        return torch.rand(count, device=device, generator=generator)
    if sampling == "beta":
        concentration1 = torch.tensor(beta_alpha, device=device)
        concentration0 = torch.tensor(beta_beta, device=device)
        times = torch.distributions.Beta(concentration1, concentration0).sample(
            (count,)
        )
        epsilon = torch.finfo(times.dtype).eps
        return times.clamp(epsilon, 1.0 - epsilon)
    raise ValueError(f"unknown timestep sampling scheme: {sampling}")
