import math

import torch
from torch import nn

LOG_2PI = math.log(2.0 * math.pi)


class ProbabilisticCircuit(nn.Module):
    def path_log_prob(
        self,
        path_values: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        if path_values.shape[0] != len(times):
            raise ValueError("path_values and times must have the same path length")
        return torch.stack(
            [
                self.log_prob(values, time)
                for values, time in zip(path_values, times, strict=True)
            ]
        )

class TimeDependentGaussianPC(ProbabilisticCircuit):
    """Gaussian sum-product circuit parameterized over flow time."""

    def __init__(self, num_components: int, dim: int, time_order: int = 4):
        super().__init__()
        self.num_components = num_components
        self.dim = dim
        self.time_order = time_order
        shape = (num_components, dim, time_order)
        self.mean_coefficients = nn.Parameter(torch.zeros(shape))
        self.log_scale_coefficients = nn.Parameter(torch.zeros(shape))
        self.mixture_logits = nn.Parameter(torch.zeros(num_components))

    def _time_basis(self, time: torch.Tensor):
        powers = torch.arange(
            1,
            self.time_order + 1,
            device=time.device,
            dtype=time.dtype,
        )
        return time**powers, powers * time ** (powers - 1)

    def leaf_parameters(self, time: torch.Tensor):
        basis, derivative = self._time_basis(time)
        mean = torch.einsum("kdp,p->kd", self.mean_coefficients, basis)
        mean_dt = torch.einsum("kdp,p->kd", self.mean_coefficients, derivative)
        log_scale = torch.einsum("kdp,p->kd", self.log_scale_coefficients, basis)
        log_scale_dt = torch.einsum(
            "kdp,p->kd", self.log_scale_coefficients, derivative
        )
        return mean, mean_dt, log_scale, log_scale_dt

    def _component_log_prob(
        self,
        values: torch.Tensor,
        mean: torch.Tensor,
        log_scale: torch.Tensor,
    ) -> torch.Tensor:
        inverse_variance = torch.exp(-2.0 * log_scale)
        quadratic = values.square() @ inverse_variance.T
        quadratic -= 2.0 * values @ (mean * inverse_variance).T
        constant = (
            mean.square() * inverse_variance + 2.0 * log_scale
        ).sum(dim=1) + self.dim * LOG_2PI
        return -0.5 * (quadratic + constant[None])

    def log_prob(self, values: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        mean, _, log_scale, _ = self.leaf_parameters(time)
        log_weights = torch.log_softmax(self.mixture_logits, dim=0)
        component_ll = self._component_log_prob(values, mean, log_scale)
        return torch.logsumexp(log_weights[None] + component_ll, dim=1)

    def velocity(self, values: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        mean, mean_dt, log_scale, log_scale_dt = self.leaf_parameters(time)
        log_weights = torch.log_softmax(self.mixture_logits, dim=0)
        component_ll = self._component_log_prob(values, mean, log_scale)
        responsibilities = torch.softmax(log_weights[None] + component_ll, dim=1)
        offset = mean_dt - log_scale_dt * mean
        return responsibilities @ offset + (responsibilities @ log_scale_dt) * values

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        time: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.sample_path(num_samples, time.reshape(1), generator)[0]

    @torch.no_grad()
    def sample_path(
        self,
        num_samples: int,
        times: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        weights = torch.softmax(self.mixture_logits, dim=0)
        components = torch.multinomial(
            weights,
            num_samples,
            replacement=True,
            generator=generator,
        )
        device = self.mean_coefficients.device
        dtype = self.mean_coefficients.dtype
        noise = torch.randn(
            num_samples,
            self.dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        path = []
        for time in times:
            mean, _, log_scale, _ = self.leaf_parameters(time)
            path.append(mean[components] + torch.exp(log_scale[components]) * noise)
        return torch.stack(path)
