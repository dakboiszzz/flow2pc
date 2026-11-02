import math

import torch
from torch import nn

from flow2pc.pc.base import ProbabilisticCircuit

LOG_2PI = math.log(2.0 * math.pi)


def discretized_gaussian_log_mass(
    values: torch.Tensor,
    mean: torch.Tensor,
    log_scale: torch.Tensor,
    num_bins: int = 256,
) -> torch.Tensor:
    """Log mass of Gaussian leaves in the quantization bins of [-1, 1].

    ``values`` has shape ``[batch, dim]`` and ``mean``/``log_scale`` have
    shape ``[dim, channels]``. The edge bins absorb the Gaussian tails so the
    resulting 256-way distribution is normalized, as required for image bpd.
    """
    if num_bins < 2:
        raise ValueError("num_bins must be at least two")
    indices = torch.round((values + 1.0) * (num_bins - 1) / 2.0)
    indices = indices.clamp(0, num_bins - 1)
    width = 2.0 / (num_bins - 1)
    lower = -1.0 + (indices - 0.5) * width
    upper = lower + width

    inverse_scale = torch.exp(-log_scale)[None]
    standardized_lower = (lower[:, :, None] - mean[None]) * inverse_scale
    standardized_upper = (upper[:, :, None] - mean[None]) * inverse_scale

    # log(Phi(upper) - Phi(lower)), evaluated through the nearer tail to avoid
    # catastrophic cancellation for intervals far from zero.
    log_cdf_upper = torch.special.log_ndtr(standardized_upper)
    log_cdf_lower = torch.special.log_ndtr(standardized_lower)
    log_survival_lower = torch.special.log_ndtr(-standardized_lower)
    log_survival_upper = torch.special.log_ndtr(-standardized_upper)

    def log_difference(log_a: torch.Tensor, log_b: torch.Tensor) -> torch.Tensor:
        ratio = torch.exp((log_b - log_a).clamp(max=-torch.finfo(log_a.dtype).eps))
        return log_a + torch.log1p(-ratio)

    lower_tail = log_difference(log_cdf_upper, log_cdf_lower)
    upper_tail = log_difference(log_survival_lower, log_survival_upper)
    log_mass = torch.where(standardized_lower >= 0.0, upper_tail, lower_tail)
    log_mass = torch.where(
        indices[:, :, None] == 0,
        log_cdf_upper,
        log_mass,
    )
    log_mass = torch.where(
        indices[:, :, None] == num_bins - 1,
        log_survival_lower,
        log_mass,
    )
    return log_mass


class TimeDependentGaussianPC(ProbabilisticCircuit):
    """Gaussian sum-product circuit parameterized over flow time."""

    def __init__(
        self,
        num_components: int,
        dim: int,
        time_order: int = 4,
        polynomial_zero_center: bool = True,
    ):
        super().__init__()
        self.num_components = num_components
        self.dim = dim
        self.time_order = time_order
        self.polynomial_zero_center = polynomial_zero_center
        num_features = time_order if polynomial_zero_center else (time_order + 1)
        shape = (num_components, dim, num_features)
        # Identical coefficients give identical responsibilities and gradients,
        # trapping the mixture in a single Gaussian. Perturb each component;
        # the time basis vanishes at t=0, preserving the exact N(0, I) boundary.
        self.mean_coefficients = nn.Parameter(0.01 * torch.randn(shape))
        self.log_scale_coefficients = nn.Parameter(0.01 * torch.randn(shape))
        self.mixture_logits = nn.Parameter(torch.zeros(num_components))

    def _time_basis(self, time: torch.Tensor):
        if self.polynomial_zero_center:
            powers = torch.arange(
                1,
                self.time_order + 1,
                device=time.device,
                dtype=time.dtype,
            )
            orig_ndim = time.ndim
            time_in = time[:, None] if orig_ndim > 0 else time
            return time_in**powers, powers * time_in ** (powers - 1)
        else:
            powers = torch.arange(
                0,
                self.time_order + 1,
                device=time.device,
                dtype=time.dtype,
            )
            orig_ndim = time.ndim
            time_in = time[:, None] if orig_ndim > 0 else time
            basis = time_in**powers
            derivative = torch.zeros_like(basis)
            if self.time_order >= 1:
                p_pos = powers[1:]
                t_pow = torch.where(
                    (p_pos == 1) & (time_in == 0.0),
                    torch.ones_like(time_in),
                    time_in ** (p_pos - 1),
                )
                if orig_ndim > 0:
                    derivative[:, 1:] = p_pos * t_pow
                else:
                    derivative[1:] = p_pos * t_pow
            return basis, derivative

    def leaf_parameters(self, time: torch.Tensor):
        basis, derivative = self._time_basis(time)
        if time.ndim == 0:
            mean = torch.einsum("kdp,p->kd", self.mean_coefficients, basis)
            mean_dt = torch.einsum("kdp,p->kd", self.mean_coefficients, derivative)
            log_scale = torch.einsum("kdp,p->kd", self.log_scale_coefficients, basis)
            log_scale_dt = torch.einsum(
                "kdp,p->kd", self.log_scale_coefficients, derivative
            )
        else:
            mean = torch.einsum("kdp,bp->bkd", self.mean_coefficients, basis)
            mean_dt = torch.einsum("kdp,bp->bkd", self.mean_coefficients, derivative)
            log_scale = torch.einsum("kdp,bp->bkd", self.log_scale_coefficients, basis)
            log_scale_dt = torch.einsum(
                "kdp,bp->bkd", self.log_scale_coefficients, derivative
            )
        return mean, mean_dt, log_scale, log_scale_dt

    def _component_log_prob(
        self,
        values: torch.Tensor,
        mean: torch.Tensor,
        log_scale: torch.Tensor,
    ) -> torch.Tensor:
        if mean.ndim == 3:
            inverse_variance = torch.exp(-2.0 * log_scale)
            return -0.5 * (
                ((values[:, None] - mean).square() * inverse_variance).sum(dim=2)
                + 2.0 * log_scale.sum(dim=2)
                + self.dim * LOG_2PI
            )
        inverse_variance = torch.exp(-2.0 * log_scale)
        quadratic = values.square() @ inverse_variance.T
        quadratic -= 2.0 * values @ (mean * inverse_variance).T
        constant = (mean.square() * inverse_variance + 2.0 * log_scale).sum(
            dim=1
        ) + self.dim * LOG_2PI
        return -0.5 * (quadratic + constant[None])

    def log_prob(self, values: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        mean, _, log_scale, _ = self.leaf_parameters(time)
        log_weights = torch.log_softmax(self.mixture_logits, dim=0)
        component_ll = self._component_log_prob(values, mean, log_scale)
        return torch.logsumexp(log_weights[None] + component_ll, dim=1)

    def discrete_log_prob(
        self,
        values: torch.Tensor,
        time: torch.Tensor,
        num_bins: int = 256,
    ) -> torch.Tensor:
        mean, _, log_scale, _ = self.leaf_parameters(time)
        leaf_log_mass = discretized_gaussian_log_mass(
            values,
            mean.T,
            log_scale.T,
            num_bins,
        )
        component_log_mass = leaf_log_mass.sum(dim=1)
        log_weights = torch.log_softmax(self.mixture_logits, dim=0)
        return torch.logsumexp(log_weights[None] + component_log_mass, dim=1)

    def velocity(self, values: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        mean, mean_dt, log_scale, log_scale_dt = self.leaf_parameters(time)
        log_weights = torch.log_softmax(self.mixture_logits, dim=0)
        component_ll = self._component_log_prob(values, mean, log_scale)
        responsibilities = torch.softmax(log_weights[None] + component_ll, dim=1)
        offset = mean_dt - log_scale_dt * mean
        if time.ndim > 0:
            weighted_offset = (responsibilities[:, :, None] * offset).sum(dim=1)
            weighted_scale_dt = (responsibilities[:, :, None] * log_scale_dt).sum(dim=1)
            return weighted_offset + weighted_scale_dt * values
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
