import torch


def t_weighting_function(t, t_sample, t_gamma):
    pairwise_sq_diff = (t[:, None] - t_sample[None, :]).square()
    return torch.exp(-pairwise_sq_diff / t_gamma).squeeze(-1)


def weighting_function(x, samples, gamma):
    # Squared distances without constructing a [batch, cells, dimensions] tensor.
    squared_distance = (
        x.square().sum(-1, keepdim=True)
        + samples.square().sum(-1).unsqueeze(0)
        - 2 * (x @ samples.T)
    ).clamp_min(0)
    return torch.exp(-squared_distance / gamma)


def compute_time_dependent_metric(x, t, samples, t_samples, gamma,
                                  t_gamma, normalize_t=False, rho=1e-6):
    weights = weighting_function(x, samples, gamma)
    if t_gamma > 0:
        temporal_weights = t_weighting_function(t, t_samples, t_gamma)
        if normalize_t:
            temporal_weights = temporal_weights / (temporal_weights.sum(-1, keepdim=True) + 1e-8)
        weights = weights * temporal_weights

    # Sum_s w_s (x - x_s)^2, expressed using weighted moments so all observed
    # cells can enter Eq. 20 without allocating a batch-by-cells-by-dimension array.
    diagonal = (
        weights @ samples.square()
        - 2 * x * (weights @ samples)
        + x.square() * weights.sum(-1, keepdim=True)
    ).clamp_min(0)
    return 1 / (diagonal + rho)
