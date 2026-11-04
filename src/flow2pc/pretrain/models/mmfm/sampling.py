"""Minibatch OT chains with the native MMFM interpolant and noise process."""

import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from flow2pc.utils.mfm_source import source_module


def couple_chain(pools):
    """Align consecutive empirical marginals; these are OT pairs, not trajectories."""
    aligned = [pools[0]]
    for pool in pools[1:]:
        rows, columns = linear_sum_assignment(
            cdist(aligned[-1].cpu().numpy(), pool.cpu().numpy(), metric="sqeuclidean")
        )
        if len(rows) != len(aligned[-1]):
            raise ValueError("MMFM OT batches must have equal marginal sizes")
        permutation = torch.empty(len(rows), dtype=torch.long, device=pool.device)
        permutation[torch.as_tensor(rows, device=pool.device)] = torch.as_tensor(columns, device=pool.device)
        aligned.append(pool[permutation])
    return torch.stack(aligned, dim=1)


class MMFMPathSampler:
    def __init__(self, settings, source_root):
        if settings.get("coupling") != "minibatch_ot":
            raise ValueError("Only MMFM minibatch_ot coupling is supported")
        cls = source_module("multi_marginal_fm", source_root, method="mmfm").MultiMarginalFlowMatcher
        self.matcher = cls(sigma=settings["flow_variance"], interpolation=settings["interpolation"])

    @torch.no_grad()
    def sample(self, data, count, device, *, validation=False, interval_index=None):
        if count < 1:
            raise ValueError("MMFM batch size must be positive")
        observed = data.observed_indices
        if len(observed) < 2:
            raise ValueError("MMFM needs at least two observed marginals")
        frames = data.validation_marginals if validation else data.train_marginals
        pools = []
        for index in observed:
            values = frames[index]
            if len(values) == 0:
                raise ValueError("MMFM cannot sample an empty observed marginal")
            pools.append(values[torch.randint(len(values), (count,))].flatten(1))
        chains = couple_chain(pools).to(device)
        anchors = data.times[observed].to(device=device, dtype=chains.dtype)
        if not torch.all(anchors[1:] > anchors[:-1]):
            raise ValueError("MMFM model times must be strictly increasing")
        if interval_index is None:
            times = anchors[0] + torch.rand(count, device=device) * (anchors[-1] - anchors[0])
            indices = torch.searchsorted(anchors, times, right=True).sub(1).clamp(0, len(anchors) - 2)
        else:
            if not 0 <= interval_index < len(anchors) - 1:
                raise ValueError("Invalid MMFM validation interval")
            indices = torch.full((count,), interval_index, device=device, dtype=torch.long)
            times = anchors[indices] + torch.rand(count, device=device) * (anchors[indices + 1] - anchors[indices])
        # Native adaptive sigma derivatives are singular exactly at the anchors.
        gap = anchors[indices + 1] - anchors[indices]
        times = times.maximum(anchors[indices] + gap * 1e-5)
        times = times.minimum(anchors[indices + 1] - gap * 1e-5)
        timepoints = anchors.unsqueeze(0).expand(count, -1)
        times, values, target, _, _ = self.matcher.sample_location_and_conditional_flow(
            chains, timepoints, t=times,
        )
        values, target = values.reshape(count, -1).float(), target.reshape(count, -1).float()
        if not torch.isfinite(values).all() or not torch.isfinite(target).all():
            raise FloatingPointError("Nonfinite native MMFM interpolant/conditional flow")
        return times, values, target
