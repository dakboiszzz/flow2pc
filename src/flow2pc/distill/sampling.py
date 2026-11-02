"""Recovered teacher-query distributions; no noise-to-image fallback or PC samples."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DistillationBatch:
    x_t: torch.Tensor
    model_t: torch.Tensor
    context: dict | None = None


def draw(pool, count, rng):
    if len(pool) == 0:
        raise ValueError("Cannot draw from a held-out or empty marginal")
    return pool[torch.randint(len(pool), (count,), generator=rng)]


def mmot_endpoints(x0, middle, x1):
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    row0, col_middle = linear_sum_assignment(
        cdist(x0.numpy(), middle.numpy(), "sqeuclidean")
    )
    row_middle, col1 = linear_sum_assignment(
        cdist(middle.numpy(), x1.numpy(), "sqeuclidean")
    )
    source, target = np.empty(len(middle), dtype=int), np.empty(len(middle), dtype=int)
    source[col_middle], target[row_middle] = row0, col1
    return x0[source], x1[target]


@torch.no_grad()
def sample_batch(teacher, dataset, pools, count, rng, sampler):
    device = next(teacher.parameters()).device
    x0, x1 = draw(pools[0], count, rng), draw(pools[-1], count, rng)
    if sampler == "mmot_middle":
        interior = [i for i in dataset.observed if 0 < i < len(pools) - 1]
        if not interior:
            raise ValueError(
                "MMOT-middle queries require an observed interior population"
            )
        index = interior[int(torch.randint(len(interior), (), generator=rng))]
        x0, x1 = mmot_endpoints(x0, draw(pools[index], count, rng), x1)
    elif sampler == "border_ot":
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial.distance import cdist

        rows, columns = linear_sum_assignment(
            cdist(x0.numpy(), x1.numpy(), "sqeuclidean")
        )
        # Native border OT samples matched edges with replacement.
        edges = torch.randint(count, (count,), generator=rng)
        x0, x1 = x0[rows][edges], x1[columns][edges]
    elif sampler != "independent":
        raise ValueError(f"Unsupported teacher query sampler {sampler!r}")
    times = torch.rand(count, generator=rng).to(device)
    if dataset.metadata.name in ("st", "knot"):
        times = times.clamp(1e-5, 1 - 1e-5)
    values = teacher.interpolate(x0.to(device), x1.to(device), times)
    return DistillationBatch(values, times)


def velocity_mse(student, teacher, batch):
    """The sole distillation objective, identically for every supported dataset."""
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError("Pass a frozen teacher in evaluation mode")
    with torch.no_grad():
        target = teacher.velocity(batch.x_t, batch.model_t, context=batch.context)
    prediction = student.velocity(batch.x_t, batch.model_t)
    if prediction.shape != target.shape or not torch.isfinite(target).all():
        raise ValueError("Invalid teacher/student velocity shapes or target")
    return (prediction - target).square().mean()
