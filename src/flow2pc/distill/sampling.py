"""Method-specific teacher queries with a shared student velocity objective."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DistillationBatch:
    x_t: torch.Tensor
    model_t: torch.Tensor
    context: dict | None = None
    target_velocity: torch.Tensor | None = None


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
def sample_ali_batch(teacher, dataset, pools, count, rng, sampler):
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


def require_frozen_teacher(teacher):
    """ALI teachers are modules; MFM wraps its frozen model in a data object."""
    model = getattr(teacher, "model", teacher)
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Pass a frozen teacher in evaluation mode")


def velocity_mse(student, teacher, batch):
    """Shared velocity objective; the teacher supplies method-specific queries."""
    require_frozen_teacher(teacher)
    if batch.target_velocity is None:
        with torch.no_grad():
            target = teacher.velocity(batch.x_t, batch.model_t, context=batch.context)
    else:
        target = batch.target_velocity.detach()
    prediction = student.velocity(batch.x_t, batch.model_t)
    if (
        prediction.shape != target.shape
        or not torch.isfinite(target).all()
        or not torch.isfinite(batch.x_t).all()
    ):
        raise ValueError("Invalid teacher/student velocity shapes or query points")
    loss = (prediction - target).square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite distillation velocity MSE")
    return loss


def _draw(values, count, device):
    return values[torch.randint(len(values), (count,))].to(device)


def sample_mfm_batch(
    teacher,
    data,
    batch_size,
    device,
    *,
    validation=False,
    interval_index=None,
):
    require_frozen_teacher(teacher)
    frames = data.validation_marginals if validation else data.train_marginals
    observed = data.observed_indices
    interval = (
        int(torch.randint(len(observed) - 1, ()).item())
        if interval_index is None
        else interval_index
    )
    left, right = observed[interval : interval + 2]
    x0, x1 = (
        _draw(frames[left], batch_size, device),
        _draw(frames[right], batch_size, device),
    )
    x0, x1 = teacher.path_sampler.couple(x0, x1)
    times, samples = teacher.path_sampler.sample(
        x0, x1, float(data.times[left]), float(data.times[right])
    )
    with torch.no_grad():
        target = teacher.velocity(samples, times).flatten(1)
    if not torch.isfinite(target).all() or not torch.isfinite(samples).all():
        raise FloatingPointError("Nonfinite metric teacher velocity/geopath points")
    return DistillationBatch(samples.flatten(1), times, target_velocity=target)


def sample_mmfm_batch(teacher, data, batch_size, device, *, validation=False, interval_index=None):
    """Query the frozen MMFM field on its own multi-marginal stochastic paths."""
    require_frozen_teacher(teacher)
    if not hasattr(teacher, "path_sampler"):
        from flow2pc.pretrain.models.mmfm.sampling import MMFMPathSampler

        teacher.path_sampler = MMFMPathSampler(teacher.metadata["matcher"], teacher.source_root)
    with torch.no_grad():
        times, values, _ = teacher.path_sampler.sample(
            data, batch_size, device, validation=validation, interval_index=interval_index,
        )
        target = teacher.velocity(values, times)
    return DistillationBatch(values, times, target_velocity=target)
