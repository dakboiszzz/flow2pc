"""Batched velocity-driven rollouts and exact population EMD from the old pipeline."""

import numpy as np
import torch


def fixed_subset(pool, count, seed):
    if count == 0 or count >= len(pool):
        return pool
    indices = torch.randperm(len(pool), generator=torch.Generator().manual_seed(seed))[
        :count
    ]
    return pool[indices]


def emd(real, predicted):
    """Previous W1 formula: uniform population masses, Euclidean costs, no square root."""
    import ot
    from scipy.spatial.distance import cdist

    a, b = (
        real.detach().cpu().double().numpy(),
        predicted.detach().cpu().double().numpy(),
    )
    if not len(a) or not len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("EMD requires nonempty finite populations")
    value, info = ot.emd2(
        np.full(len(a), 1 / len(a)),
        np.full(len(b), 1 / len(b)),
        cdist(a, b, "euclidean"),
        numItermax=10_000_000,
        log=True,
    )
    if info.get("warning") or not np.isfinite(value):
        raise RuntimeError(f"EMD solver failed: {info.get('warning')}")
    return float(value)


def raw_coordinates(values, preprocessing):
    from flow2pc.data.preprocessing import NATIVE_KIND, inverse

    if preprocessing.get("kind") == NATIVE_KIND:
        return inverse(values, preprocessing)
    return values.double() * torch.tensor(
        preprocessing["scale"], dtype=torch.float64
    ) + torch.tensor(preprocessing["shift"], dtype=torch.float64)


def metric_coordinates(values, preprocessing, space):
    from flow2pc.data.preprocessing import NATIVE_KIND, inverse

    if preprocessing.get("kind") == NATIVE_KIND:
        return inverse(values, preprocessing, retain_whitening=space == "historical")
    raw = raw_coordinates(values, preprocessing)
    if space == "historical" and preprocessing.get("whiten", False):
        if "metric_shift" not in preprocessing or "metric_scale" not in preprocessing:
            raise ValueError(
                "Saved whitening statistics needed for historical EMD units are missing. Supply the original statistics, or explicitly select evaluation.metric_space=raw; do not compare raw and historical EMD values."
            )
        raw = (
            raw - torch.tensor(preprocessing["metric_shift"], dtype=torch.float64)
        ) / torch.tensor(preprocessing["metric_scale"], dtype=torch.float64)
    return raw


@torch.no_grad()
def rollout(model, source, times, settings, *, student=False):
    """Integrate the velocity field from observed source cells; no PC sampling."""
    model.eval()
    device = next(model.parameters()).device
    paths, total_nfe = [], 0
    for start in range(0, len(source), settings["batch_size"]):
        evaluations = 0

        def velocity(t, x):
            nonlocal evaluations
            evaluations += 1
            if evaluations > settings["max_nfe"]:
                raise RuntimeError(
                    "ODE evaluation exceeded its per-batch function-evaluation budget"
                )
            result = model.velocity(x, t) if student else model.ode_velocity(t, x)
            if not torch.isfinite(result).all():
                raise FloatingPointError("Nonfinite velocity during evaluation")
            return result

        x = source[start : start + settings["batch_size"]].to(device)
        grid = times.to(device)
        if settings["solver"] == "dopri5":
            from torchdiffeq import odeint

            path = odeint(
                velocity,
                x,
                grid,
                method="dopri5",
                rtol=settings["tolerance"],
                atol=settings["tolerance"],
            )
        else:
            # Classical RK4, as in the previous ST/knot plotting scripts.
            # Include observation times exactly rather than rounding slide times
            # to the nearest integration step. Both models use the same grid.
            dense = torch.linspace(
                float(grid[0]),
                float(grid[-1]),
                settings["ode_steps"] + 1,
                device=device,
            )
            dense = torch.unique(torch.cat((dense, grid)), sorted=True)
            wanted = torch.searchsorted(dense, grid).tolist()
            records = [x.clone()]
            for index in range(len(dense) - 1):
                t, dt = dense[index], dense[index + 1] - dense[index]
                k1 = velocity(t, x)
                k2 = velocity(t + dt / 2, x + dt * k1 / 2)
                k3 = velocity(t + dt / 2, x + dt * k2 / 2)
                k4 = velocity(t + dt, x + dt * k3)
                x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
                if index + 1 in wanted:
                    records.append(x.clone())
            path = torch.stack(records)
        if not torch.isfinite(path).all():
            raise FloatingPointError("Nonfinite rollout positions")
        paths.append(path.cpu())
        total_nfe += evaluations
    return torch.cat(paths, dim=1), total_nfe
