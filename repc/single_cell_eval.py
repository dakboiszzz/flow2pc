"""Shared single-cell metrics; exact EMD is computed on CPU float64 arrays."""

import time

import numpy as np
import ot
from scipy.spatial.distance import cdist
import torch
from torchdiffeq import odeint


def fixed_subset(pool, count, seed):
    if count == 0 or count >= len(pool):
        return pool
    rng = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(len(pool), generator=rng)[:count]
    return pool[indices.to(pool.device)]


def emd(real, predicted):
    a = real.detach().cpu().double().numpy()
    b = predicted.detach().cpu().double().numpy()
    if not len(a) or not len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("EMD requires nonempty, finite populations")
    cost = cdist(a, b, "euclidean")
    value, info = ot.emd2(np.full(len(a), 1 / len(a)), np.full(len(b), 1 / len(b)),
                         cost, numItermax=10_000_000, log=True)
    if info.get("warning") or not np.isfinite(value):
        raise RuntimeError(f"EMD solver failed: {info.get('warning')}")
    return float(value)


@torch.no_grad()
def integrate(model, source, times, device, batch_size, tolerance, is_student=False):
    model.eval()
    trajectories, nfe = [], 0

    def velocity(t, values):
        nonlocal nfe
        nfe += 1
        if is_student:
            return model.velocity(values, t)
        return model(t.expand(len(values)), values)

    for start in range(0, len(source), batch_size):
        trajectory = odeint(velocity, source[start:start + batch_size].to(device),
                            times.to(device), method="dopri5",
                            rtol=tolerance, atol=tolerance)
        trajectories.append(trajectory.cpu())
    return torch.cat(trajectories, 1), nfe


@torch.no_grad()
def marginal_scores(data, num_samples, seed, times, predictions=None, direct=None,
                    student=None, device=None):
    scores = []
    for index, cloud in enumerate(data.clouds):
        real = fixed_subset(cloud, num_samples, seed + index + 1)
        generated = predictions if predictions is not None else direct
        score = {"index": index, "day": data.contract["preprocessing"]["time_order"][index],
                 "time": float(times[index]), "held_out": index == data.heldout,
                 "num_reference": len(real), "num_predicted": len(generated[index])}
        if predictions is not None:
            score["ode_emd"] = emd(data.metric_coordinates(real),
                                   data.metric_coordinates(predictions[index]))
        if direct is not None:
            score["direct_emd"] = emd(data.metric_coordinates(real),
                                      data.metric_coordinates(direct[index]))
        if student is not None:
            score["normalized_nll_nats_per_dim"] = float(
                -student.log_prob(real.to(device), times[index].to(device)).mean().cpu() / data.dim)
        scores.append(score)
    return scores


@torch.no_grad()
def direct_diagnostics(student, data, device, num_samples=512, seed=2026):
    """Monitoring only: held-out metrics never affect optimization or selection."""
    was_training = student.training
    student.eval()
    started = time.perf_counter()
    try:
        times = torch.linspace(0, 1, len(data.clouds))
        count = len(fixed_subset(data.clouds[0], num_samples, seed))
        rng = torch.Generator(device=device).manual_seed(seed)
        direct = student.sample_path(count, times.to(device), rng).cpu()
        scores = marginal_scores(data, num_samples, seed, times, direct=direct,
                                 student=student, device=device)
        return {"scores": scores, "heldout_direct_emd": scores[data.heldout]["direct_emd"],
                "num_samples": num_samples, "seed": seed,
                "metric_space": data.identity["metric_space"],
                "seconds": time.perf_counter() - started}
    finally:
        student.train(was_training)
