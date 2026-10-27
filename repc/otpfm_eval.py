"""Exact unequal-mass empirical W2, deterministic diagnostics and comparisons."""

import math
import time

import numpy as np
import ot
from scipy.spatial.distance import cdist
import torch

from repc.otpfm import sample_cache
from repc.otpfm_data import physical


def w2(real, predicted):
    a, b = real.detach().cpu().double().numpy(), predicted.detach().cpu().double().numpy()
    if (a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or not len(a) or not len(b)
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("W2 requires finite nonempty [N,D] populations")
    cost, info = ot.emd2(np.ones(len(a)) / len(a), np.ones(len(b)) / len(b),
                         cdist(a, b, "sqeuclidean"), numItermax=10_000_000, log=True)
    if info.get("warning") or not np.isfinite(cost):
        raise RuntimeError(f"W2 solve failed: {info.get('warning')}")
    return math.sqrt(max(0., float(cost)))


@torch.no_grad()
def velocity_validation(teacher, cache, rows, student=None, seed=2026, batches=8, batch_size=128):
    rng = torch.Generator().manual_seed(seed)
    device = cache["x0"].device
    error = energy = conditional_error = 0.
    was_training = student.training if student is not None else False
    if student is not None:
        student.eval()
    try:
        for _ in range(batches):
            selected = rows[torch.randint(len(rows), (batch_size,), generator=rng)].to(device)
            times = torch.rand(batch_size, generator=rng).to(device)
            values, conditional = sample_cache(teacher.model, cache, selected, times)
            target = teacher(times, values)
            conditional_error += float((target - conditional).square().mean())
            energy += float(target.square().mean())
            if student is not None:
                error += float((student.velocity(values, times) - target).square().mean())
    finally:
        if student is not None:
            student.train(was_training)
    result = {"teacher_conditional_fm_mse": conditional_error / batches,
              "teacher_velocity_energy": energy / batches}
    if student is not None:
        result.update(velocity_mse=error / batches, relative_velocity_mse=error / max(energy, 1e-12))
    return result


@torch.no_grad()
def interpolant_path(model, cache, times):
    rows = torch.arange(len(cache["x0"]), device=cache["x0"].device)
    return torch.stack([sample_cache(model, cache, rows, cache["x0"].new_full((len(rows),), t))[0].cpu()
                        for t in times])


@torch.no_grad()
def finite_teacher_path(teacher, source, observation_times, steps):
    """Native off-diagonal Euler/MeanFlow rule, split additionally at exact observations.

    This is an explicitly labeled common protocol, not the legacy truncated /
    nearby-grid minimum score. Saved potential knots are never changed.
    """
    knots = teacher.model.tks.detach().cpu().double().tolist()
    mesh = [left + (right - left) * i / steps for left, right in zip(knots[:-1], knots[1:])
            for i in range(steps)] + [1.]
    mesh = sorted(set([value for value in mesh
                       if not any(abs(value - t) < 1e-7 for t in observation_times)] + observation_times))
    values = source.clone()
    snapshots = {0.: values.cpu()}
    started = time.perf_counter()
    for left, right in zip(mesh[:-1], mesh[1:]):
        t1 = values.new_full((len(values),), left)
        t2 = values.new_full((len(values),), right)
        values = values + (t2 - t1)[:, None] * teacher.model.v_func_ema(values, t1, t2)
        if right in observation_times:
            snapshots[right] = values.cpu()
    return torch.stack([snapshots[t] for t in observation_times]), {
        "method": "finite_step_ema_exact_observation_events", "base_steps_per_potential_interval": steps,
        "nfe": len(mesh) - 1, "seconds": time.perf_counter() - started,
        "note": "observation events split native intervals; distinct from legacy evaluation"}


def score_paths(bundle, paths, student=None, device=None):
    contract = bundle["contract"]
    scores = []
    with torch.no_grad():
        for index, real in enumerate(bundle["clouds"]):
            row = {"index": index, "time": contract["times"][index],
                   "held_out": index in contract["holdout_indices"], "num_reference": len(real)}
            for name, path in paths.items():
                row[name + "_w2"] = w2(physical(real, contract), physical(path[index], contract))
                row[name + "_num_samples"] = len(path[index])
            for student_name, teacher_name in (("student_direct", "teacher_finite"), ("student_ode", "teacher_ode")):
                if student_name in paths and teacher_name in paths:
                    row[f"{student_name}_vs_{teacher_name}_w2"] = w2(
                        physical(paths[student_name][index], contract), physical(paths[teacher_name][index], contract))
            if "student_ode" in paths and "teacher_ode" in paths:
                difference = physical(paths["student_ode"][index], contract) - physical(paths["teacher_ode"][index], contract)
                row["paired_ode_position_rmse"] = float(difference.square().sum(-1).mean().sqrt())
            if student is not None:
                nll = -float(student.log_prob(real.to(device), real.new_tensor(row["time"]).to(device)).mean())
                jacobian = sum(math.log(scale) for scale in contract["normalization"]["scale"])
                row["normalized_nll_nats_per_dim"] = nll / contract["dim"]
                row["physical_nll_nats_per_dim"] = (nll + jacobian) / contract["dim"]
            scores.append(row)
    summary = {}
    for name in paths:
        summary[name + "_holdout_w2"] = float(np.mean([row[name + "_w2"] for row in scores if row["held_out"]]))
        summary[name + "_observed_w2"] = float(np.mean([row[name + "_w2"] for row in scores
                                                        if not row["held_out"] and row["index"] != 0]))
        summary[name + "_source_w2"] = scores[0][name + "_w2"]
    for key in ("student_direct_vs_teacher_finite_w2", "student_ode_vs_teacher_ode_w2"):
        if key in scores[0]:
            summary[key.replace("_w2", "_holdout_w2")] = float(np.mean([row[key] for row in scores if row["held_out"]]))
    return {"scores": scores, **summary}


def comparison_plots(bundle, paths, output, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    contract = bundle["contract"]
    count, dim = len(bundle["clouds"]), contract["dim"]
    cols = 4 if dim == 1 else 3
    fig, axes = plt.subplots(math.ceil(count / cols), cols, figsize=(4 * cols, 3 * math.ceil(count / cols)))
    colors = {"teacher_finite": "tab:blue", "interpolant": "tab:green", "student_direct": "tab:orange",
              "teacher_ode": "tab:purple", "student_ode": "tab:red"}
    for index, ax in enumerate(axes.flat):
        if index >= count:
            ax.axis("off")
            continue
        populations = {"real": physical(bundle["clouds"][index], contract).numpy(),
                       **{name: physical(path[index], contract).numpy() for name, path in paths.items()}}
        for name, values in populations.items():
            if dim == 1:
                ax.hist(values[:, 0], bins=35, density=True, histtype="step", label=name,
                        color="black" if name == "real" else colors.get(name), linewidth=1.2)
            else:
                ax.scatter(values[:, 0], values[:, 1], s=7, alpha=.45, label=name,
                           color="black" if name == "real" else colors.get(name))
        ax.set_title(f"t={contract['times'][index]:.4f}" + (" (holdout)" if index in contract["holdout_indices"] else ""))
        ax.set_xlabel("PM2.5 (physical units)" if dim == 1 else "coordinate 1 (original units)")
        if dim == 2:
            ax.set_ylabel("coordinate 2 (original units)")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"OTP-FM {contract['dataset']}: populations at exact observation times")
    fig.tight_layout()
    fig.savefig(output / "marginal_comparison.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    for name in paths:
        ax.plot(contract["times"], [row[name + "_w2"] for row in metrics["scores"]],
                marker="o", label=name, color=colors.get(name))
    for index in contract["holdout_indices"]:
        ax.axvline(contract["times"][index], color="grey", alpha=.2)
    ax.set(xlabel="global time (vertical lines: holdouts)", ylabel="W2 in physical coordinates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "w2_comparison.png", dpi=160)
    plt.close(fig)


def log_comparison(run, result, output, prefix="eval"):
    if run is None:
        return
    import wandb
    from repc.single_cell_tracking import log_files

    scalars = {f"{prefix}/{key}": value for key, value in result.items()
               if isinstance(value, (int, float))}
    for group in ("validation", "source_fit"):
        scalars.update({f"{prefix}/{group}/{key}": value for key, value in result.get(group, {}).items()
                        if isinstance(value, (int, float))})
    for row in result["scores"]:
        scalars.update({f"{prefix}/time_{row['index']}/{key}": value for key, value in row.items()
                        if key.endswith(("_w2", "nats_per_dim", "rmse"))})
    run.log(scalars)
    run.summary.update(scalars)
    images = {f"{prefix}/{name}": wandb.Image(str(output / f"{name}.png"))
              for name in ("marginal_comparison", "w2_comparison")}
    run.log(images)
    log_files(run, prefix, {p.name: p for p in output.iterdir() if p.is_file()
                           and p.suffix in (".json", ".csv", ".png", ".pt")})
