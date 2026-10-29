"""Velocity-only checkpoint selection, including recovered late tracking checks."""

import torch

from flow2pc.distill.sampling import draw, sample_batch


@torch.no_grad()
def prepare_validation(teacher, dataset, pools, settings, seed, sampler):
    rng = torch.Generator().manual_seed(seed)
    groups = {}
    batches = [
        sample_batch(
            teacher,
            dataset,
            pools,
            settings.get("validation_batch_size", 128),
            rng,
            sampler,
        )
        for _ in range(settings.get("validation_batches", 4))
    ]
    values = torch.cat([b.x_t for b in batches])
    times = torch.cat([b.model_t for b in batches])
    targets = teacher.velocity(values, times)
    groups["global"] = (values, times, targets)
    energy = float(targets.square().mean())
    if dataset.metadata.name == "cell_tracking":
        from torchdiffeq import odeint

        source = draw(pools[0], settings.get("teacher_path_particles", 128), rng).to(
            values
        )
        grid = values.new_tensor(dataset.metadata.model_times)
        paths = odeint(
            teacher.ode_velocity, source, grid, rtol=1e-5, atol=1e-5, method="dopri5"
        )
        indices = torch.nonzero(grid >= 0.8).flatten()
        late_x = paths[indices].reshape(-1, dataset.dim)
        late_t = grid[indices].repeat_interleave(len(source))
        final_t = grid[-1].expand(len(source))
        groups["late"] = (late_x, late_t, teacher.velocity(late_x, late_t))
        groups["final"] = (paths[-1], final_t, teacher.velocity(paths[-1], final_t))
    return groups, max(energy, 1e-8)


@torch.no_grad()
def validate(student, groups, energy, batch_size=128):
    was_training = student.training
    student.eval()
    try:
        scores = {}
        for name, (x, t, target) in groups.items():
            error = 0.0
            for start in range(0, len(x), batch_size):
                end = start + batch_size
                prediction = student.velocity(x[start:end], t[start:end])
                error += float((prediction - target[start:end]).square().sum())
            scores[name] = error / target.numel()
    finally:
        student.train(was_training)
    selection = max(scores.values()) / energy if "late" in groups else scores["global"]
    return {
        **{f"{name}_velocity_mse": score for name, score in scores.items()},
        "selection_score": selection,
        "teacher_velocity_energy": energy,
    }
