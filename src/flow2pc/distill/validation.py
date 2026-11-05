"""Velocity-only checkpoint selection, including recovered late tracking checks."""

import math

import torch

from flow2pc.data.base import heldout_indices
from flow2pc.distill.sampling import draw, sample_ali_batch
from flow2pc.utils.experiment_protocols import validation_scope


def temporal_validation_contract(dataset, settings):
    """Describe temporal checkpoint selection without changing dataset splits."""
    if validation_scope(dataset, settings) != "heldout_times":
        return None
    indices = heldout_indices(dataset)
    if not indices:
        raise ValueError("Held-out-time validation was requested but no times are held out")
    if (
        len(set(indices)) != len(indices)
        or any(
            type(i) is not int or not 0 <= i < len(dataset.metadata.model_times)
            for i in indices
        )
        or set(indices) & set(dataset.observed)
    ):
        raise ValueError("Temporal validation indices must be unique held-out times")
    return {
        "name": "heldout_time_velocity_mse_v1",
        "time_indices": indices,
        # Existing loaders store whole held-out populations under this label.
        "population_split": "test",
        "time_coordinates": "model_times",
        "max_cells_per_time": (
            settings.get("validation_batches", 4)
            * settings.get("validation_batch_size", 128)
        ),
        "selection": "unweighted_mean_of_per_time_velocity_MSE",
    }


@torch.no_grad()
def validate_metric(student, teacher, data, settings, device, sample_batch):
    """MFM/MMFM validation follows the dataset policy, not the teacher name."""
    from flow2pc.evaluation.metric_flow import evaluation_seed

    scope = validation_scope(data, settings)
    batch_size = int(settings.get("validation_batch_size", settings.get("batch_size", 128)))
    scores = []
    previous = student.training
    student.eval()
    try:
        with evaluation_seed(1234, device):
            if scope == "heldout_times":
                if not data.heldout_indices:
                    raise ValueError("This dataset has no held-out times for validation")
                for index in data.heldout_indices:
                    pools = getattr(data, "test_marginals", data.validation_marginals)
                    pool = pools[index]
                    if not len(pool):
                        raise ValueError(f"Empty validation population at time {index}")
                    limit = batch_size * int(settings.get("validation_batches", 4))
                    rows = torch.randperm(len(pool))[:limit]
                    values = pool[rows].flatten(1).to(device)
                    times = values.new_full((len(values),), float(data.times[index]))
                    squared_error, elements = 0.0, 0
                    for start in range(0, len(values), batch_size):
                        x, t = values[start:start + batch_size], times[start:start + batch_size]
                        target = teacher.velocity(x, t)
                        difference = student.velocity(x, t) - target
                        squared_error += float(difference.square().sum())
                        elements += difference.numel()
                    scores.append(squared_error / elements)
            else:
                from flow2pc.distill.sampling import velocity_mse

                for index in range(len(data.observed_indices) - 1):
                    for _ in range(int(settings.get("validation_batches", 4))):
                        query = sample_batch(teacher, data, batch_size, device,
                                             validation=True, interval_index=index)
                        scores.append(float(velocity_mse(student, teacher, query)))
        if not scores:
            raise ValueError("No validation times or observed intervals are available")
        score = sum(scores) / len(scores)
        if not math.isfinite(score):
            raise FloatingPointError("Nonfinite validation velocity MSE")
        return score
    finally:
        student.train(previous)


@torch.no_grad()
def prepare_validation(teacher, dataset, pools, settings, seed, sampler):
    rng = torch.Generator().manual_seed(seed)
    groups = {}
    temporal = temporal_validation_contract(dataset, settings)
    required = {0, len(pools) - 1}
    if sampler in ("mmot_middle", "chain_ot"):
        required.update(dataset.observed)
    if all(len(pools[i]) for i in required):
        # Retain the existing observed-time row-validation diagnostic.
        batches = [
            sample_ali_batch(
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
    elif temporal is None or any(len(pools[i]) for i in required):
        raise ValueError("Incomplete row-validation populations for teacher queries")

    if temporal is not None:
        parameter = next(teacher.parameters())
        batch_size = settings.get("validation_batch_size", 128)
        for index in temporal["time_indices"]:
            values = dataset.get_marginal(
                index, temporal["population_split"]
            ).materialize()
            if not len(values):
                raise ValueError(f"Temporal validation population {index} is empty")
            if len(values) > temporal["max_cells_per_time"]:
                rows = torch.randperm(len(values), generator=rng)[
                    :temporal["max_cells_per_time"]
                ]
                values = values[rows]
            values = values.to(parameter)
            times = values.new_full(
                (len(values),), dataset.metadata.model_times[index]
            )
            targets = torch.cat([
                teacher.velocity(
                    values[start:start + batch_size], times[start:start + batch_size]
                )
                for start in range(0, len(values), batch_size)
            ])
            groups[f"heldout_time_{index}"] = (values, times, targets)
        # Match the equal-time reduction used for temporal selection.
        energy = sum(
            float(groups[f"heldout_time_{i}"][2].square().mean())
            for i in temporal["time_indices"]
        ) / len(temporal["time_indices"])
    else:
        energy = float(groups["global"][2].square().mean())
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
    temporal_scores = [
        score for name, score in scores.items() if name.startswith("heldout_time_")
    ]
    if temporal_scores:
        selection = sum(temporal_scores) / len(temporal_scores)
        scores["heldout_time"] = selection
    else:
        selection = (
            max(scores.values()) / energy if "late" in groups else scores["global"]
        )
    return {
        **{f"{name}_velocity_mse": score for name, score in scores.items()},
        "selection_score": selection,
        "teacher_velocity_energy": energy,
    }
