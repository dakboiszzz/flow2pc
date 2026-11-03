"""Shared student construction and recovered real-data HCLT setup."""

import networkx as nx
import numpy as np
import torch

from flow2pc.pc.hclt_pc import TimeDependentHCLTPC


class ClampConsistentHCLTPC(TimeDependentHCLTPC):
    """Compatibility class; HCLT log scale is now unclamped for all datasets."""


def build_student_from_config(model_config):
    """Restore an MFM student without changing its checkpoint parameter layout."""
    if model_config["name"] == "gaussian":
        from flow2pc.pc.gaussian_pc import TimeDependentGaussianPC

        return TimeDependentGaussianPC(
            model_config["num_components"],
            model_config["dim"],
            model_config["time_order"],
            polynomial_zero_center=False,
        )
    if model_config["name"] != "hclt":
        raise ValueError(f"Unknown student: {model_config['name']}")
    return build_student(
        model_config["parents"],
        {
            "num_latents": model_config["num_latents"],
            "time_embedding_dim": model_config["time_embedding_dim"],
            "time_feature_seed": model_config.get("time_feature_seed", 0),
        },
        time_order=model_config["time_order"],
    )


def learn_topology(clouds, bins=32):
    values = torch.cat(clouds).detach().cpu().numpy()
    graph = nx.Graph()
    graph.add_nodes_from(range(values.shape[1]))
    for i in range(values.shape[1]):
        for j in range(i):
            counts = np.histogram2d(values[:, i], values[:, j], bins=bins)[0]
            joint = counts / counts.sum()
            independent = joint.sum(0)[None, :] * joint.sum(1)[:, None]
            mask = joint > 0
            graph.add_edge(
                i,
                j,
                weight=float(
                    (joint[mask] * np.log(joint[mask] / independent[mask])).sum()
                ),
            )
    parents = [-1] * values.shape[1]
    for parent, child in nx.bfs_edges(nx.maximum_spanning_tree(graph), 0):
        parents[child] = parent
    return parents


def build_student(parents, params, *, clamp_derivative=False, time_order=4):
    if set(params) - {"num_latents", "time_embedding_dim", "time_feature_seed"}:
        raise ValueError(
            "Unsupported student settings; the recovered recipe is a dense Gaussian/Fourier HCLT"
        )
    cls = ClampConsistentHCLTPC if clamp_derivative else TimeDependentHCLTPC
    return cls(
        parents,
        time_order=time_order,
        num_latents=params.get("num_latents", 128),
        time_conditioning="fourier",
        time_embedding_dim=params.get("time_embedding_dim", 32),
        time_feature_seed=params.get("time_feature_seed", 0),
        fourier_zero_center=False,
    )


@torch.no_grad()
def initialize_source(student, source, rng):
    if (
        source.ndim != 2
        or source.shape[1] != student.dim
        or len(source) < 2
        or not torch.isfinite(source).all()
    ):
        raise ValueError(
            "Source initialization requires finite training cells of matching dimension"
        )
    indices = torch.randint(len(source), (student.num_latents,), generator=rng)
    means = source[indices].T.to(student.mean_coefficients)
    log_scale = (source.std(0).clamp_min(0.01) * 0.5).log().to(means)
    basis, _ = student._time_basis(torch.tensor(0.0, device=means.device))
    direction = basis / basis.square().sum()
    student.mean_coefficients.copy_(means[..., None] * direction)
    student.log_scale_coefficients.copy_(log_scale[:, None, None] * direction)


@torch.no_grad()
def initialize_snapshot_source(student, source, rng):
    """Empirical initialization for new snapshot recipes, never a density loss."""
    from flow2pc.pc.gaussian_pc import TimeDependentGaussianPC

    if not isinstance(student, TimeDependentGaussianPC):
        return initialize_source(student, source, rng)
    if student.polynomial_zero_center:
        raise ValueError("Empirical sources require a free Gaussian time-zero boundary")
    if source.ndim != 2 or source.shape[1] != student.dim or len(source) < 2 or not torch.isfinite(source).all():
        raise ValueError("Expected finite source training observations")
    indices = torch.randint(len(source), (student.num_components,), generator=rng)
    student.mean_coefficients.zero_()
    student.log_scale_coefficients.zero_()
    student.mean_coefficients[..., 0].copy_(source[indices].to(student.mean_coefficients))
    log_scale = (source.std(0).clamp_min(0.01) * 0.5).log()
    student.log_scale_coefficients[..., 0].copy_(log_scale.to(student.log_scale_coefficients))


def build_optimizer(student, lr, steps):
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    return optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)


def student_config(settings, data, seed):
    structure = settings.get("pc_structure", "hclt")
    config = {
        "name": structure,
        "dim": data.dim,
        "sample_shape": list(data.sample_shape),
        "time_order": int(settings.get("time_order", 4)),
    }
    if structure == "gaussian":
        config.update(
            num_components=int(settings.get("num_components", 64)),
            polynomial_zero_center=False,
        )
    else:
        if settings.get("topology", "learned") == "balanced":
            parents = [-1] + [(i - 1) // 2 for i in range(1, data.dim)]
        else:
            from flow2pc.pc.metric_topology import parents_from_mutual_information

            matrix = torch.cat(
                [data.train_marginals[i].flatten(1) for i in data.observed_indices]
            )
            indices = torch.randperm(len(matrix))[
                : int(settings.get("topology_samples", 2048))
            ]
            matrix = matrix[indices].double()
            centered = matrix - matrix.mean(0)
            norm = centered.square().sum(0).sqrt().clamp_min(1e-12)
            standardized = centered / norm
            correlations = (standardized.T @ standardized).clamp(-1 + 1e-8, 1 - 1e-8)
            information = -0.5 * torch.log1p(-correlations.square())
            information.fill_diagonal_(0)
            parents = (
                [-1]
                if data.dim == 1
                else parents_from_mutual_information(information.float())
            )
        config.update(
            parents=parents,
            num_latents=int(settings.get("num_latents", 64)),
            sum_layer="dense",
            time_conditioning="fourier",
            fourier_zero_center=False,
            time_embedding_dim=int(settings.get("time_embedding_dim", 64)),
            time_feature_seed=seed,
            topology=settings.get("topology", "learned"),
            topology_estimator="balanced_fixed"
            if settings.get("topology") == "balanced"
            else "gaussian_mutual_information",
        )
    return config
