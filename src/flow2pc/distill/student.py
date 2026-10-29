"""Recovered real-data HCLT setup; all student parameters remain trainable."""

import networkx as nx
import numpy as np
import torch

from flow2pc.pc.hclt_pc import TimeDependentHCLTPC


class ClampConsistentHCLTPC(TimeDependentHCLTPC):
    """The existing cell-tracking subclass; its state-dict keys are unchanged."""

    def leaf_parameters(self, time):
        mean, mean_dt, logs, logs_dt = super().leaf_parameters(time)
        return mean, mean_dt, logs, logs_dt * ((logs > -20) & (logs < 20))


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


def build_student(parents, params, *, clamp_derivative=False):
    if set(params) - {"num_latents", "time_embedding_dim", "time_feature_seed"}:
        raise ValueError(
            "Unsupported student settings; the recovered recipe is a dense Gaussian/Fourier HCLT"
        )
    cls = ClampConsistentHCLTPC if clamp_derivative else TimeDependentHCLTPC
    return cls(
        parents,
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


def build_optimizer(student, lr, steps):
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    return optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
