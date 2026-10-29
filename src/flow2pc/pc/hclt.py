import time
from pathlib import Path

import networkx as nx
import torch
import torch.nn.functional as F

HCLT_MNIST_CONFIG = {
    "num_samples": 20_000,
    "num_bins": 64,
    "sigma": 0.02,
    "chunk_size": 32,
}

ICLR2022_HCLT_MNIST_CONFIG = {
    "num_samples": 60_000,
    "num_structure_categories": 8,
    "num_data_categories": 256,
    "sample_chunk_size": 1_024,
}


def collect_hclt_data(loader, num_samples: int) -> torch.Tensor:
    """Collect the shuffled training subset used to learn an HCLT backbone."""
    batches = []
    collected = 0
    for images, _ in loader:
        take = min(len(images), num_samples - collected)
        batches.append(images[:take].flatten(1))
        collected += take
        if collected == num_samples:
            break
    if collected != num_samples:
        raise ValueError(f"requested {num_samples} HCLT samples, found {collected}")
    return torch.cat(batches)


def learn_hclt_parents(
    data: torch.Tensor,
    num_bins: int,
    sigma: float,
    chunk_size: int,
) -> list[int]:
    """Learn the PyJuice/MonarchCircuits Chow--Liu image backbone."""
    # Import lazily: PyJuice has a substantial import cost and is only needed once
    # while constructing a previously unseen topology.
    from pyjuice.structures.hclt import chow_liu_tree, mutual_information_chunked

    started_at = time.perf_counter()
    print(
        "learning official HCLT topology "
        f"from x={len(data)}, bins={num_bins}, sigma={sigma}, chunk={chunk_size}",
        flush=True,
    )
    mutual_information = mutual_information_chunked(
        data,
        data,
        num_bins=num_bins,
        sigma=sigma,
        chunk_size=chunk_size,
    )
    tree = chow_liu_tree(mutual_information.numpy())
    root = nx.center(tree)[0]
    directed = nx.bfs_tree(tree, root)
    parents = [-1] * data.shape[1]
    for parent, child in directed.edges:
        parents[child] = parent
    print(
        f"learned HCLT topology in {time.perf_counter() - started_at:.1f}s; "
        f"root={root}",
        flush=True,
    )
    return parents


@torch.no_grad()
def categorical_mutual_information(
    data: torch.Tensor,
    num_categories: int,
    sample_chunk_size: int = 1_024,
    variable_chunk_size: int = 32,
) -> torch.Tensor:
    """Compute the empirical pairwise MI matrix for categorical variables.

    This is the structure-learning estimator specified in Appendix B.1 of
    Liu et al. (ICLR 2022).  The joint contingency table is accumulated with
    matrix products, then converted to mutual information in small variable
    blocks to keep peak memory bounded.
    """
    if data.ndim != 2:
        raise ValueError("categorical data must have shape [samples, variables]")
    if len(data) < 1 or data.shape[1] < 2:
        raise ValueError("categorical MI needs samples and at least two variables")
    if num_categories < 2:
        raise ValueError("num_categories must be at least two")
    if sample_chunk_size < 1 or variable_chunk_size < 1:
        raise ValueError("MI chunk sizes must be positive")

    categories = data.long()
    if categories.min().item() < 0 or categories.max().item() >= num_categories:
        raise ValueError("categorical values fall outside the configured vocabulary")

    num_samples, num_variables = categories.shape
    flattened_dim = num_variables * num_categories
    joint_counts = torch.zeros(
        flattened_dim,
        flattened_dim,
        device=categories.device,
        dtype=torch.float32,
    )
    marginal_counts = torch.zeros(
        num_variables,
        num_categories,
        device=categories.device,
        dtype=torch.float32,
    )
    for batch in categories.split(sample_chunk_size):
        indicators = F.one_hot(batch, num_classes=num_categories).to(torch.float32)
        marginal_counts.add_(indicators.sum(dim=0))
        flat_indicators = indicators.flatten(1)
        joint_counts.addmm_(flat_indicators.T, flat_indicators)

    joint_counts = joint_counts.reshape(
        num_variables,
        num_categories,
        num_variables,
        num_categories,
    )
    marginal_probabilities = marginal_counts / num_samples
    mutual_information = torch.empty(
        num_variables,
        num_variables,
        device=categories.device,
        dtype=torch.float32,
    )
    for start in range(0, num_variables, variable_chunk_size):
        stop = min(start + variable_chunk_size, num_variables)
        joint = joint_counts[start:stop] / num_samples
        denominator = (
            marginal_probabilities[start:stop, :, None, None]
            * marginal_probabilities[None, None, :, :]
        )
        positive = joint > 0.0
        ratio = torch.where(
            positive,
            joint / denominator.clamp_min(torch.finfo(joint.dtype).tiny),
            torch.ones_like(joint),
        )
        terms = torch.where(positive, joint * torch.log2(ratio), 0.0)
        mutual_information[start:stop] = terms.sum(dim=(1, 3))
    mutual_information.fill_diagonal_(0.0)
    return mutual_information.cpu()


def parents_from_mutual_information(mutual_information: torch.Tensor) -> list[int]:
    """Build and center-root the maximum-weight Chow--Liu tree."""
    if mutual_information.ndim != 2 or (
        mutual_information.shape[0] != mutual_information.shape[1]
    ):
        raise ValueError("mutual information must be a square matrix")
    if not torch.isfinite(mutual_information).all():
        raise ValueError("mutual information must contain only finite values")
    weights = mutual_information.numpy()
    complete_graph = nx.Graph()
    complete_graph.add_nodes_from(range(len(weights)))
    complete_graph.add_weighted_edges_from(
        (left, right, float(weights[left, right]))
        for left in range(len(weights))
        for right in range(left + 1, len(weights))
    )
    tree = nx.maximum_spanning_tree(complete_graph)
    if not nx.is_tree(tree):
        raise RuntimeError("mutual-information graph did not produce one tree")
    root = min(nx.center(tree))
    directed = nx.bfs_tree(tree, root, sort_neighbors=sorted)
    parents = [-1] * len(tree)
    for parent, child in directed.edges:
        parents[child] = parent
    return parents


def learn_iclr2022_hclt_parents(
    raw_data: torch.Tensor,
    sample_chunk_size: int = 1_024,
) -> list[int]:
    """Learn the MNIST HCLT backbone exactly as specified at ICLR 2022.

    Only the three most-significant pixel bits are used for mutual
    information.  Parameter learning still uses all eight pixel bits.
    """
    if raw_data.dtype != torch.uint8:
        raise ValueError("ICLR 2022 MNIST topology learning expects uint8 pixels")
    truncated = torch.bitwise_right_shift(raw_data, 5)
    mutual_information = categorical_mutual_information(
        truncated,
        num_categories=8,
        sample_chunk_size=sample_chunk_size,
    )
    return parents_from_mutual_information(mutual_information)


def load_or_learn_iclr2022_hclt_topology(
    raw_data: torch.Tensor,
    path: Path,
    device: torch.device,
    sample_chunk_size: int = 1_024,
) -> tuple[list[int], dict]:
    """Load or learn the paper's full-data, three-MSB MNIST topology."""
    config = {
        **ICLR2022_HCLT_MNIST_CONFIG,
        "num_samples": len(raw_data),
        "dim": raw_data.shape[1],
        "estimator": "empirical-pairwise-mi-log2",
        "root": "minimum-index-tree-center",
    }
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["config"] != config:
            raise ValueError(f"cached ICLR 2022 topology settings in {path} differ")
        print(f"loaded ICLR 2022 HCLT topology from {path.resolve()}", flush=True)
        return payload["parents"], config

    started_at = time.perf_counter()
    print(
        "learning ICLR 2022 HCLT topology from all "
        f"{len(raw_data)} MNIST samples using three MSBs",
        flush=True,
    )
    parents = learn_iclr2022_hclt_parents(
        raw_data.to(device),
        sample_chunk_size=sample_chunk_size,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"parents": parents, "config": config}, path)
    print(
        f"learned and cached ICLR 2022 topology in "
        f"{time.perf_counter() - started_at:.1f}s at {path.resolve()}",
        flush=True,
    )
    return parents, config


def load_or_learn_hclt_topology(
    loader,
    path: Path,
    device: torch.device,
    config: dict,
    seed: int,
) -> tuple[list[int], dict]:
    expected = {**config, "seed": seed, "dim": 28 * 28}
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["config"] != expected:
            raise ValueError(
                f"cached topology settings in {path} do not match this run"
            )
        print(f"loaded HCLT topology from {path.resolve()}", flush=True)
        return payload["parents"], expected

    data = collect_hclt_data(loader, config["num_samples"]).to(device)
    parents = learn_hclt_parents(
        data,
        num_bins=config["num_bins"],
        sigma=config["sigma"],
        chunk_size=config["chunk_size"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"parents": parents, "config": expected}, path)
    print(f"cached HCLT topology at {path.resolve()}", flush=True)
    return parents, expected
