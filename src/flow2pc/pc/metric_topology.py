"""Maximum-weight Chow–Liu topology used by the MFM student."""

import networkx as nx
import torch


def parents_from_mutual_information(mutual_information: torch.Tensor) -> list[int]:
    if mutual_information.ndim != 2 or mutual_information.shape[0] != mutual_information.shape[1]:
        raise ValueError("mutual information must be a square matrix")
    if not torch.isfinite(mutual_information).all():
        raise ValueError("mutual information must contain only finite values")
    weights = mutual_information.numpy()
    graph = nx.Graph()
    graph.add_nodes_from(range(len(weights)))
    graph.add_weighted_edges_from(
        (left, right, float(weights[left, right]))
        for left in range(len(weights))
        for right in range(left + 1, len(weights))
    )
    tree = nx.maximum_spanning_tree(graph)
    if not nx.is_tree(tree):
        raise RuntimeError("mutual-information graph did not produce one tree")
    root = min(nx.center(tree))
    directed = nx.bfs_tree(tree, root, sort_neighbors=sorted)
    parents = [-1] * len(tree)
    for parent, child in directed.edges:
        parents[child] = parent
    return parents
