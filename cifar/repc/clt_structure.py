"""Data-learned Chow-Liu tree structure over image patches (HCLT-style).

Ports the structural insight of MonarchCircuits / HCLT (Zhang et al., ICML
2025; Liu & Van den Broeck 2021) into our velocity-distillation framework:
instead of the imposed PD dyadic-rectangle region graph, the circuit
skeleton is a maximum spanning tree of the pairwise mutual-information
graph between patches, estimated from data.

Each tree node v owns its patch leaf; its region is the product of that
leaf with each child subtree. Products are binarized as a left-deep chain
so PatchDeepPC's grouped-einsum machinery applies unchanged, with a single
candidate partition per internal region (the tree IS the structure; the
sum weights only mix channels). Proposition 1 still applies verbatim —
sums with time-independent weights over time-indexed leaves — so the
velocity rules and the single distillation loss are untouched.
"""

import torch


def patch_features(x, H, W, patch, channels):
    """Per-patch mean intensity, (B, n_patch); x is (B, D) flattened CHW."""
    Ph, Pw = H // patch, W // patch
    x = x.reshape(-1, channels, Ph, patch, Pw, patch)
    return x.mean(dim=(1, 3, 5)).reshape(-1, Ph * Pw)


def mutual_information(feats, num_bins=32, sigma=0.5 / 32):
    """Soft-histogram MI between columns of feats, (K, K).

    Same estimator as MonarchCircuits' HCLT construction (Gaussian-kernel
    soft binning), applied to patch summaries instead of raw pixels.
    """
    x = feats - feats.min(dim=0, keepdim=True).values
    x = x / (x.max(dim=0, keepdim=True).values + 1e-8)
    B, K = x.shape
    bins = torch.linspace(0, 1, num_bins, device=x.device)
    xp = torch.exp(-0.5 * (x.unsqueeze(2) - bins.view(1, 1, -1)).pow(2) / sigma**2)
    joint = torch.einsum("bia,bjc->iajc", xp, xp) / B  # (K, nb, K, nb)
    marg = (xp / xp.sum(dim=2, keepdim=True)).mean(dim=0)  # (K, nb)
    joint = joint / joint.sum(dim=(1, 3), keepdim=True)
    h1 = -(marg * torch.log(marg + 1e-4)).sum(dim=1)  # (K,)
    h12 = -(joint * torch.log(joint + 1e-4)).sum(dim=(1, 3))  # (K, K)
    return h1[:, None] + h1[None, :] - h12


def maximum_spanning_tree(weights):
    """Prim's algorithm on a dense (K, K) similarity matrix -> edge list."""
    K = weights.shape[0]
    w = weights.clone().fill_diagonal_(float("-inf"))
    in_tree = [0]
    best = w[0].clone()
    parent = torch.zeros(K, dtype=torch.long)
    edges = []
    for _ in range(K - 1):
        best[torch.tensor(in_tree)] = float("-inf")
        v = int(best.argmax())
        edges.append((int(parent[v]), v))
        in_tree.append(v)
        upd = w[v] > best
        parent[upd] = v
        best = torch.maximum(best, w[v])
    return edges


def tree_center(edges, n):
    """Node minimizing BFS eccentricity (ties -> lowest index)."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    best, best_ecc = 0, n + 1
    for s in range(n):
        depth, seen, frontier, ecc = 0, {s}, [s], 0
        while frontier:
            nxt = [c for f in frontier for c in adj[f] if c not in seen]
            seen.update(nxt)
            frontier = nxt
            depth += 1
            ecc = depth if nxt else ecc
        if ecc < best_ecc:
            best, best_ecc = s, ecc
    return best, adj


def learn_patch_clt(x, H, W, patch, channels, num_bins=32):
    """Images (B, D) -> Chow-Liu tree edge list over the patch grid."""
    feats = patch_features(x.float(), H, W, patch, channels)
    mi = mutual_information(feats, num_bins=num_bins)
    return maximum_spanning_tree(mi.cpu())


def clt_region_graph(edges, n_patch):
    """Convert a patch tree into PatchDeepPC's (regions, splits, root).

    Regions 0..n_patch-1 are the patch leaves. Each internal tree node v
    contributes a left-deep chain: cur = leaf(v); for each child c:
    cur = region(cur x subtree(c)). Fake rectangles (0, scope_size, 0, 1)
    give PatchDeepPC a grouping key whose area equals the scope size, so
    its children-before-parents group ordering holds unchanged.
    """
    root, adj = tree_center(edges, n_patch)
    regions = [(0, 1, 0, 1)] * n_patch
    splits = [[] for _ in range(n_patch)]
    sub = {}  # tree node -> region index of its full subtree
    size = {}

    stack = [(root, -1, False)]
    while stack:
        v, par, expanded = stack.pop()
        children = [c for c in adj[v] if c != par]
        if not expanded:
            stack.append((v, par, True))
            stack.extend((c, v, False) for c in children)
            continue
        cur, sz = v, 1
        for c in children:
            sz += size[c]
            regions.append((0, sz, 0, 1))
            splits.append([(cur, sub[c])])
            cur = len(regions) - 1
        sub[v], size[v] = cur, sz
    return regions, splits, sub[root]
