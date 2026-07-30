"""Deep time-indexed PC with structure learning by overparameterization.

Region graph: Poon-Domingos style over the HxW image. Every region is a
rectangle reachable by recursive midpoint splits; a region with h>1 AND w>1
contains BOTH candidate partitions (horizontal and vertical split) under its
sum nodes. The DAG is heavily shared (a rectangle reached via different split
orders is the same node). Sum weights are time-INDEPENDENT (Proposition 1);
learning them IS structure learning — pruning near-zero weights afterwards
reveals the learned circuit.

Velocity: for any smooth PC the recursive leaf/product/sum rules collapse to

    u_root(z)[d] = sum_i F_{d,i}(z) * u_leaf_{d,i}(z_d),
    F_{d,i} = d log p_root / d log p_leaf_{d,i}   (PC "flows"),

because chain rule composes sum-node responsibilities exactly like the
recursion and product siblings cancel. F is one autograd call through the
circuit forward, so density and velocity share the same code path.

Leaves: per-pixel univariate Gaussians, S channels, polynomial-in-t params
with no constant term => p_0^PC = N(0, I) exactly for any weights.
"""

import math

import torch
from torch import nn

LOG2PI = math.log(2.0 * math.pi)


# --------------------------------------------------------------------------
# region graph
# --------------------------------------------------------------------------
def _cut_positions(a, b, lattice):
    """Split positions for interval [a, b): midpoint, plus lattice points
    around it when the interval is large (rich Poon-Domingos graphs)."""
    mid = (a + b) // 2
    if lattice <= 0 or b - a <= lattice:
        return [mid]
    pos = {mid}
    for p in (mid - lattice, mid + lattice):
        if a < p < b:
            pos.add(p)
    return sorted(pos)


def build_region_graph(H, W, lattice=0):
    """Returns:
    regions: list of (r0, r1, c0, c1), topologically ordered leaves-first
             (position in this list = position in the log-density buffer)
    splits:  for each region, list of candidate (childA_pos, childB_pos)
    leaf_pixel: for each leaf position, the flat pixel index d = r0*W + c0
    lattice > 0 enables multiple split positions per axis (richer DAG).
    """
    key2id = {}
    splits_by_key = {}

    def get(r0, r1, c0, c1):
        key = (r0, r1, c0, c1)
        if key in key2id:
            return key
        key2id[key] = None  # mark visited before recursing (DAG sharing)
        cands = []
        if r1 - r0 > 1:  # horizontal cuts
            for rm in _cut_positions(r0, r1, lattice):
                cands.append((get(r0, rm, c0, c1), get(rm, r1, c0, c1)))
        if c1 - c0 > 1:  # vertical cuts
            for cm in _cut_positions(c0, c1, lattice):
                cands.append((get(r0, r1, c0, cm), get(r0, r1, cm, c1)))
        splits_by_key[key] = cands
        return key

    root_key = get(0, H, 0, W)

    # order: leaves (pixels) first in pixel order, then by area ascending
    keys = list(key2id.keys())
    leaves = [k for k in keys if not splits_by_key[k]]
    leaves.sort(key=lambda k: k[0] * W + k[2])
    internal = [k for k in keys if splits_by_key[k]]
    internal.sort(key=lambda k: ((k[1] - k[0]) * (k[3] - k[2]), k))
    ordered = leaves + internal
    pos = {k: i for i, k in enumerate(ordered)}

    regions = ordered
    splits = [[(pos[a], pos[b]) for a, b in splits_by_key[k]] for k in ordered]
    leaf_pixel = [k[0] * W + k[2] for k in leaves]
    assert leaf_pixel == list(range(H * W)), "leaves must be in pixel order"
    return regions, splits, pos[root_key]


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class DeepPC(nn.Module):
    def __init__(self, H=28, W=28, S=8, order=4):
        super().__init__()
        self.H, self.W, self.D, self.S, self.order = H, W, H * W, S, order
        regions, splits, self.root_pos = build_region_graph(H, W)
        self.n_leaves = self.D
        self.n_regions = len(regions)
        self.regions = regions

        # group internal regions by shape -> batched einsums
        groups = {}
        for i in range(self.n_leaves, self.n_regions):
            r0, r1, c0, c1 = regions[i]
            groups.setdefault((r1 - r0, c1 - c0), []).append(i)
        # Buffer positions: leaves occupy 0..D-1, then internal groups are
        # appended in (area, shape) processing order. Child indices from
        # build_region_graph are remapped to these buffer positions (children
        # have strictly smaller area, so they are always remapped first).
        self.groups = []  # list of dicts, processed in area order
        remap = torch.full((self.n_regions,), -1, dtype=torch.long)
        remap[: self.n_leaves] = torch.arange(self.n_leaves)
        offset = self.n_leaves
        for shape in sorted(groups, key=lambda s: (s[0] * s[1], s)):
            idx = groups[shape]
            n_cand = len(splits[idx[0]])
            childA = remap[torch.tensor([[c[0] for c in splits[i]] for i in idx])]
            childB = remap[torch.tensor([[c[1] for c in splits[i]] for i in idx])]
            assert (childA >= 0).all() and (childB >= 0).all()
            bufpos = torch.arange(offset, offset + len(idx))
            remap[torch.tensor(idx)] = bufpos
            offset += len(idx)
            self.groups.append(
                {
                    "shape": shape,
                    "pos": torch.tensor(idx),  # original ids (for region lookup)
                    "bufpos": bufpos,  # positions in the forward buffer
                    "childA": childA,  # (n, n_cand), buffer positions
                    "childB": childB,
                    "n_cand": n_cand,
                }
            )
        self.root_pos = int(remap[self.root_pos])
        for gi, g in enumerate(self.groups):
            self.register_buffer(f"_cA{gi}", g["childA"], persistent=False)
            self.register_buffer(f"_cB{gi}", g["childB"], persistent=False)

        # sum-node weights: per internal region, S output channels over
        # n_cand * S * S product children (time-independent!)
        self.sum_logits = nn.ParameterList(
            [
                nn.Parameter(0.01 * torch.randn(len(g["pos"]), S, g["n_cand"] * S * S))
                for g in self.groups
            ]
        )
        self.root_logits = nn.Parameter(torch.zeros(S))

        # leaf polynomial params: (D, S, P), no constant term
        self.B = nn.Parameter(0.01 * torch.randn(self.D, S, order))
        self.C = nn.Parameter(0.01 * torch.randn(self.D, S, order))

    # ---------------------------------------------------------------- leaves
    def _basis(self, t):
        p = torch.arange(1, self.order + 1, device=t.device, dtype=t.dtype)
        return t**p, p * t ** (p - 1)

    def leaf_params(self, t):
        phi, dphi = self._basis(t)
        mu = torch.einsum("dsp,p->ds", self.B, phi)
        dmu = torch.einsum("dsp,p->ds", self.B, dphi)
        logsig = torch.einsum("dsp,p->ds", self.C, phi)
        dlogsig = torch.einsum("dsp,p->ds", self.C, dphi)
        return mu, dmu, logsig, dlogsig

    def leaf_log_density(self, z, t):
        """z: (B, D) -> llf (B, D, S)"""
        mu, _, logsig, _ = self.leaf_params(t)
        zc = (z[:, :, None] - mu[None]) * torch.exp(-logsig)[None]
        return -0.5 * (zc.pow(2) + LOG2PI) - logsig[None]

    # --------------------------------------------------------------- circuit
    def _forward_circuit(self, llf):
        """llf: (B, D, S) leaf log densities -> log p_root (B,)"""
        B = llf.shape[0]
        L = llf.permute(0, 1, 2).reshape(B, self.n_leaves, self.S)
        for gi, _g in enumerate(self.groups):
            cA = getattr(self, f"_cA{gi}")  # (n, n_cand)
            cB = getattr(self, f"_cB{gi}")
            n, n_cand = cA.shape
            la = L[:, cA]  # (B, n, n_cand, S)
            lb = L[:, cB]
            prod = la[..., :, None] + lb[..., None, :]  # (B, n, n_cand, S, S)
            prod = prod.reshape(B, n, n_cand * self.S * self.S)
            logw = torch.log_softmax(self.sum_logits[gi], dim=-1)  # (n, S, cSS)
            out = torch.logsumexp(logw[None] + prod[:, :, None, :], dim=-1)  # (B,n,S)
            L = torch.cat([L, out], dim=1)
        root = L[:, self.root_pos]  # (B, S)
        return torch.logsumexp(
            torch.log_softmax(self.root_logits, 0)[None] + root, dim=1
        )

    def log_prob(self, z, t):
        return self._forward_circuit(self.leaf_log_density(z, t))

    # -------------------------------------------------------------- velocity
    def velocity(self, z, t):
        """u_root(z) via leaf flows F = dlogp/dllf. z: (B, D)."""
        mu, dmu, logsig, dlogsig = self.leaf_params(t)
        with torch.enable_grad():
            llf = self.leaf_log_density(z, t)
            if not llf.requires_grad:
                llf = llf.requires_grad_(True)
            logp = self._forward_circuit(llf)
            (F,) = torch.autograd.grad(logp.sum(), llf, create_graph=self.training)
        u_leaf = dmu[None] + dlogsig[None] * (z[:, :, None] - mu[None])  # (B,D,S)
        return (F * u_leaf).sum(dim=-1)

    # -------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, n, t, generator=None):
        """Top-down ancestral sampling at time t."""
        device = self.B.device
        mu, _, logsig, _ = self.leaf_params(t)
        sig = logsig.exp()
        # per-group softmaxed weights (kept on device)
        logw = [torch.log_softmax(sl, dim=-1) for sl in self.sum_logits]
        pos2group = {}
        for gi, g in enumerate(self.groups):
            for j, p in enumerate(g["bufpos"].tolist()):
                pos2group[p] = (gi, j)
        out = torch.empty(n, self.D, device=device)
        root_w = torch.softmax(self.root_logits, 0)
        for s_idx in range(n):
            ch = torch.multinomial(root_w, 1, generator=generator).item()
            stack = [(self.root_pos, ch)]
            while stack:
                p, c = stack.pop()
                if p < self.n_leaves:
                    eps = torch.randn(1, device=device, generator=generator)
                    out[s_idx, p] = mu[p, c] + sig[p, c] * eps
                    continue
                gi, j = pos2group[p]
                k = torch.multinomial(
                    logw[gi][j, c].exp(), 1, generator=generator
                ).item()
                S = self.S
                cand, rem = divmod(k, S * S)
                i, jj = divmod(rem, S)
                g = self.groups[gi]
                stack.append((int(g["childA"][j, cand]), i))
                stack.append((int(g["childB"][j, cand]), jj))
        return out

    # ------------------------------------------------------------- structure
    @torch.no_grad()
    def structure_stats(self, threshold=1e-3):
        """Learned-structure summary: per-region H/V preference and sparsity."""
        rows = []
        total_edges, kept_edges = 0, 0
        for gi, g in enumerate(self.groups):
            w = torch.softmax(self.sum_logits[gi], dim=-1)  # (n, S, cand*S*S)
            n, S, _ = w.shape
            total_edges += w.numel()
            kept_edges += (w > threshold).sum().item()
            if g["n_cand"] == 2:
                wc = w.reshape(n, S, 2, S * S).sum(-1).mean(1)  # (n, 2) H vs V mass
                for j, p in enumerate(g["pos"].tolist()):
                    rows.append((self.regions[p], wc[j, 0].item(), wc[j, 1].item()))
        return {
            "sparsity": 1.0 - kept_edges / total_edges,
            "kept_edges": kept_edges,
            "total_edges": total_edges,
            "hv_rows": rows,
        }
