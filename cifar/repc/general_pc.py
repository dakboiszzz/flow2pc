"""General (non-image) deep PC: random balanced partition trees.

No spatial assumptions: the region graph is a root sum node mixing R
independent random balanced binary trees over the D variables. Leaves are
per-variable univariate Gaussians with polynomial-in-t parameters (shared
across trees), exactly as in the image circuits. Everything else — the three
velocity rules, the flows-autograd velocity, the single velocity-matching
loss, the p_0 = N(0,I) boundary — is unchanged and dimension-agnostic.
"""

import math

import torch
from torch import nn

LOG2PI = math.log(2.0 * math.pi)


def build_random_trees(D, R, seed=0):
    """R random balanced binary trees over variables 0..D-1.

    Returns: n_regions, splits (list of (childA, childB) for internal
    regions, buffer order: leaves 0..D-1 first, then internals in
    processing order), tree_roots (buffer positions of the R tree roots).
    """
    g = torch.Generator().manual_seed(seed)
    splits = []
    tree_roots = []
    next_pos = D

    def build(indices):
        nonlocal next_pos
        if len(indices) == 1:
            return int(indices[0])
        half = len(indices) // 2
        a = build(indices[:half])
        b = build(indices[half:])
        splits.append((a, b))
        pos = next_pos
        next_pos += 1
        return pos

    for _ in range(R):
        perm = torch.randperm(D, generator=g).tolist()
        tree_roots.append(build(perm))
    return next_pos, splits, tree_roots


class GeneralPC(nn.Module):
    """Root sum over R random trees; S channels per region; univariate
    Gaussian leaves shared across trees."""

    def __init__(self, D, R=4, S=8, order=8, seed=0):
        super().__init__()
        self.D, self.R, self.S, self.order = D, R, S, order
        n_regions, splits, tree_roots = build_random_trees(D, R, seed)
        self.n_leaves = D
        self.n_regions = n_regions
        self.register_buffer(
            "childA", torch.tensor([s[0] for s in splits]), persistent=False
        )
        self.register_buffer(
            "childB", torch.tensor([s[1] for s in splits]), persistent=False
        )
        self.register_buffer("tree_roots", torch.tensor(tree_roots), persistent=False)

        n_internal = len(splits)
        self.sum_logits = nn.Parameter(0.01 * torch.randn(n_internal, S, S * S))
        self.root_logits = nn.Parameter(torch.zeros(R * S))

        self.B = nn.Parameter(0.01 * torch.randn(D, S, order))
        self.C = nn.Parameter(0.01 * torch.randn(D, S, order))

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
        mu, _, logsig, _ = self.leaf_params(t)
        zc = (z[:, :, None] - mu[None]) * torch.exp(-logsig)[None]
        return -0.5 * (zc.pow(2) + LOG2PI) - logsig[None]

    # --------------------------------------------------------------- circuit
    def _forward_circuit(self, llf):
        """Internal regions are in topological order (children always have
        smaller buffer positions), so one sequential pass suffices; the
        per-region einsum batches over the batch dimension."""
        B = llf.shape[0]
        L = llf  # (B, n_leaves, S), grows to (B, n_regions, S)
        n_internal = self.childA.shape[0]
        # process internals in chunks whose children are already available:
        # buffer positions are assigned in build order, which is topological.
        start = 0
        while start < n_internal:
            # find the largest chunk whose children all precede n_leaves+start
            limit = self.n_leaves + start
            end = start
            while (
                end < n_internal
                and self.childA[end] < limit
                and self.childB[end] < limit
            ):
                end += 1
            if end == start:  # must process at least one
                end = start + 1
            la = L[:, self.childA[start:end]]  # (B, m, S)
            lb = L[:, self.childB[start:end]]
            prod = (la[..., :, None] + lb[..., None, :]).reshape(B, end - start, -1)
            logw = torch.log_softmax(self.sum_logits[start:end], dim=-1)
            out = torch.logsumexp(logw[None] + prod[:, :, None, :], dim=-1)
            L = torch.cat([L, out], dim=1)
            start = end
        roots = L[:, self.tree_roots]  # (B, R, S)
        flat = roots.reshape(B, -1)
        return torch.logsumexp(
            torch.log_softmax(self.root_logits, 0)[None] + flat, dim=1
        )

    def log_prob(self, z, t):
        return self._forward_circuit(self.leaf_log_density(z, t))

    # -------------------------------------------------------------- velocity
    def velocity(self, z, t):
        mu, dmu, logsig, dlogsig = self.leaf_params(t)
        with torch.enable_grad():
            llf = self.leaf_log_density(z, t)
            if not llf.requires_grad:
                llf = llf.requires_grad_(True)
            logp = self._forward_circuit(llf)
            (F,) = torch.autograd.grad(logp.sum(), llf, create_graph=self.training)
        u_leaf = dmu[None] + dlogsig[None] * (z[:, :, None] - mu[None])
        return (F * u_leaf).sum(dim=-1)

    # -------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, n, t, seed=0):
        import numpy as np

        rng = np.random.default_rng(seed)
        mu, _, logsig, _ = self.leaf_params(t)
        mu = mu.detach().cpu().numpy()
        sig = torch.exp(logsig).detach().cpu().numpy()
        w = torch.softmax(self.sum_logits, dim=-1).detach().cpu().numpy()
        root_w = torch.softmax(self.root_logits, 0).detach().cpu().numpy()
        cA = self.childA.cpu().numpy()
        cB = self.childB.cpu().numpy()
        S = self.S
        out = torch.empty(n, self.D)
        troots = self.tree_roots.cpu().numpy()
        for si in range(n):
            k = rng.choice(len(root_w), p=root_w)
            tree, ch = divmod(k, S)
            stack = [(int(troots[tree]), ch)]
            while stack:
                p, c = stack.pop()
                if p < self.n_leaves:
                    out[si, p] = float(mu[p, c] + sig[p, c] * rng.standard_normal())
                    continue
                j = p - self.n_leaves
                kk = rng.choice(S * S, p=w[j, c])
                i, jj = divmod(kk, S)
                stack.append((int(cA[j]), i))
                stack.append((int(cB[j]), jj))
        return out
