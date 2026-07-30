"""Deep PC with MULTIVARIATE Gaussian leaves over image patches.

Implements the slides' leaf rule in its full form:

    u_j(z) = mu_j'(t) + L_j'(t) L_j(t)^{-1} (z - mu_j(t)),   L_j L_j^T = Sigma_j

with a time-dependent Cholesky factor per leaf, so pixels inside a patch are
correlated (diagonal leaves = the special case used before). Boundary
conditions preserved exactly:

    mu(0) = 0,  L(0) = I   =>  p_0^PC = N(0, I)

via  mu(t) = poly(t) with no constant term,
     L(t)  = diag(exp(t * d(t))) + strictlower(t * n(t)),  d, n polynomials.

The circuit above the leaves is the same PD region graph, built over the
PATCH grid (e.g. 7x7 patches of 4x4 pixels for MNIST). Root velocity again
via PC flows: u_root over a patch = sum_s F_{p,s} u_leaf_{p,s}.
"""

import math

import torch
from torch import nn

from repc.deep_pc import build_region_graph

LOG2PI = math.log(2.0 * math.pi)


class PatchDeepPC(nn.Module):
    def __init__(
        self,
        H=28,
        W=28,
        patch=4,
        S=8,
        order=8,
        channels=1,
        lattice=0,
        time_mlp=False,
        factor_h=0,
        structure="pd",
        clt_edges=None,
    ):
        super().__init__()
        assert H % patch == 0 and W % patch == 0
        self.H, self.W, self.patch, self.S, self.order = H, W, patch, S, order
        self.channels = channels
        self.Ph, self.Pw = H // patch, W // patch
        self.n_patch = self.Ph * self.Pw
        self.pd = patch * patch * channels  # dims per patch (all channels)
        self.D = H * W * channels

        # pixel indices of each patch, (n_patch, pd)
        rows = torch.arange(H).reshape(self.Ph, patch)
        cols = torch.arange(W).reshape(self.Pw, patch)
        idx = []
        for pr in range(self.Ph):
            for pc_ in range(self.Pw):
                rr = rows[pr][:, None] * W + cols[pc_][None, :]  # (patch, patch)
                per_c = [c * H * W + rr.reshape(-1) for c in range(channels)]
                idx.append(torch.cat(per_c))
        self.register_buffer("patch_idx", torch.stack(idx), persistent=False)

        # region graph over the patch grid; leaves = patches
        if structure == "clt":
            from repc.clt_structure import clt_region_graph

            assert clt_edges is not None, (
                "structure=clt needs clt_edges (learned from data)"
            )
            regions, splits, root = clt_region_graph(clt_edges, self.n_patch)
        else:
            regions, splits, root = build_region_graph(
                self.Ph, self.Pw, lattice=lattice
            )
        self.n_leaves = self.n_patch
        self.n_regions = len(regions)
        self.regions = regions
        groups = {}
        for i in range(self.n_leaves, self.n_regions):
            r0, r1, c0, c1 = regions[i]
            groups.setdefault((r1 - r0, c1 - c0), []).append(i)
        self.groups = []
        remap = torch.full((self.n_regions,), -1, dtype=torch.long)
        remap[: self.n_leaves] = torch.arange(self.n_leaves)
        offset = self.n_leaves
        for shape in sorted(groups, key=lambda s: (s[0] * s[1], s)):
            idx_ = groups[shape]
            n_cand = len(splits[idx_[0]])
            childA = remap[torch.tensor([[c[0] for c in splits[i]] for i in idx_])]
            childB = remap[torch.tensor([[c[1] for c in splits[i]] for i in idx_])]
            assert (childA >= 0).all() and (childB >= 0).all()
            remap[torch.tensor(idx_)] = torch.arange(offset, offset + len(idx_))
            offset += len(idx_)
            self.groups.append(
                {
                    "pos": torch.tensor(idx_),
                    "childA": childA,
                    "childB": childB,
                    "n_cand": n_cand,
                }
            )
        self.root_pos = int(remap[root])
        for gi, g in enumerate(self.groups):
            self.register_buffer(f"_cA{gi}", g["childA"], persistent=False)
            self.register_buffer(f"_cB{gi}", g["childB"], persistent=False)

        # sum blocks: dense (factor_h=0) or Monarch-style two-stage
        # structured mixing (Zhang et al. 2025): per (candidate, left-channel)
        # block of S right-channels -> h hidden sums, then dense S x (cand*S*h).
        # Params: 2*cand*S^2*h vs dense cand*S^3 — S/(2h)x cheaper.
        self.factor_h = factor_h
        if factor_h > 0:
            self.sum1_logits = nn.ParameterList(
                [
                    nn.Parameter(
                        0.01 * torch.randn(len(g["pos"]), g["n_cand"] * S, factor_h, S)
                    )
                    for g in self.groups
                ]
            )
            self.sum_logits = nn.ParameterList(
                [
                    nn.Parameter(
                        0.01 * torch.randn(len(g["pos"]), S, g["n_cand"] * S * factor_h)
                    )
                    for g in self.groups
                ]
            )
        else:
            self.sum_logits = nn.ParameterList(
                [
                    nn.Parameter(
                        0.01 * torch.randn(len(g["pos"]), S, g["n_cand"] * S * S)
                    )
                    for g in self.groups
                ]
            )
        self.root_logits = nn.Parameter(torch.zeros(S))

        # leaf params: mu poly (n_patch, S, pd, P); Cholesky diag/lower polys
        P = order
        self.Bmu = nn.Parameter(0.01 * torch.randn(self.n_patch, S, self.pd, P))
        self.Cdiag = nn.Parameter(0.01 * torch.randn(self.n_patch, S, self.pd, P))
        n_low = self.pd * (self.pd - 1) // 2
        self.Clow = nn.Parameter(0.01 * torch.randn(self.n_patch, S, n_low, P))
        self.time_mlp = time_mlp
        if time_mlp:
            # learned time basis: h(t) = t * f(emb(t)), exact h(0)=0 boundary
            self.tnet = nn.Sequential(
                nn.Linear(16, 128), nn.SiLU(), nn.Linear(128, order)
            )
        tril = torch.tril_indices(self.pd, self.pd, offset=-1)
        self.register_buffer("tril_r", tril[0], persistent=False)
        self.register_buffer("tril_c", tril[1], persistent=False)

    # ---------------------------------------------------------------- leaves
    def _emb(self, t):
        k = torch.arange(8, device=t.device, dtype=t.dtype)
        ang = t * torch.pi * (2.0**k)
        return torch.cat([ang.sin(), ang.cos()])

    def _basis(self, t):
        if not self.time_mlp:
            p = torch.arange(1, self.order + 1, device=t.device, dtype=t.dtype)
            return t**p, p * t ** (p - 1)

        # h(t) = t * f(t); dh = f + t f'  via forward-mode jvp (exact)
        def h(tt):
            return tt * self.tnet(self._emb(tt))

        val, dval = torch.func.jvp(h, (t,), (torch.ones_like(t),))
        return val, dval

    def leaf_chol(self, t):
        """Returns mu, dmu (n_patch,S,pd) and L, dL (n_patch,S,pd,pd)."""
        phi, dphi = self._basis(t)
        mu = torch.einsum("psdk,k->psd", self.Bmu, phi)
        dmu = torch.einsum("psdk,k->psd", self.Bmu, dphi)
        d_poly = torch.einsum("psdk,k->psd", self.Cdiag, phi)  # t*d(t) net
        dd_poly = torch.einsum("psdk,k->psd", self.Cdiag, dphi)
        low = torch.einsum("pslk,k->psl", self.Clow, phi)
        dlow = torch.einsum("pslk,k->psl", self.Clow, dphi)
        # bounded parameterization (numerics): log-diag in [-A, A], off-diag
        # in [-A2, A2] via soft clamp; derivatives use the same chain rule so
        # (density, velocity) remain an exact continuity pair. tanh(0)=0
        # keeps L(0)=I intact.
        A, A2 = 6.0, 4.0
        th_d = torch.tanh(d_poly / A)
        d_b, dd_b = A * th_d, dd_poly * (1.0 - th_d.pow(2))
        th_l = torch.tanh(low / A2)
        low, dlow = A2 * th_l, dlow * (1.0 - th_l.pow(2))
        diag = torch.exp(d_b)  # L_ii = exp(clamped t*d(t)), =1 at t=0
        ddiag = diag * dd_b
        n, S, pd = mu.shape
        L = torch.zeros(n, S, pd, pd, device=mu.device, dtype=mu.dtype)
        dL = torch.zeros_like(L)
        ar = torch.arange(pd, device=mu.device)
        L[:, :, ar, ar] = diag
        dL[:, :, ar, ar] = ddiag
        L[:, :, self.tril_r, self.tril_c] = low
        dL[:, :, self.tril_r, self.tril_c] = dlow
        return mu, dmu, L, dL

    def leaf_log_density_and_eps(self, zp, mu, L):
        """zp: (B, n_patch, pd). Returns llf (B,n_patch,S) and
        eps = L^{-1}(z-mu) (B,n_patch,S,pd)."""
        zc = zp[:, :, None, :] - mu[None]  # (B,n,S,pd)
        eps = torch.linalg.solve_triangular(
            L[None].expand(zc.shape[0], -1, -1, -1, -1).reshape(-1, self.pd, self.pd),
            zc.reshape(-1, self.pd, 1),
            upper=False,
        ).reshape(zc.shape)
        ar = torch.arange(self.pd, device=zp.device)
        logdet = torch.log(L[:, :, ar, ar]).sum(-1)  # (n,S)
        llf = -0.5 * (eps.pow(2).sum(-1) + self.pd * LOG2PI) - logdet[None]
        return llf, eps

    # --------------------------------------------------------------- circuit
    def _forward_circuit(self, llf):
        B = llf.shape[0]
        L = llf
        for gi in range(len(self.groups)):
            cA = getattr(self, f"_cA{gi}")
            cB = getattr(self, f"_cB{gi}")
            n, n_cand = cA.shape
            la, lb = L[:, cA], L[:, cB]
            if self.factor_h > 0:
                # stage 1: per (cand, i) block over j: (B,n,cand*S,S) -> hidden (B,n,cand*S,h)
                prod = la[..., :, None] + lb[..., None, :]  # (B,n,cand,S,S)
                blk = prod.reshape(B, n, -1, 1, self.S)  # (B,n,cand*S,1,S)
                w1 = torch.log_softmax(self.sum1_logits[gi], dim=-1)  # (n,cand*S,h,S)
                hid = torch.logsumexp(w1[None] + blk, dim=-1)  # (B,n,cand*S,h)
                hid = hid.reshape(B, n, -1)  # (B,n,cand*S*h)
                logw = torch.log_softmax(self.sum_logits[gi], dim=-1)
                out = torch.logsumexp(logw[None] + hid[:, :, None, :], dim=-1)
            else:
                prod = (la[..., :, None] + lb[..., None, :]).reshape(B, n, -1)
                logw = torch.log_softmax(self.sum_logits[gi], dim=-1)
                out = torch.logsumexp(logw[None] + prod[:, :, None, :], dim=-1)
            L = torch.cat([L, out], dim=1)
        root = L[:, self.root_pos]
        return torch.logsumexp(
            torch.log_softmax(self.root_logits, 0)[None] + root, dim=1
        )

    def _to_patches(self, z):
        return z[:, self.patch_idx]  # (B, n_patch, pd)

    def log_prob(self, z, t):
        mu, _, L, _ = self.leaf_chol(t)
        llf, _ = self.leaf_log_density_and_eps(self._to_patches(z), mu, L)
        return self._forward_circuit(llf)

    def patch_marginals(self, z, t):
        """Exact log p(x_patch) for every patch, (B, n_patch).

        Marginalizing a scope in a smooth decomposable circuit = setting its
        leaf log-densities to 0 (probability one). One circuit pass per patch.
        """
        mu, _, L, _ = self.leaf_chol(t)
        llf, _ = self.leaf_log_density_and_eps(self._to_patches(z), mu, L)
        out = []
        for p in range(self.n_patch):
            masked = torch.zeros_like(llf)
            masked[:, p] = llf[:, p]
            out.append(self._forward_circuit(masked))
        return torch.stack(out, dim=1)

    # -------------------------------------------------------------- velocity
    def velocity(self, z, t):
        """u = sum_s F_{p,s} [dmu + dL L^{-1} (z-mu)] per patch, slide rule."""
        mu, dmu, L, dL = self.leaf_chol(t)
        zp = self._to_patches(z)
        with torch.enable_grad():
            llf, eps = self.leaf_log_density_and_eps(zp, mu, L)
            if not llf.requires_grad:
                llf = llf.requires_grad_(True)
            logp = self._forward_circuit(llf)
            (F,) = torch.autograd.grad(logp.sum(), llf, create_graph=self.training)
        # u_leaf = dmu + dL @ eps  (eps = L^{-1}(z-mu)), (B,n,S,pd)
        u_leaf = dmu[None] + torch.einsum(
            "psij,bpsj->bpsi", dL, eps.detach() if False else eps
        )
        u_patch = (F[..., None] * u_leaf).sum(dim=2)  # (B, n_patch, pd)
        u = torch.zeros_like(z)
        u.scatter_(
            1,
            self.patch_idx.reshape(1, -1).expand(z.shape[0], -1),
            u_patch.reshape(z.shape[0], -1),
        )
        return u

    # -------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, n, t, seed=0):
        import numpy as np

        rng = np.random.default_rng(seed)
        mu, _, L, _ = self.leaf_chol(t)
        mu = mu.detach().cpu().numpy()
        Lnp = L.detach().cpu().numpy()
        wl = [
            torch.softmax(sl, dim=-1).detach().cpu().numpy() for sl in self.sum_logits
        ]
        root_w = torch.softmax(self.root_logits, 0).detach().cpu().numpy()
        pos2group = {}
        childA, childB = [], []
        for gi, g in enumerate(self.groups):
            base = self.n_leaves + sum(len(gg["pos"]) for gg in self.groups[:gi])
            for j in range(len(g["pos"])):
                pos2group[base + j] = (gi, j)
            childA.append(g["childA"].numpy())
            childB.append(g["childB"].numpy())
        S = self.S
        w1l = None
        if self.factor_h > 0:
            w1l = [
                torch.softmax(sl, dim=-1).detach().cpu().numpy()
                for sl in self.sum1_logits
            ]
        out = torch.empty(n, self.D)
        pidx = self.patch_idx.cpu()
        for si in range(n):
            ch = rng.choice(S, p=root_w)
            stack = [(self.root_pos, ch)]
            while stack:
                p, c = stack.pop()
                if p < self.n_leaves:
                    eps = rng.standard_normal(self.pd).astype("float32")
                    val = mu[p, c] + Lnp[p, c] @ eps
                    out[si, pidx[p]] = torch.from_numpy(val)
                    continue
                gi, j = pos2group[p]
                if self.factor_h > 0:
                    k = rng.choice(wl[gi].shape[-1], p=wl[gi][j, c])
                    block, _h = divmod(k, self.factor_h)
                    cand, i = divmod(block, S)
                    jj = rng.choice(S, p=w1l[gi][j, block, _h])
                else:
                    k = rng.choice(wl[gi].shape[-1], p=wl[gi][j, c])
                    cand, rem = divmod(k, S * S)
                    i, jj = divmod(rem, S)
                stack.append((int(childA[gi][j, cand]), i))
                stack.append((int(childB[gi][j, cand]), jj))
        return out
