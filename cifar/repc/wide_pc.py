"""Deep PC with per-region-size channel counts ("wide top").

Same circuit family and same velocity rules as repc/deep_pc.py, but the
number of sum channels S is a function of region area: wide at coarse
regions (where global coherence must flow through), narrow at fine ones
(where uniform widening wastes capacity and dilutes responsibilities).

Regions are addressed as (group, row) instead of a flat buffer because
channel counts differ across groups.
"""

import math

import torch
from torch import nn

from repc.deep_pc import build_region_graph

LOG2PI = math.log(2.0 * math.pi)


def parse_schedule(spec):
    """ "392:32,98:16,1:8" -> [(392, 32), (98, 16), (1, 8)] (min-area, S)."""
    rules = []
    for part in spec.split(","):
        a, s = part.split(":")
        rules.append((int(a), int(s)))
    return sorted(rules, key=lambda r: -r[0])


class WideDeepPC(nn.Module):
    def __init__(self, H=28, W=28, schedule="392:32,98:16,1:8", order=8):
        super().__init__()
        self.H, self.W, self.D, self.order = H, W, H * W, order
        self.schedule = schedule
        rules = parse_schedule(schedule)

        def S_of(h, w):
            area = h * w
            for min_area, s in rules:
                if area >= min_area:
                    return s
            return rules[-1][1]

        regions, splits, root_old = build_region_graph(H, W)
        self.n_leaves = self.D
        self.regions = regions
        self.n_regions = len(regions)
        self.S_leaf = S_of(1, 1)

        # group internal regions by shape, ordered by (area, shape)
        by_shape = {}
        for i in range(self.n_leaves, len(regions)):
            r0, r1, c0, c1 = regions[i]
            by_shape.setdefault((r1 - r0, c1 - c0), []).append(i)

        # old region id -> (group, row); leaves are group -1, row = pixel
        addr = {i: (-1, i) for i in range(self.n_leaves)}
        self.groups = []
        for shape in sorted(by_shape, key=lambda s: (s[0] * s[1], s)):
            idx = by_shape[shape]
            gi = len(self.groups)
            for row, i in enumerate(idx):
                addr[i] = (gi, row)
            cands = []
            for c in range(len(splits[idx[0]])):
                gA, iA = zip(*[addr[splits[i][c][0]] for i in idx], strict=False)
                gB, iB = zip(*[addr[splits[i][c][1]] for i in idx], strict=False)
                assert len(set(gA)) == 1 and len(set(gB)) == 1, (
                    "candidate children same group"
                )
                cands.append(
                    {
                        "gA": gA[0],
                        "gB": gB[0],
                        "iA": torch.tensor(iA),
                        "iB": torch.tensor(iB),
                    }
                )
            self.groups.append(
                {
                    "shape": shape,
                    "pos": torch.tensor(idx),
                    "S": S_of(*shape),
                    "cands": cands,
                    "n": len(idx),
                }
            )
        self.root_group, self.root_row = addr[root_old]

        def S_of_addr(g):
            return self.S_leaf if g == -1 else self.groups[g]["S"]

        # sum weights: (n, S_out, sum_c S_A^c * S_B^c)
        self.sum_logits = nn.ParameterList()
        for g in self.groups:
            fan = sum(S_of_addr(c["gA"]) * S_of_addr(c["gB"]) for c in g["cands"])
            self.sum_logits.append(
                nn.Parameter(0.01 * torch.randn(g["n"], g["S"], fan))
            )
        self.root_logits = nn.Parameter(torch.zeros(self.groups[self.root_group]["S"]))

        for gi, g in enumerate(self.groups):
            for ci, c in enumerate(g["cands"]):
                self.register_buffer(f"_iA{gi}_{ci}", c["iA"], persistent=False)
                self.register_buffer(f"_iB{gi}_{ci}", c["iB"], persistent=False)

        self.B = nn.Parameter(0.01 * torch.randn(self.D, self.S_leaf, order))
        self.C = nn.Parameter(0.01 * torch.randn(self.D, self.S_leaf, order))

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
        """llf: (B, D, S_leaf) -> log p_root (B,). outputs[g]: (B, n_g, S_g)."""
        B = llf.shape[0]
        outputs = []
        for gi, g in enumerate(self.groups):
            prods = []
            for ci, c in enumerate(g["cands"]):
                iA = getattr(self, f"_iA{gi}_{ci}")
                iB = getattr(self, f"_iB{gi}_{ci}")
                la = llf[:, iA] if c["gA"] == -1 else outputs[c["gA"]][:, iA]
                lb = llf[:, iB] if c["gB"] == -1 else outputs[c["gB"]][:, iB]
                prod = la[..., :, None] + lb[..., None, :]  # (B, n, S_A, S_B)
                prods.append(prod.reshape(B, g["n"], -1))
            prod = torch.cat(prods, dim=-1)  # (B, n, fan)
            logw = torch.log_softmax(self.sum_logits[gi], dim=-1)
            outputs.append(torch.logsumexp(logw[None] + prod[:, :, None, :], dim=-1))
        root = outputs[self.root_group][:, self.root_row]  # (B, S_root)
        return torch.logsumexp(
            torch.log_softmax(self.root_logits, 0)[None] + root, dim=1
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
        """Numpy top-down ancestral sampler."""
        import numpy as np

        rng = np.random.default_rng(seed)
        mu, _, logsig, _ = self.leaf_params(t)
        mu = mu.detach().cpu().numpy()
        sig = torch.exp(logsig).detach().cpu().numpy()
        wl = [
            torch.softmax(sl, dim=-1).detach().cpu().numpy() for sl in self.sum_logits
        ]
        root_w = torch.softmax(self.root_logits, 0).detach().cpu().numpy()

        # per group: candidate boundaries in the flattened fan dimension
        def S_of_addr(g):
            return self.S_leaf if g == -1 else self.groups[g]["S"]

        meta = []
        for g in self.groups:
            spans = []
            off = 0
            for c in g["cands"]:
                sa, sb = S_of_addr(c["gA"]), S_of_addr(c["gB"])
                spans.append(
                    (off, sa, sb, c["gA"], c["gB"], c["iA"].numpy(), c["iB"].numpy())
                )
                off += sa * sb
            meta.append(spans)

        out = torch.empty(n, self.D)
        for si in range(n):
            ch = rng.choice(len(root_w), p=root_w)
            stack = [(self.root_group, self.root_row, ch)]
            while stack:
                gi, row, c = stack.pop()
                if gi == -1:
                    out[si, row] = float(
                        mu[row, c] + sig[row, c] * rng.standard_normal()
                    )
                    continue
                k = rng.choice(wl[gi].shape[-1], p=wl[gi][row, c])
                for off, sa, sb, gA, gB, iA, iB in meta[gi]:
                    if k < off + sa * sb:
                        i, j = divmod(k - off, sb)
                        stack.append((gA, int(iA[row]), i))
                        stack.append((gB, int(iB[row]), j))
                        break
        return out
