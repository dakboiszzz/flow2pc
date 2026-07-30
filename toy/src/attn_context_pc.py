"""Attention-based context-conditioned pair PC for Benchmark 2.

The DeepSets mean-pooled encoder (context_pair_pc.py) underfit the sharp,
locally-varying GP posterior. This variant uses cross-attention from the query
grid to the context points -- the same local-conditioning inductive bias a GP
uses -- to produce PER-QUERY context features. Each query location attends to
the observed points near it, so the epistemic uncertainty at a location can
collapse sharply once that location is well-observed.

Still the same pair-PC head and exact moments:
    Sigma_epi(c) = Cov_h(mu_h(c)) = Cov(Y1, Y2 | c)
    Sigma_ale(c) = E_h[diag(sig_h(c)^2)]

Leaves use a learned homoscedastic noise floor so the aleatoric level can be
discovered separately from component spread (the data IS f + constant-variance
noise -- a well-specified inductive bias, not a baked-in answer).
"""

import math

import torch
from torch import nn

LOG2PI = math.log(2.0 * math.pi)


class AttnContextPairPC(nn.Module):
    def __init__(self, xq, H=64, dim=96, heads=4):
        super().__init__()
        self.register_buffer("xq", xq)
        self.M, self.H, self.dim = xq.numel(), H, dim
        self.ctx_embed = nn.Sequential(
            nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.q_embed = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.empty = nn.Parameter(torch.zeros(dim))
        self.post = nn.Sequential(nn.Linear(dim, dim), nn.SiLU())
        # per-query heads: means and log per-query spread; global pi via pooled
        self.head_mu = nn.Linear(dim, H)  # (per query) -> H means
        self.head_ls = nn.Linear(dim, H)  # (per query) -> H log-scales
        self.head_pi = nn.Linear(dim, H)  # pooled -> H logits
        self.noise_floor = nn.Parameter(torch.tensor(-2.0))  # log sigma floor

    def _features(self, xc, yc):
        """xc, yc (B, n) -> per-query features (B, M, dim)."""
        B = xc.shape[0]
        q = self.q_embed(self.xq.reshape(1, self.M, 1).expand(B, -1, -1))
        if xc.shape[-1] == 0:
            return q + self.post(self.empty)[None, None]
        kv = self.ctx_embed(torch.stack([xc, yc], dim=-1))  # (B, n, dim)
        att, _ = self.attn(q, kv, kv)  # (B, M, dim)
        return self.post(att) + q

    def params(self, xc, yc):
        feat = self._features(xc, yc)  # (B, M, dim)
        mu = self.head_mu(feat).permute(0, 2, 1)  # (B, H, M)
        floor = torch.exp(self.noise_floor.clamp(-4.0, 1.0))  # in [0.018, 2.7]
        spread = torch.exp(self.head_ls(feat).clamp(-4, 2)).permute(0, 2, 1)
        sig = torch.sqrt(floor**2 + spread**2)  # noise floor + spread
        pi = torch.softmax(self.head_pi(feat.mean(1)), -1)  # (B, H)
        return pi, mu, sig

    def _leaf_logp(self, y, mu, sig):
        zc = (y[:, None, :] - mu) / sig
        return -0.5 * (zc**2 + LOG2PI).sum(-1) - torch.log(sig).sum(-1)

    def log_prob1(self, y, pi, mu, sig):
        return torch.logsumexp(torch.log(pi) + self._leaf_logp(y, mu, sig), -1)

    def log_prob2(self, y1, y2, pi, mu, sig):
        joint = self._leaf_logp(y1, mu, sig) + self._leaf_logp(y2, mu, sig)
        return torch.logsumexp(torch.log(pi) + joint, -1)

    @torch.no_grad()
    def moments(self, xc, yc):
        pi, mu, sig = self.params(xc[None], yc[None])
        pi, mu, sig = pi[0], mu[0], sig[0]
        mbar = (pi[:, None] * mu).sum(0)
        muc = mu - mbar[None]
        cov_epi = (pi[:, None, None] * muc[:, :, None] * muc[:, None, :]).sum(0)
        cov_ale = torch.diag((pi[:, None] * sig**2).sum(0))
        return {
            "mean": mbar,
            "cov_epi": cov_epi,
            "cov_ale": cov_ale,
            "cov_total": cov_epi + cov_ale,
        }
