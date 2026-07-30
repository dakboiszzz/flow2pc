"""Context-conditioned pair PC for Benchmark 2 (GP meta-regression).

A single compiled model amortized over variable-size contexts (the doc's
c -> {pi_h(c), leaf params(c)} design):

    context (xc, yc)  --DeepSets-->  c_emb
    pi_h(c)  = softmax(MLP_pi(c_emb))                 hypothesis weights
    mu_h(c)  = MLP_mu(c_emb)   in R^{H x M}           per-query means
    sig_h(c) = exp(MLP_ls(c_emb))                     per-query scales

    Q2(y1,y2|c) = sum_h pi_h prod_m N(y1_m; mu_hm, sig_hm) N(y2_m; ...)

Exact decomposition (endpoint pair law, the object Benchmark 0 showed
identifies UQ):
    Sigma_epi(c) = Cov_h(mu_h(c)) = Cov(Y1, Y2 | c)  -> GP posterior K_post
    Sigma_ale(c) = E_h[diag(sig_h(c)^2)]             -> sigma^2 I

DeepSets pooling is permutation-invariant and handles n=0 via a learned empty
embedding, so epistemic uncertainty is a smooth function of context queryable
after one compilation. Encoder/params are batched over contexts of a common
size; moments() is exact per single context.
"""

import math

import torch
from torch import nn

LOG2PI = math.log(2.0 * math.pi)


class ContextPairPC(nn.Module):
    def __init__(self, M_query, H=32, enc=64, hidden=128):
        super().__init__()
        self.M, self.H = M_query, H
        self.phi = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(), nn.Linear(hidden, enc)
        )
        self.empty = nn.Parameter(torch.zeros(enc))
        self.rho = nn.Sequential(
            nn.Linear(enc, hidden), nn.SiLU(), nn.Linear(hidden, enc), nn.SiLU()
        )
        self.head_pi = nn.Linear(enc, H)
        self.head_mu = nn.Linear(enc, H * M_query)
        self.head_ls = nn.Linear(enc, H * M_query)

    # ------------------------------------------------------- context encoder
    def encode(self, xc, yc):
        """xc, yc: (B, n) (n may be 0). Returns c_emb (B, enc)."""
        if xc.shape[-1] == 0:
            pooled = self.empty[None].expand(xc.shape[0], -1)
        else:
            pooled = self.phi(torch.stack([xc, yc], dim=-1)).mean(1)
        return self.rho(pooled)

    def params(self, xc, yc):
        c = self.encode(xc, yc)
        B = c.shape[0]
        pi = torch.softmax(self.head_pi(c), -1)  # (B, H)
        mu = self.head_mu(c).reshape(B, self.H, self.M)  # (B, H, M)
        sig = torch.exp(self.head_ls(c).reshape(B, self.H, self.M).clamp(-7, 4))
        return pi, mu, sig

    # -------------------------------------------------------------- density
    def _leaf_logp(self, y, mu, sig):
        """y (B, M), mu/sig (B, H, M) -> (B, H)."""
        zc = (y[:, None, :] - mu) / sig
        return -0.5 * (zc**2 + LOG2PI).sum(-1) - torch.log(sig).sum(-1)

    def log_prob1(self, y, pi, mu, sig):
        return torch.logsumexp(torch.log(pi) + self._leaf_logp(y, mu, sig), -1)

    def log_prob2(self, y1, y2, pi, mu, sig):
        joint = self._leaf_logp(y1, mu, sig) + self._leaf_logp(y2, mu, sig)
        return torch.logsumexp(torch.log(pi) + joint, -1)

    # -------------------------------------------------------- exact moments
    @torch.no_grad()
    def moments(self, xc, yc):
        """Single context: xc, yc are (n,). Returns cov_epi/ale/total (M,M)."""
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
