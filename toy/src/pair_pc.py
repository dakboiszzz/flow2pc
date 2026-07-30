"""Shared-channel pair PC for Benchmark 0.

One-replica law:   Q1_tau(z)      = sum_h pi_h * N(z; mu_h(tau), diag sig_h(tau)^2)
Synchronized pair: Q2_tau(z1,z2)  = sum_h pi_h * q_h(z1) * q_h(z2)

Only the hypothesis channel H is shared across replicas; pi_h is constant in
tau (time-independent sum weights, per the continuity theorem). Leaf paths
are cubic polynomials in tau by default, or a learned time-MLP basis
(h(tau) = MLP(sincos(tau)); the basis that broke the one-step wall in the
Recompile-Probabilistic experiments).

Velocity rules (slide rules, verified by the test battery):
  leaf:    u_h(z)   = mu_h' + (sig_h'/sig_h) * (z - mu_h)
  sum:     u(z)     = sum_h r_h(z) u_h(z),   r_h = responsibilities
  pair:    U_i(z1,z2) = sum_h r_h(z1,z2) u_h(z_i)   (product = concat)

Exact decomposition from parameters (law of total variance over H, at any tau):
  Sigma_epi = Cov_h(mu_h) = sum_h pi_h mu_h mu_h^T - mu_bar mu_bar^T
  Sigma_ale = E_h[diag(sig_h^2)]
  Cov(Z1, Z2) = Sigma_epi          (the identifiable epistemic covariance)
"""

import math

import torch
from torch import nn

LOG2PI = math.log(2.0 * math.pi)


class PairPC(nn.Module):
    def __init__(self, d, M=16, order=3, time_mlp=False, t_hidden=64, t_freqs=8):
        super().__init__()
        self.d, self.M, self.order = d, M, order
        self.pi_logits = nn.Parameter(torch.zeros(M))
        # basis coefficients: value at basis functions [1, b_1(tau), ...]
        P = order + 1
        self.Bmu = nn.Parameter(0.01 * torch.randn(M, d, P))
        self.Bls = nn.Parameter(0.01 * torch.randn(M, d, P))  # log-sigma
        self.time_mlp = time_mlp
        if time_mlp:
            self.t_freqs = t_freqs
            self.tnet = nn.Sequential(
                nn.Linear(2 * t_freqs, t_hidden), nn.SiLU(), nn.Linear(t_hidden, P - 1)
            )

    # ---------------------------------------------------------------- basis
    def _basis(self, tau):
        """Returns basis values and d/dtau, each (P,). tau: scalar tensor."""
        P = self.order + 1
        if self.time_mlp:

            def h(t):
                freqs = torch.arange(1, self.t_freqs + 1, device=t.device) * math.pi
                emb = torch.cat([torch.sin(freqs * t), torch.cos(freqs * t)])
                return torch.cat([torch.ones(1, device=t.device), self.tnet(emb)])

            val, dval = torch.func.jvp(h, (tau,), (torch.ones_like(tau),))
            return val, dval
        p = torch.arange(P, device=tau.device, dtype=tau.dtype)
        val = tau**p
        dval = torch.where(p > 0, p * tau ** (p - 1).clamp(min=0), torch.zeros_like(p))
        return val, dval

    def leaf_params(self, tau):
        """mu, dmu, sig, dsig at tau; shapes (M, d)."""
        val, dval = self._basis(
            torch.as_tensor(tau, dtype=self.Bmu.dtype, device=self.Bmu.device)
        )
        mu = self.Bmu @ val
        dmu = self.Bmu @ dval
        ls = self.Bls @ val
        dls = self.Bls @ dval
        sig = torch.exp(ls.clamp(-7.0, 4.0))
        return mu, dmu, sig, dls * sig  # dsig = sig * dls

    # -------------------------------------------------------------- density
    def _leaf_logp(self, z, mu, sig):
        """z (B, d) against every channel -> (B, M)."""
        zc = (z[:, None, :] - mu[None]) / sig[None]
        return -0.5 * (zc**2 + LOG2PI).sum(-1) - torch.log(sig).sum(-1)[None]

    def log_prob1(self, z, tau):
        mu, _, sig, _ = self.leaf_params(tau)
        lw = torch.log_softmax(self.pi_logits, 0)
        return torch.logsumexp(lw[None] + self._leaf_logp(z, mu, sig), dim=1)

    def log_prob2(self, z1, z2, tau):
        mu, _, sig, _ = self.leaf_params(tau)
        lw = torch.log_softmax(self.pi_logits, 0)
        joint = self._leaf_logp(z1, mu, sig) + self._leaf_logp(z2, mu, sig)
        return torch.logsumexp(lw[None] + joint, dim=1)

    # -------------------------------------------------------------- velocity
    def _leaf_velocity(self, z, mu, dmu, sig, dsig):
        return dmu[None] + (dsig / sig)[None] * (z[:, None, :] - mu[None])

    def velocity1(self, z, tau):
        mu, dmu, sig, dsig = self.leaf_params(tau)
        lw = torch.log_softmax(self.pi_logits, 0)
        r = torch.softmax(lw[None] + self._leaf_logp(z, mu, sig), dim=1)
        return (r[..., None] * self._leaf_velocity(z, mu, dmu, sig, dsig)).sum(1)

    def velocity2(self, z1, z2, tau, resp_grad_scale=1.0):
        """Pair velocities (U1, U2) with responsibilities from the pair law.

        resp_grad_scale s in [0,1]: PARTIAL stop-gradient on the
        responsibilities via r_eff = s*r + (1-s)*sg(r). The forward value is
        exactly r, but its gradient is scaled by s. s=1 lets the gradient flow
        fully through the soft assignment (theorem-faithful velocity match, but
        aleatoric scale is underdetermined and erratic); s=0 fully detaches the
        assignment (per-component regression that pins the aleatoric but hurts
        the velocity match). A moderate s trades the two off with one knob."""
        mu, dmu, sig, dsig = self.leaf_params(tau)
        lw = torch.log_softmax(self.pi_logits, 0)
        joint = self._leaf_logp(z1, mu, sig) + self._leaf_logp(z2, mu, sig)
        r = torch.softmax(lw[None] + joint, dim=1)[..., None]
        if resp_grad_scale != 1.0:
            r = resp_grad_scale * r + (1.0 - resp_grad_scale) * r.detach()
        u1 = (r * self._leaf_velocity(z1, mu, dmu, sig, dsig)).sum(1)
        u2 = (r * self._leaf_velocity(z2, mu, dmu, sig, dsig)).sum(1)
        return u1, u2

    # -------------------------------------------------------- exact moments
    @torch.no_grad()
    def moments(self, tau=1.0):
        mu, _, sig, _ = self.leaf_params(tau)
        pi = torch.softmax(self.pi_logits, 0)
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

    @torch.no_grad()
    def init_from_data(self, x):
        """Moment init: channel means from data samples, shared data scale.
        Sets the constant basis term only (so tau-paths start sensible)."""
        n = min(self.M, x.shape[0])
        self.Bmu.zero_()
        self.Bls.zero_()
        self.Bmu[:, :, 0] = x[torch.randperm(x.shape[0])[: self.M]]
        self.Bls[:, :, 0] = (
            torch.log(x.std(0).clamp_min(1e-3))[None].expand(self.M, -1) - 0.5
        )
        return n
