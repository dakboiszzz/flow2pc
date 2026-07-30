"""Genuine two-replica (pair) flow teacher for the one-loss vs two-loss test.

The earlier velocity-matching test used the ONE-replica flow velocity, which is
world-invariant and carries no epistemic signal. The plan requires a JOINT PAIR
teacher: a task-conditional flow whose pair velocity V^[2](z1,z2) has
responsibilities over the task theta that depend on BOTH replicas.

Ground truth = a K-task mixture (each task a Gaussian path from N(0,I)):

    theta_k with prior pi_k ;  p_tau(z|k) = N(z; tau*mu_k, s_tau^2 I),
    s_tau = (1-tau) + tau*sigma_a           (so p_0 = N(0,I) for every task)

    P^[2]_tau(z1,z2) = sum_k pi_k p_tau(z1|k) p_tau(z2|k)     (pair path)

Leaf velocity of task k:  u_k(z,tau) = mu_k + (s'_tau/s_tau)(z - tau*mu_k).
Pair velocity (replica j):  V^[2]_j = sum_k rho_k(z1,z2) u_k(z_j),
    rho_k(z1,z2) proportional to pi_k p_tau(z1|k) p_tau(z2|k)  <-- depends on BOTH.

This joint responsibility is exactly the epistemic signal a one-replica velocity
lacks. Exact moments at tau=1: epistemic = Cov_k(mu_k), aleatoric = sigma_a^2 I.
"""

import math

import torch

LOG2PI = math.log(2.0 * math.pi)


class PairTeacher:
    def __init__(self, d=4, K=6, sigma_a=0.5, seed=0, mu_scale=1.2):
        g = torch.Generator().manual_seed(seed)
        self.d, self.K, self.sigma_a = d, K, sigma_a
        self.mu = torch.randn(K, d, generator=g) * mu_scale  # task means
        self.pi = torch.softmax(torch.randn(K, generator=g) * 0.4, 0)

    def _st(self, tau):
        t = torch.as_tensor(tau, dtype=torch.float32)
        return (1 - t) + t * self.sigma_a, (self.sigma_a - 1.0)  # s_tau, s'

    def leaf_logp(self, z, tau):
        """z (B,d) against all K tasks -> (B,K)."""
        s, _ = self._st(tau)
        mu = tau * self.mu  # (K,d)
        zc = (z[:, None, :] - mu[None]) / s
        return -0.5 * (zc**2 + LOG2PI).sum(-1) - self.d * math.log(s)

    def log_prob2(self, z1, z2, tau):
        j = self.leaf_logp(z1, tau) + self.leaf_logp(z2, tau)
        return torch.logsumexp(torch.log(self.pi)[None] + j, 1)

    def leaf_velocity(self, z, tau):
        s, ds = self._st(tau)
        mu = tau * self.mu
        return self.mu[None] + (ds / s) * (z[:, None, :] - mu[None])  # (B,K,d)

    def pair_velocity(self, z1, z2, tau):
        """V^[2]_1, V^[2]_2 with joint responsibilities over the task."""
        j = self.leaf_logp(z1, tau) + self.leaf_logp(z2, tau)
        rho = torch.softmax(torch.log(self.pi)[None] + j, 1)[..., None]  # (B,K,1)
        v1 = (rho * self.leaf_velocity(z1, tau)).sum(1)
        v2 = (rho * self.leaf_velocity(z2, tau)).sum(1)
        return v1, v2

    @torch.no_grad()
    def sample_pair(self, n, tau, generator=None):
        k = torch.multinomial(self.pi, n, replacement=True, generator=generator)
        s, _ = self._st(tau)
        mu = tau * self.mu[k]
        z1 = mu + s * torch.randn(n, self.d, generator=generator)
        z2 = mu + s * torch.randn(n, self.d, generator=generator)
        return z1, z2

    @torch.no_grad()
    def moments(self, tau=1.0):
        mu = tau * self.mu
        mbar = (self.pi[:, None] * mu).sum(0)
        muc = mu - mbar[None]
        cov_epi = (self.pi[:, None, None] * muc[:, :, None] * muc[:, None, :]).sum(0)
        s, _ = self._st(tau)
        cov_ale = s**2 * torch.eye(self.d)
        return {
            "cov_epi": cov_epi,
            "cov_ale": cov_ale,
            "cov_total": cov_epi + cov_ale,
            "mean": mbar,
        }
