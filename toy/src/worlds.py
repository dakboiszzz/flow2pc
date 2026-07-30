"""Benchmark 0 data-generating worlds (matched marginals, different UQ).

Two worlds over replica pairs (Z1, Z2) in R^d with IDENTICAL one-replica
marginals but different pair laws:

  Shared-hypothesis world A:  Theta ~ N(0, Sigma_e);  Z_j = Theta + eps_j,
                              eps_j ~ N(0, Sigma_a) (Corr(eps1,eps2)=rho_ci)
  Independent world B:        Z_j iid ~ N(0, Sigma_e + Sigma_a)

Both are pushed through the SAME fixed triangular flow, applied per replica:

  T_tau(z_a, z_b) = (z_a, z_b + tau * s(z_a)),   s = frozen MLP

so one-replica densities and velocities agree between worlds for every tau,
while pair laws differ (Cov_A(X1,X2) != 0 = Cov_B(X1,X2)).

The true velocity field is analytic: u_tau(x) = (0, s(x_a)) since the
identity block preserves x_a = z_a.
"""

import torch
from torch import nn


class TriangularFlow(nn.Module):
    """x = T_tau(z): identity on first half, shear by s(first half) on second.

    d == 1 degenerates to a constant shift (no conditioning half); the
    matched-marginal argument is unaffected.
    """

    def __init__(self, d, hidden=64, seed=0, shear_scale=0.6):
        super().__init__()
        self.d = d
        self.da = max(1, d // 2) if d > 1 else 0
        self.db = d - self.da
        g = torch.Generator().manual_seed(seed)
        if d > 1:
            self.lin1 = nn.Linear(self.da, hidden)
            self.lin2 = nn.Linear(hidden, self.db)
            with torch.no_grad():
                self.lin1.weight.copy_(torch.randn(self.lin1.weight.shape, generator=g))
                self.lin1.bias.copy_(
                    torch.randn(self.lin1.bias.shape, generator=g) * 0.3
                )
                # scale output so shear std ~ shear_scale (was ~6 with dense sum
                # over `hidden` tanh units — too non-Gaussian for the mixture)
                self.lin2.weight.copy_(
                    torch.randn(self.lin2.weight.shape, generator=g)
                    * (shear_scale / hidden**0.5)
                )
                self.lin2.bias.zero_()
        else:
            self.register_buffer("shift", torch.randn(1, generator=g) * shear_scale)
        for p in self.parameters():
            p.requires_grad_(False)

    def s(self, za):
        return self.lin2(torch.tanh(self.lin1(za)))

    def forward(self, z, tau):
        if self.d == 1:
            return z + tau * self.shift
        za, zb = z[..., : self.da], z[..., self.da :]
        return torch.cat([za, zb + tau * self.s(za)], dim=-1)

    def velocity(self, x, tau=None):
        """u_tau(x) = dT/dtau at fixed z; independent of tau for this family."""
        if self.d == 1:
            return self.shift.expand_as(x)
        xa = x[..., : self.da]
        return torch.cat([torch.zeros_like(xa), self.s(xa)], dim=-1)


class MatchedMarginalWorlds:
    """Sampler + Monte-Carlo oracle for worlds A and B."""

    def __init__(self, d, rho_e=0.5, total_var=1.0, rho_ci=0.0, flow_seed=0, hidden=64):
        self.d = d
        self.sig_e2 = rho_e * total_var  # per-dim epistemic variance
        self.sig_a2 = (1 - rho_e) * total_var  # per-dim aleatoric variance
        self.rho_ci = rho_ci
        self.flow = TriangularFlow(d, hidden=hidden, seed=flow_seed)

    def sample_pair_z(self, n, world, generator=None):
        d = self.d
        if world == "A":
            theta = torch.randn(n, d, generator=generator) * self.sig_e2**0.5
            e1 = torch.randn(n, d, generator=generator)
            e_common = torch.randn(n, d, generator=generator)
            e2_ind = torch.randn(n, d, generator=generator)
            r = self.rho_ci
            eps1 = (r**0.5 * e_common + (1 - r) ** 0.5 * e1) * self.sig_a2**0.5
            eps2 = (r**0.5 * e_common + (1 - r) ** 0.5 * e2_ind) * self.sig_a2**0.5
            return theta + eps1, theta + eps2
        if world == "B":
            tot = (self.sig_e2 + self.sig_a2) ** 0.5
            return (
                torch.randn(n, d, generator=generator) * tot,
                torch.randn(n, d, generator=generator) * tot,
            )
        raise ValueError(world)

    def sample_pair(self, n, tau, world, generator=None):
        """Data-space pair (X1, X2) at flow time tau, plus exact velocities."""
        z1, z2 = self.sample_pair_z(n, world, generator)
        with torch.no_grad():
            x1, x2 = self.flow(z1, tau), self.flow(z2, tau)
            return x1, x2, self.flow.velocity(x1), self.flow.velocity(x2)

    @torch.no_grad()
    def oracle_moments(self, tau=1.0, world="A", n=1_000_000, seed=123):
        """MC-exact data-space moments: total covariance of one replica,
        cross-replica covariance (= epistemic + CI bias), aleatoric part."""
        g = torch.Generator().manual_seed(seed)
        x1, x2, _, _ = self.sample_pair(n, tau, world, generator=g)
        m = x1.mean(0)
        xc1, xc2 = x1 - m, x2 - x2.mean(0)
        cov_tot = xc1.T @ xc1 / (n - 1)
        cov_cross = xc1.T @ xc2 / (n - 1)
        cov_cross = 0.5 * (cov_cross + cov_cross.T)
        return {
            "mean": m,
            "cov_total": cov_tot,
            "cov_cross": cov_cross,
            "cov_ale": cov_tot - cov_cross,
        }
