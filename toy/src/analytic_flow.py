"""Analytic moving-mixture flow teacher for Benchmark 1.

P_tau(z) = sum_j w_j N(z; mu_j(tau), diag sig_j(tau)^2),  w_j time-independent.

Smooth paths (doc spec):
    mu_j(tau)        = a_j + b_j tau + c_j sin(pi tau)
    log sig_{j,k}(tau) = alpha + beta tau + gamma cos(pi tau)

For time-INDEPENDENT weights the continuity-satisfying velocity is the
responsibility-weighted leaf velocity (Proposition 1):
    U(z) = sum_j r_j(z) [ mu_j'(tau) + (sig_j'/sig_j)(z - mu_j) ]
This module returns density, velocity, and the analytic derivatives needed to
evaluate the continuity residual directly, plus a `time_dependent_weights`
switch to construct a *defective* teacher whose velocity violates continuity
(used to show the theorem's precondition is necessary).
"""

import math

import torch

LOG2PI = math.log(2.0 * math.pi)


class MovingMixture:
    def __init__(self, J=4, d=2, seed=0, time_dependent_weights=False):
        g = torch.Generator().manual_seed(seed)
        self.J, self.d = J, d
        self.tdw = time_dependent_weights
        r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
        self.a = r(J, d) * 1.5
        self.b = r(J, d) * 0.8
        self.c = r(J, d) * 0.6
        self.alpha = r(J, d) * 0.3 - 0.3
        self.beta = r(J, d) * 0.4
        self.gamma = r(J, d) * 0.3
        self.w_logits = r(J) * 0.5
        self.w_osc = r(J) * 0.5  # only used if time_dependent_weights

    # ------------------------------------------------------------- params
    def params(self, tau):
        t = torch.as_tensor(tau, dtype=torch.float32)
        s, cs = torch.sin(math.pi * t), torch.cos(math.pi * t)
        mu = self.a + self.b * t + self.c * s
        dmu = self.b + self.c * math.pi * cs
        ls = self.alpha + self.beta * t + self.gamma * cs
        dls = self.beta - self.gamma * math.pi * s
        sig = torch.exp(ls)
        dsig = sig * dls
        return mu, dmu, sig, dsig

    def weights(self, tau):
        if not self.tdw:
            return torch.softmax(self.w_logits, 0), torch.zeros(self.J)
        t = torch.as_tensor(tau, dtype=torch.float32)
        logit = self.w_logits + self.w_osc * torch.sin(math.pi * t)
        w = torch.softmax(logit, 0)
        dlogit = self.w_osc * math.pi * torch.cos(math.pi * t)
        dw = w * (dlogit - (w * dlogit).sum())  # softmax jacobian . dlogit
        return w, dw

    # ------------------------------------------------------------ density
    def _leaf_logp(self, z, mu, sig):
        zc = (z[:, None, :] - mu[None]) / sig[None]
        return -0.5 * (zc**2 + LOG2PI).sum(-1) - torch.log(sig).sum(-1)[None]

    def log_prob(self, z, tau):
        mu, _, sig, _ = self.params(tau)
        w, _ = self.weights(tau)
        return torch.logsumexp(torch.log(w)[None] + self._leaf_logp(z, mu, sig), 1)

    def responsibilities(self, z, tau):
        mu, _, sig, _ = self.params(tau)
        w, _ = self.weights(tau)
        return torch.softmax(torch.log(w)[None] + self._leaf_logp(z, mu, sig), 1)

    # ----------------------------------------------------------- velocity
    def velocity(self, z, tau):
        mu, dmu, sig, dsig = self.params(tau)
        r = self.responsibilities(z, tau)
        u_leaf = dmu[None] + (dsig / sig)[None] * (z[:, None, :] - mu[None])
        return (r[..., None] * u_leaf).sum(1)

    # -------------------------------------------------- continuity residual
    def continuity_residual(self, z, tau, velocity_offset=None):
        """Normalized residual of d_tau Q + div(Q U) = 0, in density space,
        via autograd. Returns per-point residual (B,).

        d_tau log Q is analytic (mixture of leaf log-derivatives + weight
        drift); div(Q U)/Q = div U + U . grad log Q is via autograd on U and
        log Q. Continuity <=> d_tau log Q + div U + U.grad log Q = 0.

        velocity_offset: optional constant vector c (d,) added to U. A constant
        has zero divergence, so only the U.grad-logQ term shifts by c.grad-logQ
        -- computed exactly (no heuristic scaling), so the perturbed residual is
        as rigorous as the base one.
        """
        z = z.detach().requires_grad_(True)
        tau_t = torch.as_tensor(float(tau))

        # d_tau log Q analytically: log Q = logsumexp_j (log w_j + leaf_j)
        mu, dmu, sig, dsig = self.params(tau_t)
        w, dw = self.weights(tau_t)
        leaf = self._leaf_logp(z, mu, sig)  # (B,J)
        # d_tau leaf_j = -sum_k [ (z-mu)/sig^2 * (-dmu) + ((z-mu)^2/sig^3 - 1/sig)*dsig ]
        zc = z[:, None, :] - mu[None]
        dleaf = (
            (zc / sig[None] ** 2 * dmu[None]).sum(-1)
            + ((zc**2) / sig[None] ** 3 * dsig[None]).sum(-1)
            - (dsig / sig)[None].sum(-1)
        )  # (B,J)
        r = torch.softmax(torch.log(w)[None] + leaf, 1)
        dlogQ_dt = (r * (dw / w)[None]).sum(1) + (r * dleaf).sum(1)

        # spatial terms via autograd
        logQ = torch.logsumexp(torch.log(w)[None] + leaf, 1)
        (grad_logQ,) = torch.autograd.grad(logQ.sum(), z, create_graph=True)
        U = self.velocity(z.detach(), tau).detach()  # velocity field value
        # recompute U with grad to get divergence
        z2 = z.detach().requires_grad_(True)
        Uz = self.velocity(z2, tau)
        divU = torch.zeros(z.shape[0])
        for k in range(self.d):
            (gk,) = torch.autograd.grad(Uz[:, k].sum(), z2, create_graph=True)
            divU = divU + gk[:, k]
        if velocity_offset is not None:
            U = U + velocity_offset[None]
        u_dot_grad = (U * grad_logQ.detach()).sum(-1)
        resid = dlogQ_dt + divU + u_dot_grad
        denom = dlogQ_dt.abs() + divU.abs() + u_dot_grad.abs() + 1e-6
        return (resid.abs() / denom).detach()

    @torch.no_grad()
    def sample(self, n, tau, seed=0):
        g = torch.Generator().manual_seed(seed)
        mu, _, sig, _ = self.params(tau)
        w, _ = self.weights(tau)
        h = torch.multinomial(w, n, replacement=True, generator=g)
        return mu[h] + sig[h] * torch.randn(n, self.d, generator=g)
