"""Time-indexed Probabilistic Circuit with the induced velocity field.

Structure (shallow PC, directly mirroring the slides):

  root SUM node  --  K children, time-INDEPENDENT weights w  (Proposition 1)
      |
  K PRODUCT nodes -- each factorizes over the D pixel dims (disjoint scopes)
      |
  D Gaussian LEAVES per component with time-dependent params:

      mu_{k,d}(t)      = sum_{p=1..P} B[k,d,p] t^p        =>  mu(0) = 0
      log sig_{k,d}(t) = sum_{p=1..P} C[k,d,p] t^p        =>  sig(0) = 1

  so p_0^PC = N(0, I) exactly, for any sum weights.

Velocity rules (from the slides):
  leaf:    u_{j,t}(z) = mu_j'(t) + (sig_j'(t)/sig_j(t)) (z - mu_j(t))
           (diagonal case of  mu' + L' L^{-1} (z - mu))
  product: u = concatenation of child velocities over disjoint scopes
           (implemented as the vectorized per-dim leaf formula)
  sum:     u_{j,t}(z) = sum_c [w_c p_c(z) / sum_k w_k p_k(z)] u_{c,t}(z)

By Proposition 1, (p_root_t, u_root_t) satisfies the continuity equation by
construction, so the ONLY training loss is velocity matching to the flow.
"""

import math

import torch
from torch import nn

LOG2PI = math.log(2.0 * math.pi)


class TimePolyGaussianPC(nn.Module):
    def __init__(self, K, D, order=4):
        super().__init__()
        self.K, self.D, self.order = K, D, order
        self.B = nn.Parameter(torch.zeros(K, D, order))  # mean coefficients
        self.C = nn.Parameter(torch.zeros(K, D, order))  # log-sigma coefficients
        self.w_logits = nn.Parameter(torch.zeros(K))  # time-independent sum weights

    # ---------------------------------------------------------------- leaves
    def _basis(self, t):
        """phi_p(t)=t^p, dphi_p(t)=p t^(p-1), p=1..P. t: scalar tensor."""
        p = torch.arange(1, self.order + 1, device=t.device, dtype=t.dtype)
        phi = t**p
        dphi = p * t ** (p - 1)
        return phi, dphi

    def leaf_params(self, t):
        """Returns mu, dmu/dt, logsig, dlogsig/dt, each (K, D)."""
        phi, dphi = self._basis(t)
        mu = torch.einsum("kdp,p->kd", self.B, phi)
        dmu = torch.einsum("kdp,p->kd", self.B, dphi)
        logsig = torch.einsum("kdp,p->kd", self.C, phi)
        dlogsig = torch.einsum("kdp,p->kd", self.C, dphi)
        return mu, dmu, logsig, dlogsig

    # ------------------------------------------------------------- densities
    def _component_log_prob(self, z, mu, logsig):
        """log p_k(z) for all components without materializing (B,K,D).

        log N(z; mu, sig^2) summed over D =
          -0.5 [ z^2 . inv_var - 2 z . (mu inv_var) + const_k ]
        """
        inv_var = torch.exp(-2.0 * logsig)  # (K,D)
        quad = z.pow(2) @ inv_var.T - 2.0 * (z @ (mu * inv_var).T)  # (B,K)
        const = (mu.pow(2) * inv_var + 2.0 * logsig).sum(
            dim=1
        ) + self.D * LOG2PI  # (K,)
        return -0.5 * (quad + const[None, :])

    def log_prob(self, z, t):
        """Exact PC log-density at time t. z: (B,D), t: scalar tensor."""
        mu, _, logsig, _ = self.leaf_params(t)
        log_w = torch.log_softmax(self.w_logits, dim=0)
        return torch.logsumexp(
            log_w[None, :] + self._component_log_prob(z, mu, logsig), dim=1
        )

    # -------------------------------------------------------------- velocity
    def velocity(self, z, t):
        """Root velocity u_root_t(z) via leaf/product/sum rules. z: (B,D)."""
        mu, dmu, logsig, dlogsig = self.leaf_params(t)
        log_w = torch.log_softmax(self.w_logits, dim=0)
        # sum-node responsibilities  r_k(z) = w_k p_k(z) / sum_j w_j p_j(z)
        r = torch.softmax(
            log_w[None, :] + self._component_log_prob(z, mu, logsig), dim=1
        )
        # leaf/product velocity: u_k(z) = dmu_k + dlogsig_k * (z - mu_k)
        # (avoid (B,K,D): u_root = r @ (dmu - dlogsig*mu) + (r @ dlogsig) * z)
        return r @ (dmu - dlogsig * mu) + (r @ dlogsig) * z

    # -------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, n, t, generator=None):
        """One-step ancestral sampling from the PC at time t."""
        mu, _, logsig, _ = self.leaf_params(t)
        w = torch.softmax(self.w_logits, dim=0)
        idx = torch.multinomial(w, n, replacement=True, generator=generator)
        eps = torch.randn(n, self.D, device=mu.device, generator=generator)
        return mu[idx] + torch.exp(logsig[idx]) * eps

    # ------------------------------------------------------------------ init
    @torch.no_grad()
    def init_from_data(self, x1, sigma1=0.3):
        """Initialize each component as the per-sample OT path:
        mu_k(t) = t * x1_k  (B[:,:,0] = x1, higher orders 0)
        sig_k(t) = exp(t * log sigma1)  (interpolates 1 -> sigma1)
        """
        assert x1.shape == (self.K, self.D)
        self.B.zero_()
        self.C.zero_()
        self.B[:, :, 0] = x1
        self.C[:, :, 0] = math.log(sigma1)
        self.w_logits.zero_()
