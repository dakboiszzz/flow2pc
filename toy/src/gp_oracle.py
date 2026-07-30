"""Exact GP oracle for Benchmark 2 (analytic teacher, no neural error).

RBF-kernel GP prior on a latent function f; observations Y = f(X) + noise,
noise ~ N(0, sigma^2). For a pair of fresh measurements at query locations
Xq given a context (Xc, Yc):

    Y1(Xq) = f(Xq) + eps1,   Y2(Xq) = f(Xq) + eps2   (eps indep given f)

the exact posterior is Gaussian, giving closed-form ground truth:

    epistemic  Sigma_epi* = Cov(f(Xq) | context) = K_post   (posterior cov)
    aleatoric  Sigma_ale* = sigma^2 I
    total      = K_post + sigma^2 I
    cross-replica Cov(Y1, Y2 | context) = K_post = epistemic

Bayesian behavior this makes checkable: epistemic contracts as |context|
grows; epistemic rises in input regions with no context; aleatoric = sigma^2.
"""

import torch


def rbf(x1, x2, lengthscale=0.5, signal=1.0):
    d2 = (x1[:, None] - x2[None, :]) ** 2
    return signal**2 * torch.exp(-0.5 * d2 / lengthscale**2)


class GPOracle:
    def __init__(self, xq, lengthscale=0.5, signal=1.0, sigma=0.1):
        self.xq = xq  # (M,) query grid
        self.M = xq.numel()
        self.ls, self.sig_s, self.sigma = lengthscale, signal, sigma
        self.Kqq = rbf(xq, xq, lengthscale, signal)

    def posterior(self, xc, yc):
        """Exact posterior mean/cov of f(xq) given context (xc, yc)."""
        if xc.numel() == 0:
            return torch.zeros(self.M), self.Kqq.clone()
        Kcc = rbf(xc, xc, self.ls, self.sig_s) + self.sigma**2 * torch.eye(xc.numel())
        Kqc = rbf(self.xq, xc, self.ls, self.sig_s)
        A = torch.linalg.solve(Kcc, Kqc.T)  # (nc, M)
        mean = A.T @ yc
        cov = self.Kqq - Kqc @ A
        cov = 0.5 * (cov + cov.T)
        return mean, cov

    def ground_truth(self, xc, yc):
        """epistemic/aleatoric/total covariance at the query grid."""
        _, K_post = self.posterior(xc, yc)
        ale = self.sigma**2 * torch.eye(self.M)
        return {"cov_epi": K_post, "cov_ale": ale, "cov_total": K_post + ale}

    def sample_prior_function(self, generator=None):
        """Sample f at the query grid from the GP prior."""
        L = torch.linalg.cholesky(self.Kqq + 1e-5 * torch.eye(self.M))
        return L @ torch.randn(self.M, generator=generator)

    def sample_context(self, n_ctx, generator=None, x_lo=-2.0, x_hi=2.0, hole=None):
        """Draw a context: n_ctx locations (optionally excluding an interval
        `hole`=(a,b) to test local epistemic increase), values from a prior
        function sample + observation noise, shared with the query grid."""
        fq = self.sample_prior_function(generator)  # f at query grid
        # context locations
        xc = x_lo + (x_hi - x_lo) * torch.rand(n_ctx, generator=generator)
        if hole is not None:
            a, b = hole
            mask = (xc >= a) & (xc <= b)
            while mask.any():
                xc[mask] = x_lo + (x_hi - x_lo) * torch.rand(
                    int(mask.sum()), generator=generator
                )
                mask = (xc >= a) & (xc <= b)
        # f at context via conditional draw from the joint prior with fq
        Kcc = rbf(xc, xc, self.ls, self.sig_s)
        Kcq = rbf(xc, self.xq, self.ls, self.sig_s)
        A = torch.linalg.solve(self.Kqq + 1e-5 * torch.eye(self.M), fq)
        mean_c = Kcq @ A
        cov_c = Kcc - Kcq @ torch.linalg.solve(
            self.Kqq + 1e-5 * torch.eye(self.M), Kcq.T
        )
        Lc = torch.linalg.cholesky(cov_c + 1e-5 * torch.eye(n_ctx))
        fc = mean_c + Lc @ torch.randn(n_ctx, generator=generator)
        yc = fc + self.sigma * torch.randn(n_ctx, generator=generator)
        return xc, yc, fq
