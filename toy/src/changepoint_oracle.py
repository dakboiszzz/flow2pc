"""Exact change-point oracle for Benchmark 3 (NON-Gaussian posterior).

The doc flags GPs as "too favorable because covariance fully characterizes the
posterior." A random step function has a genuinely non-Gaussian (multimodal)
posterior, so it tests whether the pair principle captures uncertainty BEYOND
second moments.

    theta = (a1, a2, xi),  a1,a2 ~ N(0,1),  xi ~ U(-1,1)
    f_theta(x) = a1 + (a2 - a1) * 1[x > xi]      (level a1 left of xi, a2 right)
    y = f(x) + eps,   eps ~ N(0, sigma^2)

Exact posterior over f at a query grid via a fine grid on xi: given xi, f is
linear in (a1,a2), so (a1,a2 | context, xi) is Gaussian (Bayesian linear
regression) and p(context | xi) is Gaussian. The posterior over f(xq) is thus a
mixture over the xi grid of Gaussians -- multimodal, non-Gaussian. Exact:
  - moments (mean, cov) via law of total covariance over the mixture
  - samples (pick xi ~ posterior, then (a1,a2) ~ Gaussian)
  - a non-Gaussianity measure (excess kurtosis of a projection)
"""

import torch


class ChangePointOracle:
    def __init__(self, xq, sigma=0.15, n_grid=201, prior_std=1.0):
        self.xq = xq
        self.M = xq.numel()
        self.sigma = sigma
        self.prior_std = prior_std
        self.xi_grid = torch.linspace(-1.0, 1.0, n_grid)
        # query design matrix per xi: B_g[m] = [1[xq_m<=xi], 1[xq_m>xi]]
        self.Bq = torch.stack(
            [
                (xq[:, None] <= self.xi_grid[None]).float(),
                (xq[:, None] > self.xi_grid[None]).float(),
            ],
            dim=-1,
        ).permute(1, 0, 2)  # (G, M, 2)

    def _design(self, xc, xi):
        return torch.stack([(xc <= xi).float(), (xc > xi).float()], dim=-1)  # (n,2)

    def posterior(self, xc, yc):
        """Returns mixture: weights w (G,), per-grid mean mu (G,M), cov S (G,M,M)."""
        G = self.xi_grid.numel()
        s2, p2 = self.sigma**2, self.prior_std**2
        prior_prec = torch.eye(2) / p2
        mu = torch.zeros(G, self.M)
        cov = torch.zeros(G, self.M, self.M)
        logw = torch.zeros(G)
        for g in range(G):
            xi = self.xi_grid[g]
            if xc.numel() > 0:
                A = self._design(xc, xi)  # (n,2)
                S = torch.linalg.inv(prior_prec + A.T @ A / s2)  # (2,2)
                m = S @ A.T @ yc / s2  # (2,)
                # marginal likelihood p(yc | xi) = N(0, A A^T p2 + s2 I)
                Cy = A @ A.T * p2 + s2 * torch.eye(xc.numel())
                logw[g] = torch.distributions.MultivariateNormal(
                    torch.zeros(xc.numel()), Cy
                ).log_prob(yc)
            else:
                S, m = torch.eye(2) * p2, torch.zeros(2)
            B = self.Bq[g]  # (M,2)
            mu[g] = B @ m
            cov[g] = B @ S @ B.T
        w = torch.softmax(logw, 0)
        return w, mu, cov

    def ground_truth(self, xc, yc):
        w, mu, cov = self.posterior(xc, yc)
        mean = (w[:, None] * mu).sum(0)
        muc = mu - mean[None]
        # total cov = E[S] + Cov(mu)  (law of total covariance)
        cov_epi = (w[:, None, None] * (cov + muc[:, :, None] * muc[:, None, :])).sum(0)
        cov_epi = 0.5 * (cov_epi + cov_epi.T)
        cov_ale = self.sigma**2 * torch.eye(self.M)
        return {
            "cov_epi": cov_epi,
            "cov_ale": cov_ale,
            "cov_total": cov_epi + cov_ale,
            "mean": mean,
            "mix_w": w,
            "mix_mu": mu,
        }

    @torch.no_grad()
    def sample_f(self, xc, yc, n, generator=None):
        """Sample f(xq) from the (non-Gaussian) posterior mixture."""
        w, mu, cov = self.posterior(xc, yc)
        gidx = torch.multinomial(w, n, replacement=True, generator=generator)
        L = torch.linalg.cholesky(cov[gidx] + 1e-5 * torch.eye(self.M))
        return mu[gidx] + torch.einsum(
            "nij,nj->ni", L, torch.randn(n, self.M, generator=generator)
        )

    @torch.no_grad()
    def sample_context(self, n_ctx, generator=None):
        a1 = torch.randn((), generator=generator) * self.prior_std
        a2 = torch.randn((), generator=generator) * self.prior_std
        xi = -1.0 + 2.0 * torch.rand((), generator=generator)
        f = lambda x: torch.where(x <= xi, a1, a2)  # noqa: E731
        xc = -1.0 + 2.0 * torch.rand(n_ctx, generator=generator)
        yc = f(xc) + self.sigma * torch.randn(n_ctx, generator=generator)
        fq = f(self.xq)
        return xc, yc, fq

    @torch.no_grad()
    def nongaussianity(self, xc, yc, n=20000, generator=None):
        """Excess kurtosis of f at the query point of max posterior variance --
        a scalar > 0 signals a non-Gaussian (here multimodal) posterior."""
        s = self.sample_f(xc, yc, n, generator)
        k = s.var(0).argmax()
        x = s[:, k]
        xc_ = x - x.mean()
        return (xc_.pow(4).mean() / xc_.pow(2).mean() ** 2 - 3.0).item()
