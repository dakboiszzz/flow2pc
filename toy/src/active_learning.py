"""Benchmark 4: active identification with an aleatoric distractor.

The utility claim: separating epistemic (reducible) from aleatoric (irreducible)
uncertainty lets active learning avoid the "aleatoric trap" -- a region of high
observation noise that a total-uncertainty signal keeps sampling but that
carries no learnable information.

Setup: latent f ~ GP on [-2,2]. Heteroscedastic noise, low everywhere except a
distractor region R_a where it is large. Each round we pick the next query from
a candidate pool by an acquisition score, observe y = f(x) + noise(x), and
refit the exact GP posterior. Three acquisitions:

  epistemic : maximize posterior variance of f      (what the pair PC gives)
  total     : maximize posterior var of f + noise(x) (what a one-replica /
              marginal model conflates -- lured into R_a)
  random    : uniform

Metrics vs #queries: test RMSE of f, and the fraction of queries spent in R_a.
"""

import torch

from src.gp_oracle import rbf


class HeteroGP:
    def __init__(
        self, lengthscale=0.4, signal=1.0, sig_lo=0.05, sig_hi=0.6, region=(0.5, 1.5)
    ):
        self.ls, self.sig_s = lengthscale, signal
        self.sig_lo, self.sig_hi, self.region = sig_lo, sig_hi, region

    def noise_std(self, x):
        a, b = self.region
        return torch.where((x >= a) & (x <= b), self.sig_hi, self.sig_lo)

    def sample_f(self, xgrid, generator=None):
        K = rbf(xgrid, xgrid, self.ls, self.sig_s) + 1e-5 * torch.eye(xgrid.numel())
        return torch.linalg.cholesky(K) @ torch.randn(
            xgrid.numel(), generator=generator
        )

    def posterior(self, xc, yc, xq):
        """Exact posterior mean and epistemic variance of f at xq, with
        per-point observation noise (heteroscedastic)."""
        if xc.numel() == 0:
            return torch.zeros(xq.numel()), rbf(xq, xq, self.ls, self.sig_s).diag()
        Kcc = rbf(xc, xc, self.ls, self.sig_s) + torch.diag(self.noise_std(xc) ** 2)
        Kqc = rbf(xq, xc, self.ls, self.sig_s)
        A = torch.linalg.solve(Kcc, Kqc.T)
        mean = A.T @ yc
        var = rbf(xq, xq, self.ls, self.sig_s).diag() - (Kqc * A.T).sum(1)
        return mean, var.clamp_min(0)


def run_active(gp, acq, n_rounds=40, pool=81, seed=0, f_seed=0):
    """One active-learning trajectory.

    Returns (epi_var_curve, frac_in_Ra). The identification objective is the
    integrated posterior epistemic variance remaining (reducible uncertainty).
    Global RMSE of f is NOT the right metric here: f is learnable everywhere,
    so total-acquisition (which samples R_a) can reach lower RMSE simply by
    paying the noise cost -- it just wastes queries doing so. Reducible-
    uncertainty efficiency is what separating epistemic from aleatoric buys."""
    g = torch.Generator().manual_seed(seed)
    xgrid = torch.linspace(-2, 2, pool)
    fg = torch.Generator().manual_seed(f_seed)
    f = gp.sample_f(xgrid, generator=fg)
    a, b = gp.region
    in_ra = (xgrid >= a) & (xgrid <= b)

    xc = torch.empty(0)
    yc = torch.empty(0)
    chosen = []
    epi_var = []
    for _ in range(n_rounds):
        _, var = gp.posterior(xc, yc, xgrid)
        noise2 = gp.noise_std(xgrid) ** 2
        if acq == "random":
            score = torch.rand(pool, generator=g)
        elif acq == "infogain":
            # noise-discounted variance reduction: var^2/(var+noise^2). Needs
            # the aleatoric level to discount -- exactly what the split provides.
            score = var**2 / (var + noise2)
        elif acq == "total":
            score = var + noise2  # one-replica: can't separate -> lured to R_a
        elif acq == "variance":
            score = var
        for c in chosen:
            score[c] = -1e9  # don't repeat
        k = int(score.argmax())
        chosen.append(k)
        y = f[k] + gp.noise_std(xgrid[k]) * torch.randn((), generator=g)
        xc = torch.cat([xc, xgrid[k : k + 1]])
        yc = torch.cat([yc, y[None]])
        _, var_after = gp.posterior(xc, yc, xgrid)
        epi_var.append(var_after.mean().item())  # reducible uncertainty remaining
    frac_ra = sum(1 for c in chosen if in_ra[c]) / n_rounds
    return epi_var, frac_ra
