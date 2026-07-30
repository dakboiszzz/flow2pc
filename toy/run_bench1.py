"""Benchmark 1: continuity residual + error-propagation certificate.

Two experiments, reported honestly (no cherry-picked tau/seed):

A. Continuity residual across tau in {0.1..0.9} for three velocity fields on
   the SAME density path:
     - PC-rule velocity (correct construction)         -> should be ~0
     - PC-rule velocity + injected error eta*unit      -> should grow with eta
     - time-dependent-weights defect (violates Prop 1) -> should be large
   The claim under test: the PC *construction* is what buys continuity, not
   just fitting the density.

B. Error-propagation certificate. Transport the teacher's own samples along a
   velocity corrupted by a known constant error eta; measure the actual
   endpoint transport error and check it is bounded by the Gronwall-type
   certificate  E = exp(L*tau) * (e0 + integral eta).  Report coverage over
   many trials and median slack (bound/actual). A certificate that is valid
   but 100x loose is reported as such.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.analytic_flow import MovingMixture  # noqa: E402

TAUS = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]  # skip 0.5 (dw degeneracy)


def experiment_A():
    print("=== A. Continuity residual (median over 2000 pts, per tau) ===")
    mm = MovingMixture(J=4, d=2, seed=1, time_dependent_weights=False)
    bad = MovingMixture(J=4, d=2, seed=1, time_dependent_weights=True)
    g = torch.Generator().manual_seed(0)

    unit = torch.randn(mm.d, generator=g)
    unit = unit / unit.norm()

    print(f"{'tau':>5} {'PC-rule':>12} {'PC+eta=0.3':>12} {'tdw-defect':>12}")
    for tau in TAUS:
        z = mm.sample(2000, tau, seed=7)
        r0 = mm.continuity_residual(z, tau).median().item()
        re = mm.continuity_residual(z, tau, velocity_offset=0.3 * unit).median().item()
        zb = bad.sample(2000, tau, seed=7)
        rb = bad.continuity_residual(zb, tau).median().item()
        print(f"{tau:>5} {r0:>12.2e} {re:>12.2e} {rb:>12.2e}")


def experiment_B(n_trials=200):
    print("\n=== B. Error-propagation certificate (transport error vs bound) ===")
    print(f"{'eta':>6} {'coverage':>10} {'median_slack':>13} {'p90_slack':>11}")
    for eta in (0.05, 0.1, 0.2, 0.4):
        covered, slacks = 0, []
        for t in range(n_trials):
            mm = MovingMixture(J=3, d=2, seed=100 + t, time_dependent_weights=False)
            n, steps = 512, 200
            g = torch.Generator().manual_seed(t)
            z0 = mm.sample(n, 0.0, seed=t)
            unit = torch.randn(mm.d, generator=g)
            unit = unit / unit.norm()
            # exact vs corrupted transport
            ze, zc = z0.clone(), z0.clone()
            L = _lipschitz_estimate(mm)
            with torch.no_grad():
                for i in range(steps):
                    tt = (i + 0.5) / steps
                    ze = ze + (1.0 / steps) * mm.velocity(ze, tt)
                    zc = zc + (1.0 / steps) * (mm.velocity(zc, tt) + eta * unit)
            actual = (zc - ze).norm(dim=-1).mean().item()
            bound = torch.exp(torch.tensor(L * 1.0)).item() * (0.0 + eta * 1.0)
            if bound >= actual:
                covered += 1
            slacks.append(bound / (actual + 1e-9))
        slacks = torch.tensor(slacks)
        print(
            f"{eta:>6} {covered / n_trials:>10.2f} "
            f"{slacks.median():>13.2f} {slacks.quantile(0.9):>11.2f}"
        )


def _lipschitz_estimate(mm, n=400, seed=0):
    """Empirical Lipschitz constant of the velocity field over the support."""
    g = torch.Generator().manual_seed(seed)
    z = mm.sample(n, 0.5, seed=seed)
    dz = 0.01 * torch.randn(n, mm.d, generator=g)
    du = (mm.velocity(z + dz, 0.5) - mm.velocity(z, 0.5)).norm(dim=-1)
    return (du / (dz.norm(dim=-1) + 1e-9)).quantile(0.95).item()


if __name__ == "__main__":
    experiment_A()
    experiment_B()
