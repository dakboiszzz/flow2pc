"""Benchmark 1 verification: continuity theorem and its precondition.

1. time-INDEPENDENT weights + PC-rule velocity => continuity residual ~ 0
   (Proposition 1 holds to numerical precision)
2. time-DEPENDENT weights => residual is large (the precondition is
   necessary, not decorative)
3. velocity transports the analytic density correctly (endpoint check)
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analytic_flow import MovingMixture  # noqa: E402


def test_continuity_residual_near_zero():
    mm = MovingMixture(J=4, d=2, seed=1, time_dependent_weights=False)
    z = mm.sample(2000, 0.5, seed=3)
    for tau in (0.2, 0.5, 0.8):
        r = mm.continuity_residual(z, tau)
        assert r.median() < 1e-3, f"tau={tau}: median residual {r.median():.2e}"


def test_time_dependent_weights_break_continuity():
    good = MovingMixture(J=4, d=2, seed=1, time_dependent_weights=False)
    bad = MovingMixture(J=4, d=2, seed=1, time_dependent_weights=True)
    # tau=0.3: cos(pi*tau) != 0 so the weight drift dw is nonzero (at tau=0.5
    # the defect vanishes exactly, cos(pi/2)=0 -- not a valid test point)
    z = good.sample(2000, 0.3, seed=3)
    rg = good.continuity_residual(z, 0.3).median()
    rb = bad.continuity_residual(z, 0.3).median()
    assert rb > 20 * rg, f"tdw residual {rb:.2e} not >> time-indep {rg:.2e}"


def test_velocity_transports_density():
    mm = MovingMixture(J=3, d=2, seed=2, time_dependent_weights=False)
    n, steps = 40000, 300
    z = mm.sample(n, 0.0, seed=5)
    with torch.no_grad():
        for i in range(steps):
            z = z + (1.0 / steps) * mm.velocity(z, (i + 0.5) / steps)
    ll_tr = mm.log_prob(z, torch.tensor(1.0)).mean()
    direct = mm.sample(n, 1.0, seed=6)
    ll_dir = mm.log_prob(direct, torch.tensor(1.0)).mean()
    assert abs(ll_tr - ll_dir) < 0.15, f"transport {ll_tr:.3f} vs direct {ll_dir:.3f}"
