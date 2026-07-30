"""Verify the two-replica teacher: boundary, moments, continuity, and the key
property that the PAIR velocity depends on both replicas (carries epistemic
signal a one-replica velocity cannot)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from torch.distributions import Independent, Normal  # noqa: E402

from src.pair_teacher import PairTeacher  # noqa: E402


def test_boundary_standard_normal():
    t = PairTeacher(d=3, K=5, seed=1)
    z = torch.randn(500, 3)
    ref = Independent(Normal(torch.zeros(3), torch.ones(3)), 1)
    # marginal at tau=0 = N(0,I): log_prob2(z,z2,0) marginalizes to it; check leaf
    lp = torch.logsumexp(torch.log(t.pi)[None] + t.leaf_logp(z, 0.0), 1)
    assert (lp - ref.log_prob(z)).abs().max() < 1e-4


def test_moments_match_montecarlo():
    t = PairTeacher(d=4, K=6, sigma_a=0.5, seed=2)
    g = torch.Generator().manual_seed(0)
    z1, z2 = t.sample_pair(400000, 1.0, generator=g)
    m = t.moments(1.0)
    cross = (z1 - z1.mean(0)).T @ (z2 - z2.mean(0)) / len(z1)
    assert (cross - m["cov_epi"]).abs().max() < 0.03, "epistemic = cross-cov"
    tot = (z1 - z1.mean(0)).T @ (z1 - z1.mean(0)) / len(z1)
    assert (tot - m["cov_total"]).abs().max() < 0.05


def test_pair_velocity_uses_both_replicas():
    """The KEY property: V^[2]_1(z1,z2) depends on z2 (joint responsibility) --
    this is the epistemic signal absent from a one-replica velocity."""
    t = PairTeacher(d=3, K=6, seed=3)
    z1 = torch.randn(200, 3)
    za, zb = torch.randn(200, 3), torch.randn(200, 3)
    v_a, _ = t.pair_velocity(z1, za, 0.6)
    v_b, _ = t.pair_velocity(z1, zb, 0.6)
    # same z1, different partner -> different velocity for replica 1
    assert (v_a - v_b).norm(dim=-1).mean() > 0.05, "pair velocity ignores partner"


def test_continuity_transport():
    """Transport p_0 along the pair velocity -> reproduces the pair law at t=1."""
    t = PairTeacher(d=2, K=4, seed=4)
    n, steps = 40000, 300
    g = torch.Generator().manual_seed(5)
    z1 = torch.randn(n, 2, generator=g)
    z2 = torch.randn(n, 2, generator=g)
    with torch.no_grad():
        for i in range(steps):
            tau = (i + 0.5) / steps
            v1, v2 = t.pair_velocity(z1, z2, tau)
            z1 = z1 + v1 / steps
            z2 = z2 + v2 / steps
    ll_tr = t.log_prob2(z1, z2, 1.0).mean()
    d1, d2 = t.sample_pair(n, 1.0, generator=g)
    ll_dir = t.log_prob2(d1, d2, 1.0).mean()
    assert abs(ll_tr - ll_dir) < 0.2, f"transport {ll_tr:.3f} vs direct {ll_dir:.3f}"
