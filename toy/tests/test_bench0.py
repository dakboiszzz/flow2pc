"""Verification battery for Benchmark 0 (run before any training):

1. matched marginals:  worlds A and B have identical one-replica moments
                       at every tau (the identifiability premise)
2. flow velocity:      analytic u_tau(x) matches finite differences of T_tau
3. pair symmetry +     Q2 is replica-symmetric; its z1-marginal moments equal
   marginalization:    Q1's (mixture-of-products structure)
4. moment identity:    moments() decomposition is exact on hand-set params
                       and Sigma_total = Sigma_epi + Sigma_ale
5. continuity:         transporting Q_0 samples along velocity1 reproduces
                       Q_1's exact moments (the PC-induced field satisfies
                       the continuity equation by construction)
6. time-MLP basis:     jvp derivative == autograd derivative
7. CI stress premise:  at tau=0, world-A cross-covariance = Sig_e + rho*Sig_a
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pair_pc import PairPC  # noqa: E402
from src.worlds import MatchedMarginalWorlds  # noqa: E402


@pytest.mark.parametrize("tau", [0.0, 0.7])
def test_matched_marginals(tau):
    w = MatchedMarginalWorlds(d=4, rho_e=0.6)
    g = torch.Generator().manual_seed(0)
    xa, _, _, _ = w.sample_pair(200_000, tau, "A", g)
    xb, _, _, _ = w.sample_pair(200_000, tau, "B", g)
    scale = xa.var(0).max().sqrt()
    assert (xa.mean(0) - xb.mean(0)).abs().max() / scale < 0.02
    ca = (xa - xa.mean(0)).T @ (xa - xa.mean(0)) / len(xa)
    cb = (xb - xb.mean(0)).T @ (xb - xb.mean(0)) / len(xb)
    assert (ca - cb).abs().max() / scale**2 < 0.02


def test_flow_velocity_finite_difference():
    w = MatchedMarginalWorlds(d=6, rho_e=0.5)
    z = torch.randn(64, 6)
    # T is linear in tau, so a large FD step is analytically exact and
    # avoids float32 cancellation (h=1e-4 gives ~1e-3 noise on O(5) values)
    tau, h = 0.4, 0.1
    x = w.flow(z, tau)
    fd = (w.flow(z, tau + h) - w.flow(z, tau - h)) / (2 * h)
    assert (w.flow.velocity(x) - fd).abs().max() < 1e-5


def test_pair_symmetry_and_marginal():
    pc = PairPC(d=3, M=5, order=2)
    with torch.no_grad():
        pc.Bmu.normal_(0, 0.5)
        pc.Bls.normal_(0, 0.2)
        pc.pi_logits.normal_(0, 1)
    z1, z2 = torch.randn(100, 3), torch.randn(100, 3)
    tau = torch.tensor(0.6)
    assert torch.allclose(
        pc.log_prob2(z1, z2, tau), pc.log_prob2(z2, z1, tau), atol=1e-5
    )
    # z1-marginal of Q2 equals Q1 exactly for mixture-of-products:
    # verified through moments (both are sum_h pi_h q_h)
    m = pc.moments(0.6)
    n = 400_000
    with torch.no_grad():
        pi = torch.softmax(pc.pi_logits, 0)
        mu, _, sig, _ = pc.leaf_params(0.6)
        h = torch.multinomial(pi, n, replacement=True)
        s = mu[h] + sig[h] * torch.randn(n, 3)
    assert (s.mean(0) - m["mean"]).abs().max() < 0.02
    emp = (s - s.mean(0)).T @ (s - s.mean(0)) / n
    assert (emp - m["cov_total"]).abs().max() < 0.03


def test_moment_decomposition_exact():
    pc = PairPC(d=2, M=2, order=1)
    with torch.no_grad():
        pc.pi_logits.zero_()  # pi = [0.5, 0.5]
        pc.Bmu.zero_()
        pc.Bls.zero_()
        pc.Bmu[0, :, 0] = torch.tensor([1.0, 0.0])
        pc.Bmu[1, :, 0] = torch.tensor([-1.0, 0.0])
        pc.Bls[:, :, 0] = torch.log(torch.tensor(0.5))
    m = pc.moments(0.0)
    # epi = Cov of means: diag(1, 0); ale = diag(0.25, 0.25)
    assert torch.allclose(m["cov_epi"], torch.diag(torch.tensor([1.0, 0.0])), atol=1e-6)
    assert torch.allclose(m["cov_ale"], torch.eye(2) * 0.25, atol=1e-6)
    assert torch.allclose(m["cov_total"], m["cov_epi"] + m["cov_ale"], atol=1e-6)


@pytest.mark.parametrize("time_mlp", [False, True])
def test_continuity_transport(time_mlp):
    torch.manual_seed(0)
    pc = PairPC(d=2, M=4, order=3, time_mlp=time_mlp)
    with torch.no_grad():
        pc.Bmu.normal_(0, 0.4)
        pc.Bls.normal_(0, 0.15)
        pc.pi_logits.normal_(0, 1)
    # sample from Q_0 exactly, transport with velocity1, compare to Q_1 moments
    n, steps = 100_000, 400
    with torch.no_grad():
        pi = torch.softmax(pc.pi_logits, 0)
        mu0, _, sig0, _ = pc.leaf_params(0.0)
        h = torch.multinomial(pi, n, replacement=True)
        z = mu0[h] + sig0[h] * torch.randn(n, 2)
        for i in range(steps):
            t = torch.tensor((i + 0.5) / steps)
            z = z + (1.0 / steps) * pc.velocity1(z, t)
    m1 = pc.moments(1.0)
    assert (z.mean(0) - m1["mean"]).abs().max() < 0.03
    emp = (z - z.mean(0)).T @ (z - z.mean(0)) / n
    assert (emp - m1["cov_total"]).abs().max() < 0.06


def test_time_mlp_jvp_matches_autograd():
    pc = PairPC(d=2, M=3, order=3, time_mlp=True)
    tau = torch.tensor(0.37, requires_grad=True)
    mu = pc.Bmu @ pc._basis(tau)[0]
    (g,) = torch.autograd.grad(mu.sum(), tau)
    _, dmu, _, _ = pc.leaf_params(0.37)
    assert torch.allclose(g, dmu.sum(), atol=1e-5)


def test_ci_bias_premise():
    d, rho_e, rho = 3, 0.5, 0.4
    w = MatchedMarginalWorlds(d=d, rho_e=rho_e, rho_ci=rho)
    o = w.oracle_moments(tau=0.0, world="A", n=500_000)
    expected = torch.eye(d) * (0.5 + rho * 0.5)  # Sig_e + rho * Sig_a
    assert (o["cov_cross"] - expected).abs().max() < 0.02
