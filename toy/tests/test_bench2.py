"""Verify the GP oracle (ground truth for Benchmark 2) before building on it.

1. epistemic contracts: mean posterior-cov trace decreases with context size
2. local epistemic: query variance inside an unobserved hole exceeds outside
3. noiseless pinning: with sigma->0 and a context point at a query location,
   posterior variance there -> 0 (a noiseless observation pins the function)
4. PSD: posterior covariance is symmetric positive semidefinite
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gp_oracle import GPOracle  # noqa: E402


def test_epistemic_contracts_with_context():
    xq = torch.linspace(-2, 2, 8)
    gp = GPOracle(xq, lengthscale=0.5, sigma=0.1)
    traces = []
    for n in (0, 1, 4, 16, 64):
        g = torch.Generator().manual_seed(0)
        acc = 0.0
        for _ in range(40):
            xc, yc, _ = gp.sample_context(n, generator=g)
            acc += gp.ground_truth(xc, yc)["cov_epi"].trace().item()
        traces.append(acc / 40)
    for a, b in zip(traces, traces[1:], strict=False):
        assert b <= a + 1e-4, f"epistemic did not contract: {traces}"
    assert traces[-1] < 0.5 * traces[0], f"weak contraction: {traces}"


def test_hole_raises_local_epistemic():
    xq = torch.linspace(-2, 2, 9)
    gp = GPOracle(xq, lengthscale=0.4, sigma=0.1)
    hole = (-0.5, 0.5)
    inside = (xq >= hole[0]) & (xq <= hole[1])
    g = torch.Generator().manual_seed(1)
    v_in, v_out = 0.0, 0.0
    for _ in range(40):
        xc, yc, _ = gp.sample_context(30, generator=g, hole=hole)
        var = gp.ground_truth(xc, yc)["cov_epi"].diag()
        v_in += var[inside].mean().item()
        v_out += var[~inside].mean().item()
    assert v_in > 1.5 * v_out, (
        f"hole variance {v_in / 40:.3f} !> outside {v_out / 40:.3f}"
    )


def test_noiseless_pinning():
    xq = torch.linspace(-2, 2, 8)
    gp = GPOracle(xq, lengthscale=0.5, sigma=1e-4)
    xc = xq[3:4].clone()  # observe exactly at query location 3
    yc = torch.tensor([0.7])
    var = gp.ground_truth(xc, yc)["cov_epi"].diag()
    assert var[3] < 1e-3, f"noiseless obs did not pin: var={var[3]:.2e}"


def test_posterior_psd():
    xq = torch.linspace(-2, 2, 8)
    gp = GPOracle(xq, lengthscale=0.5, sigma=0.1)
    g = torch.Generator().manual_seed(2)
    xc, yc, _ = gp.sample_context(10, generator=g)
    cov = gp.ground_truth(xc, yc)["cov_epi"]
    assert torch.allclose(cov, cov.T, atol=1e-5)
    assert torch.linalg.eigvalsh(cov).min() > -1e-5
