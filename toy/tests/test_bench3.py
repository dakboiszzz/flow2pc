"""Verify the change-point oracle (Benchmark 3 ground truth).

1. moments: analytic mean/cov match Monte-Carlo estimates from sample_f
2. NON-Gaussian: with few context points the posterior over f has clearly
   positive excess kurtosis (multimodal in the change-point location) -- this
   is the whole point of Benchmark 3 (a regime covariance cannot summarize)
3. epistemic contracts as context grows
4. covariance is PSD
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.changepoint_oracle import ChangePointOracle  # noqa: E402


def test_moments_match_montecarlo():
    xq = torch.linspace(-1, 1, 6)
    cp = ChangePointOracle(xq, sigma=0.15, n_grid=201)
    g = torch.Generator().manual_seed(0)
    xc, yc, _ = cp.sample_context(4, generator=g)
    gt = cp.ground_truth(xc, yc)
    s = cp.sample_f(xc, yc, 200000, generator=g)
    assert (s.mean(0) - gt["mean"]).abs().max() < 0.03
    emp = (s - s.mean(0)).T @ (s - s.mean(0)) / len(s)
    assert (emp - gt["cov_epi"]).abs().max() < 0.05


def test_posterior_is_nongaussian():
    xq = torch.linspace(-1, 1, 6)
    cp = ChangePointOracle(xq, sigma=0.15, n_grid=201)
    # average excess kurtosis over several small contexts (xi stays uncertain)
    total = 0.0
    for i in range(20):
        g = torch.Generator().manual_seed(i)
        xc, yc, _ = cp.sample_context(3, generator=g)
        total += cp.nongaussianity(xc, yc, n=20000, generator=g)
    avg = total / 20
    assert avg > 0.3, f"posterior not clearly non-Gaussian: excess kurtosis {avg:.2f}"


def test_epistemic_contracts():
    xq = torch.linspace(-1, 1, 6)
    cp = ChangePointOracle(xq, sigma=0.1, n_grid=201)
    traces = []
    for n in (1, 4, 16, 64):
        acc = 0.0
        for i in range(30):
            g = torch.Generator().manual_seed(1000 + i)
            xc, yc, _ = cp.sample_context(n, generator=g)
            acc += cp.ground_truth(xc, yc)["cov_epi"].trace().item()
        traces.append(acc / 30)
    for a, b in zip(traces, traces[1:], strict=False):
        assert b <= a + 1e-3, f"epistemic did not contract: {traces}"


def test_psd():
    xq = torch.linspace(-1, 1, 6)
    cp = ChangePointOracle(xq, sigma=0.15, n_grid=201)
    g = torch.Generator().manual_seed(3)
    xc, yc, _ = cp.sample_context(8, generator=g)
    cov = cp.ground_truth(xc, yc)["cov_epi"]
    assert torch.allclose(cov, cov.T, atol=1e-5)
    assert torch.linalg.eigvalsh(cov).min() > -1e-4
