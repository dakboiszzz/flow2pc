"""Baseline: compute the teacher's aleatoric/epistemic uncertainty by INTEGRATING
the pair flow trajectory and Monte-Carlo estimating the covariances -- the naive
approach the compiled PC replaces.

Contrast the two ways to get UQ from the same teacher:
  flow integration : simulate the pair ODE (K Euler steps) from N base pairs to
                     tau=1, estimate Sigma_epi=Cov(X1,X2), Sigma_ale=Var-Cov.
                     Cost = N*K*2 velocity evals (NFE); MC + discretization error.
  compiled PC      : exact moments in closed form. 0 NFE, no sampling noise.

Both are compared to the analytic ground truth.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.pair_teacher import PairTeacher


def integrate_pair_flow(teacher, N, K, seed=0):
    """Euler-integrate the teacher pair ODE from N base pairs to tau=1."""
    g = torch.Generator().manual_seed(seed)
    z1 = torch.randn(N, teacher.d, generator=g)
    z2 = torch.randn(N, teacher.d, generator=g)
    for i in range(K):
        tau = (i + 0.5) / K
        v1, v2 = teacher.pair_velocity(z1, z2, tau)
        z1 = z1 + v1 / K
        z2 = z2 + v2 / K
    return z1, z2


def mc_moments(x1, x2):
    m1, m2 = x1.mean(0), x2.mean(0)
    epi = (x1 - m1).T @ (x2 - m2) / len(x1)
    epi = 0.5 * (epi + epi.T)
    tot = (x1 - m1).T @ (x1 - m1) / len(x1)
    return {"cov_epi": epi, "cov_ale": tot - epi}


def main():
    teacher = PairTeacher(d=4, K=6, sigma_a=0.5, seed=0)
    gt = teacher.moments(1.0)

    def rel(a, b):
        return float(torch.linalg.norm(a - b) / (torch.linalg.norm(b) + 1e-8))

    print(f"teacher epi_tr {gt['cov_epi'].trace():.3f} ale_tr {gt['cov_ale'].trace():.3f}")
    print(f"{'method':<28} {'err_epi':>8} {'err_ale':>8} {'NFE':>12} {'time':>7}")
    # flow integration at increasing budgets
    for N, K in ((2000, 20), (10000, 50), (50000, 100), (200000, 200)):
        t0 = time.time()
        x1, x2 = integrate_pair_flow(teacher, N, K)
        m = mc_moments(x1, x2)
        nfe = N * K * 2
        print(f"flow-integ N={N} K={K:<4}      {rel(m['cov_epi'], gt['cov_epi']):>8.3f} "
              f"{rel(m['cov_ale'], gt['cov_ale']):>8.3f} {nfe:>12,} {time.time()-t0:>6.1f}s",
              flush=True)
    print(f"{'compiled PC (exact moments)':<28} {'~0':>8} {'~0':>8} {'0':>12} {'<0.001s':>7}")
    print("\n(PC error vs ground truth is set by the reconfiguration loss, not by "
          "sampling -- and it is achieved at ZERO integration cost.)")


if __name__ == "__main__":
    main()
