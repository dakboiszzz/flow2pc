"""Efficiency of three ways to get pair UQ, by literature-standard metrics:
NFE (flow/diffusion standard), FLOPs (PyTorch official counter, realistic neural
teacher), and measured wall-clock. PC circuit size via the tractable-models
edge-count convention.

Key point: MeanFlow uses 1 NFE/sample vs integration's K -- so MeanFlow is the
faster SAMPLER. Both are dwarfed by the PC, which needs 0 NFE (closed-form exact
moments). On this toy the teacher velocity is analytic (cheap); FLOPs below assume
a REAL neural teacher (velocity net ~ MeanFlow size), the setting that matters.
"""
import sys
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_meanflow_baseline import MeanFlow

from src.pair_pc import PairPC
from src.pair_teacher import PairTeacher

DEV = "cpu"


def neural_flops(module, *inp):
    fc = FlopCounterMode(display=False)
    with fc, torch.no_grad():
        module(*inp)
    return fc.get_total_flops()


def timed(fn, reps=20):
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1000  # ms


def main():
    d, M = 4, 96
    teacher = PairTeacher(d=4, K=6, sigma_a=0.5, seed=0)
    pc = PairPC(d=d, M=M, order=4)
    mf = MeanFlow(2 * d)
    net_flops = neural_flops(mf, torch.randn(1, 2 * d), torch.tensor(1.0))  # per eval

    def pc_q():
        return pc.moments(1.0)

    def mf_q(N):
        z0 = torch.randn(N, 2 * d)
        z1 = z0 + mf(z0, torch.tensor(1.0))
        x1, x2 = z1[:, :d], z1[:, d:]
        _ = (x1 - x1.mean(0)).T @ (x2 - x2.mean(0)) / N

    def fi_q(N, K):
        z1 = torch.randn(N, d)
        z2 = torch.randn(N, d)
        for i in range(K):
            v1, v2 = teacher.pair_velocity(z1, z2, (i + 0.5) / K)
            z1 = z1 + v1 / K
            z2 = z2 + v2 / K
        _ = (z1 - z1.mean(0)).T @ (z2 - z2.mean(0)) / N

    print(f"(realistic neural teacher: velocity net ~ {net_flops:,} FLOPs/eval)\n")
    print(f"{'method':<24} {'NFE':>10} {'FLOPs (neural teach)':>22} {'wall-clock':>12} {'exact?':>7}")
    print(f"{'compiled PC':<24} {'0':>10} {'0 (|PC|=' + str(M*d+M) + ' edges)':>22} "
          f"{timed(pc_q):>10.3f}ms {'YES':>7}")
    for N in (2000, 20000):
        f = N * net_flops                         # 1 NFE/sample * net
        print(f"{'MeanFlow (N=' + str(N) + ')':<24} {N:>10,} {f:>22,} "
              f"{timed(lambda N=N: mf_q(N)):>10.2f}ms {'no (MC)':>7}")
    for N, K in ((2000, 20), (50000, 100)):
        f = N * K * net_flops                      # K NFE/sample * net
        print(f"{'flow-integ N=' + str(N) + ',K=' + str(K):<24} {N*K:>10,} {f:>22,} "
              f"{timed(lambda N=N, K=K: fi_q(N, K)):>10.2f}ms {'no (MC)':>7}")
    print("\nNFE: MeanFlow (1/sample) < integration (K/sample) -- MeanFlow is the faster")
    print("sampler, as designed. Both >> the PC, which is 0-NFE, exact, and amortized.")
    print("(Toy wall-clock for integration is low only because its velocity is analytic;")
    print("the FLOPs column uses a realistic neural teacher, where integration = K x MeanFlow.)")


if __name__ == "__main__":
    main()
