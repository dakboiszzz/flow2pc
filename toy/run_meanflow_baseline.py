"""MeanFlow baseline: learn the pair flow's AVERAGE velocity so UQ needs only a
FEW (here one) function evals to reach the endpoint, instead of a long Euler
integration. Then MC-estimate the covariances from one-step samples.

Compared to the compiled PC (exact moments, 0 NFE, no sampling), MeanFlow is the
strong 'fast-integration' competitor: it amortizes to fast SAMPLING, but still
pays 1 NFE * N samples per query and carries MC noise -- it never returns exact
moments. Forward average-velocity form (noise t=0 -> data t=1):
    a(z,t) = (z_t - z_0)/t,  identity: a + t d_t a = V(z_0 + t a, t)
    one-step sample: z_1 = z_0 + a(z_0, 1).
"""
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.pair_teacher import PairTeacher

DEV = "cuda:0"


class MeanFlow(nn.Module):
    def __init__(self, d2, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d2 + 1, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, d2))

    def forward(self, z, t):
        te = t.expand(z.shape[0], 1) if t.dim() else t.reshape(1, 1).expand(z.shape[0], 1)
        return self.net(torch.cat([z, te], -1))


def teacher_pair_vel_cat(teacher, zcat, t):
    d = teacher.d
    v1, v2 = teacher.pair_velocity(zcat[:, :d], zcat[:, d:], float(t))
    return torch.cat([v1, v2], -1)


def train_meanflow(teacher, steps=30000, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    d2 = 2 * teacher.d
    mf = MeanFlow(d2).to(DEV)
    opt = torch.optim.Adam(mf.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _step in range(steps):
        z0 = torch.randn(512, d2, device=DEV)
        t = torch.rand((), device=DEV) * 0.98 + 0.02
        tt = t.clone().requires_grad_(True)
        a = mf(z0, tt)
        (da_dt,) = torch.autograd.grad(a.sum(), tt, create_graph=True)
        with torch.no_grad():
            zt = z0 + t * a.detach()
            v = teacher_pair_vel_cat(teacher, zt.cpu(), t).to(DEV)
        loss = ((a + t * da_dt - v) ** 2).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mf.parameters(), 1.0)
        opt.step()
        sched.step()
    return mf


@torch.no_grad()
def mf_moments(mf, teacher, N):
    d = teacher.d
    z0 = torch.randn(N, 2 * d, device=DEV)
    z1 = z0 + mf(z0, torch.tensor(1.0, device=DEV))  # one-step
    x1, x2 = z1[:, :d].cpu(), z1[:, d:].cpu()
    m1, m2 = x1.mean(0), x2.mean(0)
    epi = (x1 - m1).T @ (x2 - m2) / N
    tot = (x1 - m1).T @ (x1 - m1) / N
    return {"cov_epi": 0.5 * (epi + epi.T), "cov_ale": tot - epi}


def main():
    teacher = PairTeacher(d=4, K=6, sigma_a=0.5, seed=0)
    gt = teacher.moments(1.0)

    def rel(a, b):
        return float(torch.linalg.norm(a - b) / (torch.linalg.norm(b) + 1e-8))

    print(f"teacher epi_tr {gt['cov_epi'].trace():.3f} ale_tr {gt['cov_ale'].trace():.3f}")
    t0 = time.time()
    mf = train_meanflow(teacher)
    tr = time.time() - t0
    print(f"MeanFlow trained ({tr:.0f}s)")

    # verify: one-step MeanFlow moments vs ground truth (large N to isolate model err)
    m = mf_moments(mf, teacher, 200000)
    print(f"VERIFY one-step: err_epi {rel(m['cov_epi'], gt['cov_epi']):.3f} "
          f"err_ale {rel(m['cov_ale'], gt['cov_ale']):.3f} ale_tr {m['cov_ale'].trace():.2f}")

    # inference timing at a practical N (MC noise included)
    for N in (2000, 20000):
        torch.cuda.synchronize() if DEV.startswith("cuda") else None
        t0 = time.perf_counter()
        for _ in range(20):
            m = mf_moments(mf, teacher, N)
        torch.cuda.synchronize() if DEV.startswith("cuda") else None
        ms = (time.perf_counter() - t0) / 20 * 1000
        print(f"MeanFlow UQ (1-step, N={N}): {ms:.2f} ms/query, {N} NFE, "
              f"err_epi {rel(m['cov_epi'], gt['cov_epi']):.3f} "
              f"err_ale {rel(m['cov_ale'], gt['cov_ale']):.3f}")


if __name__ == "__main__":
    main()
