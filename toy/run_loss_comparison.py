"""One-loss vs two-loss: compile a student pair PC into a genuine two-replica
teacher and compare which training objective recovers the UQ decomposition.

Modes (all reconfigure the SAME pair teacher, boundary p_0=N(0,I) structural):
  velocity_only : Plan A -- L = ||V^[2] - U^[2]||^2               (one loss)
  density_only  : L = -log Q^[2]                                  (one loss)
  density+vel   : Plan B -- L = -log Q^[2] + lambda ||V-U||^2     (two loss)

Metric: student epistemic Cov_h(mu_h) at tau=1 vs teacher's exact epistemic.
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.pair_pc import PairPC
from src.pair_teacher import PairTeacher

DEV = "cuda:0"


def train(mode, teacher, M=32, steps=15000, lr=1e-3, lam=1.0, seed=0):
    torch.manual_seed(seed)
    pc = PairPC(d=teacher.d, M=M, order=4).to(DEV)
    opt = torch.optim.Adam(pc.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _step in range(steps):
        tau = float(torch.rand(1))
        with torch.no_grad():
            z1, z2 = teacher.sample_pair(512, tau)  # teacher on CPU
            v1, v2 = teacher.pair_velocity(z1, z2, tau)
            z1, z2 = z1.to(DEV), z2.to(DEV)
            v1, v2 = v1.to(DEV), v2.to(DEV)
        t = torch.tensor(tau, device=DEV)
        loss = 0.0
        if mode in ("density_only", "density+vel"):
            loss = loss - pc.log_prob2(z1, z2, t).mean()
        if mode in ("velocity_only", "density+vel"):
            u1, u2 = pc.velocity2(z1, z2, t)
            vloss = ((u1 - v1) ** 2 + (u2 - v2) ** 2).sum(-1).mean()
            loss = loss + (lam * vloss if mode == "density+vel" else vloss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pc.parameters(), 1.0)
        opt.step()
        sched.step()
    return pc


@torch.no_grad()
def velocity_defect(pc, teacher):
    """Mean ||V_teacher^[2] - U_student^[2]|| over teacher pairs, tau in (.2..8).
    Low => density fitting implicitly reconfigured the flow (velocity matches)."""
    tot, n = 0.0, 0
    for tau in (0.2, 0.4, 0.6, 0.8):
        z1, z2 = teacher.sample_pair(2000, tau)
        v1, v2 = teacher.pair_velocity(z1, z2, tau)
        t = torch.tensor(tau, device=DEV)
        u1, u2 = pc.velocity2(z1.to(DEV), z2.to(DEV), t)
        d = ((u1.cpu() - v1) ** 2 + (u2.cpu() - v2) ** 2).sum(-1).sqrt()
        tot += d.mean().item()
        n += 1
    return tot / n


def main():
    teacher = PairTeacher(d=4, K=6, sigma_a=0.5, seed=0)
    gt = teacher.moments(1.0)
    eps = 1e-8

    def rel(a, b):
        return float(torch.linalg.norm(a - b) / (torch.linalg.norm(b) + eps))

    print(
        f"teacher: epistemic trace {gt['cov_epi'].trace():.3f}, "
        f"aleatoric trace {gt['cov_ale'].trace():.3f}"
    )
    print(
        f"{'mode':<16} {'losses':>7} {'err_epi':>9} {'err_ale':>9} "
        f"{'epi_tr':>7} {'vel_defect':>11}"
    )
    for mode, nloss in (("density_only", 1), ("velocity_only", 1), ("density+vel", 2)):
        t0 = time.time()
        pc = train(mode, teacher, M=32, steps=15000)
        m = {k: v.cpu() for k, v in pc.moments(1.0).items()}
        vdef = velocity_defect(pc, teacher)
        print(
            f"{mode:<16} {nloss:>7} {rel(m['cov_epi'], gt['cov_epi']):>9.3f} "
            f"{rel(m['cov_ale'], gt['cov_ale']):>9.3f} "
            f"{m['cov_epi'].trace():>7.2f} {vdef:>11.3f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
