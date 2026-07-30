"""Stage-1 decisive ablations: matched architecture/steps/samples/compute.

Configs: V (velocity only) | D (density only) | VD0.1 / VD0.3 / VD1.0
(velocity + lam*density) | BV (best broadened velocity, b=0.25).
5 seeds each. All: M96, order 4, beta=2 tau-weighting, s=0.3 partial-stopgrad
on the velocity term, lr 5e-4 cosine, 150k steps, batch 512.

Metrics at the end of training: err_epi, err_ale, ale_tr, veldef (teacher-path
velocity fidelity), pair NLL on held-out teacher samples at tau in
{0.25,0.5,0.75,1.0} (per-dim, lower better).

Usage: run_ablation_seeds.py <config> [<config> ...]   (one process per GPU)
"""

import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.pair_pc import PairPC
from src.pair_teacher import PairTeacher

DEV = "cuda:0"
teacher = PairTeacher(d=4, K=6, sigma_a=0.5, seed=0)
gt = teacher.moments(1.0)
rel = lambda a, b: float(torch.linalg.norm(a - b) / (torch.linalg.norm(b) + 1e-8))  # noqa: E731

CONFIGS = {
    "V": dict(vel=True, den=False, lam=0.0, b=0.0),
    "D": dict(vel=False, den=True, lam=1.0, b=0.0),
    "VD0.1": dict(vel=True, den=True, lam=0.1, b=0.0),
    "VD0.3": dict(vel=True, den=True, lam=0.3, b=0.0),
    "VD1.0": dict(vel=True, den=True, lam=1.0, b=0.0),
    "BV": dict(vel=True, den=False, lam=0.0, b=0.25),
}


@torch.no_grad()
def veldef(pc):
    tot = 0.0
    for tau in (0.2, 0.4, 0.6, 0.8):
        z1, z2 = teacher.sample_pair(
            4000, tau, generator=torch.Generator().manual_seed(99)
        )
        v1, v2 = teacher.pair_velocity(z1, z2, tau)
        u1, u2 = pc.velocity2(z1.to(DEV), z2.to(DEV), torch.tensor(tau, device=DEV))
        tot += (
            (((u1.cpu() - v1) ** 2 + (u2.cpu() - v2) ** 2).sum(-1).sqrt()).mean().item()
        )
    return tot / 4


@torch.no_grad()
def pair_nll(pc):
    """Held-out per-dim pair NLL averaged over tau grid (state fidelity)."""
    tot = 0.0
    taus = (0.25, 0.5, 0.75, 1.0)
    for tau in taus:
        z1, z2 = teacher.sample_pair(
            8000, tau, generator=torch.Generator().manual_seed(7)
        )
        lp = pc.log_prob2(z1.to(DEV), z2.to(DEV), torch.tensor(tau, device=DEV))
        tot += (-lp.mean().item()) / (2 * teacher.d)
    return tot / len(taus)


def run(cfg, seed, steps=150000, M=96, s=0.3, beta=2):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 1)
    pc = PairPC(d=teacher.d, M=M, order=4).to(DEV)
    opt = torch.optim.Adam(pc.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _step in range(steps):
        u = float(torch.rand(1, generator=g))
        tau = -math.log(1 - u * (1 - math.exp(-beta))) / beta
        with torch.no_grad():
            z1, z2 = teacher.sample_pair(512, tau)
            if cfg["b"] > 0:
                z1 = z1 + cfg["b"] * torch.randn(z1.shape)
                z2 = z2 + cfg["b"] * torch.randn(z2.shape)
            v1, v2 = teacher.pair_velocity(z1, z2, tau)
            z1, z2, v1, v2 = z1.to(DEV), z2.to(DEV), v1.to(DEV), v2.to(DEV)
        t = torch.tensor(tau, device=DEV)
        loss = 0.0
        if cfg["vel"]:
            u1, u2 = pc.velocity2(z1, z2, t, resp_grad_scale=s)
            loss = loss + 0.5 * ((u1 - v1) ** 2 + (u2 - v2) ** 2).sum(-1).mean()
        if cfg["den"]:
            loss = loss + cfg["lam"] * (-pc.log_prob2(z1, z2, t).mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pc.parameters(), 1.0)
        opt.step()
        sched.step()
    m = {k: v.cpu() for k, v in pc.moments(1.0).items()}
    return dict(
        err_epi=rel(m["cov_epi"], gt["cov_epi"]),
        err_ale=rel(m["cov_ale"], gt["cov_ale"]),
        ale_tr=float(m["cov_ale"].trace()),
        veldef=veldef(pc),
        nll=pair_nll(pc),
    )


def main():
    names = sys.argv[1:] or list(CONFIGS)
    print(f"gt epi_tr {gt['cov_epi'].trace():.3f} ale_tr {gt['cov_ale'].trace():.3f}")
    for name in names:
        cfg = CONFIGS[name]
        rows = []
        for seed in range(5):
            t0 = time.time()
            r = run(cfg, seed)
            rows.append(r)
            print(
                f"{name:>6} seed {seed}: err_epi {r['err_epi']:.4f} "
                f"err_ale {r['err_ale']:.4f} ale_tr {r['ale_tr']:.3f} "
                f"veldef {r['veldef']:.4f} nll {r['nll']:.4f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
        agg = {k: torch.tensor([r[k] for r in rows]) for k in rows[0]}
        print(
            f"{name:>6} MEAN+-STD: "
            + " ".join(f"{k} {v.mean():.4f}+-{v.std():.4f}" for k, v in agg.items()),
            flush=True,
        )


if __name__ == "__main__":
    main()
