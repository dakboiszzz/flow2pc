"""Velocity-observability Gramian experiments (theory refinement).

Claims under test (from the observability analysis):
  P1  single-Gaussian velocity Hessian: lam_mean = 2, lam_gamma = 2*(sigma*)^2
      (log-scale-rate curvature is variance-weighted, NOT tail-only).
  P2  broadening preconditions the scale block: lam_gamma(b) = 2*((sigma*)^2+b^2)
      while lam_mean stays 2.
  P3  on the real K=6 pair teacher, the mixture Gramian G_tau = E[Phi^T Phi]
      has lam_min(scale block) << lam_min(mean block), worsened by overlap
      (small tau) and repaired by broadening.
  F   pair-law check: Sigma_ale = (1/2) E[(Z1-Z2)(Z1-Z2)^T] must equal the
      component formula sum_h pi_h sig_h^2 (sanity for the aleatoric metric).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.pair_teacher import PairTeacher

torch.manual_seed(0)
G = torch.Generator().manual_seed(0)


def gauss_hessian(sig_star, b=0.0, n=400000):
    """Hessian of E||dmu + dgam*(z-mu*)||^2 wrt (dmu, dgam) at 0, z broadened."""
    z = sig_star * torch.randn(n, generator=G)
    zb = z + b * torch.randn(n, generator=G)
    x = zb  # centered at mu*=0
    # loss(a) = E (a0 + a1*x)^2 -> Hessian = 2*E[[1, x],[x, x^2]]
    h11 = 2.0
    h12 = 2.0 * x.mean().item()
    h22 = 2.0 * (x**2).mean().item()
    H = torch.tensor([[h11, h12], [h12, h22]])
    return torch.linalg.eigvalsh(H)


def mixture_gramian(teacher, tau, b=0.0, n=200000):
    """G = E[Phi^T Phi] for the pair velocity, rates a={dmu_h, dgam_h} (2Kd).

    delta U_j = sum_h r_h(z1,z2) [dmu_h + dgam_h * (z_j - tau*mu_h)],
    responsibilities held at the teacher law (theory: rates don't move r).
    Returns (lam_min mean block, lam_min scale block, lam_min full).
    """
    K, d = teacher.K, teacher.d
    z1, z2 = teacher.sample_pair(n, tau, generator=G)
    if b > 0:
        z1 = z1 + b * torch.randn(n, d, generator=G)
        z2 = z2 + b * torch.randn(n, d, generator=G)
    j = teacher.leaf_logp(z1, tau) + teacher.leaf_logp(z2, tau)
    r = torch.softmax(torch.log(teacher.pi)[None] + j, 1)  # (n,K)
    mu = tau * teacher.mu  # (K,d)
    # Phi rows: replica j, dim i. Params per dim i: dmu_{h,i} (coef r_h) and
    # dgam_{h,i} (coef r_h*(z_{j,i}-mu_{h,i})). Dims decouple -> per-dim 2K x 2K
    # Gramians; assemble lam_min across dims.
    lam_mean, lam_scale, lam_full = [], [], []
    for i in range(d):
        c1 = z1[:, i : i + 1] - mu[None, :, i]  # (n,K)
        c2 = z2[:, i : i + 1] - mu[None, :, i]
        # features for replica 1 then replica 2, params [dmu(K), dgam(K)]
        f1 = torch.cat([r, r * c1], 1)  # (n,2K)
        f2 = torch.cat([r, r * c2], 1)
        Gi = (f1.T @ f1 + f2.T @ f2) / n
        lam_full.append(torch.linalg.eigvalsh(Gi)[0].item())
        lam_mean.append(torch.linalg.eigvalsh(Gi[:K, :K])[0].item())
        lam_scale.append(torch.linalg.eigvalsh(Gi[K:, K:])[0].item())
    return min(lam_mean), min(lam_scale), min(lam_full)


def main():
    print("== P1/P2: single-Gaussian velocity Hessian (theory: 2 and 2(s^2+b^2)) ==")
    print(
        f"{'sigma*':>7} {'b':>5} {'lam_mean':>9} {'lam_gamma':>10} {'theory_gam':>11}"
    )
    for s in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        ev = gauss_hessian(s)
        print(f"{s:>7} {0.0:>5} {ev[0]:>9.4f} {ev[1]:>10.4f} {2 * s**2:>11.4f}")
    for b in (0.25, 0.5, 1.0, 2.0):
        s = 0.25
        ev = gauss_hessian(s, b=b)
        lo, hi = sorted([ev[0].item(), ev[1].item()])
        th = 2 * (s**2 + b**2)
        lam_gam = hi if th > 2 else lo  # blocks swap order once s^2+b^2 > 1
        print(f"{s:>7} {b:>5} {2.0:>9.4f} {lam_gam:>10.4f} {th:>11.4f}")

    teacher = PairTeacher(d=4, K=6, sigma_a=0.5, seed=0)
    print("\n== P3: mixture pair-velocity Gramian on the K=6 teacher ==")
    print(
        f"{'tau':>5} {'b':>5} {'lam_mean':>10} {'lam_scale':>10} {'ratio':>9} {'lam_full':>10}"
    )
    for tau in (0.2, 0.4, 0.6, 0.8, 1.0):
        for b in (0.0, 0.5, 1.0):
            lm, ls, lf = mixture_gramian(teacher, tau, b=b)
            print(
                f"{tau:>5} {b:>5} {lm:>10.5f} {ls:>10.5f} "
                f"{lm / max(ls, 1e-12):>9.1f} {lf:>10.6f}"
            )

    print("\n== F: pair-law aleatoric check (teacher, tau=1) ==")
    z1, z2 = teacher.sample_pair(400000, 1.0, generator=G)
    dz = z1 - z2
    ale_pair = 0.5 * (dz[:, :, None] * dz[:, None, :]).mean(0)
    ale_comp = teacher.moments(1.0)["cov_ale"]
    print(f"trace pair {ale_pair.trace():.4f}  vs component {ale_comp.trace():.4f}")
    print(
        f"rel err {torch.linalg.norm(ale_pair - ale_comp) / torch.linalg.norm(ale_comp):.4f}"
    )


if __name__ == "__main__":
    main()
