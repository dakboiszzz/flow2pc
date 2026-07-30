"""Overparameterization null-direction experiment (the REAL gap mechanism).

Finding from run_query_observability.py: at the minimal well-specified K=6
student the velocity Hessian is well-conditioned and kappa_ale is only ~2x
kappa_epi (relative) — first-order query observability CANNOT explain the
~100x empirical aleatoric gap. Hypothesis: the gap lives in the
OVERPARAMETERIZED class (M >> K) where training actually happens.

Test at the duplicate-embedded student (each teacher component split into two
copies, weight pi/2 each — an exactly representable, singular point):
  (1) H_v gains near-zero eigenvalues (component-split directions);
  (2) F_d is strictly positive on Null(H_v)  [completion condition
      Null(H_v) cap Null(F_d) = {0} verified where it matters];
  (3) along a scale-split direction (dgamma_A = -dgamma_B): velocity loss
      stays ~0 (second order), aleatoric trace INFLATES upward by convexity
      (Jensen: pi/2(e^{2g+2a}+e^{2g-2a}) > pi e^{2g}), density loss rises —
      the mechanism of the observed upward drift through the target.
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.pair_teacher import PairTeacher

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
G = torch.Generator().manual_seed(0)

teacher = PairTeacher(d=4, K=6, sigma_a=0.5, seed=0)
d = teacher.d
# duplicate embedding: 12 components, each teacher comp twice at weight pi/2
KS = 2 * teacher.K
MU_S = teacher.mu.double().repeat_interleave(2, dim=0)  # (12, d)
PI_S = (teacher.pi.double() / 2).repeat_interleave(2, dim=0)
BETA = 2.0
N_TAU, N_SAMP = 16, 3000


def unpack(theta):
    return theta[: KS * d].view(KS, d), theta[KS * d :].view(KS, d)


def student_pair_velocity(theta, z1, z2, tau):
    A, Gm = unpack(theta)
    s = (1 - tau) + tau * torch.exp(Gm)
    ds = torch.exp(Gm) - 1.0
    mu = tau * A

    def leaf_logp(z):
        zc = (z[:, None, :] - mu[None]) / s[None]
        return (
            -0.5 * (zc**2 + math.log(2 * math.pi)).sum(-1) - torch.log(s).sum(-1)[None]
        )

    joint = leaf_logp(z1) + leaf_logp(z2)
    r = torch.softmax(torch.log(PI_S)[None] + joint, 1)[..., None]
    u1 = (r * (A[None] + (ds / s)[None] * (z1[:, None, :] - mu[None]))).sum(1)
    u2 = (r * (A[None] + (ds / s)[None] * (z2[:, None, :] - mu[None]))).sum(1)
    return torch.cat([u1, u2], -1)


def student_logq2(theta, z1, z2, tau):
    A, Gm = unpack(theta)
    s = (1 - tau) + tau * torch.exp(Gm)
    mu = tau * A

    def leaf_logp(z):
        zc = (z[:, None, :] - mu[None]) / s[None]
        return (
            -0.5 * (zc**2 + math.log(2 * math.pi)).sum(-1) - torch.log(s).sum(-1)[None]
        )

    return torch.logsumexp(torch.log(PI_S)[None] + leaf_logp(z1) + leaf_logp(z2), 1)


def moments(theta):
    A, Gm = unpack(theta)
    mbar = (PI_S[:, None] * A).sum(0)
    Ac = A - mbar[None]
    cov_epi = (PI_S[:, None, None] * Ac[:, :, None] * Ac[:, None, :]).sum(0)
    ale_diag = (PI_S[:, None] * torch.exp(2 * Gm)).sum(0)
    return cov_epi, ale_diag


def tau_grid():
    u = (torch.arange(N_TAU, dtype=torch.float64) + 0.5) / N_TAU
    return -torch.log(1 - u * (1 - math.exp(-BETA))) / BETA


def build_matrices(theta_star):
    P = theta_star.numel()
    Hv, Fd = torch.zeros(P, P), torch.zeros(P, P)
    for tau in tau_grid():
        z1, z2 = teacher.sample_pair(N_SAMP, float(tau), generator=G)
        z1, z2 = z1.double(), z2.double()
        JU = torch.func.jacfwd(
            lambda th, a=z1, b=z2, t=tau: student_pair_velocity(th, a, b, t)
        )(theta_star)
        Hv += torch.einsum("bip,biq->pq", JU, JU) / (N_SAMP * 2 * d)
        S = torch.func.jacfwd(lambda th, a=z1, b=z2, t=tau: student_logq2(th, a, b, t))(
            theta_star
        )
        Fd += torch.einsum("bp,bq->pq", S, S) / (N_SAMP * 2 * d)
    return 2 * Hv / N_TAU, Fd / N_TAU


EVAL_SETS = None  # fixed evaluation samples so loss DIFFERENCES are noise-free


def losses(theta):
    global EVAL_SETS
    if EVAL_SETS is None:
        EVAL_SETS = []
        ge = torch.Generator().manual_seed(1234)
        for tau in tau_grid()[::4]:
            z1, z2 = teacher.sample_pair(20000, float(tau), generator=ge)
            v1, v2 = teacher.pair_velocity(z1, z2, float(tau))
            EVAL_SETS.append(
                (tau, z1.double(), z2.double(), torch.cat([v1, v2], -1).double())
            )
    lv, ld = 0.0, 0.0
    for tau, z1d, z2d, V2 in EVAL_SETS:
        U = student_pair_velocity(theta, z1d, z2d, tau)
        lv += ((U - V2) ** 2).sum(-1).mean().item() / (2 * d)
        ld += -student_logq2(theta, z1d, z2d, tau).mean().item() / (2 * d)
    return lv / len(EVAL_SETS), ld / len(EVAL_SETS)


def main():
    theta_star = torch.cat(
        [MU_S.reshape(-1), torch.full((KS * d,), math.log(teacher.sigma_a))]
    )
    gt_epi, gt_ale = moments(theta_star)
    print(f"duplicate-embedded student: K_s={KS}, P={theta_star.numel()} params")
    print(f"ale trace at theta* = {gt_ale.sum():.4f} (target 1.0)")

    print("building H_v, F_d at the singular point...")
    Hv, Fd = build_matrices(theta_star)
    lam, V = torch.linalg.eigh(Hv)
    lam_f = torch.linalg.eigvalsh(Fd)
    print(f"\nspec(H_v): min {lam[0]:.2e}  p10 {lam[9]:.2e}  max {lam[-1]:.2e}")
    print(
        f"  near-null (<1e-3*max): {(lam < 1e-3 * lam[-1]).sum().item()} of {len(lam)}"
    )
    print("  (vs minimal K=6 class: lam_min 1.8e-02, cond 7.9 — no near-nulls)")
    print(f"spec(F_d): min {lam_f[0]:.2e}  max {lam_f[-1]:.2e}")

    # (2) completion condition on the near-null subspace of H_v
    k_null = int((lam < 1e-3 * lam[-1]).sum().item())
    if k_null > 0:
        Vn = V[:, :k_null]  # near-null directions of H_v
        Fn = Vn.T @ Fd @ Vn
        ev = torch.linalg.eigvalsh(Fn)
        print(
            f"\nF_d restricted to the {k_null}-dim near-null space of H_v: "
            f"min {ev[0]:.2e} max {ev[-1]:.2e}"
        )
        print(
            "completion condition Null(H_v) cap Null(F_d) = {0}: "
            + ("VERIFIED (F_d strictly positive there)" if ev[0] > 1e-6 else "FAILED")
        )
        Hn = Vn.T @ (Hv + 0.3 * Fd) @ Vn
        print(
            f"H_v + 0.3 F_d on that subspace: min eig {torch.linalg.eigvalsh(Hn)[0]:.2e}"
        )

    # (3) scale-split perturbation: dgamma_A = -dgamma_B on the heaviest comp
    h = int(torch.argmax(teacher.pi).item())
    split = torch.zeros_like(theta_star)
    base = KS * d
    split[base + (2 * h) * d : base + (2 * h + 1) * d] = 1.0
    split[base + (2 * h + 1) * d : base + (2 * h + 2) * d] = -1.0
    split = split / split.norm()
    print(
        f"\n== scale-split perturbation (comp {h}, dgamma_A=-dgamma_B) ==\n"
        f"quadratic forms: split^T H_v split = {split @ Hv @ split:.2e}, "
        f"split^T F_d split = {split @ Fd @ split:.2e}"
    )
    print(
        f"{'alpha':>6} {'L_vel':>10} {'L_den (delta)':>14} {'err_ale':>9} {'ale_tr':>8}"
    )
    lv0, ld0 = losses(theta_star)
    for alpha in (0.0, 0.5, 1.0, 2.0):
        th = theta_star + alpha * split
        lv, ld = losses(th)
        _, ale = moments(th)
        err_a = ((ale - gt_ale).norm() / gt_ale.norm()).item()
        print(
            f"{alpha:>6} {lv:>10.6f} {ld - ld0:>14.6f} {err_a:>9.4f} {ale.sum():>8.4f}"
        )
    print(
        "\nexpected: L_vel stays tiny (split is (near-)null for velocity), "
        "ale_tr INFLATES above 1.0 (Jensen), L_den rises (density sees it)."
    )

    # (4) GENERIC-POINT completion: exact duplicates are measure-zero. Jitter
    # the means slightly (as in real training, where components spread out) and
    # re-test whether F_d is positive on H_v's weak subspace there.
    print("\n== generic-point check (means jittered by 0.05) ==")
    gj = torch.Generator().manual_seed(5)
    theta_g = theta_star.clone()
    theta_g[: KS * d] += 0.05 * torch.randn(KS * d, generator=gj)
    Hv_g, Fd_g = build_matrices(theta_g)
    lam_g, V_g = torch.linalg.eigh(Hv_g)
    k_weak = int((lam_g < 1e-3 * lam_g[-1]).sum().item())
    print(
        f"spec(H_v): min {lam_g[0]:.2e}  max {lam_g[-1]:.2e}  "
        f"weak (<1e-3*max): {k_weak} of {len(lam_g)}"
    )
    kw = max(k_weak, 16)  # inspect the 16 weakest H_v directions regardless
    Vw = V_g[:, :kw]
    evw = torch.linalg.eigvalsh(Vw.T @ Fd_g @ Vw)
    print(f"F_d on the {kw} weakest H_v directions: min {evw[0]:.2e} max {evw[-1]:.2e}")
    evc = torch.linalg.eigvalsh(Vw.T @ (Hv_g + 0.3 * Fd_g) @ Vw)
    print(f"H_v + 0.3 F_d there: min {evc[0]:.2e}")
    ratio = evc[0] / max(float(lam_g[0]), 1e-30)
    print(f"weakest-direction curvature gain from density term: {ratio:.1e}x")

    # (5) QUERY-specific coefficients at the generic overparameterized point:
    # kappa_g(H) = sqrt(lam_max(J H^+ J^T)), reported RELATIVE to ||g(theta)||
    # so epi and ale are comparable (bound on the relative moment error per
    # sqrt(excess loss)).
    def kappa(J, H, eps=1e-9):
        lam2, V2 = torch.linalg.eigh(H)
        inv = torch.where(lam2 > eps * lam2.max(), 1.0 / lam2, torch.zeros_like(lam2))
        M = J @ V2 @ torch.diag(inv) @ V2.T @ J.T
        return torch.linalg.eigvalsh(M)[-1].sqrt().item()

    def vec_moments(th):
        e, a = moments(th)
        return e.reshape(-1), a

    J_epi = torch.func.jacfwd(lambda th: vec_moments(th)[0])(theta_g)
    J_ale = torch.func.jacfwd(lambda th: vec_moments(th)[1])(theta_g)
    ge_, ga_ = vec_moments(theta_g)
    ne, na = ge_.norm().item(), ga_.norm().item()
    print("\n== query-specific kappa at the generic overparameterized point ==")
    print("(relative: bound on RELATIVE moment error per sqrt(excess loss))")
    print(f"{'objective':<20} {'k_epi/|epi|':>12} {'k_ale/|ale|':>12} {'ratio':>7}")
    ke, ka = kappa(J_epi, Hv_g) / ne, kappa(J_ale, Hv_g) / na
    print(f"{'H_v':<20} {ke:>12.2f} {ka:>12.2f} {ka / ke:>7.1f}")
    for lam2 in (0.1, 0.3, 1.0):
        H2 = Hv_g + lam2 * Fd_g
        ke2, ka2 = kappa(J_epi, H2) / ne, kappa(J_ale, H2) / na
        print(
            f"{'H_v + %.1f F_d' % lam2:<20} {ke2:>12.2f} {ka2:>12.2f} {ka2 / ke2:>7.1f}"
        )


if __name__ == "__main__":
    main()
