"""Query-specific observability (Theorem 2/3 verification).

Analysis is at the TRUE reconfiguration theta* in a well-specified student
class that exactly contains the teacher (so L_v(theta*)=0 and the Gauss-Newton
Hessian is exact):

  params theta = {A_k in R^d (endpoint means), G_k in R^d (log ale-sigma)},
  pi fixed at teacher's (no weight/permutation directions),
  paths mu_k(tau)=tau*A_k, s_k(tau)=(1-tau)+tau*exp(G_k)   [teacher: A=mu, G=log sigma_a]

Quantities (all losses normalized per replica-dim, 1/(2d), per the plan):
  H_v      = 2/(2d) E_{tau~omega, X~P^[2]} [J_U^T J_U]   (exact at zero residual)
  F_d      = 1/(2d) E [score score^T]                    (Fisher, exact at theta*)
  J_epi    = d vec(Sigma_epi(1)) / d theta,  J_ale = d diag(Sigma_ale(1)) / d theta
  kappa_g(H) = sqrt(lambda_max(J_g H^+ J_g^T))           (query observability)

Predictions: kappa_ale(H_v) >> kappa_epi(H_v); adding lambda*F_d collapses
kappa_ale; worst aleatoric direction d* (leading gen-eigvec of J_ale^T J_ale
vs H_v) barely moves the velocity loss but wrecks the aleatoric moment, and
gains large curvature under H_lambda.
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
K, d = teacher.K, teacher.d
MU = teacher.mu.double()
PI = teacher.pi.double()
BETA = 2.0
N_TAU, N_SAMP = 24, 4000


def unpack(theta):
    A = theta[: K * d].view(K, d)
    Gm = theta[K * d :].view(K, d)
    return A, Gm


def student_pair_velocity(theta, z1, z2, tau):
    """U^[2] of the student class at (z1,z2,tau); responsibilities from the
    student's own pair law (full theta-dependence, no stop-grad)."""
    A, Gm = unpack(theta)
    s = (1 - tau) + tau * torch.exp(Gm)  # (K,d)
    ds = torch.exp(Gm) - 1.0
    mu = tau * A

    def leaf_logp(z):
        zc = (z[:, None, :] - mu[None]) / s[None]
        return (
            -0.5 * (zc**2 + math.log(2 * math.pi)).sum(-1) - torch.log(s).sum(-1)[None]
        )

    joint = leaf_logp(z1) + leaf_logp(z2)
    r = torch.softmax(torch.log(PI)[None] + joint, 1)[..., None]  # (B,K,1)
    u1 = (r * (A[None] + (ds / s)[None] * (z1[:, None, :] - mu[None]))).sum(1)
    u2 = (r * (A[None] + (ds / s)[None] * (z2[:, None, :] - mu[None]))).sum(1)
    return torch.cat([u1, u2], -1)  # (B, 2d)


def student_logq2(theta, z1, z2, tau):
    A, Gm = unpack(theta)
    s = (1 - tau) + tau * torch.exp(Gm)
    mu = tau * A

    def leaf_logp(z):
        zc = (z[:, None, :] - mu[None]) / s[None]
        return (
            -0.5 * (zc**2 + math.log(2 * math.pi)).sum(-1) - torch.log(s).sum(-1)[None]
        )

    return torch.logsumexp(torch.log(PI)[None] + leaf_logp(z1) + leaf_logp(z2), 1)


def moments(theta):
    A, Gm = unpack(theta)
    mbar = (PI[:, None] * A).sum(0)
    Ac = A - mbar[None]
    cov_epi = (PI[:, None, None] * Ac[:, :, None] * Ac[:, None, :]).sum(0)
    ale_diag = (PI[:, None] * torch.exp(2 * Gm)).sum(0)
    return cov_epi.reshape(-1), ale_diag  # vec(Sigma_epi), diag(Sigma_ale)


def tau_grid():
    """Deterministic omega_beta(tau) quadrature grid (midpoint)."""
    u = (torch.arange(N_TAU, dtype=torch.float64) + 0.5) / N_TAU
    return -torch.log(1 - u * (1 - math.exp(-BETA))) / BETA


def build_matrices(theta_star):
    P = theta_star.numel()
    Hv = torch.zeros(P, P)
    Fd = torch.zeros(P, P)
    taus = tau_grid()
    for tau in taus:
        z1, z2 = teacher.sample_pair(N_SAMP, float(tau), generator=G)
        z1, z2 = z1.double(), z2.double()
        # forward-mode Jacobians (P JVP passes, memory-light vs reverse vmap)
        JU = torch.func.jacfwd(
            lambda th, a=z1, b=z2, t=tau: student_pair_velocity(th, a, b, t)
        )(theta_star)  # (B, 2d, P)
        Hv += torch.einsum("bip,biq->pq", JU, JU) / (N_SAMP * 2 * d)
        S = torch.func.jacfwd(lambda th, a=z1, b=z2, t=tau: student_logq2(th, a, b, t))(
            theta_star
        )  # (B, P)
        Fd += torch.einsum("bp,bq->pq", S, S) / (N_SAMP * 2 * d)
    Hv, Fd = 2 * Hv / N_TAU, Fd / N_TAU
    return Hv, Fd


def kappa(J, H, eps=1e-10):
    """sqrt(lambda_max(J H^+ J^T)) with eigenvalue-cutoff pseudoinverse."""
    lam, V = torch.linalg.eigh(H)
    inv = torch.where(lam > eps * lam.max(), 1.0 / lam, torch.zeros_like(lam))
    M = J @ V @ torch.diag(inv) @ V.T @ J.T
    return torch.linalg.eigvalsh(M)[-1].sqrt().item()


def main():
    theta_star = torch.cat(
        [MU.reshape(-1), torch.full((K * d,), math.log(teacher.sigma_a))]
    ).requires_grad_(False)

    # sanity: exact reconfiguration
    z1, z2 = teacher.sample_pair(2000, 0.6, generator=G)
    v1, v2 = teacher.pair_velocity(z1.float(), z2.float(), 0.6)
    U = student_pair_velocity(
        theta_star, z1.double(), z2.double(), torch.tensor(0.6).double()
    )
    res = (U - torch.cat([v1, v2], -1).double()).norm() / U.norm()
    print(f"sanity: velocity residual at theta* = {res:.2e} (must be ~0)")

    print("building H_v and F_d (Gauss-Newton / Fisher at theta*)...")
    Hv, Fd = build_matrices(theta_star)
    J_epi = torch.autograd.functional.jacobian(lambda th: moments(th)[0], theta_star)
    J_ale = torch.autograd.functional.jacobian(lambda th: moments(th)[1], theta_star)

    lv = torch.linalg.eigvalsh(Hv)
    lf = torch.linalg.eigvalsh(Fd)
    print(f"\nspec(H_v):  min {lv[0]:.3e}  max {lv[-1]:.3e}  cond {lv[-1] / lv[0]:.1e}")
    print(f"spec(F_d):  min {lf[0]:.3e}  max {lf[-1]:.3e}  cond {lf[-1] / lf[0]:.1e}")

    print(
        f"\n{'objective':<24} {'kappa_epi':>10} {'kappa_ale':>10} {'ratio ale/epi':>14}"
    )
    ke0, ka0 = kappa(J_epi, Hv), kappa(J_ale, Hv)
    print(f"{'H_v (velocity only)':<24} {ke0:>10.3f} {ka0:>10.3f} {ka0 / ke0:>14.1f}")
    for lam in (0.1, 0.3, 1.0):
        H = Hv + lam * Fd
        ke, ka = kappa(J_epi, H), kappa(J_ale, H)
        print(f"{'H_v + %.1f F_d' % lam:<24} {ke:>10.3f} {ka:>10.3f} {ka / ke:>14.1f}")
    kaF = kappa(J_ale, Fd)
    keF = kappa(J_epi, Fd)
    print(f"{'F_d (density only)':<24} {keF:>10.3f} {kaF:>10.3f} {kaF / keF:>14.1f}")

    # ---- worst query direction: leading gen-eigvec of J_ale^T J_ale vs H_v
    lam, V = torch.linalg.eigh(Hv)
    inv_sqrt = torch.where(lam > 1e-10 * lam.max(), lam.rsqrt(), torch.zeros_like(lam))
    Hm = V @ torch.diag(inv_sqrt) @ V.T
    M = Hm @ J_ale.T @ J_ale @ Hm
    w = torch.linalg.eigh(M)[1][:, -1]
    dstar = Hm @ w
    dstar = dstar / (dstar @ Hv @ dstar).sqrt()  # normalize d*^T H_v d* = 1
    gt_epi, gt_ale = moments(theta_star)

    def vel_loss(theta):
        tot = 0.0
        for tau in tau_grid()[::4]:
            z1, z2 = teacher.sample_pair(3000, float(tau), generator=G)
            z1, z2 = z1.double(), z2.double()
            v1, v2 = teacher.pair_velocity(z1.float(), z2.float(), float(tau))
            U = student_pair_velocity(theta, z1, z2, tau)
            V2 = torch.cat([v1, v2], -1).double()
            tot += ((U - V2) ** 2).sum(-1).mean().item() / (2 * d)
        return tot / len(tau_grid()[::4])

    print("\n== worst-aleatoric-direction perturbation (d*^T H_v d* = 1) ==")
    print(
        f"curvature of d*: under H_v = 1.000, under F_d = "
        f"{(dstar @ Fd @ dstar).item():.3f}, under H_v+0.3F_d = "
        f"{(dstar @ (Hv + 0.3 * Fd) @ dstar).item():.3f}"
    )
    print(f"{'alpha':>6} {'L_vel':>10} {'err_epi':>9} {'err_ale':>9}")
    for alpha in (0.0, 0.1, 0.3, 1.0):
        th = theta_star + alpha * dstar
        e, a = moments(th)
        err_e = ((e - gt_epi).norm() / gt_epi.norm()).item()
        err_a = ((a - gt_ale).norm() / gt_ale.norm()).item()
        print(f"{alpha:>6} {vel_loss(th):>10.5f} {err_e:>9.4f} {err_a:>9.4f}")

    # control: same-budget perturbation along the worst EPISTEMIC direction
    M2 = Hm @ J_epi.T @ J_epi @ Hm
    w2 = torch.linalg.eigh(M2)[1][:, -1]
    dep = Hm @ w2
    dep = dep / (dep @ Hv @ dep).sqrt()
    print("\ncontrol: worst-epistemic direction, same velocity-loss budget")
    print(f"{'alpha':>6} {'L_vel':>10} {'err_epi':>9} {'err_ale':>9}")
    for alpha in (0.1, 0.3, 1.0):
        th = theta_star + alpha * dep
        e, a = moments(th)
        err_e = ((e - gt_epi).norm() / gt_epi.norm()).item()
        err_a = ((a - gt_ale).norm() / gt_ale.norm()).item()
        print(f"{alpha:>6} {vel_loss(th):>10.5f} {err_e:>9.4f} {err_a:>9.4f}")


if __name__ == "__main__":
    main()
