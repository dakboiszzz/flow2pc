"""Verification battery for every circuit family.

These are the correctness checks every model must pass before training:
  1. boundary:   p_0 = N(0, I) exactly (any weights)
  2. flows:      leaf flows F = dlogp/dllf sum to 1 per variable
  3. continuity: Euler-transporting N(0,I) along u_root reproduces the
                 circuit's own density at t=1 (Proposition 1, numerically)
  4. leaf math:  multivariate leaf density matches torch.distributions,
                 leaf velocity matches central differences (patch leaves)

Run: pytest tests/ -q      (CPU, ~2 min)
"""

import pytest
import torch
from repc.deep_pc import DeepPC
from repc.general_pc import GeneralPC
from repc.patch_pc import PatchDeepPC
from repc.pc import TimePolyGaussianPC
from repc.wide_pc import WideDeepPC
from torch.distributions import Independent, MultivariateNormal, Normal

D = 16  # toy dimension (4x4 image / 16 variables)


def make(name):
    torch.manual_seed(0)
    if name == "shallow":
        pc = TimePolyGaussianPC(K=4, D=D, order=3)
        with torch.no_grad():
            pc.B.normal_(0, 0.5)
            pc.C.normal_(0, 0.2)
            pc.w_logits.normal_(0, 1)
        return pc.eval()
    if name == "deep":
        pc = DeepPC(H=4, W=4, S=4, order=3)
    elif name == "wide":
        pc = WideDeepPC(H=4, W=4, schedule="8:8,1:4", order=3)
    elif name == "patch":
        pc = PatchDeepPC(H=4, W=4, patch=2, S=4, order=3)
    elif name == "patch_rgb":
        pc = PatchDeepPC(H=4, W=4, patch=2, S=4, order=3, channels=1)
    elif name == "patch_clt":
        pc = PatchDeepPC(
            H=4,
            W=4,
            patch=2,
            S=4,
            order=3,
            structure="clt",
            clt_edges=[(0, 1), (1, 3), (3, 2)],
        )
    elif name == "general":
        pc = GeneralPC(D=D, R=3, S=4, order=3)
    with torch.no_grad():
        for p in [pc.B, pc.C] if hasattr(pc, "B") else []:
            p.normal_(0, 0.4)
        if hasattr(pc, "Bmu"):
            pc.Bmu.normal_(0, 0.4)
            pc.Cdiag.normal_(0, 0.15)
            pc.Clow.normal_(0, 0.15)
        for sl in pc.sum_logits:
            sl.normal_(0, 1)
        pc.root_logits.normal_(0, 1)
    return pc.eval()


FAMILIES = ["shallow", "deep", "wide", "patch", "patch_clt", "general"]


def test_learned_clt_is_spanning_tree():
    from repc.clt_structure import learn_patch_clt

    torch.manual_seed(0)
    x = torch.randn(500, 3 * 8 * 8)
    edges = learn_patch_clt(x, 8, 8, 2, 3)
    n = 16  # 4x4 patches
    assert len(edges) == n - 1
    seen = {0}
    for _ in range(n):
        for u, v in edges:
            if u in seen or v in seen:
                seen.update((u, v))
    assert seen == set(range(n)), "CLT edges do not span all patches"


@pytest.mark.parametrize("name", FAMILIES)
def test_boundary_standard_normal(name):
    pc = make(name)
    z = torch.randn(200, D)
    ref = Independent(Normal(torch.zeros(D), torch.ones(D)), 1)
    err = (pc.log_prob(z, torch.tensor(0.0)) - ref.log_prob(z)).abs().max()
    assert err < 1e-4, f"{name}: p_0 != N(0,I), err={err:.2e}"


@pytest.mark.parametrize("name", [n for n in FAMILIES if n != "shallow"])
def test_flows_sum_to_one(name):
    pc = make(name)
    z = torch.randn(100, D)
    t = torch.tensor(0.8)
    if hasattr(pc, "leaf_log_density_and_eps"):
        mu, _, L, _ = pc.leaf_chol(t)
        llf, _ = pc.leaf_log_density_and_eps(pc._to_patches(z), mu, L)
        llf = llf.detach().requires_grad_(True)
    else:
        llf = pc.leaf_log_density(z, t).detach().requires_grad_(True)
    (F,) = torch.autograd.grad(pc._forward_circuit(llf).sum(), llf)
    err = (F.sum(-1) - 1).abs().max()
    assert err < 1e-4, f"{name}: flows don't sum to 1, err={err:.2e}"


@pytest.mark.parametrize("name", FAMILIES)
def test_continuity_transport(name):
    """Prop 1: transporting N(0,I) along u_root lands on the PC's own p_1."""
    pc = make(name)
    n, steps = 8000, 200
    zz = torch.randn(n, D)
    with torch.no_grad():
        for i in range(steps):
            zz = zz + (1.0 / steps) * pc.velocity(zz, torch.tensor((i + 0.5) / steps))
        t1 = torch.tensor(1.0)
        ll_tr = pc.log_prob(zz, t1).mean()
        if name == "shallow":
            direct = pc.sample(1500, t1)
        elif name in ("patch", "patch_rgb"):
            direct = pc.sample(1500, t1)
        else:
            try:
                direct = pc.sample(1500, t1)
            except TypeError:
                direct = pc.sample(1500, t1, seed=0)
        ll_dir = pc.log_prob(direct, t1).mean()
    gap = abs(ll_tr - ll_dir)
    assert gap < 0.6, f"{name}: transport LL {ll_tr:.2f} vs direct {ll_dir:.2f}"


def test_patch_leaf_density_exact():
    pc = make("patch").double()
    t = torch.tensor(0.8, dtype=torch.float64)
    mu, _, L, _ = pc.leaf_chol(t)
    z = torch.randn(100, D, dtype=torch.float64)
    zp = pc._to_patches(z)
    llf, _ = pc.leaf_log_density_and_eps(zp, mu, L)
    for p in range(pc.n_patch):
        for s in range(pc.S):
            ref = MultivariateNormal(mu[p, s], scale_tril=L[p, s]).log_prob(zp[:, p])
            assert (llf[:, p, s] - ref).abs().max() < 1e-8


def test_patch_leaf_velocity_matches_central_difference():
    pc = make("patch").double()
    t = torch.tensor(0.8, dtype=torch.float64)
    h = 1e-5
    mu, dmu, L, dL = pc.leaf_chol(t)
    mua, _, La, _ = pc.leaf_chol(t + h)
    mub, _, Lb, _ = pc.leaf_chol(t - h)
    eps0 = torch.randn(pc.pd, dtype=torch.float64)
    p, s = 1, 2
    z_t = mu[p, s] + L[p, s] @ eps0
    u_fd = ((mua[p, s] + La[p, s] @ eps0) - (mub[p, s] + Lb[p, s] @ eps0)) / (2 * h)
    rhs = torch.linalg.solve_triangular(
        L[p, s], (z_t - mu[p, s])[:, None], upper=False
    )[:, 0]
    u_an = dmu[p, s] + dL[p, s] @ rhs
    assert (u_fd - u_an).abs().max() < 1e-7


def test_velocity_shapes():
    for name in FAMILIES:
        pc = make(name)
        z = torch.randn(10, D)
        u = pc.velocity(z, torch.tensor(0.5))
        assert u.shape == z.shape, name
