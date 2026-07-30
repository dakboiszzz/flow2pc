"""Benchmark 3: NON-Gaussian change-point posterior.

Trains the attention pair PC (same architecture that worked for the GP) on
change-point data and tests two things against the exact mixture oracle:
  1. epistemic COVARIANCE recovery vs context size (2nd moment)
  2. NON-GAUSSIANITY capture: the excess kurtosis of the PC's epistemic
     predictive {mu_h, pi_h} vs the oracle's -- does the mixture PC represent
     the multimodal (beyond-covariance) posterior a Gaussian could not?
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.attn_context_pc import AttnContextPairPC
from src.changepoint_oracle import ChangePointOracle

DEV = "cuda:0"
SIZES = [1, 2, 3, 5, 8, 12, 16]


def make_episodes(cp, sizes, per_size, seed0):
    out = {}
    for n in sizes:
        xc = torch.zeros(per_size, n)
        yc = torch.zeros(per_size, n)
        Y1 = torch.zeros(per_size, cp.M)
        Y2 = torch.zeros(per_size, cp.M)
        for i in range(per_size):
            g = torch.Generator().manual_seed(seed0 + 1000 * n + i)
            cx, cy, fq = cp.sample_context(n, generator=g)
            xc[i], yc[i] = cx, cy
            Y1[i] = fq + cp.sigma * torch.randn(cp.M, generator=g)
            Y2[i] = fq + cp.sigma * torch.randn(cp.M, generator=g)
        out[n] = tuple(t.to(DEV) for t in (xc, yc, Y1, Y2))
    return out


def train(model, ep, steps, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    sizes = list(ep.keys())
    for step in range(steps):
        n = sizes[step % len(sizes)]
        xc, yc, Y1, Y2 = ep[n]
        idx = torch.randint(0, xc.shape[0], (256,), device=DEV)
        pi, mu, sig = model.params(xc[idx], yc[idx])
        loss = -model.log_prob2(Y1[idx], Y2[idx], pi, mu, sig).mean() \
               - model.log_prob1(Y1[idx], pi, mu, sig).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        sched.step()
        if step % 1000 == 0:
            print(f"  step {step}: loss {float(loss):.3f}", flush=True)


@torch.no_grad()
def pc_kurtosis(pi, mu):
    """Excess kurtosis of the epistemic predictive {mu_h, pi_h} at the
    max-variance query point (pi (H,), mu (H,M))."""
    mbar = (pi[:, None] * mu).sum(0)
    var = (pi[:, None] * (mu - mbar) ** 2).sum(0)
    k = var.argmax()
    m = mu[:, k]
    w = pi
    mu1 = (w * m).sum()
    c = m - mu1
    m2 = (w * c**2).sum()
    m4 = (w * c**4).sum()
    return (m4 / (m2**2 + 1e-9) - 3.0).item()


@torch.no_grad()
def evaluate(model, cp, seed0=77, n_eval=200):
    print("\n=== Bench3: epistemic recovery + non-Gaussianity capture ===")
    print(f"{'n':>3} {'PC epi_tr':>10} {'CP epi_tr':>10} {'epi_relF':>9} "
          f"{'PC kurt':>9} {'CP kurt':>9}")
    for n in SIZES:
        pe, ce, rf, pk, ck = 0., 0., 0., 0., 0.
        for i in range(n_eval):
            g = torch.Generator().manual_seed(seed0 + 1000 * n + i)
            cx, cy, _ = cp.sample_context(n, generator=g)
            pi, mu, sig = model.params(cx.to(DEV)[None], cy.to(DEV)[None])
            pi, mu = pi[0], mu[0]
            m = model.moments(cx.to(DEV), cy.to(DEV))
            gt = cp.ground_truth(cx, cy)
            pe += m["cov_epi"].trace().item()
            ce += gt["cov_epi"].trace().item()
            rf += (torch.linalg.norm(m["cov_epi"].cpu() - gt["cov_epi"])
                   / (torch.linalg.norm(gt["cov_epi"]) + 1e-8)).item()
            pk += pc_kurtosis(pi, mu)
            ck += cp.nongaussianity(cx, cy, n=8000, generator=g)
        k = n_eval
        print(f"{n:>3} {pe/k:>10.3f} {ce/k:>10.3f} {rf/k:>9.3f} "
              f"{pk/k:>9.3f} {ck/k:>9.3f}")


def main():
    torch.manual_seed(0)
    xq = torch.linspace(-1, 1, 8)
    cp = ChangePointOracle(xq, sigma=0.15, n_grid=161)
    model = AttnContextPairPC(xq.to(DEV), H=64, dim=96, heads=4).to(DEV)
    print("building episodes ...", flush=True)
    ep = make_episodes(cp, SIZES, per_size=5000, seed0=0)
    t0 = time.time()
    print(f"training ({time.time()-t0:.0f}s build) ...", flush=True)
    train(model, ep, steps=40000, lr=5e-4)
    evaluate(model, cp)
    out = Path("outputs/runs/bench3")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")


if __name__ == "__main__":
    main()
