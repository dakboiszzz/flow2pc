"""Benchmark 2: GP meta-regression against an EXACT Bayesian oracle.

Trains one amortized context-conditioned pair PC and tests whether it recovers
the GP posterior's aleatoric/epistemic decomposition and behaves Bayesianly:
  - epistemic trace contracts as context size grows, tracking the oracle
  - aleatoric ~ sigma^2 regardless of context
  - UQ fidelity: relative Frobenius error of Sigma_epi vs the GP posterior cov

Reports oracle<->PC error only (teacher == oracle here: the GP is analytic, so
there is no teacher-model error to separate -- stated plainly, not hidden).
"""

import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.attn_context_pc import AttnContextPairPC  # noqa: E402
from src.gp_oracle import GPOracle  # noqa: E402

DEV = "cuda:0"
SIZES = [1, 2, 4, 8, 12, 16, 20]


def make_episodes(gp, sizes, per_size, seed0):
    """Return {n: (xc(N,n), yc(N,n), Y1(N,M), Y2(N,M))} on DEV."""
    out = {}
    for n in sizes:
        xc = torch.zeros(per_size, n)
        yc = torch.zeros(per_size, n)
        Y1 = torch.zeros(per_size, gp.M)
        Y2 = torch.zeros(per_size, gp.M)
        for i in range(per_size):
            g = torch.Generator().manual_seed(seed0 + 1000 * n + i)
            cx, cy, fq = gp.sample_context(n, generator=g)
            xc[i], yc[i] = cx, cy
            Y1[i] = fq + gp.sigma * torch.randn(gp.M, generator=g)
            Y2[i] = fq + gp.sigma * torch.randn(gp.M, generator=g)
        out[n] = tuple(t.to(DEV) for t in (xc, yc, Y1, Y2))
    return out


def train(model, episodes, steps, lr, log=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    sizes = list(episodes.keys())
    for step in range(steps):
        n = sizes[step % len(sizes)]
        xc, yc, Y1, Y2 = episodes[n]
        idx = torch.randint(0, xc.shape[0], (256,), device=DEV)
        pi, mu, sig = model.params(xc[idx], yc[idx])
        nll2 = -model.log_prob2(Y1[idx], Y2[idx], pi, mu, sig).mean()
        nll1 = -model.log_prob1(Y1[idx], pi, mu, sig).mean()
        loss = nll2 + nll1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if log is not None and step % 500 == 0:
            log.log({"step": step, "nll2": float(nll2), "nll1": float(nll1)})
            print(
                f"  step {step}: nll2 {float(nll2):.3f} nll1 {float(nll1):.3f}",
                flush=True,
            )


@torch.no_grad()
def evaluate(model, gp, seed0=77, n_eval=200):
    print("\n=== Bayesian behavior: epistemic trace vs context size ===")
    print(
        f"{'n':>4} {'PC epi_tr':>10} {'GP epi_tr':>10} {'epi_relF':>9} "
        f"{'PC ale_tr':>10} {'GP ale_tr':>10}"
    )
    rows = []
    for n in SIZES:
        pc_epi, gp_epi, relf, pc_ale = 0.0, 0.0, 0.0, 0.0
        gp_ale = gp.sigma**2 * gp.M
        for i in range(n_eval):
            g = torch.Generator().manual_seed(seed0 + 1000 * n + i)
            cx, cy, _ = gp.sample_context(n, generator=g)
            m = model.moments(cx.to(DEV), cy.to(DEV))
            truth = gp.ground_truth(cx, cy)
            pc_epi += m["cov_epi"].trace().item()
            gp_epi += truth["cov_epi"].trace().item()
            relf += (
                torch.linalg.norm(m["cov_epi"].cpu() - truth["cov_epi"])
                / (torch.linalg.norm(truth["cov_epi"]) + 1e-8)
            ).item()
            pc_ale += m["cov_ale"].trace().item()
        k = n_eval
        print(
            f"{n:>4} {pc_epi / k:>10.3f} {gp_epi / k:>10.3f} {relf / k:>9.3f} "
            f"{pc_ale / k:>10.3f} {gp_ale:>10.3f}"
        )
        rows.append((n, pc_epi / k, gp_epi / k, relf / k))
    # monotone contraction check (oracle and model)
    pc_tr = [r[1] for r in rows]
    gp_tr = [r[2] for r in rows]
    print(
        f"\ncontraction: PC epi_tr {pc_tr[0]:.2f}->{pc_tr[-1]:.2f}, "
        f"GP {gp_tr[0]:.2f}->{gp_tr[-1]:.2f}"
    )
    return rows


def main():
    torch.manual_seed(0)
    xq = torch.linspace(-2, 2, 8)
    gp = GPOracle(xq, lengthscale=0.5, signal=1.0, sigma=0.1)
    model = AttnContextPairPC(xq.to(DEV), H=64, dim=96, heads=4).to(DEV)

    run = None
    if os.environ.get("WANDB_API_KEY"):
        import wandb

        run = wandb.init(
            project="replica-flow2pc", name="bench2-gp-attn", dir="outputs"
        )

    print("building episodes ...", flush=True)
    train_ep = make_episodes(gp, SIZES, per_size=6000, seed0=0)
    t0 = time.time()
    print(f"training ({time.time() - t0:.0f}s to build) ...", flush=True)
    train(model, train_ep, steps=40000, lr=1e-3, log=run)
    rows = evaluate(model, gp)

    out = Path("outputs/runs/bench2_attn")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"rows": rows, "state": model.state_dict()}, out / "result.pt")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
