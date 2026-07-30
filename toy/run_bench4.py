"""Benchmark 4: active identification with an aleatoric distractor.

Averages active-learning trajectories over many latent functions and compares
epistemic-guided vs total-uncertainty vs random acquisition. The claim: the
epistemic signal (which the pair PC recovers) identifies f faster AND avoids
the high-noise region; the total-uncertainty signal (a one-replica model) is
lured into the noise region and learns slower.
"""

import torch

from src.active_learning import HeteroGP, run_active

N_FUN = 40
ROUNDS = 40


def main():
    gp = HeteroGP(lengthscale=0.4, sig_lo=0.05, sig_hi=0.6, region=(0.5, 1.5))
    res = {
        a: {"var": torch.zeros(ROUNDS), "ra": 0.0}
        for a in ("infogain", "total", "variance", "random")
    }
    for a in res:
        for fi in range(N_FUN):
            rmse, ra = run_active(gp, a, n_rounds=ROUNDS, seed=fi, f_seed=1000 + fi)
            res[a]["var"] += torch.tensor(rmse)
            res[a]["ra"] += ra
        res[a]["var"] /= N_FUN
        res[a]["ra"] /= N_FUN

    print(f"=== Benchmark 4: active identification (avg over {N_FUN} functions) ===")
    print(
        f"{'acquisition':<12} {'epi_var@10':>11} {'epi_var@20':>11} {'epi_var@40':>11} {'%R_a':>8}"
    )
    for a in ("infogain", "total", "variance", "random"):
        r = res[a]["var"]
        print(
            f"{a:<12} {r[9]:>11.3f} {r[19]:>11.3f} {r[39]:>11.4f} {100 * res[a]['ra']:>7.0f}%"
        )
    # verdict
    e20, t20 = res["infogain"]["var"][19], res["total"]["var"][19]
    print(
        f"\ninfogain leaves {e20:.3f} reducible uncertainty at 20 queries vs "
        f"total {t20:.3f} ({t20 / e20:.1f}x more); infogain spends "
        f"{100 * res['infogain']['ra']:.0f}% in R_a vs total {100 * res['total']['ra']:.0f}% "
        f"(R_a is 25% of the domain)."
    )
    torch.save(res, "outputs/runs/bench4_result.pt")


if __name__ == "__main__":
    main()
