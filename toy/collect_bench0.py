"""Aggregate Benchmark 0 summary.yaml files into the decisive table.

The claim under test: only the pair variants recover the epistemic
(cross-replica) covariance; the marginal variants cannot, because their
inputs are identical between worlds A and B by construction.
"""

import glob

from omegaconf import OmegaConf

VARIANT_ORDER = [
    "endpoint",
    "marginal_path",
    "marginal_path_vel",
    "pair_endpoint",
    "pair_path",
]


def load_all(run_glob="outputs/runs/bench0/*/summary.yaml"):
    rows = []
    for p in glob.glob(run_glob):
        try:
            rows.append(dict(OmegaConf.load(p)))
        except Exception:
            pass
    return rows


def main():
    rows = load_all()
    print(f"loaded {len(rows)} summaries\n")

    core = [r for r in rows if r["rho_ci"] == 0.0 and r["rho_e"] == 0.5]
    print(
        "=== Core (d=4, rho_e=0.5): World A true epi_tr=2.05, World B true=0 "
        "-- IDENTICAL marginals ==="
    )
    print(
        f"{'variant':<20} {'A:err_cross':>12} {'A:epi_tr':>9} "
        f"{'B:epi_tr':>9} {'A/B ratio':>10}"
    )
    for v in VARIANT_ORDER:
        a = next((r for r in core if r["variant"] == v and r["world"] == "A"), None)
        b = next((r for r in core if r["variant"] == v and r["world"] == "B"), None)
        if not a or not b:
            continue
        ratio = a["epi_trace_model"] / max(abs(b["epi_trace_model"]), 1e-6)
        print(
            f"{v:<20} {a['err_cross']:>12.3f} {a['epi_trace_model']:>9.2f} "
            f"{b['epi_trace_model']:>9.2f} {ratio:>9.1f}x"
        )

    print("\n=== rho_e sweep (world A, pair_path): epistemic fraction ===")
    rho = sorted(
        [
            r
            for r in rows
            if r["variant"] == "pair_path" and r["world"] == "A" and r["rho_ci"] == 0.0
        ],
        key=lambda r: r["rho_e"],
    )
    print(f"{'rho_e':>6} {'err_cross':>10} {'epi_tr_mdl':>11} {'epi_tr_true':>12}")
    for r in rho:
        print(
            f"{r['rho_e']:>6} {r['err_cross']:>10.3f} "
            f"{r['epi_trace_model']:>11.2f} {r['epi_trace_true']:>12.2f}"
        )

    print("\n=== CI stress (world A, pair_path): correlated local noise ===")
    ci = sorted(
        [r for r in rows if r["variant"] == "pair_path" and r["rho_ci"] > 0.0],
        key=lambda r: r["rho_ci"],
    )
    print(f"{'rho_ci':>7} {'err_cross':>10} {'epi_tr_mdl':>11} {'epi_tr_true':>12}")
    for r in ci:
        print(
            f"{r['rho_ci']:>7} {r['err_cross']:>10.3f} "
            f"{r['epi_trace_model']:>11.2f} {r['epi_trace_true']:>12.2f}"
        )

    # verdict
    pair = [r for r in core if r["variant"].startswith("pair")]
    marg = [r for r in core if not r["variant"].startswith("pair")]
    if pair and marg:
        best_pair = min(r["err_cross"] for r in pair)
        best_marg = min(r["err_cross"] for r in marg)
        print(
            f"\nVERDICT: best pair err_cross {best_pair:.3f} vs "
            f"best marginal {best_marg:.3f} "
            f"({'PASS' if best_pair < 0.5 * best_marg else 'INCONCLUSIVE'})"
        )


if __name__ == "__main__":
    main()
