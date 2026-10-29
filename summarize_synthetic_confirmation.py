import csv
import json
import statistics
from pathlib import Path

CANDIDATES = {
    "uniform-lambda10": (10.0, "uniform", None, None),
    "beta-a1-b0p2-lambda10": (10.0, "beta", 1.0, 0.2),
    "beta-a1-b0p05-lambda10": (10.0, "beta", 1.0, 0.05),
}
SEEDS = (0, 1, 2)
METRICS = (
    "global_velocity_mse",
    "endpoint_kl_source_to_student",
    "endpoint_density_l1",
    "endpoint_sliced_wasserstein",
    "conditional_mean_error",
    "conditional_variance_error",
)


def config_key(config: dict) -> tuple[float, str, float | None, float | None]:
    if config["timestep_sampling"] == "uniform":
        return float(config["density_loss_weight"]), "uniform", None, None
    return (
        float(config["density_loss_weight"]),
        "beta",
        float(config["timestep_beta_alpha"]),
        float(config["timestep_beta_beta"]),
    )


def collect_rows(runs_dir: Path) -> list[dict]:
    matched = {}
    keys_to_names = {value: key for key, value in CANDIDATES.items()}
    for results_path in runs_dir.glob("*/results.json"):
        payload = json.loads(results_path.read_text())
        config = payload["config"]
        seed = config.get("seed")
        candidate = keys_to_names.get(config_key(config))
        if candidate is None or seed not in SEEDS:
            continue
        result = next(
            (
                item
                for item in payload["results"]
                if item["kind"] == "trained" and item["capacity"] == 8
            ),
            None,
        )
        if result is None:
            continue
        row = {
            "candidate": candidate,
            "seed": seed,
            "global_velocity_mse": result["global_velocity_mse"],
            "endpoint_kl_source_to_student": result["endpoint_kl_source_to_student"],
            "endpoint_density_l1": result["endpoint_density_l1"],
            "endpoint_sliced_wasserstein": result["endpoint_sliced_wasserstein"],
            "conditional_mean_error": result["query_errors"]["conditional_mean_y"],
            "conditional_variance_error": result["query_errors"][
                "conditional_variance_y"
            ],
            "run_dir": str(results_path.parent),
        }
        match_key = candidate, seed
        previous = matched.get(match_key)
        if previous is None or results_path.stat().st_mtime > previous[0]:
            matched[match_key] = results_path.stat().st_mtime, row
    expected = {(candidate, seed) for candidate in CANDIDATES for seed in SEEDS}
    missing = sorted(expected - set(matched))
    if missing:
        raise RuntimeError(f"missing confirmation runs: {missing}")
    return [matched[(candidate, seed)][1] for candidate in CANDIDATES for seed in SEEDS]


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for candidate in CANDIDATES:
        candidate_rows = [row for row in rows if row["candidate"] == candidate]
        summary = {"candidate": candidate, "seeds": len(candidate_rows)}
        for metric in METRICS:
            values = [row[metric] for row in candidate_rows]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_std"] = statistics.stdev(values)
        summaries.append(summary)
    return summaries


def write_outputs(rows: list[dict], summaries: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_mean = min(
        summaries,
        key=lambda item: item["endpoint_kl_source_to_student_mean"],
    )["candidate"]
    robust = min(
        summaries,
        key=lambda item: (
            item["endpoint_kl_source_to_student_mean"]
            + item["endpoint_kl_source_to_student_std"]
        ),
    )["candidate"]
    recommendation = {
        "best_mean_endpoint_kl": best_mean,
        "robust_minimum_mean_plus_std_endpoint_kl": robust,
    }
    with (output_dir / "per_seed.csv").open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "recommendation": recommendation,
                "rows": rows,
                "summaries": summaries,
            },
            indent=2,
        )
        + "\n"
    )
    lines = [
        "# Synthetic density/timestep confirmation (seeds 0-2)",
        "",
        f"Best mean endpoint KL: `{best_mean}`.",
        "",
        (
            "Robust recommendation (minimum mean + sample standard deviation "
            f"of endpoint KL): `{robust}`."
        ),
        "",
        "| Candidate | Velocity MSE | KL(p‖q) | Density L1 | SWD | Cond. mean | Cond. var. |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        values = [summary["candidate"]]
        for metric in METRICS:
            values.append(
                f"{summary[f'{metric}_mean']:.6f} ± {summary[f'{metric}_std']:.6f}"
            )
        lines.append("| " + " | ".join(values) + " |")
    (output_dir / "metrics.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = collect_rows(Path("outputs/runs/synthetic"))
    summaries = summarize(rows)
    output_dir = Path("outputs/benchmark/synthetic-density-timestep-confirmation")
    write_outputs(rows, summaries, output_dir)
    print(f"confirmation summary saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
