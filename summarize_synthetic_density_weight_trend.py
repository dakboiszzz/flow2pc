import argparse
import csv
import json
import math
import statistics
from pathlib import Path

WEIGHTS = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0)
SEEDS = (0, 1, 2)
EVALUATION_METRICS = (
    "global_velocity_mse",
    "endpoint_kl_source_to_student",
    "endpoint_density_l1",
    "endpoint_sliced_wasserstein",
    "conditional_mean_error",
    "conditional_variance_error",
)
TRAINING_METRICS = (
    "final_train_loss_mean_100",
    "final_train_velocity_loss",
    "final_train_density_nll",
)
METRICS = EVALUATION_METRICS + TRAINING_METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("outputs/runs/synthetic"),
    )
    parser.add_argument(
        "--decision-file",
        type=Path,
        default=Path("outputs/benchmark/synthetic-timestep-significance/summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/benchmark/synthetic-density-weight-trend"),
    )
    parser.add_argument("--velocity-tolerance", type=float, default=0.10)
    parser.add_argument("--check-run", action="store_true")
    parser.add_argument("--timestep-sampling", choices=("uniform", "beta"))
    parser.add_argument("--density-loss-weight", type=float)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def selected_timestep(decision_file: Path) -> tuple[str, float | None, float | None]:
    payload = json.loads(decision_file.read_text())
    selected = payload["decision"]["selected_timestep_sampling"]
    if selected == "uniform":
        return "uniform", None, None
    if selected == "beta-a1-b0p2":
        return "beta", 1.0, 0.2
    raise RuntimeError(f"unsupported timestep decision: {selected}")


def config_matches(
    config: dict,
    sampling: str,
    alpha: float | None,
    beta: float | None,
    weight: float,
    seed: int,
) -> bool:
    if config.get("seed") != seed:
        return False
    if config.get("timestep_sampling") != sampling:
        return False
    if not math.isclose(float(config.get("density_loss_weight", -1.0)), weight):
        return False
    if sampling == "uniform":
        return True
    return math.isclose(
        float(config["timestep_beta_alpha"]), float(alpha)
    ) and math.isclose(float(config["timestep_beta_beta"]), float(beta))


def result_for_capacity(payload: dict) -> dict | None:
    return next(
        (
            item
            for item in payload["results"]
            if item["kind"] == "trained" and item["capacity"] == 8
        ),
        None,
    )


def collect_rows(
    runs_dir: Path,
    sampling: str,
    alpha: float | None,
    beta: float | None,
) -> list[dict]:
    matched = {}
    for results_path in runs_dir.glob("*/results.json"):
        payload = json.loads(results_path.read_text())
        config = payload["config"]
        match = next(
            (
                (weight, seed)
                for weight in WEIGHTS
                for seed in SEEDS
                if config_matches(config, sampling, alpha, beta, weight, seed)
            ),
            None,
        )
        if match is None:
            continue
        result = result_for_capacity(payload)
        if result is None:
            continue
        history_path = results_path.parent / "training_history.json"
        history = json.loads(history_path.read_text())
        final_training = next(
            item
            for item in reversed(history)
            if item["capacity"] == 8 and item["step"] == 5000
        )
        weight, seed = match
        row = {
            "density_loss_weight": weight,
            "seed": seed,
            "global_velocity_mse": result["global_velocity_mse"],
            "endpoint_kl_source_to_student": result["endpoint_kl_source_to_student"],
            "endpoint_density_l1": result["endpoint_density_l1"],
            "endpoint_sliced_wasserstein": result["endpoint_sliced_wasserstein"],
            "conditional_mean_error": result["query_errors"]["conditional_mean_y"],
            "conditional_variance_error": result["query_errors"][
                "conditional_variance_y"
            ],
            "final_train_loss_mean_100": final_training["loss_mean_100"],
            "final_train_velocity_loss": final_training["velocity_loss"],
            "final_train_density_nll": final_training["density_nll"],
            "run_dir": str(results_path.parent),
        }
        modified = results_path.stat().st_mtime
        if match not in matched or modified > matched[match][0]:
            matched[match] = modified, row
    return [
        matched[(weight, seed)][1]
        for weight in WEIGHTS
        for seed in SEEDS
        if (weight, seed) in matched
    ]


def has_run(
    rows: list[dict],
    weight: float | None,
    seed: int | None,
) -> bool:
    if weight is None or seed is None:
        raise ValueError("--check-run requires --density-loss-weight and --seed")
    return any(
        math.isclose(row["density_loss_weight"], weight) and row["seed"] == seed
        for row in rows
    )


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for weight in WEIGHTS:
        weight_rows = [
            row for row in rows if math.isclose(row["density_loss_weight"], weight)
        ]
        summary = {"density_loss_weight": weight, "seeds": len(weight_rows)}
        for metric in METRICS:
            values = [row[metric] for row in weight_rows]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_std"] = statistics.stdev(values)
        summaries.append(summary)
    return summaries


def select_weight(summaries: list[dict], velocity_tolerance: float) -> dict:
    baseline = next(item for item in summaries if item["density_loss_weight"] == 0.0)
    velocity_limit = baseline["global_velocity_mse_mean"] * (1.0 + velocity_tolerance)
    for item in summaries:
        item["within_velocity_tolerance"] = (
            item["global_velocity_mse_mean"] <= velocity_limit
        )
        item["pareto_optimal"] = not any(
            other is not item
            and other["global_velocity_mse_mean"] <= item["global_velocity_mse_mean"]
            and other["endpoint_kl_source_to_student_mean"]
            <= item["endpoint_kl_source_to_student_mean"]
            and (
                other["global_velocity_mse_mean"] < item["global_velocity_mse_mean"]
                or other["endpoint_kl_source_to_student_mean"]
                < item["endpoint_kl_source_to_student_mean"]
            )
            for other in summaries
        )
    eligible = [item for item in summaries if item["within_velocity_tolerance"]]
    selected = min(
        eligible,
        key=lambda item: (
            item["endpoint_kl_source_to_student_mean"]
            + item["endpoint_kl_source_to_student_std"]
        ),
    )
    selected["selected"] = True
    for item in summaries:
        item.setdefault("selected", False)
    return {
        "rule": (
            "minimum mean-plus-standard-deviation endpoint KL among weights "
            "whose mean velocity MSE is within 10% of the lambda=0 baseline"
        ),
        "velocity_tolerance": velocity_tolerance,
        "baseline_velocity_mse_mean": baseline["global_velocity_mse_mean"],
        "velocity_limit": velocity_limit,
        "selected_density_loss_weight": selected["density_loss_weight"],
    }


def plot_trends(summaries: list[dict], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    weights = [item["density_loss_weight"] for item in summaries]
    panels = (
        ("endpoint_kl_source_to_student", "Endpoint KL(p‖q)"),
        ("global_velocity_mse", "Evaluation velocity MSE"),
        ("final_train_density_nll", "Final sampled density NLL"),
    )
    figure, axes = plt.subplots(1, len(panels), figsize=(13, 3.8))
    for axis, (metric, label) in zip(axes, panels, strict=True):
        means = [item[f"{metric}_mean"] for item in summaries]
        errors = [item[f"{metric}_std"] for item in summaries]
        axis.errorbar(weights, means, yerr=errors, marker="o", capsize=3)
        axis.set_xscale("symlog", linthresh=1.0)
        axis.set_xlabel("Density weight λ")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def format_mean_std(item: dict, metric: str) -> str:
    return f"{item[f'{metric}_mean']:.6f} ± {item[f'{metric}_std']:.6f}"


def write_outputs(
    rows: list[dict],
    summaries: list[dict],
    selection: dict,
    sampling_label: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_seed.csv").open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "timestep_sampling": sampling_label,
                "selection": selection,
                "summaries": summaries,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )

    lines = [
        f"# Density-weight trend with {sampling_label} timesteps (seeds 0-2)",
        "",
        (
            "Selection minimizes mean + sample standard deviation of endpoint "
            "KL, subject to mean evaluation velocity MSE remaining within 10% "
            "of the λ=0 baseline."
        ),
        "",
        (
            "The total training loss scales mechanically with λ. Compare the "
            "separately reported velocity and density terms when reading the trend."
        ),
        "",
        f"Selected density weight: `{selection['selected_density_loss_weight']:g}`.",
        "",
        "| Selected | Pareto | λ | Eval velocity MSE | Endpoint KL | Density L1 | SWD | Final train velocity | Final density NLL | Total loss (100-step mean) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {selected} | {pareto} | {weight:g} | {mse} | {kl} | {density} | "
            "{swd} | {train_velocity} | {train_density} | {train_total} |".format(
                selected="yes" if item["selected"] else "",
                pareto="yes" if item["pareto_optimal"] else "",
                weight=item["density_loss_weight"],
                mse=format_mean_std(item, "global_velocity_mse"),
                kl=format_mean_std(item, "endpoint_kl_source_to_student"),
                density=format_mean_std(item, "endpoint_density_l1"),
                swd=format_mean_std(item, "endpoint_sliced_wasserstein"),
                train_velocity=format_mean_std(item, "final_train_velocity_loss"),
                train_density=format_mean_std(item, "final_train_density_nll"),
                train_total=format_mean_std(item, "final_train_loss_mean_100"),
            )
        )
    (output_dir / "metrics.md").write_text("\n".join(lines) + "\n")
    plot_trends(summaries, output_dir / "weight_trends.png")


def main() -> None:
    args = parse_args()
    if args.check_run:
        sampling = args.timestep_sampling
        if sampling is None:
            raise ValueError("--check-run requires --timestep-sampling")
        alpha, beta = (1.0, 0.2) if sampling == "beta" else (None, None)
    else:
        sampling, alpha, beta = selected_timestep(args.decision_file)
    rows = collect_rows(args.runs_dir, sampling, alpha, beta)
    if args.check_run:
        raise SystemExit(0 if has_run(rows, args.density_loss_weight, args.seed) else 1)

    expected_count = len(WEIGHTS) * len(SEEDS)
    if len(rows) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} completed weight runs, found {len(rows)}"
        )
    summaries = summarize(rows)
    selection = select_weight(summaries, args.velocity_tolerance)
    sampling_label = "uniform" if sampling == "uniform" else "Beta(1, 0.2)"
    write_outputs(rows, summaries, selection, sampling_label, args.output_dir)
    print(f"selected density weight: {selection['selected_density_loss_weight']:g}")
    print(f"weight-trend summary saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
