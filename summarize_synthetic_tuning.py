import argparse
import csv
import json
import math
from pathlib import Path

# The coarse optimum landed on both search boundaries. These extra candidates
# bracket the density weight and endpoint bias without changing architecture,
# optimizer, or training budget.
EXPECTED_CANDIDATES = (
    (0.0, "uniform", None, None),
    (0.1, "uniform", None, None),
    (1.0, "uniform", None, None),
    (10.0, "uniform", None, None),
    (30.0, "uniform", None, None),
    (100.0, "uniform", None, None),
    (0.0, "beta", 1.0, 0.8),
    (0.1, "beta", 1.0, 0.8),
    (1.0, "beta", 1.0, 0.8),
    (10.0, "beta", 1.0, 0.8),
    (30.0, "beta", 1.0, 0.8),
    (100.0, "beta", 1.0, 0.8),
    (10.0, "beta", 1.0, 0.5),
    (30.0, "beta", 1.0, 0.5),
    (100.0, "beta", 1.0, 0.5),
    (10.0, "beta", 1.0, 0.2),
    (30.0, "beta", 1.0, 0.2),
    (100.0, "beta", 1.0, 0.2),
    (10.0, "beta", 1.0, 0.1),
    (10.0, "beta", 1.0, 0.05),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("outputs/runs/synthetic"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/benchmark/synthetic-density-timestep-tuning"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--velocity-tolerance", type=float, default=0.10)
    return parser.parse_args()


def candidate_key(config: dict) -> tuple[float, str, float | None, float | None]:
    weight = float(config["density_loss_weight"])
    sampling = config["timestep_sampling"]
    if sampling == "uniform":
        return weight, sampling, None, None
    return (
        weight,
        sampling,
        float(config["timestep_beta_alpha"]),
        float(config["timestep_beta_beta"]),
    )


def collect_rows(args: argparse.Namespace) -> list[dict]:
    candidates = {}
    for results_path in args.runs_dir.glob("*/results.json"):
        payload = json.loads(results_path.read_text())
        config = payload["config"]
        if config.get("seed") != args.seed:
            continue
        key = candidate_key(config)
        matching_key = next(
            (
                expected
                for expected in EXPECTED_CANDIDATES
                if key[:2] == expected[:2]
                and all(
                    actual is expected_value
                    or (
                        actual is not None
                        and expected_value is not None
                        and math.isclose(actual, expected_value)
                    )
                    for actual, expected_value in zip(
                        key[2:], expected[2:], strict=True
                    )
                )
            ),
            None,
        )
        if matching_key is None:
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
            "density_loss_weight": key[0],
            "timestep_sampling": key[1],
            "timestep_beta_alpha": (
                config["timestep_beta_alpha"] if key[1] == "beta" else ""
            ),
            "timestep_beta_beta": (
                config["timestep_beta_beta"] if key[1] == "beta" else ""
            ),
            "global_velocity_mse": result["global_velocity_mse"],
            "endpoint_kl_source_to_student": result["endpoint_kl_source_to_student"],
            "endpoint_density_l1": result["endpoint_density_l1"],
            "endpoint_sliced_wasserstein": result["endpoint_sliced_wasserstein"],
            "conditional_mean_error": result["query_errors"]["conditional_mean_y"],
            "conditional_variance_error": result["query_errors"][
                "conditional_variance_y"
            ],
            "event_probability_error": result["query_errors"][
                "event_probability_x_gt_threshold"
            ],
            "run_dir": str(results_path.parent),
        }
        previous = candidates.get(matching_key)
        if previous is None or results_path.stat().st_mtime > previous[0]:
            candidates[matching_key] = (results_path.stat().st_mtime, row)
    return [candidates[key][1] for key in EXPECTED_CANDIDATES if key in candidates]


def mark_selection(rows: list[dict], velocity_tolerance: float) -> dict:
    baseline = next(
        row
        for row in rows
        if row["density_loss_weight"] == 0.0 and row["timestep_sampling"] == "uniform"
    )
    velocity_limit = baseline["global_velocity_mse"] * (1.0 + velocity_tolerance)
    for row in rows:
        row["within_velocity_tolerance"] = row["global_velocity_mse"] <= velocity_limit
        row["pareto_optimal"] = not any(
            other is not row
            and other["global_velocity_mse"] <= row["global_velocity_mse"]
            and other["endpoint_kl_source_to_student"]
            <= row["endpoint_kl_source_to_student"]
            and (
                other["global_velocity_mse"] < row["global_velocity_mse"]
                or other["endpoint_kl_source_to_student"]
                < row["endpoint_kl_source_to_student"]
            )
            for other in rows
        )
    eligible = [row for row in rows if row["within_velocity_tolerance"]]
    selected = min(eligible, key=lambda row: row["endpoint_kl_source_to_student"])
    selected["selected"] = True
    for row in rows:
        row.setdefault("selected", False)
    return {
        "baseline_velocity_mse": baseline["global_velocity_mse"],
        "velocity_tolerance": velocity_tolerance,
        "velocity_limit": velocity_limit,
        "selected": selected,
    }


def write_outputs(rows: list[dict], selection: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with (output_dir / "metrics.csv").open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps({"selection": selection, "rows": rows}, indent=2) + "\n"
    )

    lines = [
        "# Synthetic density-weight and timestep tuning",
        "",
        (
            "Selection rule: minimum endpoint KL among runs with velocity MSE "
            f"within {selection['velocity_tolerance']:.0%} of the velocity-only "
            "uniform baseline."
        ),
        "",
        "| Selected | Pareto | λ_den | Time | Velocity MSE | KL(p‖q) | Density L1 | SWD | Cond. mean err. | Cond. var. err. |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        time_label = row["timestep_sampling"]
        if time_label == "beta":
            time_label = (
                f"Beta({row['timestep_beta_alpha']:g}, {row['timestep_beta_beta']:g})"
            )
        lines.append(
            "| {selected} | {pareto} | {weight:g} | {time} | {mse:.6f} | "
            "{kl:.6f} | {density:.6f} | {swd:.6f} | {mean:.6f} | {variance:.6f} |".format(
                selected="yes" if row["selected"] else "",
                pareto="yes" if row["pareto_optimal"] else "",
                weight=row["density_loss_weight"],
                time=time_label,
                mse=row["global_velocity_mse"],
                kl=row["endpoint_kl_source_to_student"],
                density=row["endpoint_density_l1"],
                swd=row["endpoint_sliced_wasserstein"],
                mean=row["conditional_mean_error"],
                variance=row["conditional_variance_error"],
            )
        )
    (output_dir / "metrics.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    rows = collect_rows(args)
    expected_count = len(EXPECTED_CANDIDATES)
    if len(rows) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} completed candidates, found {len(rows)}"
        )
    selection = mark_selection(rows, args.velocity_tolerance)
    write_outputs(rows, selection, args.output_dir)
    selected = selection["selected"]
    print(
        f"selected lambda={selected['density_loss_weight']:g} "
        f"time={selected['timestep_sampling']} "
        f"mse={selected['global_velocity_mse']:.6f} "
        f"kl={selected['endpoint_kl_source_to_student']:.6f}"
    )
    print(f"summary saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
