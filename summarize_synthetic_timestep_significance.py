import csv
import itertools
import json
import math
import statistics
from pathlib import Path

CANDIDATES = {
    "uniform": (10.0, "uniform", None, None),
    "beta-a1-b0p2": (10.0, "beta", 1.0, 0.2),
}
SEEDS = tuple(range(8))
METRICS = (
    "global_velocity_mse",
    "endpoint_kl_source_to_student",
    "endpoint_density_l1",
    "endpoint_sliced_wasserstein",
    "conditional_mean_error",
    "conditional_variance_error",
)
PRIMARY_METRIC = "endpoint_kl_source_to_student"
SIGNIFICANCE_LEVEL = 0.05


def config_key(config: dict) -> tuple[float, str, float | None, float | None]:
    if config["timestep_sampling"] == "uniform":
        return float(config["density_loss_weight"]), "uniform", None, None
    return (
        float(config["density_loss_weight"]),
        "beta",
        float(config["timestep_beta_alpha"]),
        float(config["timestep_beta_beta"]),
    )


def keys_match(
    actual: tuple[float, str, float | None, float | None],
    expected: tuple[float, str, float | None, float | None],
) -> bool:
    if actual[1] != expected[1] or not math.isclose(actual[0], expected[0]):
        return False
    return all(
        left is right
        or (
            left is not None
            and right is not None
            and math.isclose(float(left), float(right))
        )
        for left, right in zip(actual[2:], expected[2:], strict=True)
    )


def collect_rows(runs_dir: Path) -> list[dict]:
    matched = {}
    for results_path in runs_dir.glob("*/results.json"):
        payload = json.loads(results_path.read_text())
        config = payload["config"]
        seed = config.get("seed")
        if seed not in SEEDS:
            continue
        actual_key = config_key(config)
        candidate = next(
            (
                name
                for name, expected_key in CANDIDATES.items()
                if keys_match(actual_key, expected_key)
            ),
            None,
        )
        if candidate is None:
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
        modified = results_path.stat().st_mtime
        if previous is None or modified > previous[0]:
            matched[match_key] = modified, row

    expected = {(candidate, seed) for candidate in CANDIDATES for seed in SEEDS}
    missing = sorted(expected - set(matched))
    if missing:
        raise RuntimeError(f"missing paired timestep runs: {missing}")
    return [matched[(candidate, seed)][1] for seed in SEEDS for candidate in CANDIDATES]


def exact_paired_sign_flip_test(differences: list[float]) -> float:
    """Return the exact two-sided randomization p-value for paired differences."""
    observed = abs(statistics.mean(differences))
    permuted = (
        abs(statistics.mean(sign * value for sign, value in zip(signs, differences)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    )
    extreme = sum(value >= observed - 1e-15 for value in permuted)
    return extreme / (2 ** len(differences))


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    aggregates = []
    by_key = {(row["candidate"], row["seed"]): row for row in rows}
    for candidate in CANDIDATES:
        candidate_rows = [by_key[(candidate, seed)] for seed in SEEDS]
        aggregate = {"candidate": candidate, "seeds": len(candidate_rows)}
        for metric in METRICS:
            values = [row[metric] for row in candidate_rows]
            aggregate[f"{metric}_mean"] = statistics.mean(values)
            aggregate[f"{metric}_std"] = statistics.stdev(values)
        aggregates.append(aggregate)

    paired_tests = []
    for metric in METRICS:
        differences = [
            by_key[("uniform", seed)][metric] - by_key[("beta-a1-b0p2", seed)][metric]
            for seed in SEEDS
        ]
        difference_mean = statistics.mean(differences)
        difference_std = statistics.stdev(differences)
        uniform_mean = statistics.mean(
            by_key[("uniform", seed)][metric] for seed in SEEDS
        )
        paired_tests.append(
            {
                "metric": metric,
                "difference_definition": "uniform - beta",
                "mean_paired_difference": difference_mean,
                "std_paired_difference": difference_std,
                "relative_beta_improvement": (
                    difference_mean / uniform_mean if uniform_mean != 0.0 else None
                ),
                "cohen_dz": (
                    difference_mean / difference_std if difference_std != 0.0 else None
                ),
                "exact_two_sided_p": exact_paired_sign_flip_test(differences),
            }
        )
    return aggregates, paired_tests


def write_outputs(
    rows: list[dict],
    aggregates: list[dict],
    paired_tests: list[dict],
    output_dir: Path,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_test = next(
        test for test in paired_tests if test["metric"] == PRIMARY_METRIC
    )
    beta_is_better = primary_test["mean_paired_difference"] > 0.0
    significant = primary_test["exact_two_sided_p"] < SIGNIFICANCE_LEVEL
    selected = "beta-a1-b0p2" if beta_is_better and significant else "uniform"
    decision = {
        "primary_metric": PRIMARY_METRIC,
        "significance_level": SIGNIFICANCE_LEVEL,
        "rule": (
            "select beta only when its mean endpoint KL is lower and the exact "
            "paired two-sided sign-flip p-value is below 0.05; otherwise select "
            "uniform"
        ),
        "selected_timestep_sampling": selected,
        "beta_is_better": beta_is_better,
        "significant": significant,
    }

    with (output_dir / "per_seed.csv").open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "decision": decision,
                "aggregates": aggregates,
                "paired_tests": paired_tests,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )

    lines = [
        "# Uniform vs Beta(1, 0.2) timestep sampling (seeds 0-7)",
        "",
        (
            "Predefined decision: select Beta only if its endpoint KL is lower "
            "with an exact paired two-sided sign-flip `p < 0.05`; otherwise "
            "select uniform."
        ),
        "",
        f"Selected timestep scheme: `{selected}`.",
        "",
        "| Candidate | Velocity MSE | KL(p‖q) | Density L1 | SWD | Cond. mean | Cond. var. |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for aggregate in aggregates:
        values = [aggregate["candidate"]]
        for metric in METRICS:
            values.append(
                f"{aggregate[f'{metric}_mean']:.6f} ± {aggregate[f'{metric}_std']:.6f}"
            )
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Positive paired differences mean Beta has the lower error.",
            "",
            "| Metric | Mean paired difference | Relative Beta improvement | Cohen dz | Exact two-sided p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for test in paired_tests:
        relative = test["relative_beta_improvement"]
        effect = test["cohen_dz"]
        lines.append(
            "| {metric} | {difference:.6f} | {relative} | {effect} | {p:.6f} |".format(
                metric=test["metric"],
                difference=test["mean_paired_difference"],
                relative=(f"{relative:.1%}" if relative is not None else "n/a"),
                effect=(f"{effect:.3f}" if effect is not None else "n/a"),
                p=test["exact_two_sided_p"],
            )
        )
    (output_dir / "metrics.md").write_text("\n".join(lines) + "\n")
    return selected


def main() -> None:
    rows = collect_rows(Path("outputs/runs/synthetic"))
    aggregates, paired_tests = summarize(rows)
    output_dir = Path("outputs/benchmark/synthetic-timestep-significance")
    selected = write_outputs(rows, aggregates, paired_tests, output_dir)
    print(f"selected timestep scheme: {selected}")
    print(f"significance summary saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
