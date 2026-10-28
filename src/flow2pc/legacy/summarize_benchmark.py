import argparse
import csv
import json
from pathlib import Path

COLUMNS = (
    "run",
    "step",
    "parameters",
    "train_seconds",
    "test_bpd",
    "fid_mnist",
    "kid_mnist",
    "precision",
    "recall",
    "class_entropy",
    "sample_images_per_second",
    "likelihood_examples_per_second",
    "fid_euler20",
    "precision_euler20",
    "recall_euler20",
    "euler20_images_per_second",
    "fid_rk4_20",
    "precision_rk4_20",
    "recall_rk4_20",
    "rk4_20_images_per_second",
    "fid_dopri5",
    "precision_dopri5",
    "recall_dopri5",
    "dopri5_images_per_second",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("outputs/runs/distill"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/benchmark/vanilla-noem"),
    )
    return parser.parse_args()


def metric(metrics: dict, key: str):
    return metrics.get(key, "")


def collect_rows(runs_root: Path) -> list[dict]:
    rows = []
    for metrics_path in sorted(runs_root.glob("*/eval/*/metrics.json")):
        run_name = metrics_path.parents[2].name
        if "vanilla-noem-step20000" not in run_name:
            continue
        metrics = json.loads(metrics_path.read_text())
        rows.append(
            {
                "run": run_name,
                "step": metric(metrics, "model/checkpoint_step"),
                "parameters": metric(metrics, "model/parameter_count"),
                "train_seconds": metric(metrics, "performance/train_seconds"),
                "test_bpd": metric(metrics, "likelihood/test_bpd"),
                "fid_mnist": metric(metrics, "sample/fid_mnist"),
                "kid_mnist": metric(metrics, "sample/kid_mnist_mean"),
                "precision": metric(metrics, "sample/precision_mnist"),
                "recall": metric(metrics, "sample/recall_mnist"),
                "class_entropy": metric(
                    metrics,
                    "sample/class_entropy_normalized",
                ),
                "sample_images_per_second": metric(
                    metrics,
                    "performance/sample_images_per_second",
                ),
                "likelihood_examples_per_second": metric(
                    metrics,
                    "performance/likelihood_examples_per_second",
                ),
                "fid_euler20": metric(metrics, "sample_euler_20/fid_mnist"),
                "precision_euler20": metric(
                    metrics,
                    "sample_euler_20/precision_mnist",
                ),
                "recall_euler20": metric(
                    metrics,
                    "sample_euler_20/recall_mnist",
                ),
                "euler20_images_per_second": metric(
                    metrics,
                    "performance/euler_20_images_per_second",
                ),
                "fid_rk4_20": metric(metrics, "sample_rk4_20/fid_mnist"),
                "precision_rk4_20": metric(
                    metrics,
                    "sample_rk4_20/precision_mnist",
                ),
                "recall_rk4_20": metric(
                    metrics,
                    "sample_rk4_20/recall_mnist",
                ),
                "rk4_20_images_per_second": metric(
                    metrics,
                    "performance/rk4_20_images_per_second",
                ),
                "fid_dopri5": metric(metrics, "sample_dopri5/fid_mnist"),
                "precision_dopri5": metric(
                    metrics,
                    "sample_dopri5/precision_mnist",
                ),
                "recall_dopri5": metric(
                    metrics,
                    "sample_dopri5/recall_mnist",
                ),
                "dopri5_images_per_second": metric(
                    metrics,
                    "performance/dopri5_images_per_second",
                ),
            }
        )
    return sorted(rows, key=lambda row: (row["run"], row["step"]))


def format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.runs_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "metrics.csv"
    with csv_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = args.output_dir / "metrics.md"
    lines = [
        "| " + " | ".join(COLUMNS) + " |",
        "| " + " | ".join("---" for _ in COLUMNS) + " |",
    ]
    lines.extend(
        "| " + " | ".join(format_value(row[column]) for column in COLUMNS) + " |"
        for row in rows
    )
    markdown_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(rows)} evaluations to {csv_path} and {markdown_path}")


if __name__ == "__main__":
    main()
