import argparse
import csv
import json
from pathlib import Path

CHECKPOINT_STEPS = (5_000, 10_000, 15_000, 20_000)
COLUMNS = (
    "run",
    "num_latents",
    "time_conditioning",
    "time_features",
    "learning_rate",
    "parameters",
    "parameter_ratio",
    "train_seconds",
    "final_train_loss",
    "selected_step",
    "selected_validation_bpd",
    "selected_validation_mse",
    "final_validation_bpd",
    "test_bpd_step20000",
    "full_fid_mnist",
    "full_precision",
    "full_recall",
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
        default=Path("outputs/benchmark/dense-hclt-time-tuning"),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as input_file:
        return list(csv.DictReader(input_file))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def float_or_blank(value):
    return "" if value in (None, "") else float(value)


def collect_rows(runs_root: Path) -> tuple[list[dict], dict | None]:
    rows = []
    selections = []
    pattern = "hclt-dense-h*-*-vanilla-noem-tune-step20000-seed0"
    for run_dir in sorted(runs_root.glob(pattern)):
        config = read_json(run_dir / "config.json")
        train_rows = read_csv(run_dir / "metrics.csv")
        validation_rows = read_csv(run_dir / "validation_metrics.csv")
        if not config or not train_rows or not validation_rows:
            continue

        checkpoint_validation = [
            row for row in validation_rows if int(row["step"]) in CHECKPOINT_STEPS
        ]
        if not checkpoint_validation:
            continue
        selected = min(
            checkpoint_validation,
            key=lambda row: (float(row["endpoint_bpd"]), int(row["step"])),
        )
        selected_step = int(selected["step"])
        checkpoint_name = (
            "last.pt" if selected_step == 20_000 else f"step_{selected_step:06d}.pt"
        )
        selections.append(
            {
                "run": run_dir.name,
                "num_latents": config.get("num_latents", ""),
                "run_dir": str(run_dir),
                "checkpoint": checkpoint_name,
                "step": selected_step,
                "validation_bpd": float(selected["endpoint_bpd"]),
                "validation_mse": float(selected["distill_mse"]),
            }
        )

        final_train = train_rows[-1]
        final_validation = validation_rows[-1]
        final_test = read_json(run_dir / "eval" / "last" / "metrics.json")
        selected_eval_stem = Path(checkpoint_name).stem
        selected_test = read_json(
            run_dir / "eval" / selected_eval_stem / "metrics.json"
        )
        if "sample/fid_mnist" not in selected_test:
            quality_screen = read_json(
                run_dir / "eval" / "quality-screen-direct" / "metrics.json"
            )
            if quality_screen.get("model/checkpoint_step") == selected_step:
                selected_test = quality_screen
        time_conditioning = config.get("time_conditioning", "polynomial")
        time_features = (
            config.get("time_embedding_dim", 16)
            if time_conditioning == "fourier"
            else config.get("time_order", 4)
        )
        rows.append(
            {
                "run": run_dir.name,
                "num_latents": config.get("num_latents", ""),
                "time_conditioning": time_conditioning,
                "time_features": time_features,
                "learning_rate": config.get("lr", ""),
                "parameters": config.get(
                    "parameter_count",
                    final_test.get("model/parameter_count", ""),
                ),
                "parameter_ratio": config.get("student_to_teacher_parameter_ratio", ""),
                "train_seconds": float_or_blank(final_train.get("elapsed_seconds")),
                "final_train_loss": float_or_blank(final_train.get("loss")),
                "selected_step": selected_step,
                "selected_validation_bpd": float(selected["endpoint_bpd"]),
                "selected_validation_mse": float(selected["distill_mse"]),
                "final_validation_bpd": float(final_validation["endpoint_bpd"]),
                "test_bpd_step20000": final_test.get("likelihood/test_bpd", ""),
                "full_fid_mnist": selected_test.get("sample/fid_mnist", ""),
                "full_precision": selected_test.get("sample/precision_mnist", ""),
                "full_recall": selected_test.get("sample/recall_mnist", ""),
            }
        )

    best = None
    if selections:
        best = min(
            selections,
            key=lambda item: (item["validation_bpd"], item["step"]),
        )
    return rows, best


def format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    args = parse_args()
    rows, best = collect_rows(args.runs_root)
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

    selection_path = args.output_dir / "best_selection.json"
    selection_path.write_text(json.dumps(best, indent=2) + "\n")
    print(f"wrote {len(rows)} tuning rows to {csv_path} and {markdown_path}")
    print(f"best validation selection: {best}")


if __name__ == "__main__":
    main()
