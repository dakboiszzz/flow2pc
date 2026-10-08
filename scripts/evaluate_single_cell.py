"""Evaluate one teacher first, or compare a student against its saved reference."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from repc.single_cell import (DEFAULT_DATA_DIR, DEFAULT_TEACHER, ROOT,
                             atomic_json, build_single_cell_student,
                             load_single_cell_data, load_single_cell_models,
                             sha256_file, validation_metrics)
from repc.single_cell_eval import fixed_subset, integrate, marginal_scores
from repc.single_cell_tracking import (add_tracking_arguments, dataset_run_name,
                                       init_tracking, log_evaluation)


def make_plot(data, predictions, output):
    fig, axes = plt.subplots(1, len(data.clouds), figsize=(16, 4), sharex=True, sharey=True)
    for index, ax in enumerate(axes):
        real = data.metric_coordinates(data.clouds[index])
        ax.scatter(real[::max(1, len(real) // 1500), 0],
                   real[::max(1, len(real) // 1500), 1], s=2, alpha=.3, label="Observed")
        predicted = data.metric_coordinates(predictions[index])
        ax.scatter(predicted[:, 0], predicted[:, 1], s=2, alpha=.3, label="Predicted")
        day = data.contract["preprocessing"]["time_order"][index]
        ax.set_title(f"Day {day}" + (" (held out)" if index == data.heldout else ""))
        ax.set_xlabel("PCA component 1 (metric space)")
    axes[0].set_ylabel("PCA component 2 (metric space)")
    axes[-1].legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/reference/cite_5D/day3_seed42")
    parser.add_argument("--student-checkpoint", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--num-samples", type=int, default=2048, help="0 uses all cells")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--wandb-run-dir", type=Path,
                        help="resume the student's training W&B run from this directory")
    add_tracking_arguments(parser)
    args = parser.parse_args()
    if (args.num_samples < 0 or args.batch_size < 1 or args.threads < 1
            or args.tolerance <= 0 or args.wandb_service_wait <= 0):
        parser.error("sample count must be nonnegative; batch, threads, and tolerance positive")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error(f"CUDA device {args.device} is unavailable")
        torch.cuda.set_device(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_single_cell_data(args.teacher_checkpoint, args.data_dir)
    teacher, generator = load_single_cell_models(args.teacher_checkpoint, device)
    protocol = {"num_samples": args.num_samples, "seed": args.seed, "source_index": 0,
                "solver": "dopri5", "tolerance": args.tolerance, "batch_size": args.batch_size,
                "metric_space": data.identity["metric_space"]}
    source = fixed_subset(data.clouds[0], args.num_samples, args.seed)
    times = torch.linspace(0, 1, len(data.clouds))
    reference = None
    if args.reference:
        reference = json.loads(args.reference.read_text())
        if (reference.get("identity", {}).get("checkpoint_sha256") != data.identity.get("checkpoint_sha256") or reference["protocol"] != protocol
                or reference.get("model") != "teacher"):
            raise ValueError("reference checkpoint/data/protocol differs from this evaluation")
    student = None
    checkpoint_step = 0
    if args.student_checkpoint:
        if reference is None:
            parser.error("student evaluation requires --reference")
        checkpoint = torch.load(args.student_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("data_identity", {}).get("checkpoint_sha256") != data.identity.get("checkpoint_sha256"):
            raise ValueError("student was trained against a different teacher/data contract")
        if checkpoint["reference_sha256"] != sha256_file(args.reference):
            raise ValueError("student was trained with a different reference file")
        student = build_single_cell_student(checkpoint["model_config"]).to(device)
        student.load_state_dict(checkpoint["state_dict"])
        checkpoint_step = checkpoint.get("step", 0)
    if args.wandb_run_dir and student is None:
        parser.error("--wandb-run-dir is only for student evaluation")
    cached_result = None
    existing_metrics = args.output_dir / "metrics.json"
    if student is None and existing_metrics.exists():
        cached_result = json.loads(existing_metrics.read_text())
        if (cached_result.get("model") != "teacher" or cached_result.get("identity", {}).get("checkpoint_sha256") != data.identity.get("checkpoint_sha256")
                or cached_result["protocol"] != protocol):
            raise ValueError("existing teacher reference differs; use a new --output-dir")
    tracking = None
    atomic_json(args.output_dir / "status.json", {"status": "running", "wandb_mode": args.wandb_mode})
    try:
        tracking = init_tracking(
            args, args.wandb_run_dir or args.output_dir,
            {"evaluation_protocol": protocol, "teacher_identity": data.identity,
             "student_checkpoint": str(args.student_checkpoint.resolve()) if student is not None else None},
            "student-evaluation" if student is not None else "teacher-evaluation",
            name=dataset_run_name(data.contract, (args.wandb_run_dir or args.output_dir).name),
        )
        if cached_result is not None:
            # Keep the reference byte-for-byte immutable for student resumes.
            log_evaluation(tracking, cached_result, args.output_dir)
            print(f"Reused teacher reference: {existing_metrics}", flush=True)
        else:
            evaluate(args, data, teacher, generator, student, reference, source, times,
                     protocol, device, tracking, checkpoint_step)
    except BaseException as exc:
        atomic_json(args.output_dir / "status.json", {"status": "failed", "error": str(exc),
                                                      "wandb_mode": args.wandb_mode})
        if tracking is not None:
            tracking.summary["evaluation_error"] = str(exc)
        raise
    finally:
        if tracking is not None:
            tracking.finish()
    atomic_json(args.output_dir / "status.json", {"status": "complete", "wandb_mode": args.wandb_mode})


def evaluate(args, data, teacher, generator, student, reference, source, times,
             protocol, device, tracking, checkpoint_step):
    started = time.perf_counter()
    predictions, nfe = integrate(student or teacher, source, times, device,
                                  args.batch_size, args.tolerance, student is not None)
    integration_seconds = time.perf_counter() - started
    direct = None
    if student is not None:
        sample_rng = torch.Generator(device=device).manual_seed(args.seed)
        sample_started = time.perf_counter()
        with torch.no_grad():
            direct = student.sample_path(len(source), times.to(device), sample_rng).cpu()
        direct_seconds = time.perf_counter() - sample_started
    scores = marginal_scores(data, args.num_samples, args.seed, times,
                             predictions=predictions, direct=direct, student=student, device=device)
    for score in scores:
        print(json.dumps(score), flush=True)
    result = {"model": "student" if student is not None else "teacher",
              "identity": data.identity, "protocol": protocol, "data_contract": data.contract,
              "scores": scores, "heldout_ode_emd": scores[data.heldout]["ode_emd"],
              "integration_seconds": integration_seconds, "ode_nfe_total": nfe}
    if student is not None:
        result.update({"student_checkpoint": str(args.student_checkpoint.resolve()),
                       "student_checkpoint_sha256": sha256_file(args.student_checkpoint),
                       "checkpoint_step": checkpoint_step,
                       "heldout_direct_emd": scores[data.heldout]["direct_emd"],
                       "reference_heldout_ode_emd": reference["heldout_ode_emd"],
                       "direct_sampling_seconds": direct_seconds,
                       "validation": validation_metrics(student, teacher, generator, data,
                                                          device, args.seed)})
    atomic_json(args.output_dir / "metrics.json", result)
    make_plot(data, predictions, args.output_dir / "ode_marginals.png")
    if direct is not None:
        make_plot(data, direct, args.output_dir / "direct_marginals.png")
    log_evaluation(tracking, result, args.output_dir, checkpoint_step)
    print(f"Saved {args.output_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
