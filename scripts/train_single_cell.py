"""Distill one evaluated single-cell ALI-CFM bundle into a dense HCLT."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from repc.single_cell import (atomic_json, build_single_cell_student,
                             initialize_source, learn_topology,
                             load_single_cell_data, load_single_cell_models,
                             observed_batch, sample_paths, sha256_file,
                             validation_metrics)
from repc.single_cell_eval import direct_diagnostics
from repc.single_cell_tracking import (add_tracking_arguments, dataset_run_name, emd_scalars, init_tracking,
                                       log_files, log_training)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON overrides for student settings")
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-latents", type=int, default=64)
    parser.add_argument("--time-embedding-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--emd-every", type=int, default=5000,
                        help="direct-sampling EMD monitoring interval; 0 disables it")
    parser.add_argument("--emd-samples", type=int, default=512)
    add_tracking_arguments(parser)
    preliminary = parser.parse_args()
    if preliminary.config:
        overrides = json.loads(preliminary.config.read_text())
        allowed = {"num_latents", "time_embedding_dim", "lr", "steps",
                   "batch_size", "seed", "validate_every",
                   "checkpoint_every", "log_every", "emd_every", "emd_samples",
                   "wandb_mode", "wandb_project", "wandb_entity", "wandb_group", "wandb_service_wait"}
        if set(overrides) - allowed:
            parser.error(f"unsupported student settings: {sorted(set(overrides) - allowed)}")
        parser.set_defaults(**overrides)
    args = parser.parse_args()
    for name in ("steps", "batch_size", "threads", "validate_every", "checkpoint_every", "log_every"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if (args.num_latents < 2 or args.time_embedding_dim < 2 or args.time_embedding_dim % 2
            or args.lr <= 0):
        parser.error("invalid circuit capacity or learning rate")
    if args.wandb_mode not in ("disabled", "offline", "online"):
        parser.error("invalid wandb_mode")
    if args.emd_every < 0 or args.emd_samples < 1 or args.wandb_service_wait <= 0:
        parser.error("EMD interval must be nonnegative; EMD samples and W&B timeout positive")
    return args


def atomic_checkpoint(path, checkpoint):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def update(student, optimizer, loss):
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite training objective")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    if not torch.isfinite(gradient):
        raise FloatingPointError("non-finite gradient")
    optimizer.step()
    return float(gradient)


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {args.device} is unavailable")
        torch.cuda.set_device(device)
    reference = json.loads(args.reference.read_text())
    data = load_single_cell_data(args.teacher_checkpoint, args.data_dir)
    if reference.get("model") != "teacher" or reference.get("identity", {}).get("checkpoint_sha256") != data.identity.get("checkpoint_sha256"):
        raise ValueError("evaluate this teacher/data contract before starting distillation")
    teacher, generator = load_single_cell_models(args.teacher_checkpoint, device)
    reference_hash = sha256_file(args.reference)
    output = args.output_dir
    checkpoints = output / "checkpoints"
    last_path = checkpoints / "last.pt"
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"{output} is not empty; use --resume or a new output directory")
    saved = None
    if args.resume and last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        model_config = saved["model_config"]
    else:
        model_config = {"name": "hclt", "parents": learn_topology(data),
                        "num_latents": args.num_latents, "time_conditioning": "fourier",
                        "time_embedding_dim": args.time_embedding_dim,
                        "time_feature_seed": args.seed, "fourier_zero_center": False,
                        "topology_protocol": "observed_train_histogram_mi_32_bins"}
    student = build_single_cell_student(model_config).to(device)
    settings = {key: value for key, value in vars(args).items()
                if key not in ("config", "resume", "device", "threads", "output_dir")}
    settings = {key: str(value.resolve()) if isinstance(value, Path) else value
                for key, value in settings.items()}
    settings.update({"objective": "velocity_mse", "selection_rule": "validation_velocity_mse"})
    if saved:
        saved_config = saved.get("training_config", {}).copy()
        current_settings = settings.copy()
        for k in ("teacher_checkpoint", "data_dir", "reference"):
            saved_config.pop(k, None)
            current_settings.pop(k, None)
        if (saved.get("data_identity", {}).get("checkpoint_sha256") != data.identity.get("checkpoint_sha256")
                or saved_config != current_settings
                or saved["reference_sha256"] != reference_hash):
            raise ValueError("resume inputs/configuration/reference differ from the saved run")
    checkpoints.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "config.json", {"training": settings, "model": model_config,
                                        "data_identity": data.identity,
                                        "data_contract": data.contract,
                                        "reference_sha256": reference_hash})
    tracking = None
    try:
        tracking = init_tracking(
            args, output,
            {**settings, "model_config": model_config,
             "teacher_identity": data.identity, "data_contract": data.contract,
             "reference_protocol": reference["protocol"],
             "emd_monitor_protocol": {"num_samples": args.emd_samples,
                                      "seed": 2026, "method": "direct_ancestral",
                                      "metric_space": data.identity["metric_space"]}},
            "distillation",
            name=dataset_run_name(data.contract, output.name),
        )
        if tracking is not None:
            tracking.log(emd_scalars(reference, "reference"))
            tracking.summary.update(emd_scalars(reference, "reference"))
        train(args, data, student, teacher, generator, model_config, settings,
              reference, reference_hash, saved, tracking)
    except BaseException as exc:
        atomic_json(output / "status.json", {"status": "failed", "error": str(exc)})
        if tracking is not None:
            tracking.summary.update({"training_status": "failed", "training_error": str(exc)})
        raise
    finally:
        if tracking is not None:
            tracking.finish()


def train(args, data, student, teacher, generator, model_config, settings,
          reference, reference_hash, saved, tracking):
    device = torch.device(args.device)
    output = args.output_dir
    checkpoints = output / "checkpoints"
    last_path = checkpoints / "last.pt"
    started = time.perf_counter()
    path_rng = torch.Generator().manual_seed(args.seed + 10000)
    marginal_rng = torch.Generator().manual_seed(args.seed + 20000)
    best_score = math.inf
    best_metrics = None
    start_step = 1
    if saved:
        student.load_state_dict(saved["state_dict"])
    else:
        # Set initial Gaussian means/scales directly; there is no likelihood training phase.
        initialize_source(student, data.train[0], marginal_rng)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    if saved:
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        path_rng.set_state(saved["path_rng_state"])
        marginal_rng.set_state(saved["marginal_rng_state"])
        best_score = saved["best_validation_score"]
        best_metrics = saved["best_validation_metrics"]
        start_step = saved["step"] + 1

    def checkpoint(step):
        return {"model_type": "distill", "model_config": model_config,
                "state_dict": student.state_dict(), "data_identity": data.identity,
                "data_contract": data.contract, "training_config": settings,
                "reference_sha256": reference_hash, "step": step,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "path_rng_state": path_rng.get_state(), "marginal_rng_state": marginal_rng.get_state(),
                "best_validation_score": best_score, "best_validation_metrics": best_metrics}

    atomic_json(output / "status.json", {"status": "running", "start_step": start_step})
    metrics_path = output / "metrics.csv"
    fields = ["step", "loss", "velocity_mse", "observed_nll_nats_per_dim", "gradient_norm",
              "learning_rate", "elapsed_seconds", "validation_score",
              "validation_velocity_mse", "validation_relative_velocity_mse", "validation_observed_nll",
              "monitor_heldout_direct_emd", "monitor_seconds"]
    try:
        with metrics_path.open("a" if saved else "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if not saved:
                writer.writeheader()
            for step in range(start_step, args.steps + 1):
                student.train()
                samples, times = sample_paths(data, data.train, generator,
                                              args.batch_size, path_rng, device)
                with torch.no_grad():
                    target = teacher(times, samples)
                velocity_mse = (student.velocity(samples, times) - target).square().mean()
                observed, observed_times = observed_batch(data, data.train, args.batch_size,
                                                           marginal_rng, device)
                with torch.no_grad():
                    nll = -student.log_prob(observed, observed_times).mean() / data.dim
                loss = velocity_mse
                learning_rate = optimizer.param_groups[0]["lr"]
                gradient = update(student, optimizer, loss)
                scheduler.step()
                row = {"step": step, "loss": loss.item(), "velocity_mse": velocity_mse.item(),
                       "observed_nll_nats_per_dim": nll.item(), "gradient_norm": gradient,
                       "learning_rate": learning_rate, "elapsed_seconds": time.perf_counter() - started}
                if step == 1 or step % args.validate_every == 0 or step == args.steps:
                    validation = validation_metrics(student, teacher, generator, data, device, 2026)
                    # Selection uses teacher fidelity on observed validation cells.
                    score = validation["velocity_mse"]
                    row.update({"validation_score": score,
                                "validation_velocity_mse": validation["velocity_mse"],
                                "validation_relative_velocity_mse": validation["relative_velocity_mse"],
                                "validation_observed_nll": validation["observed_nll_nats_per_dim"]})
                    if score < best_score:
                        best_score, best_metrics = score, validation
                        atomic_checkpoint(checkpoints / "best.pt", checkpoint(step))
                    print(f"validation {step}: mse={validation['velocity_mse']:.6f} "
                          f"relative_mse={validation['relative_velocity_mse']:.4f} score={score:.6f}", flush=True)
                if args.emd_every and (step % args.emd_every == 0 or step == args.steps):
                    diagnostics = direct_diagnostics(student, data, device, args.emd_samples, 2026)
                    diagnostics["step"] = step
                    atomic_json(output / "monitor" / f"step_{step:06d}.json", diagnostics)
                    emd_path = output / "emd.csv"
                    needs_header = not emd_path.exists()
                    with emd_path.open("a", newline="") as emd_stream:
                        emd_writer = csv.DictWriter(emd_stream, fieldnames=["step", *diagnostics["scores"][0]])
                        if needs_header:
                            emd_writer.writeheader()
                        emd_writer.writerows({"step": step, **score} for score in diagnostics["scores"])
                    row.update({"monitor_heldout_direct_emd": diagnostics["heldout_direct_emd"],
                                "monitor_seconds": diagnostics["seconds"]})
                    if tracking is not None:
                        tracking.log({"train/step": step, **emd_scalars(diagnostics, "monitor")})
                    print(f"EMD monitor {step}: heldout_direct_emd={diagnostics['heldout_direct_emd']:.6f} "
                          f"samples={args.emd_samples}", flush=True)
                if step == 1 or step % args.checkpoint_every == 0 or step == args.steps:
                    current = checkpoint(step)
                    atomic_checkpoint(last_path, current)
                    if step % args.checkpoint_every == 0:
                        atomic_checkpoint(checkpoints / f"step_{step:06d}.pt", current)
                if (step == 1 or step % args.log_every == 0 or step == args.steps
                        or "validation_score" in row or "monitor_heldout_direct_emd" in row):
                    writer.writerow(row)
                    stream.flush()
                    log_training(tracking, row)
                    print(f"step {step}/{args.steps}: loss={loss.item():.6f} "
                          f"elapsed={row['elapsed_seconds']:.1f}s", flush=True)
        atomic_json(output / "status.json", {"status": "complete", "steps": args.steps,
                                             "best_validation_score": best_score,
                                             "best_validation_metrics": best_metrics,
                                             "seconds_this_invocation": time.perf_counter() - started})
        if tracking:
            tracking.summary.update({"best_validation_score": best_score,
                                     "reference_heldout_emd": reference["heldout_ode_emd"],
                                     "training_status": "complete"})
            if best_metrics is not None:
                tracking.summary.update({f"best_validation/{key}": value
                                         for key, value in best_metrics.items()})
            log_files(tracking, "student", {
                "best.pt": checkpoints / "best.pt", "last.pt": last_path,
                "config.json": output / "config.json", "metrics.csv": metrics_path,
                "emd.csv": output / "emd.csv", "status.json": output / "status.json",
                "reference.json": args.reference,
                **{str(path.relative_to(output)): path for path in sorted((output / "monitor").glob("step_*.json"))}})
    except BaseException as exc:
        atomic_json(output / "status.json", {"status": "failed", "error": str(exc)})
        if tracking is not None:
            tracking.summary.update({"training_status": "failed", "training_error": str(exc)})
        raise
    print(f"Completed {output}; best validation score {best_score:.6f}", flush=True)


if __name__ == "__main__":
    main()
