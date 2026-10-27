"""Distill diagonal EMA velocity on cached OTP tuples using the shared ALI recipe."""

import argparse
import csv
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from repc.otpfm import DEFAULT_OTP_ROOT, load_teacher, move_cache, sample_cache
from repc.otpfm_eval import comparison_plots, log_comparison, score_paths, velocity_validation
from repc.otpfm_student import build_student, initialize_student, build_optimizer
from repc.single_cell import atomic_json, sha256_file
from repc.single_cell_eval import integrate
from repc.single_cell_tracking import add_tracking_arguments, init_tracking, log_files
from scripts.prepare_otpfm import atomic_torch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--otp-root", type=Path, default=DEFAULT_OTP_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--student-recipe", choices=("ali", "anchored"), default="ali")
    parser.add_argument("--direct-sampling", action=argparse.BooleanOptionalAction, default=False)
    defaults = {"num_latents": (int, 64), "time_embedding_dim": (int, 16), "batch_size": (int, 64),
                "lr": (float, .001), "seed": (int, 0), "steps": (int, 20000),
                "validate_every": (int, 500), "checkpoint_every": (int, 1000),
                "log_every": (int, 100), "source_steps": (int, 1000), "source_lr": (float, .001)}
    for name, (kind, value) in defaults.items():
        parser.add_argument("--" + name.replace("_", "-"), type=kind, default=value)
    add_tracking_arguments(parser)
    preliminary = parser.parse_args(argv)
    if preliminary.config:
        config = json.loads(preliminary.config.read_text())
        allowed = set(defaults) | {"student_recipe", "direct_sampling", "wandb_mode", "wandb_project", "wandb_entity", "wandb_group", "wandb_service_wait"}
        if set(config) - allowed:
            parser.error(f"unknown student settings: {set(config) - allowed}")
        parser.set_defaults(**config)
    args = parser.parse_args(argv)
    if args.student_recipe not in ("ali", "anchored") or not isinstance(args.direct_sampling, bool):
        parser.error("invalid student recipe or direct-sampling setting")
    if any(getattr(args, name) < 1 for name in ("num_latents", "time_embedding_dim", "batch_size", "steps",
                                                "validate_every", "checkpoint_every", "log_every", "threads", "source_steps")):
        parser.error("counts must be positive")
    if args.num_latents < 2 or args.time_embedding_dim % 2 or args.lr <= 0 or args.source_lr <= 0:
        parser.error("invalid capacity or learning rate")
    return args


def execute(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reference_path = args.reference_dir / "metrics.json"
    reference = json.loads(reference_path.read_text())
    for name, digest in reference["identity"]["integration_source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f"reference integration code changed ({name}); prepare a new reference")
    if json.loads((args.reference_dir / "status.json").read_text())["status"] != "complete":
        raise ValueError("teacher/interpolant reference is incomplete")
    if sha256_file(args.reference_dir / "cache.pt") != reference["cache_sha256"]:
        raise ValueError("reference cache changed")
    bundle = torch.load(args.reference_dir / "cache.pt", map_location="cpu", weights_only=False)
    dataset = bundle["contract"]["dataset"]
    teacher, native_config, entry, provenance = load_teacher(args.otp_root, dataset, device)
    if (entry["checkpoint_sha256"] != reference["identity"]["checkpoint_sha256"]
            or entry["config_sha256"] != reference["identity"]["config_sha256"]
            or bundle["identity"] != reference["identity"]
            or provenance["native_source_sha256"] != reference["identity"]["native_source_sha256"]):
        raise ValueError("teacher/config/cache/source identity differs from reference")
    cache = move_cache(bundle["cache"], device)
    args.wandb_group = f"otpfm-{dataset}-w2-teacher{entry['seed']}"
    training = {key: getattr(args, key) for key in ("num_latents", "time_embedding_dim", "batch_size", "lr",
                "seed", "steps", "validate_every", "checkpoint_every", "log_every", "source_steps", "source_lr",
                "student_recipe", "direct_sampling")}
    binding = {"training": training, "reference_sha256": sha256_file(reference_path),
               "source_policy": ("source_MLE_then_frozen_emissions_and_circuit_weights" if args.student_recipe == "anchored"
                                 else "ALI_source_initialization_only_all_student_parameters_trainable"),
               "objective": "diagonal_EMA_velocity_MSE_on_conditional_otpfm_cached",
               "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                   [Path(__file__).resolve(), ROOT / "repc/otpfm_student.py", ROOT / "repc/otpfm_eval.py"]}}
    old_config = json.loads((output / "config.json").read_text()) if (output / "config.json").exists() else None
    if old_config is not None and (not args.resume or old_config != binding):
        raise ValueError("run exists or settings changed; use --resume with unchanged inputs, or a new directory")
    atomic_json(output / "config.json", binding)
    (output / "checkpoints").mkdir(exist_ok=True)
    last_path, best_path = output / "checkpoints/last.pt", output / "checkpoints/best.pt"
    saved = torch.load(last_path, map_location="cpu", weights_only=False) if args.resume and last_path.exists() else None
    if saved is not None and saved["binding"] != binding:
        raise ValueError("checkpoint binding differs from inputs")
    run = None
    atomic_json(output / "status.json", {"status": "running"})
    try:
        run = init_tracking(args, output, {**binding, "teacher": entry, "data_contract": bundle["contract"]},
                            "otpfm-distillation", name=f"otpfm-{dataset}-{output.name}")
        if run is not None:
            run.define_metric("source/step")
            run.define_metric("source/*", step_metric="source/step")
        student = build_student(entry["dim"], args.num_latents, args.time_embedding_dim,
                                args.seed, args.student_recipe).to(device)
        source_rows, val_rows = bundle["train_rows"], bundle["validation_rows"]
        source_started = time.perf_counter()
        if saved:
            student.load_state_dict(saved["state_dict"])
            source_fit = saved["source_fit"]
        else:
            # Source initialization is independent of the Fourier feature count and velocity LR.
            torch.manual_seed(args.seed + 50000)
            source_log_path = output / "source_metrics.csv"
            with source_log_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["source/step", "source/validation_nll_normalized"])
                writer.writeheader()

                def log_source(row):
                    writer.writerow(row)
                    stream.flush()
                    if run is not None:
                        run.log(row)

                source_fit = initialize_student(student, cache["x0"][source_rows.to(device)],
                    cache["x0"][val_rows.to(device)], args.student_recipe, args.seed,
                    args.source_steps, args.source_lr, log_source)
            source_fit["seconds"] = time.perf_counter() - source_started
            atomic_json(output / "source_fit.json", source_fit)
        boundary = {key: value.detach().clone() for key, value in student.state_dict().items()
                    if args.student_recipe == "anchored" and key in
                    ("source_mean", "source_log_scale", "root_logits", "transition_logits")}
        optimizer, scheduler = build_optimizer(student, args.student_recipe, args.lr, args.steps)
        rng = torch.Generator().manual_seed(args.seed + 10000)
        best = float("inf")
        start = 1
        previous_seconds = 0.
        if saved:
            optimizer.load_state_dict(saved["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(saved["scheduler"])
            rng.set_state(saved["rng"])
            best, start = saved["best_validation_mse"], saved["step"] + 1
            previous_seconds = saved["train_seconds"]
        started = time.perf_counter()

        def checkpoint(step):
            return {"state_dict": student.state_dict(), "binding": binding, "step": step,
                    "student_recipe": args.student_recipe,
                    "model_config": {"dim": entry["dim"], "num_latents": args.num_latents,
                                     "time_embedding_dim": args.time_embedding_dim, "seed": args.seed},
                    "source_fit": source_fit, "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None, "rng": rng.get_state(),
                    "best_validation_mse": best, "train_seconds": previous_seconds + time.perf_counter() - started}

        columns = ["step", "loss", "gradient_norm", "learning_rate", "elapsed_seconds", "velocity_mse",
                   "relative_velocity_mse", "teacher_conditional_fm_mse", "teacher_velocity_energy"]
        metrics_path = output / "metrics.csv"
        with metrics_path.open("a" if saved else "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            if not saved:
                writer.writeheader()
            for step in range(start, args.steps + 1):
                student.train()
                rows = source_rows[torch.randint(len(source_rows), (args.batch_size,), generator=rng)].to(device)
                times = torch.rand(args.batch_size, generator=rng).to(device)
                with torch.no_grad():
                    values, _ = sample_cache(teacher.model, cache, rows, times)
                    target = teacher(times, values)
                loss = (student.velocity(values, times) - target).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite velocity distillation objective")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(student.parameters(), 1., error_if_nonfinite=True)
                learning_rate = optimizer.param_groups[0]["lr"]
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                row = {"step": step, "loss": float(loss), "gradient_norm": float(grad),
                       "learning_rate": learning_rate,
                       "elapsed_seconds": previous_seconds + time.perf_counter() - started}
                if step == 1 or step % args.validate_every == 0 or step == args.steps:
                    validation = velocity_validation(teacher, cache, val_rows, student)
                    row.update(validation)
                    if validation["velocity_mse"] < best:
                        best = validation["velocity_mse"]
                        atomic_torch(best_path, checkpoint(step))
                    print(f"{dataset} step {step}: loss={row['loss']:.6g}, validation={validation['velocity_mse']:.6g}", flush=True)
                if step == 1 or step % args.checkpoint_every == 0 or step == args.steps:
                    atomic_torch(last_path, checkpoint(step))
                if step == 1 or step % args.log_every == 0 or "velocity_mse" in row:
                    writer.writerow(row)
                    stream.flush()
                    if run is not None:
                        run.log({"train/step": step, **{f"{'validation' if key in validation_keys() else 'train'}/{key}": value
                                                                      for key, value in row.items() if key != "step"}})
        for key, value in boundary.items():
            if not torch.equal(student.state_dict()[key], value):
                raise RuntimeError(f"source boundary changed during velocity-only training: {key}")
        selected = torch.load(best_path, map_location="cpu", weights_only=False)
        student.load_state_dict(selected["state_dict"])
        student.eval()
        with torch.no_grad():
            times_check = cache["x0"].new_tensor(bundle["contract"]["times"])
            log_scale = student.leaf_parameters(times_check)[2]
            clamp_fraction = float((log_scale.abs() >= 20).float().mean())
        paths = dict(bundle["reference_paths"])
        sample_started = time.perf_counter()
        if args.direct_sampling:
            with torch.no_grad():
                paths["student_direct"] = student.sample_path(len(cache["x0"]),
                    cache["x0"].new_tensor(bundle["contract"]["times"]),
                    torch.Generator(device=device).manual_seed(2026)).cpu()
        sample_seconds = time.perf_counter() - sample_started
        if "teacher_ode" in paths:
            paths["student_ode"], student_nfe = integrate(student, cache["x0"].cpu(),
                torch.tensor(bundle["contract"]["times"]), device, 256, 1e-5, is_student=True)
        else:
            student_nfe = 0
        final = {**score_paths(bundle, paths, student, device),
                 "checkpoint_step": selected["step"], "best_validation_mse": best,
                 "reference_sha256": binding["reference_sha256"],
                 "source_fit": source_fit, "student_recipe": args.student_recipe,
                 "student_ode_nfe": student_nfe,
                 "scale_clamp_saturation_fraction": clamp_fraction,
                 "train_seconds": previous_seconds + time.perf_counter() - started,
                 "distillation_teacher_calls": args.steps,
                 "student_parameter_count": sum(p.numel() for p in student.parameters()),
                 "student_trainable_parameters": sum(p.numel() for p in student.parameters() if p.requires_grad),
                 "teacher_parameter_count_including_live_and_ema": sum(p.numel() for p in teacher.parameters()),
                 "validation": velocity_validation(teacher, cache, val_rows, student)}
        if args.direct_sampling:
            final.update(student_sampling_seconds=sample_seconds,
                         student_samples_per_second=len(cache["x0"]) / max(sample_seconds, 1e-12),
                         student_direct_nfe=0)
            final["student_direct_minus_teacher_finite_holdout_w2"] = final["student_direct_holdout_w2"] - final["teacher_finite_holdout_w2"]
        if student_nfe:
            final["student_ode_minus_teacher_ode_holdout_w2"] = final["student_ode_holdout_w2"] - final["teacher_ode_holdout_w2"]
        if device.type == "cuda":
            final["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(device)
        eval_output = output / "eval"
        atomic_json(eval_output / "metrics.json", final)
        atomic_torch(eval_output / "predictions.pt", {"paths": paths, "times": bundle["contract"]["times"]})
        with (eval_output / "per_time_metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(final["scores"][0]))
            writer.writeheader()
            writer.writerows(final["scores"])
        comparison_plots(bundle, paths, eval_output, final)
        log_comparison(run, final, eval_output)
        log_files(run, "student", {"best.pt": best_path, "last.pt": last_path,
                                  "config.json": output / "config.json",
                                  "source_fit.json": output / "source_fit.json", "metrics.csv": metrics_path,
                                  "source_metrics.csv": output / "source_metrics.csv"})
        if run is not None:
            run.summary["training_status"] = "complete"
            run.summary["best_validation_mse"] = best
        atomic_json(output / "status.json", {"status": "complete", "steps": args.steps,
                                             "best_validation_mse": best,
                                             "selected_checkpoint_step": selected["step"]})
    except BaseException as error:
        atomic_json(output / "status.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        if run is not None:
            run.finish()
        lock.close()


def validation_keys():
    return {"velocity_mse", "relative_velocity_mse", "teacher_conditional_fm_mse", "teacher_velocity_energy"}


if __name__ == "__main__":
    execute(parse_args())
