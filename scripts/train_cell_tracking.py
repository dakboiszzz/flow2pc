"""Velocity-only PC distillation of the selected frozen cell-tracking checkpoint."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from repc.cell_tracking import (ROOT, DEFAULT_ALI_ROOT, EXPECTED_SHA256, atomic_torch, code_identity,
    load_teacher, sample_interpolant, build_student, initialize_student, validation_metrics,
    rollout, score_paths, overlay_images, comparison_image)
from repc.single_cell import atomic_json, sha256_file
from repc.single_cell_tracking import add_tracking_arguments, init_tracking, log_files
from scripts.run_otpfm_sweep import read_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ali-root", type=Path, default=DEFAULT_ALI_ROOT)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_ALI_ROOT / "changed_files/ali_cfm_model.pth")
    p.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    p.add_argument("--config", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    settings = {"num_latents": (int, 128), "time_embedding_dim": (int, 32), "lr": (float, .001),
        "batch_size": (int, 64), "steps": (int, 50000), "seed": (int, 0),
        "validate_every": (int, 1000), "checkpoint_every": (int, 1000), "log_every": (int, 100)}
    for key, (kind, value) in settings.items(): p.add_argument("--" + key.replace("_", "-"), type=kind, default=value)
    add_tracking_arguments(p)
    preliminary = p.parse_args(argv)
    if preliminary.config:
        config = json.loads(preliminary.config.read_text())
        allowed = set(settings) | {"wandb_mode", "wandb_project", "wandb_entity", "wandb_group", "wandb_service_wait"}
        if set(config) - allowed: p.error(f"unknown settings: {sorted(set(config) - allowed)}")
        p.set_defaults(**config)
    args = p.parse_args(argv)
    if any(getattr(args, k) < 1 for k in settings if k not in ("seed", "lr")) or args.lr <= 0 or args.threads < 1:
        p.error("invalid positive training settings")
    if args.num_latents < 2 or args.time_embedding_dim % 2: p.error("invalid PC capacity")
    return args


def execute(args):
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda": torch.cuda.set_device(device)
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".worker.lock").open("a"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reference = read_json(args.reference_dir / "metrics.json")
    if read_json(args.reference_dir / "status.json").get("status") != "complete":
        raise ValueError("prepare the teacher reference first")
    if reference["identity"]["code_sha256"] != code_identity() or sha256_file(args.reference_dir / "cache.pt") != reference["cache_sha256"]:
        raise ValueError("reference code/cache changed; use a new output root")
    bundle = torch.load(args.reference_dir / "cache.pt", map_location="cpu", weights_only=False)
    if bundle["identity"] != reference["identity"] or args.expected_sha256 != bundle["identity"]["checkpoint_sha256"]:
        raise ValueError("teacher reference identity differs")
    if (sha256_file(args.ali_root / "ali_cfm/nets.py") != bundle["identity"]["native_nets_sha256"] or
        sha256_file(args.ali_root / "ali_cfm/land_metric.py") != bundle["identity"]["native_land_metric_sha256"]):
        raise ValueError("native teacher implementation changed since reference preparation")
    for index, digest in bundle["identity"]["image_sha256"].items():
        if sha256_file(Path(bundle["identity"]["image_dir"]) / f"t{int(index):03d}.tif") != digest:
            raise ValueError("microscopy display image changed since reference preparation")
    teacher, generator, _, minimum, maximum = load_teacher(args.ali_root, args.checkpoint, args.expected_sha256, device)
    if not torch.equal(minimum, bundle["minimum"]) or not torch.equal(maximum - minimum, bundle["scale"]):
        raise ValueError("normalization differs from teacher reference")
    training = {k: getattr(args, k) for k in ("num_latents", "time_embedding_dim", "lr", "batch_size", "steps", "seed", "validate_every", "checkpoint_every", "log_every")}
    binding = {"training": training, "reference_sha256": sha256_file(args.reference_dir / "metrics.json"),
        "code_sha256": code_identity(), "objective": "frozen_CFM_global_time_velocity_MSE_on_frozen_interpolant",
        "selection_rule": "worst normalized velocity MSE of global, late and final teacher-state validation"}
    previous = read_json(out / "config.json")
    if previous and (not args.resume or previous != binding): raise ValueError("run settings changed; use a new run directory")
    atomic_json(out / "config.json", binding)
    checkpoints = out / "checkpoints"; checkpoints.mkdir(exist_ok=True)
    saved = torch.load(checkpoints / "last.pt", map_location="cpu", weights_only=False) if args.resume and (checkpoints / "last.pt").exists() else None
    if saved and saved["binding"] != binding: raise ValueError("checkpoint binding differs")
    run = None; atomic_json(out / "status.json", {"status": "running"})
    try:
        args.wandb_project = "flow2pc"; args.wandb_group = f"cell-tracking-teacher-{args.expected_sha256[:12]}"
        run = init_tracking(args, out, {**binding, "teacher": bundle["identity"]}, "cell-tracking-distillation",
                            name=f"cell-tracking-{out.name}")
        torch.manual_seed(args.seed)
        student = build_student(args.num_latents, args.time_embedding_dim, args.seed).to(device)
        model_config = {"num_latents": args.num_latents, "time_embedding_dim": args.time_embedding_dim, "seed": args.seed}
        optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
        rng = torch.Generator().manual_seed(args.seed + 10000)
        best, start, previous_seconds = float("inf"), 1, 0.
        if saved:
            student.load_state_dict(saved["state_dict"]); optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"]); rng.set_state(saved["rng"])
            best, start, previous_seconds = saved["best_selection_score"], saved["step"] + 1, saved["train_seconds"]
        else: initialize_student(student, bundle, args.seed)
        began = time.perf_counter()

        def checkpoint(step):
            return {"state_dict": student.state_dict(), "binding": binding, "model_config": model_config,
                "step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": rng.get_state(),
                "best_selection_score": best, "train_seconds": previous_seconds + time.perf_counter() - began}

        fields = ["step", "loss", "gradient_norm_before_clip", "learning_rate", "elapsed_seconds",
                  "global_velocity_mse", "late_velocity_mse", "final_velocity_mse", "teacher_velocity_energy", "relative_velocity_mse", "selection_score"]
        with (out / "metrics.csv").open("a" if saved else "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if not saved: writer.writeheader()
            for step in range(start, args.steps + 1):
                student.train()
                values, times = sample_interpolant(generator, bundle, args.batch_size, rng, device)
                with torch.no_grad(): target = teacher(times, values)
                loss = (student.velocity(values, times) - target).square().mean()
                if not torch.isfinite(loss): raise FloatingPointError("nonfinite velocity objective")
                optimizer.zero_grad(set_to_none=True); loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(student.parameters(), 1., error_if_nonfinite=True)
                lr = optimizer.param_groups[0]["lr"]; optimizer.step(); scheduler.step()
                row = {"step": step, "loss": float(loss), "gradient_norm_before_clip": float(grad), "learning_rate": lr,
                       "elapsed_seconds": previous_seconds + time.perf_counter() - began}
                validate = step == 1 or step % args.validate_every == 0 or step == args.steps
                if validate:
                    row.update(validation_metrics(student, bundle, device))
                    if row["selection_score"] < best:
                        best = row["selection_score"]; atomic_torch(checkpoints / "best.pt", checkpoint(step))
                    print(f"cell-tracking step {step}: global={row['global_velocity_mse']:.6g}, late={row['late_velocity_mse']:.6g}, final={row['final_velocity_mse']:.6g}", flush=True)
                if step == 1 or step % args.checkpoint_every == 0 or step == args.steps:
                    atomic_torch(checkpoints / "last.pt", checkpoint(step))
                if validate or step % args.log_every == 0:
                    writer.writerow(row); stream.flush()
                    if run is not None:
                        run.log({"train/step": step, **{f"{'validation' if k.endswith('velocity_mse') or k in ('selection_score', 'teacher_velocity_energy') else 'train'}/{k}": v for k,v in row.items() if k != "step"}})
        selected = torch.load(checkpoints / "best.pt", map_location="cpu", weights_only=False)
        student.load_state_dict(selected["state_dict"]); student.eval()
        path, nfe = rollout(student, bundle["clouds"][0], bundle["times"], device,
            bundle["eval_batch_size"], bundle["ode_tolerance"], is_student=True)
        result = {**score_paths(bundle, path), "checkpoint_step": selected["step"], "best_selection_score": best,
            "validation": validation_metrics(student, bundle, device), "student_ode_nfe": nfe,
            "teacher_ode_nfe": bundle["teacher_nfe"], "student_parameters": sum(p.numel() for p in student.parameters()),
            "source_particles": len(path[0]), "reference_sha256": binding["reference_sha256"]}
        with torch.no_grad():
            logs = student.leaf_parameters(bundle["times"].to(device))[2]
            result["scale_clamp_fraction"] = float((logs.abs() >= 20).float().mean())
        evaluation = out / "eval"; evaluation.mkdir(exist_ok=True)
        atomic_json(evaluation / "metrics.json", result)
        atomic_torch(evaluation / "predictions.pt", {"times": bundle["times"], "student_ode": path})
        with (evaluation / "per_frame_metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(result["scores"][0])); writer.writeheader(); writer.writerows(result["scores"])
        overlay_images(bundle, path, evaluation / "student_overlay.png", "PC student")
        comparison_image(bundle, path, evaluation / "comparison_overlay.png")
        if run is not None:
            import wandb
            scalars = {f"eval/{k}": v for k,v in result.items() if isinstance(v, (int,float))}
            scalars.update({f"eval/validation/{k}":v for k,v in result["validation"].items()})
            for row in result["scores"]:
                scalars.update({f"eval/frame_{row['frame_number']}/{k}":v for k,v in row.items() if isinstance(v,(int,float)) and k.endswith(("pixels", "count"))})
            run.log({"eval/checkpoint_step": selected["step"], **scalars,
                "eval/student_overlay": wandb.Image(str(evaluation / "student_overlay.png")),
                "eval/teacher_student_overlay": wandb.Image(str(evaluation / "comparison_overlay.png")),
                "eval/teacher_overlay": wandb.Image(str(args.reference_dir / "teacher_overlay.png"))})
            run.summary.update(scalars); run.summary["training_status"] = "complete"
            log_files(run, "student", {"best.pt": checkpoints / "best.pt", "last.pt": checkpoints / "last.pt", "config.json": out / "config.json", "metrics.csv": out / "metrics.csv"})
            log_files(run, "eval", {p.name:p for p in evaluation.iterdir() if p.is_file()})
        atomic_json(out / "status.json", {"status": "complete", "steps": args.steps, "best_selection_score": best,
            "selected_checkpoint_step": selected["step"], "validation": result["validation"]})
    except BaseException as error:
        atomic_json(out / "status.json", {"status": "failed", "error": str(error)}); raise
    finally:
        if run is not None: run.finish()
        lock.close()


if __name__ == "__main__": execute(parse_args())
