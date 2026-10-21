"""Export, train, distill, and evaluate ordinary MNIST latent flow matching.

All runtime artifacts default to /data/lthung/flow2pc. Training only starts
when this entry point is explicitly invoked; importing it has no side effects.
"""

import argparse
import csv
from datetime import datetime, timezone
import logging
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from repc.data import mnist_latent_loader
from repc.flow import build_teacher
from repc.hclt_pc import TimeDependentHCLTPC
from repc.mnist_latent import (atomic_save, check_contract, export_latents,
                               learn_latent_topology, load_eqvae,
                               load_latent_bundle, sha256_file, write_json)
from repc.mnist_latent_eval import (decoded_evaluation, generate_images,
                                    comparison_tables, reconstruction_reference,
                                    resolve_classifier_checkpoint, save_panel,
                                    validation_metrics)
from repc.pc import TimeDependentGaussianPC

ARTIFACT_ROOT = Path("/data/lthung/flow2pc")
REPO_ROOT = Path(__file__).resolve().parent


def resolve_device(name):
    """Resolve bare CUDA to the current visible device before strict CUDA APIs."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        index = torch.cuda.current_device() if device.index is None else device.index
        torch.cuda.set_device(index)
        device = torch.device("cuda", index)
    return device


def should_log_images(step, steps, image_every):
    """A zero interval means images only after the last training update."""
    return step == steps or (image_every > 0 and step % image_every == 0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("export", "flow", "distill", "evaluate"))
    parser.add_argument("--eqvae-root", type=Path, default=REPO_ROOT.parent / "eqvae")
    parser.add_argument("--vae-checkpoint", type=Path, default=Path("/data/lthung/eqvae/mnist-eqvae-seed0/best.pt"))
    parser.add_argument("--bundle-dir", type=Path, default=ARTIFACT_ROOT / "latents/mnist-eqvae-seed0")
    parser.add_argument("--data-dir", type=Path, default=ARTIFACT_ROOT / "data/mnist")
    parser.add_argument("--download", action="store_true", help="Download MNIST during export only")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Teacher or student checkpoint for evaluation")
    parser.add_argument("--classifier-checkpoint", type=Path,
                        help="Fixed trained MNIST evaluator; required for evaluate (standard location auto-detected)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--image-every", type=int, default=0,
                        help="Image interval; 0 (default) logs only at the end")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--teacher-time-embedding-dim", type=int, default=64)
    parser.add_argument("--pc-structure", choices=("hclt", "gaussian"), default="hclt")
    parser.add_argument("--pc-layer", choices=("dense", "monarch"), default="dense")
    parser.add_argument("--num-latents", type=int, default=64, help="HCLT hidden states per variable")
    parser.add_argument("--num-components", type=int, default=256, help="Gaussian control mixture components")
    parser.add_argument("--time-conditioning", choices=("polynomial", "fourier"), default="fourier")
    parser.add_argument("--time-order", type=int, default=4)
    parser.add_argument("--time-embedding-dim", type=int, default=16)
    parser.add_argument("--leaf-components", type=int, default=1)
    parser.add_argument("--hclt-samples", type=int, default=20000)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--num-generated", type=int, default=10000)
    parser.add_argument("--ode-steps", type=int, default=100)
    parser.add_argument("--kid-subsets", type=int, default=100)
    parser.add_argument("--kid-subset-size", type=int, default=1000)
    parser.add_argument("--is-splits", type=int, default=10)
    parser.add_argument("--precision-recall-k", type=int, default=3)
    parser.add_argument("--metric-chunk-size", type=int, default=512)
    parser.add_argument("--wandb-project", default="flow2pc")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-group", default="mnist-latent-distillation")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    for name in ("threads", "batch_size", "steps", "log_every", "validate_every", "checkpoint_every",
                 "hidden_dim", "num_blocks", "time_order", "leaf_components",
                 "hclt_samples", "ode_steps", "kid_subsets", "kid_subset_size", "num_components",
                 "is_splits", "precision_recall_k", "metric_chunk_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0 or args.image_every < 0 or args.lr <= 0 or args.weight_decay < 0 or args.gradient_clip <= 0:
        parser.error("Invalid workers, learning rate, weight decay, or gradient clipping")
    if args.num_latents < 2 or args.hclt_samples < 2:
        parser.error("At least two HCLT states and topology samples are required")
    for name in ("time_embedding_dim", "teacher_time_embedding_dim"):
        if getattr(args, name) < 2 or getattr(args, name) % 2:
            parser.error(f"{name} must be positive and even")
    if args.pc_layer == "monarch" and math.isqrt(args.num_latents)**2 != args.num_latents:
        parser.error("Monarch requires a square number of HCLT hidden states")
    if args.num_generated < 4 or args.kid_subset_size < 2:
        parser.error("Use at least four generated samples and two KID subset samples")
    if args.task == "evaluate" and args.precision_recall_k >= min(args.num_generated, 5000 if args.split == "validation" else 10000):
        parser.error("precision-recall-k must be smaller than both sample counts")
    if args.task == "distill" and args.teacher_checkpoint is None:
        parser.error("distill requires --teacher-checkpoint (a latent flow, not the EQ-VAE)")
    if args.task == "evaluate" and args.checkpoint is None:
        parser.error("evaluate requires --checkpoint")
    return args


def configure_output(args):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output_dir or ARTIFACT_ROOT / "outputs/mnist-latent" / f"{args.task}-seed{args.seed}-{timestamp}").resolve()
    if output == REPO_ROOT or REPO_ROOT in output.parents:
        raise ValueError("Runtime artifacts must be outside the code checkout")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a fresh --output-dir; {output} is not empty")
    code_roots = (REPO_ROOT, args.eqvae_root.resolve())
    for path in (output, args.bundle_dir.resolve(), args.data_dir.resolve()):
        if any(path == root or root in path.parents for root in code_roots):
            raise ValueError("Runtime artifacts must be outside both code checkouts")
    output.mkdir(parents=True, exist_ok=True)
    # Override inherited W&B paths so an old environment cannot log in the repo.
    for variable, directory in (
        ("WANDB_DIR", "wandb"), ("WANDB_DATA_DIR", ".wandb-data"),
        ("WANDB_CACHE_DIR", ".wandb-cache"), ("WANDB_CONFIG_DIR", ".wandb-config"),
        ("MPLCONFIGDIR", ".matplotlib"), ("XDG_CACHE_HOME", ".cache"),
        ("XDG_CONFIG_HOME", ".config"),
        ("TMPDIR", "tmp"),
    ):
        path = output / directory
        path.mkdir(exist_ok=True)
        os.environ[variable] = str(path)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(output / "run.log"), logging.StreamHandler()])
    return output


def init_tracking(args, output, config):
    if args.wandb_mode == "disabled":
        return None
    import wandb

    # Concurrent GPU workers share only this short initialization lock; model
    # training and evaluation remain fully parallel.
    from contextlib import nullcontext
    import fcntl

    lock_path = os.environ.get("FLOW2PC_WANDB_INIT_LOCK")
    context = Path(lock_path).open("a") if lock_path else nullcontext()
    with context as lock:
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_EX)
        name = args.wandb_run_name or output.name
        name = name if name.startswith("mnist") else f"mnist-{name}"
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         group=args.wandb_group, name=name,
                         job_type=f"mnist-latent-{args.task}", tags=["mnist-latent", "vanilla-cfm", args.task],
                         mode=args.wandb_mode, dir=str(output), config=config,
                         settings=wandb.Settings(init_timeout=300))
    (output / "wandb_run_id.txt").write_text(run.id + "\n")
    return run


def log_metrics(run, metrics, step=None, panel=None, tables=None):
    if run is None:
        return
    payload = dict(metrics)
    if panel is not None:
        import wandb

        payload["images/panel"] = wandb.Image(str(panel), caption="Real references and VAE reconstructions; unconditional teacher/student samples")
    if tables:
        import wandb

        payload.update({f"eval/{name}": wandb.Table(columns=table["columns"], data=table["data"])
                        for name, table in tables.items()})
    run.log(payload, step=step)


def build_latent_student(config):
    if config["name"] == "gaussian":
        model = TimeDependentGaussianPC(config["num_components"], config["dim"], config["time_order"])
        # Break component symmetry while keeping the t=0 distribution N(0,I).
        with torch.no_grad():
            model.mean_coefficients.normal_(std=0.01)
            model.log_scale_coefficients.normal_(std=0.01)
        return model
    return TimeDependentHCLTPC(**{key: value for key, value in config.items() if key not in ("name", "topology_config")})


def load_latent_teacher(path, contract, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("model_type") != "flow"
            or checkpoint.get("model_config", {}).get("name") != "latent_residual_mlp"):
        raise ValueError("Expected a trained latent_residual_mlp flow checkpoint")
    check_contract(checkpoint["data_contract"], contract)
    if checkpoint["model_config"]["dim"] != contract["mean"].shape[1]:
        raise ValueError("Teacher architecture dimension differs from the latent bundle")
    if checkpoint["flow_matching_config"] != {"name": "vanilla_cfm", "sigma": 0.0}:
        raise ValueError("This entry point requires ordinary straight-line CFM with sigma=0")
    teacher = build_teacher(checkpoint["model_config"]).to(device)
    teacher.load_state_dict(checkpoint["state_dict"], strict=True)
    teacher.eval().requires_grad_(False)
    return teacher, checkpoint


def student_config(args, values, contract, output):
    if args.pc_structure == "gaussian":
        return {"name": "gaussian", "dim": values.shape[1], "num_components": args.num_components,
                "time_order": args.time_order}
    parents, topology = learn_latent_topology(values, args.hclt_samples, args.seed)
    topology["bundle_manifest_sha256"] = contract["manifest_sha256"]
    atomic_save({"parents": parents, "config": topology}, output / "topology.pt")
    return {"name": "hclt", "parents": parents, "num_latents": args.num_latents,
            "time_order": args.time_order, "sum_layer": args.pc_layer,
            "time_conditioning": args.time_conditioning, "time_embedding_dim": args.time_embedding_dim,
            "time_feature_seed": args.seed, "fourier_zero_center": True,
            "leaf_distribution": "gaussian" if args.leaf_components == 1 else "gaussian-mixture",
            "leaf_components": args.leaf_components, "topology_config": topology}


def train_stage(args, codes, contract, vae, output, device, settings):
    # Reuse the existing ordinary CFM and distillation objectives. No generator
    # or marginal indices are passed: paths are (1-t)*noise + t*latent_code.
    from train import build_flow_matcher, distillation_loss, flow_matching_loss

    matcher_config = {"name": "vanilla_cfm", "sigma": 0.0}
    matcher = build_flow_matcher(matcher_config)
    teacher_identity = None
    if args.task == "flow":
        model_config = {"name": "latent_residual_mlp", "dim": contract["manifest"]["dim"], "hidden_dim": args.hidden_dim,
                        "num_blocks": args.num_blocks, "time_embedding_dim": args.teacher_time_embedding_dim}
        model = teacher = build_teacher(model_config).to(device)
        student = None
    else:
        teacher, teacher_checkpoint = load_latent_teacher(args.teacher_checkpoint, contract, device)
        teacher_identity = {"path": str(args.teacher_checkpoint.resolve()),
                            "sha256": sha256_file(args.teacher_checkpoint),
                            "model_config": teacher_checkpoint["model_config"],
                            "wandb_run_id": teacher_checkpoint.get("wandb_run_id")}
        model_config = student_config(args, codes["train"]["values"], contract, output)
        model = student = build_latent_student(model_config).to(device)
    settings.update(model_config=model_config, flow_matching_config=matcher_config,
                    data_contract_manifest=contract["manifest"],
                    parameter_count=sum(p.numel() for p in model.parameters()),
                    teacher=teacher_identity,
                    objective="flow_matching_mse" if student is None else "latent_teacher_velocity_mse",
                    selection_rule="validation_flow_mse" if student is None else "validation_student_velocity_mse")
    write_json(settings, output / "config.json")
    run = init_tracking(args, output, settings)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    loader = mnist_latent_loader(codes, "train", args.batch_size, args.num_workers, args.seed)
    iterator = iter(loader)
    best, started = float("inf"), time.perf_counter()
    real = reconstructed = None
    try:
        with (output / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["step", "loss", "gradient_norm", "lr", "training_seconds",
                                                        "validation_teacher_flow_mse", "validation_student_velocity_mse",
                                                        "validation_student_latent_nll_nats"])
            writer.writeheader()
            for step in range(1, args.steps + 1):
                model.train()
                try:
                    values, _ = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    values, _ = next(iterator)
                values = values.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = (flow_matching_loss(model, matcher, values) if student is None else
                        distillation_loss(student, teacher, matcher, values))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training objective at step {step}")
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                if not torch.isfinite(gradient):
                    raise FloatingPointError(f"Non-finite gradient at step {step}")
                optimizer.step()
                scheduler.step()
                metrics = {"train/loss": loss.item(), "train/gradient_norm": gradient.item(),
                           "train/lr": optimizer.param_groups[0]["lr"], "train/seconds": time.perf_counter() - started}
                should_validate = step % args.validate_every == 0 or step == args.steps
                improved = False
                validation = {}
                if should_validate:
                    model.eval()
                    validation = validation_metrics(teacher, student, codes["validation"]["values"],
                                                     args.batch_size, device, args.seed)
                    if not all(math.isfinite(value) for value in validation.values()):
                        raise FloatingPointError("Non-finite latent validation metrics")
                    metrics.update({f"validation/{key}": value for key, value in validation.items()})
                    score = validation["teacher_flow_mse" if student is None else "student_velocity_mse"]
                    improved = score < best
                    best = min(best, score)
                panel = None
                if should_log_images(step, args.steps, args.image_every):
                    model.eval()
                    if real is None:
                        real, reconstructed = reconstruction_reference(
                            vae, codes, contract, args.data_dir, "validation", device)
                    # Keep monitoring noise fixed without perturbing training RNG.
                    with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
                        images, _ = generate_images(teacher, student, vae, contract, 100,
                                                   args.batch_size, device, args.seed + 2718, args.ode_steps)
                    sections = [("Real MNIST (validation)", real), ("Frozen VAE reconstructions", reconstructed),
                                (f"Teacher: RK4 ODE ({args.ode_steps} steps)", images["teacher"])]
                    if student is not None:
                        sections.append((f"Student: velocity ODE ({args.ode_steps} steps)", images["student"]))
                    panel = output / "images" / f"step{step:07d}.png"
                    save_panel(sections, panel)
                if improved or step % args.checkpoint_every == 0 or step == args.steps:
                    checkpoint = {"model_type": "flow" if student is None else "pc", "model_config": model_config,
                                  "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                                  "scheduler": scheduler.state_dict(), "step": step, "best_validation_score": best,
                                  "data_contract": contract, "flow_matching_config": matcher_config,
                                  "teacher": teacher_identity, "training_config": settings,
                                  "validation_metrics": validation, "wandb_run_id": run.id if run else None}
                    if improved:
                        atomic_save(checkpoint, output / "checkpoints/best.pt")
                    if step % args.checkpoint_every == 0 or step == args.steps:
                        atomic_save(checkpoint, output / f"checkpoints/step{step:07d}.pt")
                        atomic_save(checkpoint, output / "checkpoints/last.pt")
                if step % args.log_every == 0 or should_validate or panel:
                    log_metrics(run, metrics, step, panel)
                    row = {"step": step, "loss": metrics["train/loss"], "gradient_norm": metrics["train/gradient_norm"],
                           "lr": metrics["train/lr"], "training_seconds": metrics["train/seconds"]}
                    row.update({f"validation_{key}": value for key, value in validation.items()})
                    writer.writerow(row)
                    handle.flush()
                    logging.info("step=%d loss=%.6f validation=%s", step, loss.item(), validation)
        write_json({"status": "complete", "step": args.steps, "best_validation_score": best,
                    "best_checkpoint": str(output / "checkpoints/best.pt"),
                    "wandb_run_id": run.id if run else None}, output / "training_summary.json")
    finally:
        if run is not None:
            run.finish()


def evaluate_stage(args, codes, contract, vae, output, device, settings):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    check_contract(checkpoint["data_contract"], contract)
    student = None
    if checkpoint["model_type"] == "flow":
        teacher, _ = load_latent_teacher(args.checkpoint, contract, device)
    elif checkpoint["model_type"] == "pc":
        if checkpoint["flow_matching_config"] != {"name": "vanilla_cfm", "sigma": 0.0}:
            raise ValueError("Student checkpoint has a different flow matching contract")
        student = build_latent_student(checkpoint["model_config"]).to(device).eval()
        student.load_state_dict(checkpoint["state_dict"], strict=True)
        identity = checkpoint["teacher"]
        teacher_path = args.teacher_checkpoint or Path(identity["path"])
        if sha256_file(teacher_path) != identity["sha256"]:
            raise ValueError("Student's frozen teacher checkpoint has changed")
        teacher, _ = load_latent_teacher(teacher_path, contract, device)
    else:
        raise ValueError("Expected a latent teacher or student checkpoint")
    settings.update(checkpoint_sha256=sha256_file(args.checkpoint), data_contract_manifest=contract["manifest"])
    write_json(settings, output / "config.json")
    run = init_tracking(args, output, settings)
    try:
        metrics, panel = decoded_evaluation(args, teacher, student, vae, codes, contract, output, device)
        metrics["model/checkpoint_bytes"] = args.checkpoint.stat().st_size
        write_json(metrics, output / "metrics.json")
        log_metrics(run, metrics, panel=panel, tables=comparison_tables(metrics))
        if student is not None:
            for metric in ("fid_mnist", "kid_mnist_mean", "inception_score_mnist_mean",
                           "precision_mnist", "recall_mnist"):
                logging.info("%s teacher=%.6f student=%.6f student_minus_teacher=%+.6f",
                             metric, metrics[f"teacher/{metric}"], metrics[f"student/{metric}"],
                             metrics[f"comparison/{metric}_student_minus_teacher"])
        logging.info("Evaluation saved to %s", output)
    finally:
        if run is not None:
            run.finish()


def main():
    args = parse_args()
    if args.task == "evaluate":
        args.classifier_checkpoint = resolve_classifier_checkpoint(args.classifier_checkpoint)
    output = configure_output(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    settings = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings["output_dir"], settings["device"] = str(output), str(device)
    write_json(settings, output / "config.json")
    if args.task == "export":
        manifest = export_latents(args, device)
        logging.info("Exported latent bundle to %s; normalization=%s", args.bundle_dir, manifest["normalization"])
        return
    codes, contract = load_latent_bundle(args.bundle_dir)
    manifest = contract["manifest"]
    # Decoder identity comes from the bundle, not an unrelated CLI default.
    vae = load_eqvae(args.eqvae_root, manifest["encoder_checkpoint"], device, manifest["encoder_sha256"])
    if args.task == "evaluate":
        evaluate_stage(args, codes, contract, vae, output, device, settings)
    else:
        train_stage(args, codes, contract, vae, output, device, settings)


if __name__ == "__main__":
    main()
