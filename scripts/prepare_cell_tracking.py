"""Prepare a frozen Cell4 teacher reference once; never optimize the teacher."""
import argparse
import csv
import fcntl
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from repc.cell_tracking import (DEFAULT_ALI_ROOT, EXPECTED_SHA256, DISPLAY_FRAMES,
    atomic_torch, code_identity, input_identity, load_teacher, load_masks,
    sample_interpolant, rollout, score_paths, overlay_images)
from repc.single_cell import atomic_json, sha256_file
from repc.single_cell_tracking import add_tracking_arguments, init_tracking, log_files
from scripts.run_otpfm_sweep import read_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ali-root", type=Path, default=DEFAULT_ALI_ROOT)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_ALI_ROOT / "changed_files/ali_cfm_model.pth")
    p.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    p.add_argument("--mask-stack", type=Path, default=DEFAULT_ALI_ROOT / "data/cell_tracking/exports/Cell4_masks/mask_cell4_stack.npy")
    p.add_argument("--image-dir", type=Path, default=DEFAULT_ALI_ROOT / "data/cell_tracking/PhC-C2DH-U373/01")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--split-seed", type=int, default=2026)
    p.add_argument("--validation-fraction", type=float, default=.1)
    p.add_argument("--metric-samples", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--ode-tolerance", type=float, default=1e-5)
    add_tracking_arguments(p)
    args = p.parse_args(argv)
    if min(args.threads, args.metric_samples, args.eval_batch_size) < 1 or not 0 < args.validation_fraction < .5 or args.ode_tolerance <= 0:
        p.error("invalid reference preparation settings")
    return args


def execute(args):
    torch.set_num_threads(args.threads)
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".prepare.lock").open("a"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = input_identity(args.ali_root, args.checkpoint, args.mask_stack, args.image_dir, args.expected_sha256)
    identity.update(code_sha256=code_identity(), split_seed=args.split_seed,
        validation_fraction=args.validation_fraction, metric_samples=args.metric_samples,
        eval_batch_size=args.eval_batch_size, ode_tolerance=args.ode_tolerance,
        protocol="full-source empirical ODE; frozen border-OT interpolant training samples; global time")
    existing = read_json(out / "metrics.json")
    if existing:
        if existing.get("identity") != identity or sha256_file(out / "cache.pt") != existing.get("cache_sha256"):
            raise ValueError("reference inputs/code changed; choose a new output root")
        if read_json(out / "status.json").get("status") == "complete":
            print(f"Reused immutable teacher reference: {out}", flush=True); lock.close(); return
    device = torch.device(args.device)
    if device.type == "cuda": torch.cuda.set_device(device)
    run = None
    atomic_json(out / "status.json", {"status": "running"})
    try:
        args.wandb_group = f"cell-tracking-teacher-{args.expected_sha256[:12]}"
        args.wandb_project = "flow2pc"
        run = init_tracking(args, out, identity, "cell-tracking-teacher-reference",
                            name=f"cell-tracking-teacher-{args.expected_sha256[:12]}-reference")
        teacher, generator, config, minimum, maximum = load_teacher(args.ali_root, args.checkpoint, args.expected_sha256, device)
        bundle = load_masks(args.mask_stack, minimum, maximum, args.split_seed, args.validation_fraction)
        bundle.update(identity=identity, teacher_config=config, eval_batch_size=args.eval_batch_size,
                      ode_tolerance=args.ode_tolerance)
        bundle["teacher_path"], bundle["teacher_nfe"] = rollout(teacher, bundle["clouds"][0], bundle["times"], device,
            args.eval_batch_size, args.ode_tolerance)
        rng = torch.Generator().manual_seed(args.split_seed + 7000)
        values, times, targets = [], [], []
        for _ in range(8):
            x, t = sample_interpolant(generator, bundle, 128, rng, device, validation=True)
            with torch.no_grad(): y = teacher(t, x)
            values.append(x.cpu()); times.append(t.cpu()); targets.append(y.cpu())
        validation = {"global": {"values": torch.cat(values), "times": torch.cat(times), "target": torch.cat(targets)}}
        probe = bundle["validation_source_rows"][:128]
        for name, indices in (("late", list(range(92, 115))), ("final", [114])):
            x = bundle["teacher_path"][indices][:, probe].reshape(-1, 2)
            t = bundle["times"][indices, None].expand(-1, len(probe)).reshape(-1)
            with torch.no_grad(): y = teacher(t.to(device), x.to(device)).cpu()
            validation[name] = {"values": x, "times": t, "target": y}
        bundle["velocity_validation"] = validation
        rng = torch.Generator().manual_seed(args.split_seed + 8000)
        bundle["metric_source_rows"] = torch.randperm(len(bundle["clouds"][0]), generator=rng)[:args.metric_samples]
        bundle["metric_real_rows"] = {i: torch.randperm(len(bundle["clouds"][i]), generator=rng)[:args.metric_samples] for i in DISPLAY_FRAMES}
        bundle["metric_real_rows"][0] = bundle["metric_source_rows"].clone()
        atomic_torch(out / "cache.pt", bundle)
        result = {"identity": identity, "cache_sha256": sha256_file(out / "cache.pt"),
            "teacher_nfe": bundle["teacher_nfe"], "source_particles": len(bundle["clouds"][0]),
            "counts": [len(x) for x in bundle["clouds"]], **score_paths(bundle)}
        overlay_images(bundle, bundle["teacher_path"], out / "teacher_overlay.png", "Frozen CFM teacher")
        atomic_json(out / "metrics.json", result)
        with (out / "per_frame_metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(result["scores"][0])); writer.writeheader(); writer.writerows(result["scores"])
        if run is not None:
            import wandb
            run.log({"reference/teacher_overlay": wandb.Image(str(out / "teacher_overlay.png")),
                     "reference/final_emd_pixels": result["teacher_final_emd_pixels"],
                     "reference/display_mean_emd_pixels": result["teacher_display_mean_emd_pixels"]})
            log_files(run, "reference", {p.name: p for p in out.iterdir() if p.suffix in (".json", ".csv", ".png")})
        atomic_json(out / "status.json", {"status": "complete"})
    except BaseException as error:
        atomic_json(out / "status.json", {"status": "failed", "error": str(error)}); raise
    finally:
        if run is not None: run.finish()
        lock.close()


if __name__ == "__main__": execute(parse_args())
