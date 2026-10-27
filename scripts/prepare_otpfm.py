"""Prepare immutable OTP-FM data/correction cache and measured teacher reference."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from repc.otpfm import DEFAULT_OTP_ROOT, build_cache, load_teacher
from repc.otpfm_data import couple_observed, load_data
from repc.otpfm_eval import (comparison_plots, finite_teacher_path, interpolant_path,
                             log_comparison, score_paths, velocity_validation)
from repc.single_cell import atomic_json, sha256_file
from repc.single_cell_tracking import add_tracking_arguments, init_tracking, log_files
from repc.single_cell_eval import integrate


def atomic_torch(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("gom", "beijing"), required=True)
    parser.add_argument("--otp-root", type=Path, default=DEFAULT_OTP_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--validation-fraction", type=float, default=.05)
    parser.add_argument("--ode-comparison", action=argparse.BooleanOptionalAction, default=True)
    add_tracking_arguments(parser)
    args = parser.parse_args()
    if args.threads < 1 or not 0 < args.validation_fraction < .5:
        parser.error("invalid thread count or validation fraction")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".prepare.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    teacher, config, entry, provenance = load_teacher(args.otp_root, args.dataset, device)
    args.wandb_group = f"otpfm-{args.dataset}-w2-teacher{entry['seed']}"
    identity = {**provenance, "dataset": args.dataset, "checkpoint_sha256": entry["checkpoint_sha256"],
                "config_sha256": entry["config_sha256"], "data_manifest_sha256": entry["data_manifest_sha256"],
                "path_type": "conditional_otpfm_cached", "alpha": entry["fixed_interpolant_alpha"],
                "split_seed": args.split_seed, "validation_fraction": args.validation_fraction,
                "ode_comparison": args.ode_comparison,
                "integration_source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                    [ROOT / "repc/otpfm.py", ROOT / "repc/otpfm_data.py", ROOT / "repc/otpfm_eval.py",
                     Path(__file__).resolve()]}}
    status_path, reference_path, cache_path = output / "status.json", output / "metrics.json", output / "cache.pt"
    if reference_path.exists():
        existing = json.loads(reference_path.read_text())
        if existing.get("identity") != identity or sha256_file(cache_path) != existing["cache_sha256"]:
            raise ValueError("reference inputs/cache changed; choose a new output directory")
        if status_path.exists() and json.loads(status_path.read_text())["status"] == "complete":
            print(f"Reused immutable reference: {reference_path}", flush=True)
            return
    run = None
    atomic_json(status_path, {"status": "running"})
    try:
        run = init_tracking(args, output, {"identity": identity, "teacher": entry, "native_config": config},
                            "otpfm-teacher-reference", name=f"otpfm-{args.dataset}-w2-teacher{entry['seed']}-reference")
        if reference_path.exists():
            log_comparison(run, existing, output, "reference")
            atomic_json(status_path, {"status": "complete"})
            return
        data = load_data(entry, config, args.split_seed, args.validation_fraction)
        tuples, chains = couple_observed(data["clouds"], data["contract"]["observed_indices"])
        cache = build_cache(teacher.model, tuples.to(device), entry["fixed_interpolant_alpha"])
        times = data["contract"]["times"]
        paths = {"interpolant": interpolant_path(teacher.model, cache, times)}
        finite, sampler = finite_teacher_path(teacher, cache["x0"], times, entry["recorded_eval_steps_per_marginal"])
        paths["teacher_finite"] = finite
        if args.ode_comparison:
            ode, nfe = integrate(teacher, cache["x0"].cpu(), torch.tensor(times), device, 256, 1e-5)
            paths["teacher_ode"] = ode
            sampler["diagonal_ode"] = {"solver": "dopri5", "rtol": 1e-5, "atol": 1e-5, "nfe": nfe}
        bundle = {**data, "cache": {key: value.cpu() if isinstance(value, torch.Tensor) else value
                                    for key, value in cache.items()}, "chains": chains, "tuples": tuples,
                  "identity": identity, "entry": entry, "native_config": config,
                  "reference_paths": paths, "sampler": sampler}
        atomic_torch(cache_path, bundle)
        diagnostics = velocity_validation(teacher, cache, data["validation_rows"])
        result = {"model": "otpfm_teacher_and_conditional_interpolant", "identity": identity,
                  "data_contract": data["contract"], "cache_sha256": sha256_file(cache_path),
                  "linear_residual": cache["relative_linear_residual"], "condition_number": cache["condition_number"],
                  "coupling": "native_POT_sqeuclidean_EMD_row_argmax_chain_observed_only",
                  "coupling_sha256": hashlib.sha256(chains.contiguous().numpy().tobytes()).hexdigest(),
                  "sampler": sampler,
                  "legacy_recorded_finite_step": {"holdout_w2": entry["recorded_holdout_w2"],
                                                   "observed_w2": entry["recorded_train_w2"],
                                                   "selected_epoch": entry["best_epoch"],
                                                   "protocol": "original_truncation_or_nearby_grid_minimum; not common reference"},
                  **diagnostics, **score_paths(bundle, paths)}
        atomic_json(reference_path, result)
        atomic_json(output / "data_contract.json", data["contract"])
        comparison_plots(bundle, paths, output, result)
        log_comparison(run, result, output, "reference")
        log_files(run, "reference-cache", {"cache.pt": cache_path})
        atomic_json(status_path, {"status": "complete"})
        print(f"Prepared {args.dataset}: residual={result['linear_residual']:.3g}, "
              f"interpolant holdout W2={result['interpolant_holdout_w2']:.6g}", flush=True)
    except BaseException as error:
        atomic_json(status_path, {"status": "failed", "error": str(error)})
        raise
    finally:
        if run is not None:
            run.finish()
        lock.close()


if __name__ == "__main__":
    main()
