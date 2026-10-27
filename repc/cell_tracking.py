"""Cell4 teacher contract, frozen-interpolant samples and velocity-only PC evaluation."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import torch
from torch import nn
from torchdiffeq import odeint

from repc.hclt_pc import TimeDependentHCLTPC
from repc.single_cell import initialize_source, sha256_file
from repc.single_cell_eval import emd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALI_ROOT = ROOT.parent / "ali-cfm"
EXPECTED_SHA256 = "d2c071a2b2fea1931c5c515d4dbc35e8f436043f58322fa0c8083b3503049129"
DISPLAY_FRAMES = [0, 16, 32, 48, 65, 81, 97, 114]


def atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def code_identity():
    paths = [ROOT / "repc/cell_tracking.py", ROOT / "repc/hclt_pc.py",
             ROOT / "repc/single_cell.py", ROOT / "repc/single_cell_eval.py",
             ROOT / "scripts/prepare_cell_tracking.py", ROOT / "scripts/train_cell_tracking.py"]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}


def input_identity(ali_root, checkpoint, mask_stack, image_dir, expected_sha256):
    checkpoint, mask_stack, image_dir, ali_root = map(Path, (checkpoint, mask_stack, image_dir, ali_root))
    digest = sha256_file(checkpoint)
    if digest != expected_sha256:
        raise ValueError(f"Selected root checkpoint hash mismatch: expected {expected_sha256}, got {digest}")
    return {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": digest,
            "mask_stack": str(mask_stack.resolve()), "mask_sha256": sha256_file(mask_stack),
            "image_dir": str(image_dir.resolve()), "native_nets_sha256": sha256_file(ali_root / "ali_cfm/nets.py"),
            "native_land_metric_sha256": sha256_file(ali_root / "ali_cfm/land_metric.py"),
            "image_sha256": {str(i): sha256_file(image_dir / f"t{i:03d}.tif") for i in DISPLAY_FRAMES}}


class CellTeacher(nn.Module):
    """The checkpoint CFM consumes [normalized x, normalized y, GLOBAL t]."""
    def __init__(self, model):
        super().__init__()
        self.model = model.eval().requires_grad_(False)

    def forward(self, time, values):
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("expected normalized [N,2] coordinates")
        time = torch.as_tensor(time, device=values.device, dtype=values.dtype).reshape(-1)
        if time.numel() == 1:
            time = time.expand(len(values))
        if len(time) != len(values):
            raise ValueError("one global time per particle is required")
        return self.model(torch.cat([values, time[:, None]], -1))


def load_teacher(ali_root, checkpoint, expected_sha256, device):
    ali_root = Path(ali_root).resolve()
    if sha256_file(Path(checkpoint)) != expected_sha256:
        raise ValueError("checkpoint identity differs from the selected root teacher")
    sys.path.insert(0, str(ali_root))
    native = importlib.import_module("ali_cfm.nets")
    if Path(native.__file__).resolve() != ali_root / "ali_cfm/nets.py":
        raise ImportError("ali_cfm.nets came from a different checkout")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    if int(config["dim"]) != 2 or not config.get("normalize_dataset") or config.get("whiten"):
        raise ValueError("expected the selected 2D, normalized, non-whitened cell teacher")
    if config.get("cfm_ot") != "border":
        raise ValueError("this sampler implements the saved border-OT endpoint contract")
    model = native.MLP(dim=2, w=int(config["cfm_dim"]), time_varying=True)
    model.load_state_dict(saved["cfm_model"], strict=True)
    generator = native.TrainableInterpolant(dim=2, h_dim=int(config["net_hidden"]),
        t_smooth=float(config.get("t_smooth", .01)), time_varying=True,
        regulariser=config.get("regulariser", "piecewise"))
    generator.load_state_dict(saved["interpolant"], strict=True)
    teacher = CellTeacher(model).to(device)
    generator = generator.to(device).eval().requires_grad_(False)
    minimum, maximum = [torch.as_tensor(v, dtype=torch.float32).cpu() for v in saved["min_max"]]
    if minimum.shape != (2,) or maximum.shape != (2,) or not (maximum > minimum).all():
        raise ValueError("invalid saved pixel min-max normalization")
    return teacher, generator, config, minimum, maximum


def load_masks(path, minimum, maximum, split_seed=2026, validation_fraction=.1):
    masks = np.load(path, mmap_mode="r", allow_pickle=False)
    if masks.ndim != 3 or masks.shape[0] != 115:
        raise ValueError(f"expected 115 Cell4 masks, got {masks.shape}")
    pixels = [torch.from_numpy(np.argwhere(mask)[:, [1, 0]].astype(np.float32)) for mask in masks]
    if any(len(x) < 2 for x in pixels):
        raise ValueError("empty or degenerate cell mask")
    lower = torch.stack([x.min(0).values for x in pixels]).min(0).values
    upper = torch.stack([x.max(0).values for x in pixels]).max(0).values
    if not torch.equal(lower, minimum) or not torch.equal(upper, maximum):
        raise ValueError("mask bounds differ from checkpoint; do not refit normalization")
    if not 0 < validation_fraction < .5:
        raise ValueError("invalid endpoint-row validation fraction")
    clouds = [(x - minimum) / (maximum - minimum) for x in pixels]
    splits = []
    for index in (0, 114):
        order = torch.randperm(len(clouds[index]), generator=torch.Generator().manual_seed(split_seed + index))
        n = max(1, round(len(order) * validation_fraction))
        splits.append((order[n:], order[:n]))
    return {"clouds": clouds, "train_source_rows": splits[0][0], "validation_source_rows": splits[0][1],
            "train_end_rows": splits[1][0], "validation_end_rows": splits[1][1],
            "minimum": minimum, "scale": maximum - minimum,
            "times": torch.linspace(0, 1, 115), "mask_shape": list(masks.shape[1:]),
            "split_seed": split_seed, "validation_fraction": validation_fraction,
            "split_unit": "disjoint source and final endpoint pixels; student fidelity, not unseen-teacher data"}


def endpoint_batch(bundle, count, rng, validation=False):
    """Uniform minibatch OT, then iid sampling of its edges with replacement.

    For equal uniform batches the exact squared-cost OT plan is a permutation.
    Sampling its matched edges reproduces native border-OT sample_plan behavior.
    Full-mask pools broaden the original teacher's ten-point training pools.
    """
    prefix = "validation" if validation else "train"
    a = bundle[f"{prefix}_source_rows"]
    b = bundle[f"{prefix}_end_rows"]
    x0 = bundle["clouds"][0][a[torch.randint(len(a), (count,), generator=rng)]]
    x1 = bundle["clouds"][-1][b[torch.randint(len(b), (count,), generator=rng)]]
    row, col = linear_sum_assignment(cdist(x0.numpy(), x1.numpy(), "sqeuclidean"))
    edges = torch.randint(count, (count,), generator=rng)
    return x0[torch.as_tensor(row)[edges]], x1[torch.as_tensor(col)[edges]]


@torch.no_grad()
def sample_interpolant(generator, bundle, count, rng, device, validation=False):
    x0, x1 = endpoint_batch(bundle, count, rng, validation)
    times = torch.rand(count, generator=rng).to(device)
    values = generator(x0.to(device), x1.to(device), times[:, None], training=False)
    return values, times


class CellStudent(TimeDependentHCLTPC):
    """Standard free-source PC; clamp-consistent derivatives at saturated widths."""
    def leaf_parameters(self, time):
        mean, mean_dt, logs, logs_dt = super().leaf_parameters(time)
        return mean, mean_dt, logs, logs_dt * ((logs > -20) & (logs < 20))


def build_student(num_latents, time_embedding_dim, seed):
    return CellStudent(parents=[-1, 0], num_latents=num_latents, time_conditioning="fourier",
        time_embedding_dim=time_embedding_dim, time_feature_seed=seed, fourier_zero_center=False)


def initialize_student(student, bundle, seed):
    initialize_source(student, bundle["clouds"][0][bundle["train_source_rows"]],
                      torch.Generator().manual_seed(seed + 20000))


@torch.no_grad()
def validation_metrics(student, bundle, device, batch_size=128):
    was_training = student.training
    student.eval()
    results = {}
    for name in ("global", "late", "final"):
        data = bundle["velocity_validation"][name]
        error = energy = 0.
        for start in range(0, len(data["values"]), batch_size):
            x, t, target = [data[k][start:start + batch_size].to(device) for k in ("values", "times", "target")]
            error += float((student.velocity(x, t) - target).square().sum())
            energy += float(target.square().sum())
        results[f"{name}_velocity_mse"] = error / data["target"].numel()
        if name == "global":
            results["teacher_velocity_energy"] = energy / data["target"].numel()
    norm = max(results["teacher_velocity_energy"], 1e-8)
    results["relative_velocity_mse"] = results["global_velocity_mse"] / norm
    results["selection_score"] = max(results[f"{name}_velocity_mse"] for name in ("global", "late", "final")) / norm
    student.train(was_training)
    return results


@torch.no_grad()
def rollout(model, source, times, device, batch_size=512, tolerance=1e-5, is_student=False, max_nfe=50000):
    model.eval()
    paths, total_nfe = [], 0
    for start in range(0, len(source), batch_size):
        nfe = 0

        def velocity(time, values):
            nonlocal nfe
            nfe += 1
            if nfe > max_nfe:
                raise RuntimeError("ODE evaluation exceeded its per-batch function-evaluation budget")
            result = model.velocity(values, time) if is_student else model(time, values)
            if not torch.isfinite(result).all():
                raise FloatingPointError("nonfinite velocity during ODE evaluation")
            return result

        path = odeint(velocity, source[start:start + batch_size].to(device), times.to(device),
                       method="dopri5", rtol=tolerance, atol=tolerance)
        if not torch.isfinite(path).all():
            raise FloatingPointError("nonfinite ODE positions")
        paths.append(path.cpu()); total_nfe += nfe
    return torch.cat(paths, 1), total_nfe


def pixels(values, bundle):
    return values.double() * bundle["scale"].double() + bundle["minimum"].double()


def make_metric_rows(bundle, student_path=None):
    teacher_path = bundle["teacher_path"]
    rows = []
    for i in range(115):
        row = {"frame_index": i, "frame_number": i + 1, "time": float(bundle["times"][i]),
               "is_display_frame": i in DISPLAY_FRAMES, "num_mask_pixels": len(bundle["clouds"][i]),
               "num_source_particles": len(teacher_path[0]), "teacher_emd_pixels": None,
               "student_emd_pixels": None, "student_teacher_emd_pixels": None,
               "paired_position_rmse_pixels": None}
        if i in DISPLAY_FRAMES:
            real_idx = bundle["metric_real_rows"][i]
            source_idx = bundle["metric_source_rows"]
            real = pixels(bundle["clouds"][i][real_idx], bundle)
            teacher = pixels(teacher_path[i][source_idx], bundle)
            row["teacher_emd_pixels"] = emd(real, teacher)
            row["metric_reference_count"] = len(real_idx)
            row["metric_particle_count"] = len(source_idx)
            if student_path is not None:
                student = pixels(student_path[i][source_idx], bundle)
                row["student_emd_pixels"] = emd(real, student)
                row["student_teacher_emd_pixels"] = emd(teacher, student)
        else:
            row["metric_reference_count"] = row["metric_particle_count"] = 0
        if student_path is not None:
            difference = pixels(student_path[i], bundle) - pixels(teacher_path[i], bundle)
            row["paired_position_rmse_pixels"] = float(difference.square().sum(-1).mean().sqrt())
        rows.append(row)
    return rows


def score_paths(bundle, student_path=None):
    rows = make_metric_rows(bundle, student_path)
    displayed = [r for r in rows if r["is_display_frame"] and r["frame_index"] != 0]
    result = {"scores": rows, "teacher_display_mean_emd_pixels": float(np.mean([r["teacher_emd_pixels"] for r in displayed])),
              "teacher_final_emd_pixels": rows[-1]["teacher_emd_pixels"],
              "emd_protocol": "W1 on fixed deterministic subsamples at eight display frames; pixel Euclidean costs; not full-mask EMD"}
    if student_path is not None:
        late = [r for r in rows if r["time"] >= .8]
        result.update(student_display_mean_emd_pixels=float(np.mean([r["student_emd_pixels"] for r in displayed])),
            student_final_emd_pixels=rows[-1]["student_emd_pixels"],
            student_teacher_display_mean_emd_pixels=float(np.mean([r["student_teacher_emd_pixels"] for r in displayed])),
            final_paired_rmse_pixels=rows[-1]["paired_position_rmse_pixels"],
            late_paired_rmse_pixels=float(np.mean([r["paired_position_rmse_pixels"] for r in late])),
            max_paired_rmse_pixels=max(r["paired_position_rmse_pixels"] for r in rows))
    return result


def overlay_images(bundle, path, output, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, i in zip(axes.flat, DISPLAY_FRAMES):
        with Image.open(Path(bundle["identity"]["image_dir"]) / f"t{i:03d}.tif") as raw:
            image = np.asarray(raw)
        if list(image.shape[:2]) != bundle["mask_shape"]:
            raise ValueError("microscopy frame shape differs from mask stack")
        real = pixels(bundle["clouds"][i], bundle).numpy()
        predicted = pixels(path[i], bundle).numpy()
        ax.imshow(image, cmap="gray", origin="upper")
        ax.scatter(real[:, 0], real[:, 1], s=1, alpha=.12, color="cyan")
        ax.scatter(predicted[:, 0], predicted[:, 1], s=2, alpha=.35, color="red")
        ax.set_xlim(0, image.shape[1]); ax.set_ylim(image.shape[0], 0)
        ax.set_title(f"Frame {i+1} (t={i/114:.3f})"); ax.axis("off")
    fig.suptitle(f"Cell tracking | {label} | cyan: real mask, red: ODE prediction", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(output, dpi=150, bbox_inches="tight"); plt.close(fig)


def comparison_image(bundle, student_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 8, figsize=(26, 7))
    for col, i in enumerate(DISPLAY_FRAMES):
        with Image.open(Path(bundle["identity"]["image_dir"]) / f"t{i:03d}.tif") as raw:
            image = np.asarray(raw)
        real = pixels(bundle["clouds"][i], bundle).numpy()
        for row, (path, label) in enumerate(((bundle["teacher_path"], "Teacher"), (student_path, "Student"))):
            ax = axes[row, col]; predicted = pixels(path[i], bundle).numpy()
            ax.imshow(image, cmap="gray", origin="upper")
            ax.scatter(real[:, 0], real[:, 1], s=.7, alpha=.12, color="cyan")
            ax.scatter(predicted[:, 0], predicted[:, 1], s=1.2, alpha=.35, color="red")
            ax.set_xlim(0, image.shape[1]); ax.set_ylim(image.shape[0], 0)
            ax.set_title(f"{label} | frame {i+1}", fontsize=10); ax.axis("off")
    fig.suptitle("Cell tracking: identical source particles | cyan: actual mask, red: velocity-driven ODE", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(output, dpi=150, bbox_inches="tight"); plt.close(fig)
