import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import torch
import wandb
from scipy.stats import pearsonr, spearmanr

from repc.pc import TimeDependentGaussianPC
from repc.synthetic import (
    BranchingGaussianFlow,
    sample_path_times,
    standard_branching_flow,
)

BASIS_QUERY_NAMES = (
    "mean_x",
    "mean_y",
    "second_x",
    "second_y",
    "evidence_mass",
    "evidence_y_first",
    "evidence_y_second",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analytic-oracle Flow2PC benchmark with configurable velocity and "
            "path-density objectives"
        )
    )
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--capacities", default="2,4,8")
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--time-order", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--density-loss-weight",
        type=float,
        default=0.0,
        help="lambda_den in L_vel + lambda_den * L_den",
    )
    parser.add_argument(
        "--timestep-sampling",
        choices=("uniform", "beta"),
        default="uniform",
    )
    parser.add_argument("--timestep-beta-alpha", type=float, default=1.0)
    parser.add_argument("--timestep-beta-beta", type=float, default=0.8)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-samples", type=int, default=16_384)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--sliced-wasserstein-projections", type=int, default=128)
    parser.add_argument("--certificate-samples", type=int, default=512)
    parser.add_argument("--certificate-times", type=int, default=17)
    parser.add_argument("--certificate-ode-steps", type=int, default=24)
    parser.add_argument(
        "--max-pruned-variants",
        type=int,
        default=8,
        help="number of leave-one-component-out variants from the largest PC",
    )
    parser.add_argument("--evidence-center", type=float, default=3.1)
    parser.add_argument("--evidence-scale", type=float, default=0.45)
    parser.add_argument("--event-threshold", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--wandb-project", default="flow2pc")
    parser.add_argument("--wandb-entity")
    parser.add_argument(
        "--wandb-group",
        default="synthetic-query-faithfulness",
    )
    parser.add_argument("--wandb-run-name")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    args = parser.parse_args()
    args.capacities = tuple(int(value) for value in args.capacities.split(","))
    if not args.capacities or any(value < 1 for value in args.capacities):
        parser.error("--capacities must contain positive comma-separated integers")
    if args.steps < 1 or args.batch_size < 1:
        parser.error("--steps and --batch-size must be positive")
    if args.density_loss_weight < 0.0:
        parser.error("--density-loss-weight cannot be negative")
    if args.timestep_beta_alpha <= 0.0 or args.timestep_beta_beta <= 0.0:
        parser.error("Beta timestep concentrations must be positive")
    if args.certificate_samples < 0:
        parser.error("--certificate-samples cannot be negative")
    if args.certificate_samples and args.certificate_times < 2:
        parser.error("--certificate-times must be at least two")
    if args.certificate_ode_steps < 1:
        parser.error("--certificate-ode-steps must be positive")
    if args.output_dir is None:
        objective_label = (
            "distill-only"
            if args.density_loss_weight == 0.0
            else f"coupled-lambda{args.density_loss_weight:g}".replace(".", "p")
        )
        if args.timestep_sampling == "beta":
            objective_label += (
                f"-beta-a{args.timestep_beta_alpha:g}-b{args.timestep_beta_beta:g}"
            ).replace(".", "p")
        args.output_dir = Path(
            f"outputs/runs/synthetic/branching-{objective_label}-seed{args.seed}"
        )
    if args.wandb_run_name is None:
        objective_label = (
            "distill-only"
            if args.density_loss_weight == 0.0
            else f"coupled-lambda{args.density_loss_weight:g}".replace(".", "p")
        )
        if args.timestep_sampling == "beta":
            objective_label += (
                f"-beta-a{args.timestep_beta_alpha:g}-b{args.timestep_beta_beta:g}"
            ).replace(".", "p")
        args.wandb_run_name = f"synthetic-branching-{objective_label}-seed{args.seed}"
    return args


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def initialize_student(student: TimeDependentGaussianPC, seed: int) -> None:
    """Break mixture symmetry without using source parameters or query targets."""
    generator = torch.Generator(device=student.mean_coefficients.device).manual_seed(
        seed
    )
    with torch.no_grad():
        student.mean_coefficients.normal_(0.0, 0.02, generator=generator)
        student.log_scale_coefficients.normal_(0.0, 0.02, generator=generator)
        student.mixture_logits.normal_(0.0, 0.02, generator=generator)


def train_student(
    source: BranchingGaussianFlow,
    capacity: int,
    args: argparse.Namespace,
    tracking_run,
    global_step_offset: int,
) -> tuple[TimeDependentGaussianPC, list[dict]]:
    torch.manual_seed(args.seed + 10_000 + capacity)
    student = TimeDependentGaussianPC(
        num_components=capacity,
        dim=source.dim,
        time_order=args.time_order,
    ).to(source.endpoint_means.device)
    initialize_student(student, args.seed + 20_000 + capacity)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    generator = torch.Generator(device=source.endpoint_means.device).manual_seed(
        args.seed + 30_000 + capacity
    )
    history = []
    started_at = time.perf_counter()
    recent_losses = []
    student.train()
    for step in range(1, args.steps + 1):
        times = sample_path_times(
            args.batch_size,
            source.endpoint_means.device,
            generator,
            args.timestep_sampling,
            args.timestep_beta_alpha,
            args.timestep_beta_beta,
        )
        values = source.sample(args.batch_size, times, generator)
        with torch.no_grad():
            source_velocity = source.velocity(values, times)
        student_velocity = student.velocity(values, times)
        velocity_loss = torch.nn.functional.mse_loss(
            student_velocity,
            source_velocity,
        )
        density_nll = -student.log_prob(values, times).mean()
        weighted_density_loss = args.density_loss_weight * density_nll
        loss = velocity_loss + weighted_density_loss
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss for capacity {capacity} at step {step}"
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                args.gradient_clip,
            )
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        recent_losses.append(float(loss))
        recent_losses = recent_losses[-100:]

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            elapsed = time.perf_counter() - started_at
            record = {
                "capacity": capacity,
                "step": step,
                "loss": float(loss),
                "velocity_loss": float(velocity_loss),
                "density_nll": float(density_nll),
                "weighted_density_loss": float(weighted_density_loss),
                "loss_mean_100": sum(recent_losses) / len(recent_losses),
                "learning_rate": learning_rate,
                "gradient_norm": gradient_norm,
                "timestep_mean": float(times.mean()),
                "timestep_std": float(times.std(unbiased=False)),
                "timestep_fraction_t_ge_0p8": float((times >= 0.8).float().mean()),
                "elapsed_seconds": elapsed,
                "samples_per_second": step * args.batch_size / elapsed,
            }
            history.append(record)
            tracking_run.log(
                {
                    "train/global_step": global_step_offset + step,
                    "train/local_step": step,
                    "train/capacity": capacity,
                    "train/loss": record["loss"],
                    "train/velocity_loss": record["velocity_loss"],
                    "train/density_nll": record["density_nll"],
                    "train/weighted_density_loss": record["weighted_density_loss"],
                    "train/loss_mean_100": record["loss_mean_100"],
                    "train/learning_rate": learning_rate,
                    "train/gradient_norm": gradient_norm,
                    "train/timestep_mean": record["timestep_mean"],
                    "train/timestep_std": record["timestep_std"],
                    "train/timestep_fraction_t_ge_0p8": record[
                        "timestep_fraction_t_ge_0p8"
                    ],
                    "performance/samples_per_second": record["samples_per_second"],
                    "performance/elapsed_seconds": elapsed,
                }
            )
            print(
                f"capacity={capacity} step={step}/{args.steps} "
                f"loss={float(loss):.6f} vel={float(velocity_loss):.6f} "
                f"nll={float(density_nll):.4f} tmean={float(times.mean()):.3f} "
                f"grad={gradient_norm:.4f} "
                f"samples/s={record['samples_per_second']:.0f}",
                flush=True,
            )
    student.eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    return student, history


def clone_pruned_student(
    student: TimeDependentGaussianPC,
    keep_components: list[int],
) -> TimeDependentGaussianPC:
    pruned = TimeDependentGaussianPC(
        num_components=len(keep_components),
        dim=student.dim,
        time_order=student.time_order,
    ).to(student.mean_coefficients.device)
    with torch.no_grad():
        pruned.mean_coefficients.copy_(student.mean_coefficients[keep_components])
        pruned.log_scale_coefficients.copy_(
            student.log_scale_coefficients[keep_components]
        )
        pruned.mixture_logits.copy_(student.mixture_logits[keep_components])
    pruned.eval()
    for parameter in pruned.parameters():
        parameter.requires_grad_(False)
    return pruned


def mixture_query_values(
    weights: torch.Tensor,
    means: torch.Tensor,
    scales: torch.Tensor,
    evidence_center: float,
    evidence_scale: float,
    event_threshold: float,
) -> dict[str, float]:
    mean_x = (weights * means[:, 0]).sum()
    mean_y = (weights * means[:, 1]).sum()
    second_x = (weights * (means[:, 0].square() + scales[:, 0].square())).sum()
    second_y = (weights * (means[:, 1].square() + scales[:, 1].square())).sum()

    evidence_variance = evidence_scale**2 + scales[:, 0].square()
    evidence_by_component = (
        evidence_scale
        / evidence_variance.sqrt()
        * torch.exp(-0.5 * (means[:, 0] - evidence_center).square() / evidence_variance)
    )
    weighted_evidence = weights * evidence_by_component
    evidence_mass = weighted_evidence.sum()
    evidence_y_first = (weighted_evidence * means[:, 1]).sum()
    evidence_y_second = (
        weighted_evidence * (means[:, 1].square() + scales[:, 1].square())
    ).sum()
    conditional_mean = evidence_y_first / evidence_mass
    conditional_variance = evidence_y_second / evidence_mass - conditional_mean.square()
    event_probability = (
        weights * torch.special.ndtr((means[:, 0] - event_threshold) / scales[:, 0])
    ).sum()
    return {
        "mean_x": float(mean_x),
        "mean_y": float(mean_y),
        "second_x": float(second_x),
        "second_y": float(second_y),
        "evidence_mass": float(evidence_mass),
        "evidence_y_first": float(evidence_y_first),
        "evidence_y_second": float(evidence_y_second),
        "conditional_mean_y": float(conditional_mean),
        "conditional_variance_y": float(conditional_variance),
        "event_probability_x_gt_threshold": float(event_probability),
    }


def student_mixture_parameters(
    student: TimeDependentGaussianPC,
    time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    means, _, log_scales, _ = student.leaf_parameters(time)
    return torch.softmax(student.mixture_logits, dim=0), means, log_scales.exp()


def sliced_wasserstein_distance(
    source_samples: torch.Tensor,
    student_samples: torch.Tensor,
    num_projections: int,
) -> float:
    angles = torch.arange(
        num_projections,
        device=source_samples.device,
        dtype=source_samples.dtype,
    )
    angles = math.pi * (angles + 0.5) / num_projections
    projections = torch.stack((angles.cos(), angles.sin()), dim=1)
    source_projected = (source_samples @ projections.T).sort(dim=0).values
    student_projected = (student_samples @ projections.T).sort(dim=0).values
    return float((source_projected - student_projected).abs().mean())


@torch.no_grad()
def endpoint_density_l1(
    source: BranchingGaussianFlow,
    student: TimeDependentGaussianPC,
    grid_size: int,
) -> float:
    coordinates = torch.linspace(
        -5.0,
        5.0,
        grid_size,
        device=source.endpoint_means.device,
    )
    grid_x, grid_y = torch.meshgrid(coordinates, coordinates, indexing="ij")
    points = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1)
    endpoint = torch.tensor(1.0, device=points.device)
    source_density = source.log_prob(points, endpoint).exp()
    student_density = student.log_prob(points, endpoint).exp()
    spacing = float(coordinates[1] - coordinates[0])
    return float((source_density - student_density).abs().sum() * spacing**2)


@torch.no_grad()
def evaluate_variant(
    name: str,
    kind: str,
    source: BranchingGaussianFlow,
    student: TimeDependentGaussianPC,
    args: argparse.Namespace,
    source_queries: dict[str, float],
) -> dict:
    device = source.endpoint_means.device
    generator = torch.Generator(device=device).manual_seed(args.seed + 100_003)
    times = torch.rand(args.eval_samples, device=device, generator=generator)
    path_samples = source.sample(args.eval_samples, times, generator)
    source_velocity = source.velocity(path_samples, times)
    student_velocity = student.velocity(path_samples, times)
    velocity_mse = float((source_velocity - student_velocity).square().mean())

    endpoint = torch.tensor(1.0, device=device)
    source_samples = source.sample(args.eval_samples, endpoint, generator)
    student_generator = torch.Generator(device=device).manual_seed(args.seed + 200_003)
    student_samples = student.sample(args.eval_samples, endpoint, student_generator)
    source_log_prob = source.log_prob(source_samples, endpoint)
    student_log_prob = student.log_prob(source_samples, endpoint)
    source_nll = float(-source_log_prob.mean())
    student_nll = float(-student_log_prob.mean())
    kl_source_to_student = float((source_log_prob - student_log_prob).mean())
    swd_samples = min(args.eval_samples, 8_192)
    sliced_wasserstein = sliced_wasserstein_distance(
        source_samples[:swd_samples],
        student_samples[:swd_samples],
        args.sliced_wasserstein_projections,
    )
    weights, means, scales = student_mixture_parameters(student, endpoint)
    student_queries = mixture_query_values(
        weights,
        means,
        scales,
        args.evidence_center,
        args.evidence_scale,
        args.event_threshold,
    )
    query_errors = {
        query: abs(source_queries[query] - value)
        for query, value in student_queries.items()
    }
    return {
        "name": name,
        "kind": kind,
        "capacity": student.num_components,
        "parameter_count": sum(parameter.numel() for parameter in student.parameters()),
        "global_velocity_mse": velocity_mse,
        "endpoint_source_nll": source_nll,
        "endpoint_student_nll_on_source": student_nll,
        "endpoint_kl_source_to_student": kl_source_to_student,
        "endpoint_density_l1": endpoint_density_l1(
            source,
            student,
            args.grid_size,
        ),
        "endpoint_sliced_wasserstein": sliced_wasserstein,
        "query_values": student_queries,
        "query_errors": query_errors,
    }


def smooth_basis_queries(
    values: torch.Tensor,
    evidence_center: float,
    evidence_scale: float,
) -> torch.Tensor:
    x_values = values[:, 0]
    y_values = values[:, 1]
    evidence = torch.exp(
        -0.5 * ((x_values - evidence_center) / evidence_scale).square()
    )
    return torch.stack(
        (
            x_values,
            y_values,
            x_values.square(),
            y_values.square(),
            evidence,
            y_values * evidence,
            y_values.square() * evidence,
        ),
        dim=1,
    )


def integrate_student_flow(
    student: TimeDependentGaussianPC,
    values: torch.Tensor,
    start_time: float,
    end_time: float,
    steps: int,
) -> torch.Tensor:
    if start_time == end_time:
        return values
    step_size = (end_time - start_time) / steps
    state = values
    for step in range(steps):
        current_time = start_time + step * step_size
        t0 = torch.tensor(current_time, device=values.device, dtype=values.dtype)
        tm = torch.tensor(
            current_time + 0.5 * step_size,
            device=values.device,
            dtype=values.dtype,
        )
        t1 = torch.tensor(
            current_time + step_size,
            device=values.device,
            dtype=values.dtype,
        )
        k1 = student.velocity(state, t0)
        k2 = student.velocity(state + 0.5 * step_size * k1, tm)
        k3 = student.velocity(state + 0.5 * step_size * k2, tm)
        k4 = student.velocity(state + step_size * k3, t1)
        state = state + step_size * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return state


def query_certificates(
    source: BranchingGaussianFlow,
    student: TimeDependentGaussianPC,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Estimate Eq. (16) with a trapezoidal time grid and RK4 adjoints."""
    if args.certificate_samples == 0:
        return {}
    device = source.endpoint_means.device
    generator = torch.Generator(device=device).manual_seed(args.seed + 300_003)
    time_grid = torch.linspace(0.0, 1.0, args.certificate_times, device=device)
    accumulated = torch.zeros(len(BASIS_QUERY_NAMES), device=device)
    time_step = 1.0 / (args.certificate_times - 1)

    for time_index, path_time in enumerate(time_grid):
        time_value = float(path_time)
        with torch.no_grad():
            source_values = source.sample(
                args.certificate_samples,
                path_time,
                generator,
            )
            residual = source.velocity(source_values, path_time) - student.velocity(
                source_values,
                path_time,
            )
        values = source_values.detach().requires_grad_(True)
        terminal_values = integrate_student_flow(
            student,
            values,
            time_value,
            1.0,
            args.certificate_ode_steps,
        )
        outputs = smooth_basis_queries(
            terminal_values,
            args.evidence_center,
            args.evidence_scale,
        )
        integrand = []
        for query_index in range(len(BASIS_QUERY_NAMES)):
            gradient = torch.autograd.grad(
                outputs[:, query_index].sum(),
                values,
                retain_graph=query_index + 1 < len(BASIS_QUERY_NAMES),
            )[0]
            integrand.append((residual * gradient).sum(dim=1).mean().detach())
        quadrature_weight = 0.5 if time_index in (0, len(time_grid) - 1) else 1.0
        accumulated += quadrature_weight * torch.stack(integrand)
    accumulated *= time_step
    return {
        name: float(value)
        for name, value in zip(BASIS_QUERY_NAMES, accumulated, strict=True)
    }


def safe_correlation(function, left: list[float], right: list[float]) -> float:
    if len(left) < 3 or max(left) == min(left) or max(right) == min(right):
        return float("nan")
    return float(function(left, right).statistic)


def correlation_summary(results: list[dict], source_queries: dict[str, float]) -> dict:
    per_query = {}
    signed_actual = []
    signed_certificate = []
    relative_errors = []
    for query in BASIS_QUERY_NAMES:
        actual = [result["query_errors"][query] for result in results]
        global_mse = [result["global_velocity_mse"] for result in results]
        certificate = [abs(result["certificates"][query]) for result in results]
        per_query[query] = {
            "global_mse_spearman": safe_correlation(spearmanr, global_mse, actual),
            "certificate_spearman": safe_correlation(
                spearmanr,
                certificate,
                actual,
            ),
        }
        for result in results:
            delta = source_queries[query] - result["query_values"][query]
            estimate = result["certificates"][query]
            signed_actual.append(delta)
            signed_certificate.append(estimate)
            relative_errors.append(abs(estimate - delta) / (abs(delta) + 1e-4))
    global_correlations = [
        value["global_mse_spearman"]
        for value in per_query.values()
        if math.isfinite(value["global_mse_spearman"])
    ]
    certificate_correlations = [
        value["certificate_spearman"]
        for value in per_query.values()
        if math.isfinite(value["certificate_spearman"])
    ]
    return {
        "per_query": per_query,
        "global_mse_spearman_mean": (
            sum(global_correlations) / len(global_correlations)
            if global_correlations
            else float("nan")
        ),
        "certificate_spearman_mean": (
            sum(certificate_correlations) / len(certificate_correlations)
            if certificate_correlations
            else float("nan")
        ),
        "certificate_signed_pearson": safe_correlation(
            pearsonr,
            signed_certificate,
            signed_actual,
        ),
        "certificate_median_relative_error": float(
            torch.tensor(relative_errors).median()
        ),
    }


def select_comparable_pair(results: list[dict]) -> dict:
    candidates = [result for result in results if result["kind"] == "pruned"]
    if len(candidates) < 2:
        candidates = results
    if len(candidates) < 2:
        return {}
    pairs = []
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            mse_ratio = max(
                left["global_velocity_mse"],
                right["global_velocity_mse"],
            ) / min(left["global_velocity_mse"], right["global_velocity_mse"])
            left_error = left["query_errors"]["conditional_mean_y"]
            right_error = right["query_errors"]["conditional_mean_y"]
            error_ratio = (max(left_error, right_error) + 1e-8) / (
                min(left_error, right_error) + 1e-8
            )
            pairs.append((mse_ratio, -error_ratio, left, right, error_ratio))
    comparable = [pair for pair in pairs if pair[0] <= 1.20]
    selected = min(comparable or pairs, key=lambda pair: (pair[1], pair[0]))
    mse_ratio, _, left, right, error_ratio = selected
    return {
        "left": left["name"],
        "right": right["name"],
        "global_velocity_mse_ratio": mse_ratio,
        "conditional_mean_error_ratio": error_ratio,
        "left_global_velocity_mse": left["global_velocity_mse"],
        "right_global_velocity_mse": right["global_velocity_mse"],
        "left_conditional_mean_error": left["query_errors"]["conditional_mean_y"],
        "right_conditional_mean_error": right["query_errors"]["conditional_mean_y"],
    }


def save_results_csv(results: list[dict], path: Path) -> None:
    rows = []
    for result in results:
        row = {
            key: value
            for key, value in result.items()
            if key not in ("query_values", "query_errors", "certificates")
        }
        for query, value in result["query_values"].items():
            row[f"query_value/{query}"] = value
        for query, value in result["query_errors"].items():
            row[f"query_error/{query}"] = value
        for query, value in result["certificates"].items():
            row[f"certificate/{query}"] = value
        rows.append(row)
    with path.open("w", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(
    source: BranchingGaussianFlow,
    student: TimeDependentGaussianPC,
    results: list[dict],
    source_queries: dict[str, float],
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".matplotlib"))
    import matplotlib.pyplot as plt

    device = source.endpoint_means.device
    endpoint = torch.tensor(1.0, device=device)
    source_generator = torch.Generator(device=device).manual_seed(args.seed + 400_003)
    student_generator = torch.Generator(device=device).manual_seed(args.seed + 500_003)
    source_samples = source.sample(3_000, endpoint, source_generator).cpu()
    student_samples = student.sample(3_000, endpoint, student_generator).cpu()
    sample_path = args.output_dir / "endpoint_samples.png"
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for axis, samples, title in zip(
        axes,
        (source_samples, student_samples),
        ("Frozen analytic source", f"Distilled PC (K={student.num_components})"),
        strict=True,
    ):
        axis.scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.25)
        axis.set(xlim=(-5, 5), ylim=(-5, 5), title=title, xlabel="x", ylabel="y")
        axis.set_aspect("equal")
    figure.savefig(sample_path, dpi=180)
    plt.close(figure)

    scatter_path = args.output_dir / "query_faithfulness.png"
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for query_index, query in enumerate(BASIS_QUERY_NAMES):
        actual = [result["query_errors"][query] + 1e-10 for result in results]
        mse = [result["global_velocity_mse"] for result in results]
        certificate = [abs(result["certificates"][query]) + 1e-10 for result in results]
        axes[0].scatter(mse, actual, s=18, color=colors(query_index), label=query)
        axes[1].scatter(
            certificate,
            actual,
            s=18,
            color=colors(query_index),
            label=query,
        )
    minimum = min(
        abs(source_queries[query] - result["query_values"][query]) + 1e-10
        for query in BASIS_QUERY_NAMES
        for result in results
    )
    maximum = max(
        abs(source_queries[query] - result["query_values"][query]) + 1e-10
        for query in BASIS_QUERY_NAMES
        for result in results
    )
    axes[1].plot([minimum, maximum], [minimum, maximum], "k--", linewidth=1)
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="Global velocity MSE",
        ylabel="Absolute query error",
        title="Global score",
    )
    axes[1].set(
        xscale="log",
        yscale="log",
        xlabel="Absolute adjoint certificate",
        ylabel="Absolute query error",
        title="Query-sensitive score",
    )
    axes[1].legend(fontsize=7, loc="best")
    figure.savefig(scatter_path, dpi=180)
    plt.close(figure)
    return sample_path, scatter_path


def serializable_args(args: argparse.Namespace) -> dict:
    result = vars(args).copy()
    result["capacities"] = list(args.capacities)
    result["output_dir"] = str(args.output_dir)
    result.update(
        {
            "objective": (
                "velocity_distillation_mse_only"
                if args.density_loss_weight == 0.0
                else "velocity_distillation_plus_path_density_nll"
            ),
            "endpoint_pretraining": "none",
            "source": "analytic_vanilla_cfm_gaussian_mixture",
        }
    )
    return result


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "results.json").exists():
        raise FileExistsError(
            f"completed results already exist in {args.output_dir}; choose another directory"
        )
    config = serializable_args(args)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    source = standard_branching_flow(device=device)
    endpoint = torch.tensor(1.0, device=device)
    source_means, source_scales = source.marginal_parameters(endpoint)
    source_queries = mixture_query_values(
        source.weights,
        source_means,
        source_scales,
        args.evidence_center,
        args.evidence_scale,
        args.event_threshold,
    )

    wandb_data_dir = args.output_dir / ".wandb-data"
    wandb_data_dir.mkdir(exist_ok=True)
    os.environ.setdefault("WANDB_DATA_DIR", str(wandb_data_dir))
    started_at = time.perf_counter()
    with wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=args.wandb_run_name,
        job_type="synthetic-distill",
        tags=[
            "synthetic",
            "analytic-oracle",
            "query-faithfulness",
            (
                "distill-only"
                if args.density_loss_weight == 0.0
                else "coupled-density-velocity"
            ),
            f"lambda-density-{args.density_loss_weight:g}",
            f"timestep-{args.timestep_sampling}",
        ],
        config=config,
        mode=args.wandb_mode,
        dir=str(args.output_dir),
    ) as tracking_run:
        tracking_run.define_metric("train/global_step")
        tracking_run.define_metric("train/*", step_metric="train/global_step")
        tracking_run.define_metric("performance/*", step_metric="train/global_step")
        tracking_run.log_code(root=str(Path.cwd()))

        trained_students = {}
        training_history = []
        for capacity_index, capacity in enumerate(args.capacities):
            student, history = train_student(
                source,
                capacity,
                args,
                tracking_run,
                capacity_index * args.steps,
            )
            trained_students[capacity] = student
            training_history.extend(history)

        variants = [
            (f"trained-k{capacity}", "trained", student)
            for capacity, student in trained_students.items()
        ]
        largest_capacity = max(trained_students)
        largest_student = trained_students[largest_capacity]
        prune_count = min(args.max_pruned_variants, largest_capacity)
        for dropped_component in range(prune_count):
            keep = [
                component
                for component in range(largest_capacity)
                if component != dropped_component
            ]
            variants.append(
                (
                    f"pruned-k{largest_capacity}-drop{dropped_component}",
                    "pruned",
                    clone_pruned_student(largest_student, keep),
                )
            )

        results = []
        for variant_index, (name, kind, student) in enumerate(variants, start=1):
            print(f"evaluating {name} ({variant_index}/{len(variants)})", flush=True)
            result = evaluate_variant(
                name,
                kind,
                source,
                student,
                args,
                source_queries,
            )
            result["certificates"] = query_certificates(source, student, args)
            results.append(result)
            tracking_run.log(
                {
                    "evaluation/variant_index": variant_index,
                    "evaluation/global_velocity_mse": result["global_velocity_mse"],
                    "evaluation/endpoint_kl_source_to_student": result[
                        "endpoint_kl_source_to_student"
                    ],
                    "evaluation/endpoint_density_l1": result["endpoint_density_l1"],
                    "evaluation/endpoint_sliced_wasserstein": result[
                        "endpoint_sliced_wasserstein"
                    ],
                    "evaluation/conditional_mean_error": result["query_errors"][
                        "conditional_mean_y"
                    ],
                    "evaluation/conditional_variance_error": result["query_errors"][
                        "conditional_variance_y"
                    ],
                    "evaluation/event_probability_error": result["query_errors"][
                        "event_probability_x_gt_threshold"
                    ],
                }
            )

        correlations = (
            correlation_summary(results, source_queries)
            if args.certificate_samples
            else {}
        )
        comparable_pair = select_comparable_pair(results)
        sample_plot, faithfulness_plot = make_plots(
            source,
            largest_student,
            results,
            source_queries,
            args,
        )
        elapsed_seconds = time.perf_counter() - started_at
        payload = {
            "config": config,
            "source_queries": source_queries,
            "results": results,
            "correlations": correlations,
            "comparable_pair": comparable_pair,
            "elapsed_seconds": elapsed_seconds,
        }
        (args.output_dir / "results.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
        (args.output_dir / "training_history.json").write_text(
            json.dumps(training_history, indent=2) + "\n"
        )
        save_results_csv(results, args.output_dir / "results.csv")
        checkpoint_path = args.output_dir / "students.pt"
        torch.save(
            {
                "model_type": "synthetic_oracle_flow2pc",
                "config": config,
                "source": {
                    "weights": source.weights.cpu(),
                    "endpoint_means": source.endpoint_means.cpu(),
                    "endpoint_scales": source.endpoint_scales.cpu(),
                },
                "students": {
                    str(capacity): student.state_dict()
                    for capacity, student in trained_students.items()
                },
            },
            checkpoint_path,
        )

        table_columns = [
            "variant",
            "kind",
            "capacity",
            "parameters",
            "velocity_mse",
            "kl_pq",
            "density_l1",
            "sliced_wasserstein",
            "conditional_mean_error",
            "conditional_variance_error",
            "event_error",
        ]
        table = wandb.Table(
            columns=table_columns,
            data=[
                [
                    result["name"],
                    result["kind"],
                    result["capacity"],
                    result["parameter_count"],
                    result["global_velocity_mse"],
                    result["endpoint_kl_source_to_student"],
                    result["endpoint_density_l1"],
                    result["endpoint_sliced_wasserstein"],
                    result["query_errors"]["conditional_mean_y"],
                    result["query_errors"]["conditional_variance_y"],
                    result["query_errors"]["event_probability_x_gt_threshold"],
                ]
                for result in results
            ],
        )
        tracking_run.log(
            {
                "evaluation/results": table,
                "qualitative/endpoint_samples": wandb.Image(str(sample_plot)),
                "qualitative/query_faithfulness": wandb.Image(str(faithfulness_plot)),
            }
        )
        if correlations:
            tracking_run.summary["evidence/global_mse_spearman_mean"] = correlations[
                "global_mse_spearman_mean"
            ]
            tracking_run.summary["evidence/certificate_spearman_mean"] = correlations[
                "certificate_spearman_mean"
            ]
            tracking_run.summary["evidence/certificate_signed_pearson"] = correlations[
                "certificate_signed_pearson"
            ]
            tracking_run.summary["evidence/certificate_median_relative_error"] = (
                correlations["certificate_median_relative_error"]
            )
        for key, value in comparable_pair.items():
            tracking_run.summary[f"evidence/comparable_pair/{key}"] = value
        tracking_run.summary["performance/total_seconds"] = elapsed_seconds
        objective_label = (
            "distill-only"
            if args.density_loss_weight == 0.0
            else f"coupled-lambda{args.density_loss_weight:g}".replace(".", "p")
        )
        time_label = args.timestep_sampling
        if args.timestep_sampling == "beta":
            time_label += (
                f"-a{args.timestep_beta_alpha:g}-b{args.timestep_beta_beta:g}"
            ).replace(".", "p")
        artifact = wandb.Artifact(
            name=(
                f"synthetic-branching-{objective_label}-{time_label}-seed{args.seed}"
            ),
            type="model",
            description="PCs for the analytic-oracle synthetic benchmark",
        )
        artifact.add_file(str(checkpoint_path))
        artifact.add_file(str(args.output_dir / "results.json"))
        tracking_run.log_artifact(artifact, aliases=["latest", f"seed-{args.seed}"])

    print(json.dumps({"correlations": correlations, "pair": comparable_pair}, indent=2))
    print(f"run saved to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
