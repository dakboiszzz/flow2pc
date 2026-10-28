import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import torch
import wandb
from torch import nn
from torchdiffeq import odeint
from torchvision.utils import save_image

from flow2pc.data.real.mnist import MNIST_DIM, MNIST_SHAPE, mnist_loader, quantize_mnist
from flow2pc.legacy.evaluation import (
    class_distribution_metrics,
    frechet_distance,
    improved_precision_recall,
    inception_score,
    kernel_distance,
)
from flow2pc.legacy.mnist_classifier import MNISTClassifier, normalize_for_classifier
from flow2pc.legacy.train import (
    build_flow_matcher,
    build_student,
    load_teacher,
    pair_flow_endpoints,
    resolve_device,
)


def parse_times(value: str) -> list[float]:
    times = [float(item) for item in value.split(",")]
    if (
        not times
        or any(time < 0.0 or time > 1.0 for time in times)
        or times != sorted(set(times))
    ):
        raise argparse.ArgumentTypeError(
            "times must be unique, increasing comma-separated values in [0, 1]"
        )
    return times


def parse_pc_sampling_methods(value: str) -> list[str]:
    valid = ("direct", "euler", "rk4", "dopri5")
    methods = [item.strip() for item in value.split(",") if item.strip()]
    if not methods or len(methods) != len(set(methods)):
        raise argparse.ArgumentTypeError("PC sampling methods must be unique")
    invalid = [method for method in methods if method not in valid]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown PC sampling methods {invalid}; choose from {valid}"
        )
    if "direct" not in methods:
        raise argparse.ArgumentTypeError("PC sampling methods must include direct")
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="checkpoint relative to run-dir or an absolute checkpoint path",
    )
    parser.add_argument(
        "--output-name",
        help="optional evaluation subdirectory name (defaults to checkpoint stem)",
    )
    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        default=Path(
            "outputs/evaluation/mnist-classifier-pytorch-example-seed1/"
            "checkpoints/last.pt"
        ),
    )
    parser.add_argument(
        "--times",
        type=parse_times,
        default=parse_times("0,0.25,0.5,0.75,1"),
    )
    parser.add_argument("--num-generated", type=int, default=10_000)
    parser.add_argument("--num-grid-samples", type=int, default=64)
    parser.add_argument("--num-nearest-neighbors", type=int, default=16)
    parser.add_argument("--max-test-examples", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kid-subsets", type=int, default=100)
    parser.add_argument("--kid-subset-size", type=int, default=1000)
    parser.add_argument("--precision-recall-k", type=int, default=3)
    parser.add_argument("--metric-chunk-size", type=int, default=512)
    parser.add_argument("--ode-method", default="dopri5")
    parser.add_argument("--ode-rtol", type=float, default=1e-5)
    parser.add_argument("--ode-atol", type=float, default=1e-5)
    parser.add_argument(
        "--pc-ode-steps",
        "--pc-trajectory-steps",
        dest="pc_ode_steps",
        type=int,
        default=20,
        help="fixed Euler/RK4 steps for sampling from the PC velocity",
    )
    parser.add_argument(
        "--pc-sampling-methods",
        type=parse_pc_sampling_methods,
        default=parse_pc_sampling_methods("direct,euler,rk4,dopri5"),
        help="comma-separated PC endpoint samplers evaluated on the full protocol",
    )
    parser.add_argument("--skip-sample-metrics", action="store_true")
    parser.add_argument("--wandb-project", default="flow2pc")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    args = parser.parse_args()
    positive_counts = (
        args.num_generated,
        args.num_grid_samples,
        args.num_nearest_neighbors,
        args.batch_size,
        args.kid_subsets,
        args.kid_subset_size,
        args.precision_recall_k,
        args.metric_chunk_size,
        args.pc_ode_steps,
    )
    if any(value < 1 for value in positive_counts):
        parser.error("sample counts, batch sizes, and metric counts must be positive")
    if args.max_test_examples is not None and args.max_test_examples < 2:
        parser.error("--max-test-examples must be at least two")
    if 1.0 not in args.times:
        parser.error("--times must include endpoint t=1")
    return args


def time_label(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_checkpoint(run_dir: Path, requested: Path | None) -> Path:
    if requested is None:
        return run_dir / "checkpoints" / "last.pt"
    if requested.is_absolute():
        return requested
    candidate = run_dir / requested
    if candidate.exists():
        return candidate
    return run_dir / "checkpoints" / requested


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict, object | None]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint["model_type"] == "distill":
        model = build_student(checkpoint["model_config"]).to(device).eval()
        model.load_state_dict(checkpoint["state_dict"])
        matcher = build_flow_matcher(checkpoint["flow_matching_config"])
        return model, checkpoint, matcher
    if checkpoint["model_type"] == "flow":
        model, _ = load_teacher(checkpoint_path, device)
        return model, checkpoint, None
    raise ValueError(f"unsupported checkpoint type: {checkpoint['model_type']}")


def load_classifier(path: Path, device: torch.device) -> tuple[MNISTClassifier, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_type") != "mnist_classifier":
        raise ValueError(f"{path} is not an MNIST classifier checkpoint")
    classifier = MNISTClassifier().to(device).eval()
    classifier.load_state_dict(checkpoint["state_dict"])
    return classifier, checkpoint


class TeacherODE(nn.Module):
    def __init__(self, teacher: nn.Module):
        super().__init__()
        self.teacher = teacher
        self.nfe = 0

    def forward(self, time_value: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        times = time_value.expand(values.shape[0])
        return self.teacher(times, values)


class PCVelocityODE(nn.Module):
    """Expose the analytically derived PC velocity to an ODE solver."""

    def __init__(self, pc: nn.Module):
        super().__init__()
        self.pc = pc
        self.nfe = 0

    def forward(self, time_value: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        velocity = self.pc.velocity(values.flatten(1), time_value)
        return velocity.reshape_as(values)


@torch.no_grad()
def sample_model(
    model: nn.Module,
    model_type: str,
    count: int,
    batch_size: int,
    times: torch.Tensor,
    device: torch.device,
    seed: int,
    ode_method: str,
    ode_rtol: float,
    ode_atol: float,
    pc_ode_steps: int,
    grid_count: int,
) -> tuple[torch.Tensor, torch.Tensor, float, int, int]:
    generator = torch.Generator(device=device).manual_seed(seed)
    endpoint_samples = []
    endpoint_nfe = 0
    endpoint_time = times.new_tensor(1.0)
    integration_times = times.new_tensor((0.0, 1.0))
    synchronize(device)
    started_at = time.perf_counter()
    for start in range(0, count, batch_size):
        current_count = min(batch_size, count - start)
        if model_type == "distill":
            endpoint = model.sample(
                current_count,
                endpoint_time,
                generator=generator,
            ).reshape(current_count, *MNIST_SHAPE)
        else:
            source = torch.randn(
                current_count,
                *MNIST_SHAPE,
                device=device,
                generator=generator,
            )
            dynamics = TeacherODE(model)
            endpoint = odeint(
                dynamics,
                source,
                integration_times,
                rtol=ode_rtol,
                atol=ode_atol,
                method=ode_method,
            )[-1]
            endpoint_nfe += dynamics.nfe
        endpoint_samples.append(endpoint.detach().cpu())
    synchronize(device)
    elapsed = time.perf_counter() - started_at
    samples = torch.cat(endpoint_samples)[:count]

    grid_count = min(grid_count, count)
    grid_generator = torch.Generator(device=device).manual_seed(seed)
    grid_source = torch.randn(
        grid_count,
        *MNIST_SHAPE,
        device=device,
        generator=grid_generator,
    )
    if model_type == "distill":
        grid_dynamics = PCVelocityODE(model)
        grid_path = odeint(
            grid_dynamics,
            grid_source,
            times,
            method="rk4",
            options={"step_size": 1.0 / pc_ode_steps},
        )
    else:
        grid_dynamics = TeacherODE(model)
        grid_path = odeint(
            grid_dynamics,
            grid_source,
            times,
            rtol=ode_rtol,
            atol=ode_atol,
            method=ode_method,
        )
    grid_path = grid_path.detach().cpu()
    return samples, grid_path, elapsed, endpoint_nfe, grid_dynamics.nfe


@torch.no_grad()
def sample_pc_velocity_endpoint(
    pc: nn.Module,
    method: str,
    count: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    fixed_steps: int,
    rtol: float,
    atol: float,
) -> tuple[torch.Tensor, float, int]:
    """Sample p1 by integrating the velocity analytically induced by the PC."""
    if method not in ("euler", "rk4", "dopri5"):
        raise ValueError(f"unsupported PC velocity sampler: {method}")
    generator = torch.Generator(device=device).manual_seed(seed)
    integration_times = torch.tensor((0.0, 1.0), device=device)
    samples = []
    total_nfe = 0
    synchronize(device)
    started_at = time.perf_counter()
    for start in range(0, count, batch_size):
        current_count = min(batch_size, count - start)
        source = torch.randn(
            current_count,
            *MNIST_SHAPE,
            device=device,
            generator=generator,
        )
        dynamics = PCVelocityODE(pc)
        solver_arguments = {}
        if method in ("euler", "rk4"):
            solver_arguments["options"] = {"step_size": 1.0 / fixed_steps}
        else:
            solver_arguments.update({"rtol": rtol, "atol": atol})
        endpoint = odeint(
            dynamics,
            source,
            integration_times,
            method=method,
            **solver_arguments,
        )[-1]
        samples.append(endpoint.detach().cpu())
        total_nfe += dynamics.nfe
    synchronize(device)
    elapsed = time.perf_counter() - started_at
    return torch.cat(samples)[:count], elapsed, total_nfe


@torch.no_grad()
def classifier_outputs(
    classifier: MNISTClassifier,
    images: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = []
    probabilities = []
    for batch in images.split(batch_size):
        values = normalize_for_classifier(batch.to(device))
        batch_features = classifier.features(values)
        features.append(batch_features.cpu())
        probabilities.append(classifier.fc2(batch_features).softmax(dim=1).cpu())
    return torch.cat(features), torch.cat(probabilities)


@torch.no_grad()
def sample_quality_metrics(
    classifier: MNISTClassifier,
    generated_images: torch.Tensor,
    real_features: torch.Tensor,
    real_labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
    kid_subsets: int,
    kid_subset_size: int,
    precision_recall_k: int,
    metric_chunk_size: int,
    seed: int,
) -> tuple[dict[str, float | list[float]], torch.Tensor, torch.Tensor]:
    fake_features, fake_probabilities = classifier_outputs(
        classifier,
        generated_images,
        batch_size,
        device,
    )
    fid = frechet_distance(real_features, fake_features)
    kid_mean, kid_std = kernel_distance(
        real_features,
        fake_features,
        device,
        subsets=kid_subsets,
        subset_size=kid_subset_size,
        seed=seed,
    )
    precision, recall = improved_precision_recall(
        real_features,
        fake_features,
        device,
        nearest_neighbors=precision_recall_k,
        chunk_size=metric_chunk_size,
    )
    score_mean, score_std = inception_score(fake_probabilities)
    class_metrics = class_distribution_metrics(fake_probabilities, real_labels)
    quality = {
        "fid_mnist": fid,
        "kid_mnist_mean": kid_mean,
        "kid_mnist_std": kid_std,
        "precision_mnist": precision,
        "recall_mnist": recall,
        "inception_score_mnist_mean": score_mean,
        "inception_score_mnist_std": score_std,
        "classifier_confidence": class_metrics["classifier_confidence"],
        "class_entropy_normalized": class_metrics["class_entropy_normalized"],
        "class_jsd_to_test": class_metrics["class_jsd_to_test"],
        "class_distribution": class_metrics["class_distribution"],
    }
    return quality, fake_features, fake_probabilities


def sampling_label(method: str, fixed_steps: int) -> str:
    if method in ("euler", "rk4"):
        return f"{method}_{fixed_steps}"
    return method


@torch.no_grad()
def nearest_training_images(
    classifier: MNISTClassifier,
    fake_images: torch.Tensor,
    fake_features: torch.Tensor,
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> torch.Tensor:
    count = len(fake_images)
    best_distances = torch.full((count,), float("inf"), device=device)
    best_images = torch.empty_like(fake_images)
    train_loader = mnist_loader(
        data_dir,
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    queries = fake_features.to(device)
    for images, _ in train_loader:
        images = images.to(device, non_blocking=True)
        features = classifier.features(normalize_for_classifier(images))
        distances = (
            queries.square().sum(dim=1, keepdim=True)
            + features.square().sum(dim=1)[None]
            - 2.0 * queries @ features.T
        )
        values, indices = distances.min(dim=1)
        update = values < best_distances
        best_distances[update] = values[update]
        best_images[update.cpu()] = images[indices[update]].cpu()
    pairs = torch.stack((fake_images, best_images), dim=1)
    return pairs.flatten(0, 1)


def training_seconds(run_dir: Path, checkpoint_step: int | None) -> float | None:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    with metrics_path.open(newline="") as metrics_file:
        selected = None
        for row in csv.DictReader(metrics_file):
            selected = row
            if checkpoint_step is not None and int(row["step"]) == checkpoint_step:
                break
    if selected is None:
        return None
    if selected.get("elapsed_seconds"):
        return float(selected["elapsed_seconds"])
    return float(selected["examples_seen"]) / float(selected["samples_per_second"])


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    run_dir = args.run_dir.resolve()
    device = resolve_device(args.device)
    checkpoint_path = resolve_checkpoint(run_dir, args.checkpoint).resolve()
    model, checkpoint, flow_matcher = load_model(checkpoint_path, device)
    model_type = checkpoint["model_type"]
    output_name = args.output_name or checkpoint_path.stem
    if Path(output_name).name != output_name or output_name in (".", ".."):
        raise ValueError("--output-name must be a single directory name")
    output_dir = run_dir / "eval" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = Path(checkpoint["training_config"]["data_dir"])
    data_dir = data_dir.resolve()
    test_loader = mnist_loader(
        data_dir,
        train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    times = torch.tensor(args.times, device=device)
    endpoint_index = args.times.index(1.0)
    real_images = []
    real_labels = []
    path_likelihoods = [[] for _ in args.times]
    discrete_likelihoods = []
    seen = 0
    likelihood_started_at = time.perf_counter()
    path_generator = torch.Generator(device=device).manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for images, labels in test_loader:
            if args.max_test_examples is not None:
                remaining = args.max_test_examples - seen
                images = images[:remaining]
                labels = labels[:remaining]
            if len(images) == 0:
                break
            real_images.append(images)
            real_labels.append(labels)
            if model_type == "distill":
                images_device = images.to(device, non_blocking=True)
                noise = torch.randn(
                    images_device.shape,
                    device=device,
                    dtype=images_device.dtype,
                    generator=path_generator,
                )
                noise, paired_images = pair_flow_endpoints(
                    flow_matcher,
                    noise,
                    images_device,
                )
                path = torch.stack(
                    [
                        (1.0 - time_value) * noise + time_value * paired_images
                        for time_value in times
                    ]
                ).flatten(2)
                path_values = model.path_log_prob(path, times).cpu()
                for index, values in enumerate(path_values):
                    path_likelihoods[index].append(values)
                discrete_likelihoods.append(
                    model.discrete_log_prob(
                        images_device.flatten(1),
                        times[endpoint_index],
                    ).cpu()
                )
            seen += len(images)
            if args.max_test_examples is not None and seen >= args.max_test_examples:
                break
    synchronize(device)
    likelihood_elapsed = time.perf_counter() - likelihood_started_at
    real_images = torch.cat(real_images)
    real_labels = torch.cat(real_labels)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metrics: dict[str, float | int | list[float] | list[str] | str] = {
        "protocol/num_test": len(real_images),
        "model/parameter_count": parameter_count,
        "model/checkpoint_step": checkpoint.get("step", 0),
        "model/checkpoint_bytes": checkpoint_path.stat().st_size,
    }
    run_training_seconds = training_seconds(run_dir, checkpoint.get("step"))
    if run_training_seconds is not None:
        metrics["performance/train_seconds"] = run_training_seconds

    if model_type == "distill":
        path_means = []
        path_stds = []
        for index, time_value in enumerate(args.times):
            values = torch.cat(path_likelihoods[index])
            mean = values.mean().item()
            std = values.std(unbiased=False).item()
            path_means.append(mean)
            path_stds.append(std)
            metrics[f"path/log_prob_t{time_label(time_value)}"] = mean
            print(
                f"t={time_value:.3f}: continuous log-likelihood "
                f"{mean:.2f} ± {std:.2f} nats"
            )
        discrete_values = torch.cat(discrete_likelihoods)
        discrete_nll = -discrete_values.mean().item()
        bpd = discrete_nll / (MNIST_DIM * math.log(2.0))
        metrics["likelihood/test_discrete_nll_nats"] = discrete_nll
        metrics["likelihood/test_bpd"] = bpd
        metrics["likelihood/test_continuous_nll_nats"] = -path_means[endpoint_index]
        metrics["performance/likelihood_examples_per_second"] = (
            len(real_images) / likelihood_elapsed
        )
        print(f"endpoint: discrete NLL={discrete_nll:.2f} nats, bpd={bpd:.4f}")
        torch.save(
            {
                "times": torch.tensor(args.times),
                "mean_log_likelihood": torch.tensor(path_means),
                "std_log_likelihood": torch.tensor(path_stds),
                "log_likelihoods": [torch.cat(values) for values in path_likelihoods],
                "discrete_log_likelihoods": discrete_values,
            },
            output_dir / "likelihoods.pt",
        )

    generated_images = None
    image_paths: dict[str, Path] = {}
    if not args.skip_sample_metrics:
        classifier, classifier_checkpoint = load_classifier(
            args.classifier_checkpoint.resolve(),
            device,
        )
        (
            continuous_images,
            grid_path,
            sample_elapsed,
            endpoint_nfe,
            trajectory_nfe,
        ) = sample_model(
            model,
            model_type,
            args.num_generated,
            args.batch_size,
            times,
            device,
            args.seed,
            args.ode_method,
            args.ode_rtol,
            args.ode_atol,
            args.pc_ode_steps,
            args.num_grid_samples,
        )
        generated_images = quantize_mnist(continuous_images)
        metrics["protocol/num_generated"] = len(generated_images)
        metrics["performance/sample_images_per_second"] = (
            len(generated_images) / sample_elapsed
        )
        metrics["performance/endpoint_total_nfe"] = endpoint_nfe
        metrics["performance/trajectory_nfe"] = trajectory_nfe
        if model_type == "flow":
            metrics["performance/teacher_mean_nfe_per_batch"] = (
                endpoint_nfe / math.ceil(len(generated_images) / args.batch_size)
            )

        endpoint_path = output_dir / "samples_endpoint.png"
        save_image(
            generated_images[: args.num_grid_samples].clamp(-1, 1) * 0.5 + 0.5,
            endpoint_path,
            nrow=8,
        )
        image_paths["qualitative/endpoint_samples"] = endpoint_path
        for index, time_value in enumerate(args.times):
            sample_path = output_dir / f"samples_t{time_label(time_value)}.png"
            save_image(
                grid_path[index].clamp(-1, 1) * 0.5 + 0.5,
                sample_path,
                nrow=8,
            )
        trajectory_path = output_dir / "sample_trajectories.png"
        trajectory_count = min(8, grid_path.shape[1])
        trajectory_images = grid_path[:, :trajectory_count].transpose(0, 1)
        save_image(
            trajectory_images.flatten(0, 1).clamp(-1, 1) * 0.5 + 0.5,
            trajectory_path,
            nrow=len(args.times),
        )
        image_paths["qualitative/sample_trajectories"] = trajectory_path

        real_features, _ = classifier_outputs(
            classifier,
            real_images,
            args.batch_size,
            device,
        )
        direct_quality, fake_features, fake_probabilities = sample_quality_metrics(
            classifier,
            generated_images,
            real_features,
            real_labels,
            args.batch_size,
            device,
            args.kid_subsets,
            args.kid_subset_size,
            args.precision_recall_k,
            args.metric_chunk_size,
            args.seed,
        )
        metrics.update(
            {f"sample/{key}": value for key, value in direct_quality.items()}
        )
        metrics.update(
            {
                "protocol/classifier_test_accuracy": classifier_checkpoint[
                    "test_accuracy"
                ],
                "protocol/feature_dimension": real_features.shape[1],
                "protocol/kid_subsets": args.kid_subsets,
                "protocol/kid_subset_size": min(
                    args.kid_subset_size,
                    len(real_features),
                    len(fake_features),
                ),
                "protocol/precision_recall_k": args.precision_recall_k,
            }
        )
        print(
            f"direct samples: FID-MNIST={direct_quality['fid_mnist']:.4f}, "
            f"KID-MNIST={direct_quality['kid_mnist_mean']:.6f}±"
            f"{direct_quality['kid_mnist_std']:.6f}, "
            f"precision={direct_quality['precision_mnist']:.4f}, "
            f"recall={direct_quality['recall_mnist']:.4f}, "
            f"IS-MNIST={direct_quality['inception_score_mnist_mean']:.4f}"
        )

        if model_type == "distill":
            ancestral_generator = torch.Generator(device=device).manual_seed(args.seed)
            ancestral_path = (
                model.sample_path(
                    min(args.num_grid_samples, args.num_generated),
                    times,
                    generator=ancestral_generator,
                )
                .reshape(
                    len(times),
                    min(args.num_grid_samples, args.num_generated),
                    *MNIST_SHAPE,
                )
                .cpu()
            )
            ancestral_trajectory_path = (
                output_dir / "sample_trajectories_ancestral_path.png"
            )
            ancestral_count = min(8, ancestral_path.shape[1])
            save_image(
                ancestral_path[:, :ancestral_count]
                .transpose(0, 1)
                .flatten(0, 1)
                .clamp(-1, 1)
                * 0.5
                + 0.5,
                ancestral_trajectory_path,
                nrow=len(args.times),
            )
            image_paths["qualitative/sample_trajectories_ancestral"] = (
                ancestral_trajectory_path
            )

            for method in args.pc_sampling_methods:
                if method == "direct":
                    continue
                method_label = sampling_label(method, args.pc_ode_steps)
                continuous_method_images, method_elapsed, method_nfe = (
                    sample_pc_velocity_endpoint(
                        model,
                        method,
                        args.num_generated,
                        args.batch_size,
                        device,
                        args.seed,
                        args.pc_ode_steps,
                        args.ode_rtol,
                        args.ode_atol,
                    )
                )
                method_images = quantize_mnist(continuous_method_images)
                method_quality, method_features, method_probabilities = (
                    sample_quality_metrics(
                        classifier,
                        method_images,
                        real_features,
                        real_labels,
                        args.batch_size,
                        device,
                        args.kid_subsets,
                        args.kid_subset_size,
                        args.precision_recall_k,
                        args.metric_chunk_size,
                        args.seed,
                    )
                )
                method_prefix = f"sample_{method_label}"
                metrics.update(
                    {
                        f"{method_prefix}/{key}": value
                        for key, value in method_quality.items()
                    }
                )
                metrics[f"performance/{method_label}_images_per_second"] = (
                    len(method_images) / method_elapsed
                )
                metrics[f"performance/{method_label}_total_nfe"] = method_nfe
                metrics[f"performance/{method_label}_mean_nfe_per_batch"] = (
                    method_nfe / math.ceil(len(method_images) / args.batch_size)
                )
                method_image_path = output_dir / f"samples_endpoint_{method_label}.png"
                save_image(
                    method_images[: args.num_grid_samples].clamp(-1, 1) * 0.5 + 0.5,
                    method_image_path,
                    nrow=8,
                )
                image_paths[f"qualitative/endpoint_samples_{method_label}"] = (
                    method_image_path
                )
                torch.save(
                    {
                        "generated_images": method_images,
                        "generated_features": method_features,
                        "generated_class_probabilities": method_probabilities,
                    },
                    output_dir / f"samples_and_features_{method_label}.pt",
                )
                print(
                    f"{method_label} samples: "
                    f"FID-MNIST={method_quality['fid_mnist']:.4f}, "
                    f"KID-MNIST={method_quality['kid_mnist_mean']:.6f}±"
                    f"{method_quality['kid_mnist_std']:.6f}, "
                    f"precision={method_quality['precision_mnist']:.4f}, "
                    f"recall={method_quality['recall_mnist']:.4f}, "
                    f"images/s={len(method_images) / method_elapsed:.1f}, "
                    f"NFE/batch={method_nfe / math.ceil(len(method_images) / args.batch_size):.1f}"
                )

        neighbor_count = min(args.num_nearest_neighbors, len(generated_images))
        neighbor_images = nearest_training_images(
            classifier,
            generated_images[:neighbor_count],
            fake_features[:neighbor_count],
            data_dir,
            args.batch_size,
            args.num_workers,
            device,
        )
        neighbor_path = output_dir / "generated_and_nearest_train.png"
        save_image(
            neighbor_images.clamp(-1, 1) * 0.5 + 0.5,
            neighbor_path,
            nrow=2,
        )
        image_paths["qualitative/generated_and_nearest_train"] = neighbor_path
        torch.save(
            {
                "generated_images": generated_images,
                "real_features": real_features,
                "generated_features": fake_features,
                "generated_class_probabilities": fake_probabilities,
            },
            output_dir / "samples_and_features.pt",
        )

    if device.type == "cuda":
        metrics["performance/eval_peak_gpu_memory_bytes"] = (
            torch.cuda.max_memory_allocated(device)
        )
    metrics["protocol/model_type"] = model_type
    metrics["protocol/checkpoint"] = str(checkpoint_path)
    metrics["protocol/feature_space"] = (
        "official-pytorch-mnist-classifier-fc1-128"
        if not args.skip_sample_metrics
        else "not-computed"
    )
    metrics["protocol/sampling"] = (
        "direct-t1-pc-ancestral-then-8bit-quantize"
        if model_type == "distill"
        else f"{args.ode_method}-ode-then-8bit-quantize"
    )
    metrics["protocol/pc_trajectory_solver"] = (
        f"rk4-{args.pc_ode_steps}-steps"
        if model_type == "distill"
        else "not-applicable"
    )
    metrics["protocol/pc_sampling_methods"] = (
        args.pc_sampling_methods if model_type == "distill" else []
    )
    metrics["protocol/ode_method"] = args.ode_method
    metrics["protocol/ode_rtol"] = args.ode_rtol
    metrics["protocol/ode_atol"] = args.ode_atol
    metrics["protocol/pc_fixed_steps"] = (
        args.pc_ode_steps if model_type == "distill" else 0
    )
    metrics["protocol/ancestral_path_endpoint_law"] = (
        "identical-to-direct-p1-marginal"
        if model_type == "distill"
        else "not-applicable"
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    model_name = (
        "teacher"
        if model_type == "flow"
        else "-".join(
            filter(
                None,
                (
                    checkpoint["model_config"]["name"],
                    checkpoint["model_config"].get("sum_layer"),
                ),
            )
        )
    )
    run_name = args.wandb_run_name or (
        f"eval-{model_name}-step{checkpoint.get('step', 0)}-seed{args.seed}"
    )
    wandb_data_dir = output_dir / ".wandb-data"
    wandb_data_dir.mkdir(exist_ok=True)
    os.environ.setdefault("WANDB_DATA_DIR", str(wandb_data_dir))
    with wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        group="standard-evaluation",
        job_type=f"{model_type}-eval",
        tags=("mnist", "evaluation", "standard-metrics", model_name),
        config={
            **vars(args),
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint_path),
            "classifier_checkpoint": str(args.classifier_checkpoint.resolve()),
            "data_dir": str(data_dir),
            "model_config": checkpoint["model_config"],
        },
        mode=args.wandb_mode,
        dir=str(output_dir),
    ) as run:
        scalar_metrics = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (float, int))
        }
        run.log(scalar_metrics)
        for key, path in image_paths.items():
            run.log({key: wandb.Image(str(path))})
        if generated_images is not None:
            distribution = metrics["sample/class_distribution"]
            run.log(
                {
                    "sample/class_distribution_table": wandb.Table(
                        columns=["digit", "fraction"],
                        data=[
                            [digit, fraction]
                            for digit, fraction in enumerate(distribution)
                        ],
                    )
                }
            )
        artifact = wandb.Artifact(
            f"flow2pc-eval-{model_name}-step{checkpoint.get('step', 0)}-seed{args.seed}",
            type="evaluation",
            metadata=scalar_metrics,
        )
        artifact.add_file(str(metrics_path))
        for path in image_paths.values():
            artifact.add_file(str(path))
        run.log_artifact(artifact)
    print(f"evaluation artifacts: {output_dir}")


if __name__ == "__main__":
    main()
