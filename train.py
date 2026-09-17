import argparse
import csv
import json
import math
import os
import time
import sys
from collections import deque
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

# Ensure local repository root takes precedence over any installed site-packages
_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path or sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)

import torch
import wandb
from torch import nn
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)
from torchvision.datasets import MNIST

from repc.data import MNIST_DIM, mnist_loader, multimarginal_loader
from repc.flow import build_teacher, torchcfm_mnist_teacher_config
from repc.hclt import (
    HCLT_MNIST_CONFIG,
    load_or_learn_hclt_topology,
    load_or_learn_iclr2022_hclt_topology,
)
from repc.hclt_pc import TimeDependentHCLTPC
from repc.pc import TimeDependentGaussianPC
from repc.region_pc import RegionGraphPC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("flow", "distill"))
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument(
        "--pc-structure",
        "--student-model",
        dest="pc_structure",
        choices=("hclt", "pd", "gaussian"),
        default="hclt",
        help="student circuit backbone (HCLT is the literature-standard default)",
    )
    parser.add_argument(
        "--pc-layer",
        choices=("dense", "monarch"),
        default="dense",
        help="HCLT latent transition parameterization",
    )
    parser.add_argument(
        "--capacity-profile",
        choices=("standard", "published"),
        default="standard",
        help="resolve a predeclared student width unless one is given explicitly",
    )
    parser.add_argument(
        "--student",
        dest="legacy_student",
        choices=("gaussian", "region"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--flow-matcher",
        choices=("vanilla", "ot"),
        default="ot",
        help="teacher probability path; distillation inherits it from the checkpoint",
    )
    parser.add_argument(
        "--dataset",
        choices=("mnist", "knot", "ht206b1"),
        default="mnist",
        help="dataset for training or distillation",
    )
    parser.add_argument(
        "--teacher-arch",
        choices=("torchcfm", "alicfm"),
        default="torchcfm",
        help="teacher model architecture family",
    )
    parser.add_argument(
        "--num-marginals",
        type=int,
        default=60,
        help="number of temporal marginal snapshots for multi-marginal datasets",
    )
    parser.add_argument(
        "--fourier-zero-center",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="forces t=0 to N(0, I) by centering the cosine term in Fourier PC embeddings",
    )
    parser.add_argument(
        "--polynomial-zero-center",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forces t=0 to N(0, I) by omitting the t^0 basis term in standard PCs.",
    )
    parser.add_argument(
        "--multimarginal-time-sampling",
        choices=("uniform_global", "uniform_interval"),
        default="uniform_interval",
        help="timestep sampling mode for multi-marginal snapshot distillation",
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float)
    parser.add_argument(
        "--optimizer",
        choices=("auto", "adam", "adamw"),
        default="auto",
    )
    parser.add_argument(
        "--scheduler",
        choices=("auto", "constant", "cosine"),
        default="auto",
    )
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5000,
        help="save numbered student checkpoints at this step interval",
    )
    parser.add_argument("--num-components", type=int)
    parser.add_argument("--num-channels", type=int)
    parser.add_argument("--num-latents", type=int)
    parser.add_argument("--time-order", type=int, default=4)
    parser.add_argument(
        "--time-conditioning",
        choices=("polynomial", "fourier"),
        default="polynomial",
        help="HCLT leaf-path time basis",
    )
    parser.add_argument(
        "--time-embedding-dim",
        type=int,
        default=16,
        help="even output dimension of the fixed Fourier feature layer",
    )
    parser.add_argument(
        "--hclt-leaf-distribution",
        choices=("gaussian", "gaussian-mixture"),
        default="gaussian",
        help="normalized univariate distribution used by each HCLT emission",
    )
    parser.add_argument(
        "--hclt-leaf-components",
        type=int,
        default=1,
        help="number of fixed-weight path components in each HCLT leaf",
    )
    parser.add_argument(
        "--endpoint-pretraining",
        choices=("none", "em"),
        default="none",
        help="optional t=1 PC pretraining stage before velocity distillation",
    )
    parser.add_argument(
        "--density-weight",
        type=float,
        default=0.0,
        help="Weight for negative log-likelihood density loss at t=1 (0 = disabled).",
    )
    parser.add_argument(
        "--endpoint-em-epochs",
        type=int,
        default=20,
        help="number of full-batch soft-EM epochs at t=1",
    )
    parser.add_argument(
        "--endpoint-em-pseudocount",
        type=float,
        default=0.1,
        help="Laplace pseudocount for endpoint EM updates",
    )
    parser.add_argument(
        "--endpoint-em-min-std",
        type=float,
        default=0.01,
        help="minimum Gaussian standard deviation during endpoint EM",
    )
    parser.add_argument(
        "--timestep-sampling",
        choices=("uniform", "beta"),
        default="uniform",
        help="distribution used to sample distillation timesteps",
    )
    parser.add_argument(
        "--timestep-beta-alpha",
        type=float,
        default=1.0,
        help="alpha concentration for --timestep-sampling beta",
    )
    parser.add_argument(
        "--timestep-beta-beta",
        type=float,
        default=1.0,
        help="beta concentration for --timestep-sampling beta",
    )
    parser.add_argument(
        "--hclt-topology",
        type=Path,
        help="cache file for the learned Chow--Liu tree",
    )
    parser.add_argument(
        "--hclt-topology-protocol",
        choices=("pyjuice-kde", "iclr2022-categorical"),
        default="pyjuice-kde",
        help="published estimator used to learn the HCLT backbone",
    )
    parser.add_argument(
        "--hclt-mi-sample-chunk-size",
        type=int,
        default=1_024,
        help="sample chunk size for exact categorical mutual information",
    )
    parser.add_argument(
        "--hclt-samples",
        type=int,
        default=HCLT_MNIST_CONFIG["num_samples"],
    )
    parser.add_argument(
        "--hclt-num-bins",
        type=int,
        default=HCLT_MNIST_CONFIG["num_bins"],
    )
    parser.add_argument(
        "--hclt-sigma",
        type=float,
        default=HCLT_MNIST_CONFIG["sigma"],
    )
    parser.add_argument(
        "--hclt-chunk-size",
        type=int,
        default=HCLT_MNIST_CONFIG["chunk_size"],
    )
    parser.add_argument("--wandb-project", default="flow2pc")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-group")
    parser.add_argument(
        "--wandb-tags",
        default="",
        help="comma-separated extra tags",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument(
        "--validation-size",
        type=int,
        default=0,
        help="fixed held-out subset of MNIST training images used for tuning",
    )
    parser.add_argument(
        "--validate-every",
        type=int,
        default=1000,
        help="validation interval in optimizer steps",
    )
    args = parser.parse_args()
    if args.legacy_student is not None:
        args.pc_structure = {
            "gaussian": "gaussian",
            "region": "pd",
        }[args.legacy_student]
    if args.task == "distill" and args.teacher_checkpoint is None:
        parser.error("--teacher-checkpoint is required for distillation")
    if args.task == "flow":
        if args.steps is not None and args.epochs is not None:
            parser.error("use either --epochs or --steps for flow training, not both")
        if args.steps is None and args.epochs is None:
            args.epochs = 3
    else:
        if args.epochs is not None:
            parser.error("--epochs is only supported for flow training")
        if args.steps is None:
            args.steps = 20_000
        if args.pc_structure != "hclt" and args.pc_layer != "dense":
            parser.error("--pc-layer only applies to --pc-structure hclt")
        if args.pc_structure == "hclt" and args.num_latents is None:
            args.num_latents = {
                ("standard", "dense"): 64,
                ("standard", "monarch"): 64,
                ("published", "dense"): 256,
                ("published", "monarch"): 1024,
            }[(args.capacity_profile, args.pc_layer)]
        if args.pc_structure == "pd" and args.num_channels is None:
            args.num_channels = {
                "standard": 8,
                "published": 64,
            }[args.capacity_profile]
        if args.pc_structure == "gaussian" and args.num_components is None:
            args.num_components = {
                "standard": 256,
                "published": 256,
            }[args.capacity_profile]
        if (
            args.pc_structure == "hclt"
            and args.pc_layer == "monarch"
            and math.isqrt(args.num_latents) ** 2 != args.num_latents
        ):
            parser.error("Monarch HCLT requires --num-latents to be a square")
    counts = (
        args.steps,
        args.epochs,
        args.batch_size,
        args.log_every,
        args.checkpoint_every,
        args.hclt_samples,
        args.hclt_num_bins,
        args.hclt_chunk_size,
        args.hclt_mi_sample_chunk_size,
        args.validate_every,
        args.endpoint_em_epochs,
    )
    if any(value is not None and value < 1 for value in counts):
        parser.error("steps, epochs, batch size, and log interval must be positive")
    if args.lr is not None and args.lr <= 0:
        parser.error("learning rate must be positive")
    if args.weight_decay is not None and args.weight_decay < 0:
        parser.error("weight decay must be non-negative")
    if args.gradient_clip <= 0:
        parser.error("gradient clipping threshold must be positive")
    if args.endpoint_em_pseudocount < 0:
        parser.error("endpoint EM pseudocount must be non-negative")
    if args.endpoint_em_min_std <= 0:
        parser.error("endpoint EM minimum standard deviation must be positive")
    if args.hclt_sigma <= 0:
        parser.error("HCLT sigma must be positive")
    if args.validation_size < 0 or args.validation_size >= 60_000:
        parser.error("validation size must be in [0, 60000)")
    if args.time_embedding_dim < 2 or args.time_embedding_dim % 2:
        parser.error("time embedding dimension must be a positive even integer")
    if args.task == "flow" and args.validation_size:
        parser.error("validation monitoring is currently supported for distillation")
    if args.task == "flow" and args.timestep_sampling != "uniform":
        parser.error(
            "non-uniform timestep sampling is currently supported for distillation"
        )
    if args.task == "flow" and args.endpoint_pretraining != "none":
        parser.error("endpoint pretraining is only supported for distillation")
    if args.timestep_beta_alpha <= 0 or args.timestep_beta_beta <= 0:
        parser.error("Beta timestep concentrations must be positive")
    if args.pc_structure != "hclt" and args.time_conditioning != "polynomial":
        parser.error("--time-conditioning currently applies to HCLT students")
    if args.pc_structure != "hclt" and (
        args.hclt_leaf_distribution != "gaussian" or args.hclt_leaf_components != 1
    ):
        parser.error("HCLT leaf options require --pc-structure hclt")
    if args.hclt_leaf_distribution == "gaussian" and args.hclt_leaf_components != 1:
        parser.error("Gaussian HCLT leaves require exactly one component")
    if (
        args.hclt_leaf_distribution == "gaussian-mixture"
        and args.hclt_leaf_components < 2
    ):
        parser.error("Gaussian-mixture HCLT leaves require at least two components")
    if args.endpoint_pretraining == "em" and (
        args.pc_structure != "hclt"
        or args.pc_layer != "dense"
        or args.hclt_leaf_distribution != "gaussian"
    ):
        parser.error("endpoint EM requires a dense HCLT with Gaussian leaves")
    selected_widths = (
        args.num_latents,
        args.num_channels,
        args.num_components,
        args.hclt_leaf_components,
    )
    if any(value is not None and value < 1 for value in selected_widths):
        parser.error("student widths and component counts must be positive")
    return args


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def build_student(config: dict) -> nn.Module:
    if config["name"] == "gaussian":
        return TimeDependentGaussianPC(
            num_components=config["num_components"],
            dim=config["dim"],
            time_order=config["time_order"],
            polynomial_zero_center=config.get("polynomial_zero_center", True),
        )
    if config["name"] in ("pd", "region"):
        return RegionGraphPC(
            height=config["height"],
            width=config["width"],
            num_channels=config["num_channels"],
            time_order=config["time_order"],
            polynomial_zero_center=config.get("polynomial_zero_center", True),
        )
    if config["name"] == "hclt":
        return TimeDependentHCLTPC(
            parents=config["parents"],
            num_latents=config["num_latents"],
            time_order=config["time_order"],
            sum_layer=config["sum_layer"],
            time_conditioning=config.get("time_conditioning", "polynomial"),
            time_embedding_dim=config.get("time_embedding_dim", 16),
            time_feature_seed=config.get("time_feature_seed", 0),
            leaf_distribution=config.get("leaf_distribution", "gaussian"),
            leaf_components=config.get("leaf_components", 1),
            num_categories=config.get("num_categories", 256),
            fourier_zero_center=config.get("fourier_zero_center", True),
        )
    raise ValueError(f"unknown student: {config['name']}")


def build_flow_matcher(config: dict) -> ConditionalFlowMatcher:
    if config.get("name") == "vanilla_cfm":
        return ConditionalFlowMatcher(sigma=config["sigma"])
    if config.get("name") == "exact_ot_cfm":
        return ExactOptimalTransportConditionalFlowMatcher(sigma=config["sigma"])
    raise ValueError(f"unknown flow matcher: {config.get('name')}")


def pair_flow_endpoints(
    flow_matcher: ConditionalFlowMatcher,
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(flow_matcher, ExactOptimalTransportConditionalFlowMatcher):
        return flow_matcher.ot_sampler.sample_plan(source, target)
    return source, target


def load_teacher(
    path: Path,
    device: torch.device,
    teacher_arch: str = "torchcfm",
) -> tuple[nn.Module, dict]:
    if teacher_arch == "alicfm":
        teacher = build_teacher({"name": "alicfm", "checkpoint_path": path}).to(device)
        flow_matching_config = {"name": "vanilla_cfm", "sigma": 0.0}
        return teacher.eval(), flow_matching_config
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and checkpoint.get("model_config", {}).get("name") == "alicfm":
        teacher = build_teacher(checkpoint["model_config"]).to(device)
        flow_matching_config = checkpoint.get("flow_matching_config", {"name": "vanilla_cfm", "sigma": 0.0})
        return teacher.eval(), flow_matching_config
    if checkpoint["model_type"] != "flow":
        raise ValueError(f"{path} is not a flow checkpoint")
    teacher = build_teacher(checkpoint["model_config"]).to(device)
    teacher.load_state_dict(checkpoint["state_dict"])
    return teacher.eval(), checkpoint["flow_matching_config"]


def flow_matching_loss(
    model: nn.Module,
    flow_matcher: ConditionalFlowMatcher,
    data: torch.Tensor,
    next_data: torch.Tensor | None = None,
) -> torch.Tensor:
    if next_data is not None:
        noise = data
        data = next_data
    else:
        noise = torch.randn_like(data)
    time, path_sample, target_velocity = (
        flow_matcher.sample_location_and_conditional_flow(noise, data)
    )
    return nn.functional.mse_loss(model(time, path_sample), target_velocity)


def sample_distillation_timesteps(
    count: int,
    device: torch.device,
    dtype: torch.dtype,
    sampling: str = "uniform",
    beta_alpha: float = 1.0,
    beta_beta: float = 1.0,
    k_indices: torch.Tensor | None = None,
    K_total: int | None = None,
) -> torch.Tensor:
    """Sample path times; t=0 is noise and t=1 is the MNIST data endpoint."""
    if sampling in ("multimarginal_interval", "uniform_interval") and k_indices is not None:
        if K_total is None or K_total < 2:
            raise ValueError("K_total must be at least 2 for interval time sampling")
        k_float = k_indices.to(device=device, dtype=dtype)
        k_clamped = torch.clamp(k_float, 0, K_total - 2)
        t_lower = k_clamped / (K_total - 1)
        t_upper = (k_clamped + 1.0) / (K_total - 1)
        return t_lower + torch.rand(count, device=device, dtype=dtype) * (t_upper - t_lower)
    if sampling in ("uniform", "uniform_global"):
        return torch.rand(count, device=device, dtype=dtype)
    if sampling == "beta":
        concentration1 = torch.tensor(beta_alpha, device=device, dtype=dtype)
        concentration0 = torch.tensor(beta_beta, device=device, dtype=dtype)
        times = torch.distributions.Beta(concentration1, concentration0).sample(
            (count,)
        )
        epsilon = torch.finfo(dtype).eps
        return times.clamp(epsilon, 1.0 - epsilon)
    raise ValueError(f"unknown timestep sampling distribution: {sampling}")


def distillation_loss(
    pc: nn.Module,
    teacher: nn.Module,
    flow_matcher: ConditionalFlowMatcher,
    data: torch.Tensor,
    next_data: torch.Tensor | None = None,
    timestep_sampling: str = "uniform",
    timestep_beta_alpha: float = 1.0,
    timestep_beta_beta: float = 1.0,
    return_timesteps: bool = False,
    k_indices: torch.Tensor | None = None,
    K_total: int | None = None,
    density_weight: float = 0.0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict]:
    if next_data is not None:
        noise = data
        data = next_data
        tau_sampling = (
            "uniform"
            if timestep_sampling in ("multimarginal_interval", "uniform_interval", "uniform_global")
            else timestep_sampling
        )
        times = sample_distillation_timesteps(
            data.shape[0],
            data.device,
            data.dtype,
            tau_sampling,
            timestep_beta_alpha,
            timestep_beta_beta,
            k_indices=None,
            K_total=None,
        )
        _, path_sample, _ = flow_matcher.sample_location_and_conditional_flow(
            noise,
            data,
            times,
        )
        
        global_times = (k_indices.to(data.dtype) + times) / (K_total - 1)

        with torch.no_grad():
            teacher_velocity = teacher(global_times, path_sample)
        pc_velocity = pc.velocity(path_sample.flatten(1), global_times)
        loss = nn.functional.mse_loss(pc_velocity, teacher_velocity.flatten(1))
        
        metrics = {"distill_mse": loss.item()}
        if density_weight > 0.0:
            density_loss = -pc.log_prob(
                data.flatten(1), torch.tensor(1.0, device=data.device)
            ).mean()
            metrics["density_loss"] = density_loss.item()
            loss = loss + density_weight * density_loss
        if return_timesteps:
            return loss, global_times.detach(), metrics
        return loss

    noise = torch.randn_like(data)
    times = sample_distillation_timesteps(
        data.shape[0],
        data.device,
        data.dtype,
        timestep_sampling,
        timestep_beta_alpha,
        timestep_beta_beta,
        k_indices=k_indices,
        K_total=K_total,
    )
    _, path_sample, _ = flow_matcher.sample_location_and_conditional_flow(
        noise,
        data,
        times,
    )
    with torch.no_grad():
        teacher_velocity = teacher(times, path_sample)
    pc_velocity = pc.velocity(path_sample.flatten(1), times)
    loss = nn.functional.mse_loss(pc_velocity, teacher_velocity.flatten(1))
    metrics = {"distill_mse": loss.item()}
    if density_weight > 0.0:
        density_loss = -pc.log_prob(
            data.flatten(1), torch.tensor(1.0, device=data.device)
        ).mean()
        metrics["density_loss"] = density_loss.item()
        loss = loss + density_weight * density_loss
    if return_timesteps:
        return loss, times.detach(), metrics
    return loss


def distillation_validation_metrics(
    pc: nn.Module,
    teacher: nn.Module,
    flow_matcher: ConditionalFlowMatcher,
    loader,
    device: torch.device,
    seed: int,
) -> tuple[float, float]:
    """Evaluate a deterministic held-out velocity MSE and endpoint bpd."""
    was_training = pc.training
    pc.eval()
    generator = torch.Generator(device=device).manual_seed(seed + 1_000_003)
    squared_error = 0.0
    velocity_values = 0
    endpoint_log_prob = 0.0
    examples = 0
    endpoint_time = torch.tensor(1.0, device=device)
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        noise = torch.randn(
            images.shape,
            device=device,
            dtype=images.dtype,
            generator=generator,
        )
        times = torch.rand(
            images.shape[0],
            device=device,
            generator=generator,
        )
        _, path_sample, _ = flow_matcher.sample_location_and_conditional_flow(
            noise,
            images,
            times,
        )
        with torch.no_grad():
            teacher_velocity = teacher(times, path_sample).flatten(1)
        pc_velocity = pc.velocity(path_sample.flatten(1), times)
        squared_error += (pc_velocity.detach() - teacher_velocity).square().sum().item()
        velocity_values += teacher_velocity.numel()
        with torch.no_grad():
            endpoint_log_prob += (
                pc.discrete_log_prob(
                    images.flatten(1),
                    endpoint_time,
                )
                .sum()
                .item()
            )
        examples += len(images)
    pc.train(was_training)
    mse = squared_error / velocity_values
    bpd = -endpoint_log_prob / (examples * MNIST_DIM * math.log(2.0))
    return mse, bpd


@torch.no_grad()
def endpoint_discrete_bpd(
    pc: TimeDependentHCLTPC,
    loader,
    device: torch.device,
) -> float:
    """Evaluate exact discretized likelihood at the MNIST data endpoint."""
    endpoint = torch.tensor(1.0, device=device)
    log_probability = 0.0
    examples = 0
    was_training = pc.training
    pc.eval()
    for images, _ in loader:
        images = images.to(device, non_blocking=True).flatten(1)
        log_probability += pc.discrete_log_prob(images, endpoint).sum().item()
        examples += images.shape[0]
    pc.train(was_training)
    return -log_probability / (examples * MNIST_DIM * math.log(2.0))


def endpoint_em_pretraining(
    pc: TimeDependentHCLTPC,
    loader,
    validation_loader,
    device: torch.device,
    epochs: int,
    pseudocount: float,
    min_std: float,
    run_dir: Path,
    tracking_run,
    checkpoint_metadata: dict,
) -> None:
    """Fit the HCLT endpoint with exact full-batch soft EM."""
    metrics_path = run_dir / "endpoint_em_metrics.csv"
    checkpoint_path = run_dir / "checkpoints" / "endpoint_em.pt"
    started_at = time.perf_counter()
    tracking_run.define_metric("endpoint_em/epoch")
    tracking_run.define_metric(
        "endpoint_em/*",
        step_metric="endpoint_em/epoch",
    )
    with metrics_path.open("w", newline="") as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow(
            (
                "epoch",
                "train_continuous_nll_nats",
                "train_continuous_nll_nats_per_dim",
                "validation_endpoint_bpd",
                "samples_per_second",
                "elapsed_seconds",
            )
        )
        for epoch in range(1, epochs + 1):
            aggregate = None
            epoch_log_likelihood = 0.0
            epoch_examples = 0
            epoch_started_at = time.perf_counter()
            pc.eval()
            for images, _ in loader:
                images = images.to(device, non_blocking=True).flatten(1)
                statistics = pc.endpoint_em_statistics(images)
                epoch_log_likelihood += statistics["log_likelihood"].item()
                epoch_examples += images.shape[0]
                if aggregate is None:
                    aggregate = {
                        key: value.detach().clone()
                        for key, value in statistics.items()
                        if key not in ("log_likelihood", "num_examples")
                    }
                else:
                    for key in aggregate:
                        aggregate[key].add_(statistics[key])
            pc.endpoint_em_update(
                aggregate,
                pseudocount=pseudocount,
                min_std=min_std,
            )
            validation_bpd = endpoint_discrete_bpd(
                pc,
                validation_loader if validation_loader is not None else loader,
                device,
            )
            epoch_elapsed = time.perf_counter() - epoch_started_at
            elapsed = time.perf_counter() - started_at
            continuous_nll = -epoch_log_likelihood / epoch_examples
            samples_per_second = epoch_examples / epoch_elapsed
            tracking_run.log(
                {
                    "endpoint_em/epoch": epoch,
                    "endpoint_em/train_continuous_nll_nats": continuous_nll,
                    "endpoint_em/train_continuous_nll_nats_per_dim": (
                        continuous_nll / MNIST_DIM
                    ),
                    "endpoint_em/validation_endpoint_bpd": validation_bpd,
                    "endpoint_em/samples_per_second": samples_per_second,
                    "endpoint_em/elapsed_seconds": elapsed,
                }
            )
            writer.writerow(
                (
                    epoch,
                    continuous_nll,
                    continuous_nll / MNIST_DIM,
                    validation_bpd,
                    samples_per_second,
                    elapsed,
                )
            )
            metrics_file.flush()
            print(
                f"endpoint EM epoch {epoch}/{epochs}: "
                f"continuous_nll={continuous_nll:.4f} "
                f"validation_bpd={validation_bpd:.4f} "
                f"samples/s={samples_per_second:.1f}",
                flush=True,
            )

    checkpoint = {
        **checkpoint_metadata,
        "state_dict": pc.state_dict(),
        "step": 0,
        "training_stage": "endpoint_em",
        "endpoint_em_epochs": epochs,
    }
    torch.save(checkpoint, checkpoint_path)
    tracking_run.summary["endpoint_em/final_validation_bpd"] = validation_bpd
    tracking_run.summary["endpoint_em/total_seconds"] = time.perf_counter() - started_at


def serializable_config(args: argparse.Namespace) -> dict:
    config = vars(args).copy()
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value.resolve())
    return config


def run_directory(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S-%f")
    return Path("outputs") / "runs" / args.task / timestamp


def gradient_norm(parameters) -> float:
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            norm = parameter.grad.detach().float().norm(2).item()
            squared_norm += norm * norm
    return squared_norm**0.5


def wandb_tags(args: argparse.Namespace) -> list[str]:
    # Dynamically pull the dataset and teacher, falling back to original defaults
    dataset = getattr(args, "dataset", "mnist")
    teacher = getattr(args, "teacher_arch", args.flow_matcher)

    automatic_tags = [dataset, teacher]

    if args.task == "flow":
        automatic_tags.extend(("teacher", teacher))
    else:
        # Dynamically pull the sampling strategy
        sampling = getattr(args, "multimarginal_time_sampling", args.timestep_sampling)

        automatic_tags.extend(
            (
                "student",
                args.pc_structure,
                args.pc_layer,
                args.capacity_profile,
                f"endpoint-{args.endpoint_pretraining}",
                f"timesteps-{sampling}",
            )
        )

    extra_tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    return list(dict.fromkeys(automatic_tags + extra_tags))


def artifact_name(args: argparse.Namespace) -> str:
    dataset = getattr(args, "dataset", "mnist")
    if args.task == "flow":
        return f"{dataset}-flow-teacher-{args.flow_matcher}"
    return f"{dataset}-pc-student-{args.pc_structure}-{args.pc_layer}-{args.flow_matcher}"


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    run_dir = run_directory(args)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "mnist":
        loader = mnist_loader(
            args.data_dir,
            train=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            validation_size=args.validation_size,
            split_seed=args.seed,
        )
    else:
        loader = multimarginal_loader(
            args.dataset,
            args.batch_size,
            K=args.num_marginals,
            num_workers=args.num_workers,
            data_dir=args.data_dir,
            seed=args.seed,
        )
    validation_loader = None
    if args.dataset == "mnist" and args.task == "distill" and args.validation_size:
        validation_loader = mnist_loader(
            args.data_dir,
            train=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            validation=True,
            validation_size=args.validation_size,
            split_seed=args.seed,
        )
    if args.task == "flow":
        model_config = torchcfm_mnist_teacher_config()
        matcher_name = {
            "vanilla": "vanilla_cfm",
            "ot": "exact_ot_cfm",
        }[args.flow_matcher]
        flow_matching_config = {"name": matcher_name, "sigma": 0.0}
        flow_matcher = build_flow_matcher(flow_matching_config)
        model = build_teacher(model_config).to(device)
        loss_function = partial(flow_matching_loss, model, flow_matcher)
        total_steps = args.steps or args.epochs * len(loader)
        lr = args.lr or 1e-3
    else:
        teacher, flow_matching_config = load_teacher(
            args.teacher_checkpoint,
            device,
            teacher_arch=args.teacher_arch,
        )
        if flow_matching_config["name"] in ("vanilla_cfm", "exact_ot_cfm"):
            args.flow_matcher = {
                "vanilla_cfm": "vanilla",
                "exact_ot_cfm": "ot",
            }[flow_matching_config["name"]]
        flow_matcher = build_flow_matcher(flow_matching_config)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        if args.pc_structure == "gaussian":
            data_dim = 2 if args.dataset in ("knot", "ht206b1") else MNIST_DIM
            model_config = {
                "name": "gaussian",
                "num_components": args.num_components,
                "dim": data_dim,
                "time_order": args.time_order,
                "polynomial_zero_center": args.polynomial_zero_center,
            }
        elif args.pc_structure == "pd":
            model_config = {
                "name": "pd",
                "height": 28,
                "width": 28,
                "num_channels": args.num_channels,
                "time_order": args.time_order,
                "polynomial_zero_center": args.polynomial_zero_center,
            }
        elif args.pc_structure == "hclt":
            if args.dataset in ("knot", "ht206b1") and args.hclt_topology is None:
                parents = [-1, 0]
                resolved_topology_config = {"topology": "2d_chain"}
                topology_path = run_dir / "topology_2d.pt"
            else:
                topology_path = args.hclt_topology
                if args.hclt_topology_protocol == "iclr2022-categorical":
                    if topology_path is None:
                        topology_path = (
                            args.data_dir
                            / "MNIST"
                            / "hclt"
                            / "topology_iclr2022_msb3_n60000.pt"
                        )
                    raw_topology_data = MNIST(
                        root=args.data_dir,
                        train=True,
                        download=True,
                    ).data.flatten(1)
                    parents, resolved_topology_config = (
                        load_or_learn_iclr2022_hclt_topology(
                            raw_topology_data,
                            topology_path,
                            device,
                            sample_chunk_size=args.hclt_mi_sample_chunk_size,
                        )
                    )
                else:
                    topology_config = {
                        "num_samples": args.hclt_samples,
                        "num_bins": args.hclt_num_bins,
                        "sigma": args.hclt_sigma,
                        "chunk_size": args.hclt_chunk_size,
                    }
                    if args.validation_size:
                        topology_config["validation_size"] = args.validation_size
                    if topology_path is None:
                        sigma_label = str(args.hclt_sigma).replace(".", "p")
                        topology_path = (
                            args.data_dir
                            / "MNIST"
                            / "hclt"
                            / (
                                f"topology_x{args.hclt_samples}_"
                                f"b{args.hclt_num_bins}_s{sigma_label}_"
                                f"c{args.hclt_chunk_size}_seed{args.seed}.pt"
                            )
                        )
                    topology_loader = mnist_loader(
                        args.data_dir,
                        train=True,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        validation_size=args.validation_size,
                        split_seed=args.seed,
                    )
                    parents, resolved_topology_config = load_or_learn_hclt_topology(
                        topology_loader,
                        topology_path,
                        device,
                        topology_config,
                        args.seed,
                    )
            model_config = {
                "name": "hclt",
                "parents": parents,
                "num_latents": args.num_latents,
                "time_order": args.time_order,
                "sum_layer": args.pc_layer,
                "time_conditioning": args.time_conditioning,
                "time_embedding_dim": args.time_embedding_dim,
                "time_feature_seed": args.seed,
                "leaf_distribution": args.hclt_leaf_distribution,
                "leaf_components": args.hclt_leaf_components,
                "fourier_zero_center": args.fourier_zero_center,
                "topology_config": resolved_topology_config,
                "topology_path": str(topology_path.resolve()),
                "endpoint_pretraining": args.endpoint_pretraining,
            }
        model = build_student(model_config).to(device)
        timestep_sampling = (
            args.multimarginal_time_sampling
            if args.dataset != "mnist"
            else args.timestep_sampling
        )
        loss_function = partial(
            distillation_loss,
            model,
            teacher,
            flow_matcher,
            timestep_sampling=timestep_sampling,
            timestep_beta_alpha=args.timestep_beta_alpha,
            timestep_beta_beta=args.timestep_beta_beta,
            density_weight=args.density_weight,
            return_timesteps=True,
        )
        total_steps = args.steps
        lr = args.lr or 1e-3

    batches = iter(loader)

    optimizer_name = args.optimizer
    if optimizer_name == "auto":
        optimizer_name = "adam" if args.task == "flow" else "adamw"
    scheduler_name = args.scheduler
    if scheduler_name == "auto":
        scheduler_name = "constant" if args.task == "flow" else "cosine"
    weight_decay = args.weight_decay
    if weight_decay is None:
        weight_decay = 0.0 if optimizer_name == "adam" else 0.01

    config = serializable_config(args)
    config["resolved_steps"] = total_steps
    config["resolved_optimizer"] = optimizer_name
    config["resolved_scheduler"] = scheduler_name
    config["resolved_weight_decay"] = weight_decay
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    optimizer_class = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }[optimizer_name]
    optimizer = optimizer_class(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            total_steps,
        )
    else:
        scheduler = None
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    teacher_parameter_count = (
        parameter_count
        if args.task == "flow"
        else sum(parameter.numel() for parameter in teacher.parameters())
    )
    parameter_ratio = parameter_count / teacher_parameter_count
    config.update(
        {
            "parameter_count": parameter_count,
            "teacher_parameter_count": teacher_parameter_count,
            "student_to_teacher_parameter_ratio": parameter_ratio,
        }
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    tracking_config = {
        **config,
        "model_config": model_config,
        "flow_matching_config": flow_matching_config,
        "optimizer": optimizer.__class__.__name__,
        "parameter_count": parameter_count,
        "teacher_parameter_count": teacher_parameter_count,
        "student_to_teacher_parameter_ratio": parameter_ratio,
    }
    group = args.wandb_group or (
        "teacher-pretraining" if args.task == "flow" else "student-distillation"
    )
    job_type = "teacher-train" if args.task == "flow" else "student-distill"
    checkpoint_path = checkpoint_dir / "last.pt"
    metrics_path = run_dir / "metrics.csv"
    wandb_data_dir = run_dir / ".wandb-data"
    wandb_data_dir.mkdir(exist_ok=True)
    os.environ.setdefault("WANDB_DATA_DIR", str(wandb_data_dir))

    with wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        group=group,
        job_type=job_type,
        tags=wandb_tags(args),
        config=tracking_config,
        mode=args.wandb_mode,
        dir=str(run_dir),
    ) as tracking_run:
        tracking_run.define_metric("train/step")
        tracking_run.define_metric("train/*", step_metric="train/step")
        tracking_run.define_metric("train/loss", summary="min")
        tracking_run.define_metric("train/loss_mean_100", summary="min")
        tracking_run.define_metric("train/loss_mean_500", summary="min")
        tracking_run.define_metric("performance/*", step_metric="train/step")
        tracking_run.define_metric("validation/*", step_metric="train/step")
        tracking_run.summary["model/parameter_count"] = parameter_count
        tracking_run.summary["model/teacher_parameter_count"] = teacher_parameter_count
        tracking_run.summary["model/student_to_teacher_parameter_ratio"] = (
            parameter_ratio
        )
        tracking_run.log_code(root=str(Path.cwd()))

        if args.task == "distill" and args.endpoint_pretraining == "em":
            endpoint_loader = mnist_loader(
                args.data_dir,
                train=True,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                validation_size=args.validation_size,
                split_seed=args.seed,
            )
            endpoint_em_pretraining(
                model,
                endpoint_loader,
                validation_loader,
                device,
                args.endpoint_em_epochs,
                args.endpoint_em_pseudocount,
                args.endpoint_em_min_std,
                run_dir,
                tracking_run,
                {
                    "model_type": args.task,
                    "model_config": model_config,
                    "flow_matching_config": flow_matching_config,
                    "training_config": config,
                },
            )

        model.train()
        examples_seen = 0
        started_at = time.perf_counter()
        final_loss = None
        recent_losses = deque(maxlen=500)
        best_validation_bpd = float("inf")
        validation_metrics_file = None
        validation_writer = None
        if validation_loader is not None:
            validation_metrics_file = (run_dir / "validation_metrics.csv").open(
                "w", newline=""
            )
            validation_writer = csv.writer(validation_metrics_file)
            validation_writer.writerow(("step", "distill_mse", "endpoint_bpd"))
        with metrics_path.open("w", newline="") as metrics_file:
            writer = csv.writer(metrics_file)
            writer.writerow(
                (
                    "step",
                    "epoch",
                    "loss",
                    "learning_rate",
                    "gradient_norm",
                    "timestep_mean",
                    "timestep_std",
                    "timestep_fraction_t_le_0p2",
                    "timestep_fraction_t_ge_0p8",
                    "examples_seen",
                    "samples_per_second",
                    "steps_per_second",
                    "elapsed_seconds",
                )
            )
            is_multimarginal = args.dataset != "mnist"
            for step in range(1, total_steps + 1):
                try:
                    batch = next(batches)
                except StopIteration:
                    batches = iter(loader)
                    batch = next(batches)
                if is_multimarginal:
                    images, next_images, k_indices = batch
                    images = images.to(device, non_blocking=True)
                    next_images = next_images.to(device, non_blocking=True)
                    k_indices = k_indices.to(device, non_blocking=True)
                    if args.task == "distill":
                        loss_result = loss_function(
                            images,
                            next_data=next_images,
                            k_indices=k_indices,
                            K_total=args.num_marginals,
                        )
                    else:
                        loss_result = loss_function(images, next_data=next_images)
                else:
                    images, _ = batch
                    images = images.to(device, non_blocking=True)
                    loss_result = loss_function(images)
                sampled_timesteps = None
                batch_metrics = {}
                if isinstance(loss_result, tuple):
                    if len(loss_result) == 3:
                        loss, sampled_timesteps, batch_metrics = loss_result
                    else:
                        loss, sampled_timesteps = loss_result
                else:
                    loss = loss_result
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.task == "distill":
                    step_gradient_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            args.gradient_clip,
                        )
                    )
                else:
                    step_gradient_norm = gradient_norm(model.parameters())
                learning_rate = optimizer.param_groups[0]["lr"]
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                final_loss = loss.item()
                recent_losses.append(final_loss)
                examples_seen += images.shape[0]
                elapsed = time.perf_counter() - started_at
                epoch = step / len(loader)
                samples_per_second = examples_seen / elapsed
                steps_per_second = step / elapsed
                metrics = {
                    "train/step": step,
                    "train/epoch": epoch,
                    "train/loss": final_loss,
                    "train/loss_mean_100": sum(list(recent_losses)[-100:])
                    / min(len(recent_losses), 100),
                    "train/loss_mean_500": sum(recent_losses) / len(recent_losses),
                    "train/learning_rate": learning_rate,
                    "train/gradient_norm": step_gradient_norm,
                    "train/examples_seen": examples_seen,
                    "performance/samples_per_second": samples_per_second,
                    "performance/steps_per_second": steps_per_second,
                    "performance/elapsed_seconds": elapsed,
                }
                for k, v in batch_metrics.items():
                    metrics[f"train/{k}"] = v
                timestep_mean = ""
                timestep_std = ""
                timestep_low_fraction = ""
                timestep_high_fraction = ""
                if sampled_timesteps is not None:
                    timestep_mean = sampled_timesteps.mean().item()
                    timestep_std = sampled_timesteps.std(unbiased=False).item()
                    timestep_low_fraction = (
                        (sampled_timesteps <= 0.2).float().mean().item()
                    )
                    timestep_high_fraction = (
                        (sampled_timesteps >= 0.8).float().mean().item()
                    )
                    metrics.update(
                        {
                            "train/timestep_mean": timestep_mean,
                            "train/timestep_std": timestep_std,
                            "train/timestep_fraction_t_le_0p2": (timestep_low_fraction),
                            "train/timestep_fraction_t_ge_0p8": (
                                timestep_high_fraction
                            ),
                        }
                    )
                    if step == 1 or step % args.log_every == 0 or step == total_steps:
                        metrics["train/timestep_histogram"] = wandb.Histogram(
                            sampled_timesteps.cpu().numpy()
                        )
                tracking_run.log(metrics)
                writer.writerow(
                    (
                        step,
                        epoch,
                        final_loss,
                        learning_rate,
                        step_gradient_norm,
                        timestep_mean,
                        timestep_std,
                        timestep_low_fraction,
                        timestep_high_fraction,
                        examples_seen,
                        samples_per_second,
                        steps_per_second,
                        elapsed,
                    )
                )
                if (
                    args.task == "distill"
                    and step % args.checkpoint_every == 0
                    and step != total_steps
                ):
                    intermediate_checkpoint = {
                        "model_type": args.task,
                        "model_config": model_config,
                        "flow_matching_config": flow_matching_config,
                        "state_dict": model.state_dict(),
                        "step": step,
                        "training_config": config,
                    }
                    torch.save(
                        intermediate_checkpoint,
                        checkpoint_dir / f"step_{step:06d}.pt",
                    )
                if validation_loader is not None and (
                    step == 1 or step % args.validate_every == 0 or step == total_steps
                ):
                    validation_mse, validation_bpd = distillation_validation_metrics(
                        model,
                        teacher,
                        flow_matcher,
                        validation_loader,
                        device,
                        args.seed,
                    )
                    best_validation_bpd = min(
                        best_validation_bpd,
                        validation_bpd,
                    )
                    tracking_run.log(
                        {
                            "train/step": step,
                            "validation/distill_mse": validation_mse,
                            "validation/endpoint_bpd": validation_bpd,
                        }
                    )
                    validation_writer.writerow((step, validation_mse, validation_bpd))
                    validation_metrics_file.flush()
                    print(
                        f"validation step {step}: mse={validation_mse:.6f} "
                        f"endpoint_bpd={validation_bpd:.4f}",
                        flush=True,
                    )
                if step == 1 or step % args.log_every == 0 or step == total_steps:
                    print(
                        f"step {step}/{total_steps} loss={final_loss:.6f} "
                        f"grad_norm={step_gradient_norm:.4f} "
                        f"samples/s={samples_per_second:.1f}",
                        flush=True,
                    )

        if validation_metrics_file is not None:
            validation_metrics_file.close()

        checkpoint = {
            "model_type": args.task,
            "model_config": model_config,
            "flow_matching_config": flow_matching_config,
            "state_dict": model.state_dict(),
            "step": total_steps,
            "training_config": config,
        }
        torch.save(checkpoint, checkpoint_path)
        tracking_run.summary["train/final_loss"] = final_loss
        tracking_run.summary["train/total_examples"] = examples_seen
        tracking_run.summary["train/total_seconds"] = time.perf_counter() - started_at
        if validation_loader is not None:
            tracking_run.summary["validation/final_distill_mse"] = validation_mse
            tracking_run.summary["validation/final_endpoint_bpd"] = validation_bpd
            tracking_run.summary["validation/best_endpoint_bpd"] = best_validation_bpd
        model_artifact = wandb.Artifact(
            name=artifact_name(args),
            type="model",
            description="Self-describing Flow2PC training checkpoint",
            metadata={
                "model_config": model_config,
                "flow_matching_config": flow_matching_config,
                "step": total_steps,
            },
        )
        model_artifact.add_file(str(checkpoint_path))
        tracking_run.log_artifact(
            model_artifact,
            aliases=["latest", f"seed-{args.seed}"],
        )

        if args.dataset == "knot":
            try:
                import subprocess
                import sys
                print("Generating final ODE path plot for student...")
                output_img = run_dir / "student_trajectories.png"
                cmd = [
                    sys.executable, "scripts/plot_ode_path.py",
                    "--model-type", "student",
                    "--checkpoint", str(checkpoint_path),
                    "--output", str(output_img)
                ]
                subprocess.run(cmd, check=True)
                tracking_run.log({
                    "student/ode_path": wandb.Image(str(output_img), caption="Student ODE Path")
                })
            except Exception as e:
                print(f"Failed to plot ODE path: {e}")
                
    print(f"run saved to {run_dir.resolve()}")


if __name__ == "__main__":
    main()
