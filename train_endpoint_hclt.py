"""Train the categorical MNIST HCLT endpoint from Liu et al. (ICLR 2022).

The default command reproduces the smallest model in Table 2: M=16,
256-category leaves, batch size 1024, 100 mini-batch EM epochs with a linear
0.15 -> 0.05 step-size schedule, and 20 exact full-batch EM epochs.
"""

import argparse
import csv
import gc
import json
import math
import os
import time
from pathlib import Path

import torch
import wandb
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from repc.hclt import load_or_learn_iclr2022_hclt_topology
from repc.hclt_pc import TimeDependentHCLTPC
from repc.published_hclt import PUBLISHED_MNIST_HCLT_PARENTS

MNIST_DIM = 28 * 28
REFERENCE_PAPER = "https://starai.cs.ucla.edu/papers/LiuICLR22.pdf"
REFERENCE_REPOSITORY = "https://github.com/Tractables/PressedJuice.jl"
TABLE2_TARGETS = {
    16: {"reported_parameters": 3_300_000, "theoretical_bpd": 1.26},
    24: {"reported_parameters": 5_100_000, "theoretical_bpd": 1.22},
    32: {"reported_parameters": 7_000_000, "theoretical_bpd": 1.20},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-latents", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--mini-batch-epochs", type=int, default=100)
    parser.add_argument("--full-batch-epochs", type=int, default=20)
    parser.add_argument("--validation-size", type=int, default=0)
    parser.add_argument("--evaluate-every", type=int, default=5)
    parser.add_argument("--em-step-start", type=float, default=0.15)
    parser.add_argument("--em-step-end", type=float, default=0.05)
    parser.add_argument("--em-pseudocount", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--topology-source",
        choices=("iclr2022", "pic-aistats2024"),
        default="iclr2022",
    )
    parser.add_argument(
        "--topology-cache",
        type=Path,
        default=Path("data/MNIST/hclt/topology_iclr2022_msb3_n60000.pt"),
    )
    parser.add_argument("--topology-checkpoint", type=Path)
    parser.add_argument("--mi-sample-chunk-size", type=int, default=1_024)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--wandb-project", default="flow2pc")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-group", default="endpoint-hclt-iclr2022-table2")
    parser.add_argument(
        "--wandb-tags",
        default="hclt,iclr-2022,table-2,categorical-endpoint,hybrid-em",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    args = parser.parse_args(argv)

    positive_integers = (
        args.num_latents,
        args.batch_size,
        args.mini_batch_epochs,
        args.full_batch_epochs,
        args.evaluate_every,
        args.mi_sample_chunk_size,
        args.log_every,
        args.num_samples,
    )
    if any(value < 1 for value in positive_integers):
        parser.error("counts, epochs, and intervals must be positive")
    if not 0 <= args.validation_size < 60_000:
        parser.error("validation size must be in [0, 60000)")
    if not 0.0 < args.em_step_end <= args.em_step_start <= 1.0:
        parser.error("require 0 < em-step-end <= em-step-start <= 1")
    if args.em_pseudocount < 0.0:
        parser.error("EM pseudocount must be non-negative")
    return args


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def load_mnist(
    data_dir: Path,
    validation_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    all_training_data = MNIST(root=data_dir, train=True, download=True).data.flatten(1)
    test_data = MNIST(root=data_dir, train=False, download=True).data.flatten(1)
    if validation_size == 0:
        return all_training_data, all_training_data, None, test_data
    permutation = torch.randperm(
        len(all_training_data),
        generator=torch.Generator().manual_seed(seed),
    )
    validation_indices = permutation[:validation_size]
    training_indices = permutation[validation_size:]
    return (
        all_training_data,
        all_training_data[training_indices],
        all_training_data[validation_indices],
        test_data,
    )


def load_topology_checkpoint(path: Path) -> list[int]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    parents = payload["parents"]
    if len(parents) != MNIST_DIM:
        raise ValueError(f"expected {MNIST_DIM} topology nodes, found {len(parents)}")
    return parents


@torch.no_grad()
def evaluate_bpd(
    model: TimeDependentHCLTPC,
    data: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    endpoint = torch.tensor(1.0, device=device)
    log_likelihood = 0.0
    for batch in data.split(batch_size):
        log_likelihood += (
            model.discrete_log_prob(
                batch.to(device, non_blocking=True),
                endpoint,
            )
            .sum()
            .item()
        )
    model.train(was_training)
    return -log_likelihood / (len(data) * MNIST_DIM * math.log(2.0))


def add_statistics(
    accumulated: dict[str, torch.Tensor] | None,
    update: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Accumulate batch sufficient statistics without retaining autograd state."""
    if accumulated is None:
        return {name: value.detach().clone() for name, value in update.items()}
    if accumulated.keys() != update.keys():
        raise ValueError("EM statistic dictionaries do not have the same fields")
    for name, value in update.items():
        accumulated[name].add_(value)
    return accumulated


def linear_em_step_size(
    update_index: int,
    total_updates: int,
    start: float,
    end: float,
) -> float:
    if total_updates <= 1:
        return end
    progress = update_index / (total_updates - 1)
    return start + progress * (end - start)


def checkpoint_payload(
    model: TimeDependentHCLTPC,
    model_config: dict,
    training_config: dict,
    phase: str,
    epoch: int,
    update: int,
    monitor_bpd: float,
) -> dict:
    return {
        "model_type": "endpoint_hclt",
        "model_config": model_config,
        "state_dict": model.state_dict(),
        "phase": phase,
        "epoch": epoch,
        "step": update,
        "monitor_bpd": monitor_bpd,
        "training_config": training_config,
        "reference": {
            "paper": REFERENCE_PAPER,
            "repository": REFERENCE_REPOSITORY,
            "protocol": "ICLR 2022 Appendix B.1/B.3 and Table 2",
        },
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    label = (
        f"hclt-categorical-m{args.num_latents}-bs{args.batch_size}-"
        f"hybrid-em{args.mini_batch_epochs}p{args.full_batch_epochs}-seed{args.seed}"
    )
    run_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path("outputs/runs/endpoint") / label
    ).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run: {run_dir}")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    all_training_data, train_data, validation_data, test_data = load_mnist(
        args.data_dir,
        args.validation_size,
        args.seed,
    )
    topology_started_at = time.perf_counter()
    if args.topology_checkpoint is not None:
        parents = load_topology_checkpoint(args.topology_checkpoint)
        topology_source = str(args.topology_checkpoint.resolve())
        topology_config = {"source": "explicit-checkpoint"}
    elif args.topology_source == "pic-aistats2024":
        parents = list(PUBLISHED_MNIST_HCLT_PARENTS)
        topology_source = "gengala/pic:TREE_MNIST@51863e4"
        topology_config = {"source": "pic-aistats2024"}
    else:
        parents, topology_config = load_or_learn_iclr2022_hclt_topology(
            all_training_data,
            args.topology_cache,
            device,
            sample_chunk_size=args.mi_sample_chunk_size,
        )
        topology_source = str(args.topology_cache.resolve())
    topology_seconds = time.perf_counter() - topology_started_at
    del all_training_data
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model_config = {
        "name": "hclt",
        "parents": parents,
        "num_latents": args.num_latents,
        "time_order": 1,
        "sum_layer": "dense",
        "time_conditioning": "polynomial",
        "time_embedding_dim": 16,
        "time_feature_seed": args.seed,
        "leaf_distribution": "categorical",
        "leaf_components": 1,
        "num_categories": 256,
        "topology_source": topology_source,
        "topology_config": topology_config,
    }
    model = TimeDependentHCLTPC(
        parents=parents,
        num_latents=args.num_latents,
        time_order=1,
        sum_layer="dense",
        leaf_distribution="categorical",
        num_categories=256,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    table_target = TABLE2_TARGETS.get(args.num_latents)

    training_config = vars(args).copy()
    for key, value in training_config.items():
        if isinstance(value, Path):
            training_config[key] = str(value.resolve())
    training_config.update(
        {
            "parameter_count": parameter_count,
            "paper_table2_target": table_target,
            "topology_source_resolved": topology_source,
            "topology_config": topology_config,
            "topology_seconds": topology_seconds,
            "num_training_examples": len(train_data),
            "num_validation_examples": 0
            if validation_data is None
            else len(validation_data),
            "reference_paper": REFERENCE_PAPER,
            "reference_repository": REFERENCE_REPOSITORY,
        }
    )
    (run_dir / "config.json").write_text(json.dumps(training_config, indent=2) + "\n")

    metrics_path = run_dir / "metrics.csv"
    last_checkpoint_path = checkpoint_dir / "last.pt"
    final_checkpoint_path = checkpoint_dir / "final.pt"
    best_checkpoint_path = checkpoint_dir / "best.pt"
    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    wandb_data_dir = run_dir / ".wandb-data"
    wandb_data_dir.mkdir(exist_ok=True)
    os.environ.setdefault("WANDB_DATA_DIR", str(wandb_data_dir))

    batches_per_epoch = math.ceil(len(train_data) / args.batch_size)
    total_mini_batch_updates = args.mini_batch_epochs * batches_per_epoch
    shuffle_generator = torch.Generator().manual_seed(args.seed)

    with wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or label,
        group=args.wandb_group,
        job_type="endpoint-hclt-hybrid-em",
        tags=["mnist", *tags],
        config={**training_config, "model_config": model_config},
        mode=args.wandb_mode,
        dir=str(run_dir),
    ) as tracking_run:
        tracking_run.define_metric("train/update")
        tracking_run.define_metric("train/*", step_metric="train/update")
        tracking_run.define_metric("validation/*", step_metric="train/update")
        tracking_run.define_metric("test/*", step_metric="train/update")
        tracking_run.define_metric("performance/*", step_metric="train/update")
        tracking_run.summary["model/parameter_count"] = parameter_count
        tracking_run.summary["structure/learning_seconds"] = topology_seconds
        tracking_run.summary["reference/paper"] = REFERENCE_PAPER
        if table_target is not None:
            tracking_run.summary["reference/table2_theoretical_bpd"] = table_target[
                "theoretical_bpd"
            ]
            tracking_run.summary["reference/table2_reported_parameters"] = table_target[
                "reported_parameters"
            ]
        tracking_run.log_code(root=str(Path.cwd()))

        started_at = time.perf_counter()
        recent_times: list[float] = []
        update_index = 0
        latest_monitor_bpd = float("inf")
        best_validation_bpd = float("inf")
        model.train()

        with metrics_path.open("w", newline="") as metrics_file:
            writer = csv.writer(metrics_file)
            writer.writerow(
                (
                    "phase",
                    "epoch",
                    "update",
                    "batch",
                    "batch_bpd",
                    "epoch_online_bpd",
                    "validation_bpd",
                    "test_bpd",
                    "em_step_size",
                    "batch_seconds",
                    "elapsed_seconds",
                    "max_memory_gib",
                )
            )

            for epoch in range(1, args.mini_batch_epochs + 1):
                permutation = torch.randperm(
                    len(train_data), generator=shuffle_generator
                )
                epoch_log_likelihood = 0.0
                epoch_examples = 0
                for batch_number, indices in enumerate(
                    permutation.split(args.batch_size),
                    start=1,
                ):
                    batch_started_at = time.perf_counter()
                    batch = train_data[indices].to(device, non_blocking=True)
                    em_step_size = linear_em_step_size(
                        update_index,
                        total_mini_batch_updates,
                        args.em_step_start,
                        args.em_step_end,
                    )
                    statistics = model.endpoint_em_statistics(batch)
                    batch_log_likelihood = statistics["log_likelihood"].item()
                    batch_bpd = -batch_log_likelihood / (
                        len(batch) * MNIST_DIM * math.log(2.0)
                    )
                    epoch_log_likelihood += batch_log_likelihood
                    epoch_examples += len(batch)
                    model.endpoint_em_update(
                        statistics,
                        pseudocount=args.em_pseudocount,
                        step_size=em_step_size,
                    )
                    update_index += 1

                    batch_seconds = time.perf_counter() - batch_started_at
                    recent_times.append(batch_seconds)
                    recent_times = recent_times[-100:]
                    elapsed_seconds = time.perf_counter() - started_at
                    max_memory_gib = (
                        torch.cuda.max_memory_allocated(device) / 1024**3
                        if device.type == "cuda"
                        else 0.0
                    )
                    tracking_run.log(
                        {
                            "train/update": update_index,
                            "train/phase": 0,
                            "train/epoch": epoch,
                            "train/batch_bpd": batch_bpd,
                            "train/em_step_size": em_step_size,
                            "performance/batch_seconds": batch_seconds,
                            "performance/updates_per_second_mean_100": (
                                len(recent_times) / sum(recent_times)
                            ),
                            "performance/elapsed_seconds": elapsed_seconds,
                            "performance/max_memory_gib": max_memory_gib,
                        }
                    )
                    writer.writerow(
                        (
                            "mini_batch",
                            epoch,
                            update_index,
                            batch_number,
                            batch_bpd,
                            "",
                            "",
                            "",
                            em_step_size,
                            batch_seconds,
                            elapsed_seconds,
                            max_memory_gib,
                        )
                    )

                    if update_index == 1 or update_index % args.log_every == 0:
                        print(
                            f"mini epoch {epoch}/{args.mini_batch_epochs} "
                            f"batch {batch_number}/{batches_per_epoch} "
                            f"update={update_index}/{total_mini_batch_updates} "
                            f"bpd={batch_bpd:.4f} eta={em_step_size:.5f} "
                            f"updates/s={len(recent_times) / sum(recent_times):.2f}",
                            flush=True,
                        )

                epoch_online_bpd = -epoch_log_likelihood / (
                    epoch_examples * MNIST_DIM * math.log(2.0)
                )
                validation_bpd: float | str = ""
                test_bpd: float | str = ""
                should_evaluate = (
                    epoch == 1
                    or epoch % args.evaluate_every == 0
                    or epoch == args.mini_batch_epochs
                )
                if should_evaluate:
                    test_bpd = evaluate_bpd(model, test_data, args.batch_size, device)
                    latest_monitor_bpd = test_bpd
                    if validation_data is not None:
                        validation_bpd = evaluate_bpd(
                            model,
                            validation_data,
                            args.batch_size,
                            device,
                        )
                        latest_monitor_bpd = validation_bpd
                        if validation_bpd < best_validation_bpd:
                            best_validation_bpd = validation_bpd
                            torch.save(
                                checkpoint_payload(
                                    model,
                                    model_config,
                                    training_config,
                                    "mini_batch",
                                    epoch,
                                    update_index,
                                    validation_bpd,
                                ),
                                best_checkpoint_path,
                            )
                    torch.save(
                        checkpoint_payload(
                            model,
                            model_config,
                            training_config,
                            "mini_batch",
                            epoch,
                            update_index,
                            latest_monitor_bpd,
                        ),
                        last_checkpoint_path,
                    )
                epoch_metrics = {
                    "train/update": update_index,
                    "train/epoch_online_bpd": epoch_online_bpd,
                }
                if validation_bpd != "":
                    epoch_metrics["validation/endpoint_bpd"] = validation_bpd
                if test_bpd != "":
                    epoch_metrics["test/endpoint_bpd_diagnostic"] = test_bpd
                tracking_run.log(epoch_metrics)
                writer.writerow(
                    (
                        "mini_batch_epoch",
                        epoch,
                        update_index,
                        "",
                        "",
                        epoch_online_bpd,
                        validation_bpd,
                        test_bpd,
                        em_step_size,
                        "",
                        time.perf_counter() - started_at,
                        max_memory_gib,
                    )
                )
                metrics_file.flush()
                evaluation_text = "" if test_bpd == "" else f" test_bpd={test_bpd:.4f}"
                print(
                    f"mini epoch {epoch} complete: online_bpd={epoch_online_bpd:.4f}"
                    f"{evaluation_text}",
                    flush=True,
                )

            torch.save(
                checkpoint_payload(
                    model,
                    model_config,
                    training_config,
                    "mini_batch_final",
                    args.mini_batch_epochs,
                    update_index,
                    latest_monitor_bpd,
                ),
                checkpoint_dir / "mini_batch_final.pt",
            )

            for epoch in range(1, args.full_batch_epochs + 1):
                epoch_started_at = time.perf_counter()
                accumulated_statistics = None
                for batch_number, batch in enumerate(
                    train_data.split(args.batch_size),
                    start=1,
                ):
                    batch_started_at = time.perf_counter()
                    statistics = model.endpoint_em_statistics(
                        batch.to(device, non_blocking=True)
                    )
                    batch_bpd = -statistics["log_likelihood"].item() / (
                        len(batch) * MNIST_DIM * math.log(2.0)
                    )
                    accumulated_statistics = add_statistics(
                        accumulated_statistics,
                        statistics,
                    )
                    update_index += 1
                    batch_seconds = time.perf_counter() - batch_started_at
                    elapsed_seconds = time.perf_counter() - started_at
                    max_memory_gib = (
                        torch.cuda.max_memory_allocated(device) / 1024**3
                        if device.type == "cuda"
                        else 0.0
                    )
                    writer.writerow(
                        (
                            "full_batch_e_step",
                            epoch,
                            update_index,
                            batch_number,
                            batch_bpd,
                            "",
                            "",
                            "",
                            1.0,
                            batch_seconds,
                            elapsed_seconds,
                            max_memory_gib,
                        )
                    )
                    if batch_number == 1 or batch_number % args.log_every == 0:
                        print(
                            f"full epoch {epoch}/{args.full_batch_epochs} E-step "
                            f"batch {batch_number}/{batches_per_epoch}",
                            flush=True,
                        )

                if accumulated_statistics is None:
                    raise RuntimeError("full-batch EM received no training data")
                pre_update_bpd = -accumulated_statistics["log_likelihood"].item() / (
                    len(train_data) * MNIST_DIM * math.log(2.0)
                )
                model.endpoint_em_update(
                    accumulated_statistics,
                    pseudocount=args.em_pseudocount,
                    step_size=1.0,
                )
                test_bpd = evaluate_bpd(model, test_data, args.batch_size, device)
                latest_monitor_bpd = test_bpd
                validation_bpd = ""
                if validation_data is not None:
                    validation_bpd = evaluate_bpd(
                        model,
                        validation_data,
                        args.batch_size,
                        device,
                    )
                    latest_monitor_bpd = validation_bpd
                    if validation_bpd < best_validation_bpd:
                        best_validation_bpd = validation_bpd
                        torch.save(
                            checkpoint_payload(
                                model,
                                model_config,
                                training_config,
                                "full_batch",
                                epoch,
                                update_index,
                                validation_bpd,
                            ),
                            best_checkpoint_path,
                        )
                payload = checkpoint_payload(
                    model,
                    model_config,
                    training_config,
                    "full_batch",
                    epoch,
                    update_index,
                    latest_monitor_bpd,
                )
                torch.save(payload, last_checkpoint_path)
                tracking_run.log(
                    {
                        "train/update": update_index,
                        "train/phase": 1,
                        "train/full_batch_epoch": epoch,
                        "train/full_batch_pre_update_bpd": pre_update_bpd,
                        "test/endpoint_bpd_diagnostic": test_bpd,
                        "performance/full_batch_epoch_seconds": (
                            time.perf_counter() - epoch_started_at
                        ),
                        **(
                            {"validation/endpoint_bpd": validation_bpd}
                            if validation_bpd != ""
                            else {}
                        ),
                    }
                )
                writer.writerow(
                    (
                        "full_batch_epoch",
                        epoch,
                        update_index,
                        "",
                        "",
                        pre_update_bpd,
                        validation_bpd,
                        test_bpd,
                        1.0,
                        "",
                        time.perf_counter() - started_at,
                        max_memory_gib,
                    )
                )
                metrics_file.flush()
                print(
                    f"full epoch {epoch} complete: pre_update_bpd={pre_update_bpd:.4f} "
                    f"test_bpd={test_bpd:.4f}",
                    flush=True,
                )

        final_train_bpd = evaluate_bpd(model, train_data, args.batch_size, device)
        final_test_bpd = evaluate_bpd(model, test_data, args.batch_size, device)
        final_validation_bpd = (
            evaluate_bpd(model, validation_data, args.batch_size, device)
            if validation_data is not None
            else None
        )
        final_payload = checkpoint_payload(
            model,
            model_config,
            training_config,
            "complete",
            args.full_batch_epochs,
            update_index,
            final_validation_bpd or final_test_bpd,
        )
        torch.save(final_payload, final_checkpoint_path)
        torch.save(final_payload, last_checkpoint_path)

        generator = torch.Generator(device=device).manual_seed(args.seed)
        samples = model.sample(
            args.num_samples,
            torch.tensor(1.0, device=device),
            generator=generator,
        ).reshape(-1, 1, 28, 28)
        sample_path = run_dir / "samples_t1.png"
        save_image(
            samples,
            sample_path,
            nrow=10,
            normalize=True,
            value_range=(-1.0, 1.0),
        )

        summary = {
            "final_train_bpd": final_train_bpd,
            "final_validation_bpd": final_validation_bpd,
            "final_test_bpd": final_test_bpd,
            "paper_table2_theoretical_bpd": (
                None if table_target is None else table_target["theoretical_bpd"]
            ),
            "parameter_count": parameter_count,
            "paper_table2_reported_parameters": (
                None if table_target is None else table_target["reported_parameters"]
            ),
            "topology_seconds": topology_seconds,
            "training_seconds": time.perf_counter() - started_at,
            "total_e_step_batches": update_index,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        tracking_run.summary.update(summary)
        tracking_run.log(
            {
                "train/update": update_index,
                "train/final_endpoint_bpd": final_train_bpd,
                "test/final_endpoint_bpd": final_test_bpd,
                "qualitative/t1_samples": wandb.Image(str(sample_path)),
                **(
                    {"validation/final_endpoint_bpd": final_validation_bpd}
                    if final_validation_bpd is not None
                    else {}
                ),
            }
        )

        artifact = wandb.Artifact(
            name=f"mnist-endpoint-{label}",
            type="model",
            description="ICLR 2022 Table 2 categorical HCLT endpoint",
            metadata=summary,
        )
        artifact.add_file(str(final_checkpoint_path))
        artifact.add_file(str(run_dir / "summary.json"))
        artifact.add_file(str(sample_path))
        tracking_run.log_artifact(artifact, aliases=["final", f"seed-{args.seed}"])

    print(
        f"final train BPD={final_train_bpd:.4f}; test BPD={final_test_bpd:.4f}; "
        f"run saved to {run_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
