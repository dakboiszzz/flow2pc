import argparse
import csv
import json
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
from torch import nn

from repc.data import MNIST_DIM, mnist_loader
from repc.flow import UNet
from repc.pc import TimeDependentGaussianPC
from repc.region_pc import RegionGraphPC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("flow", "distill"))
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--student", choices=("gaussian", "region"), default="gaussian")
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--time-dim", type=int, default=256)
    parser.add_argument("--num-components", type=int, default=512)
    parser.add_argument("--num-channels", type=int, default=8)
    parser.add_argument("--time-order", type=int, default=4)
    args = parser.parse_args()
    if args.task == "distill" and args.teacher_checkpoint is None:
        parser.error("--teacher-checkpoint is required for distillation")
    if args.steps < 1 or args.batch_size < 1 or args.log_every < 1:
        parser.error("steps, batch size, and log interval must be positive")
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
        )
    if config["name"] == "region":
        return RegionGraphPC(
            height=config["height"],
            width=config["width"],
            num_channels=config["num_channels"],
            time_order=config["time_order"],
        )
    raise ValueError(f"unknown student: {config['name']}")


def load_teacher(path: Path, device: torch.device) -> UNet:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["model_type"] != "flow":
        raise ValueError(f"{path} is not a flow checkpoint")
    teacher = UNet(**checkpoint["model_config"]).to(device)
    teacher.load_state_dict(checkpoint["state_dict"])
    return teacher.eval()


def flow_matching_loss(model: nn.Module, data: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(data)
    time = torch.rand(data.shape[0], device=data.device)
    broadcast_time = time.reshape(-1, *([1] * (data.ndim - 1)))
    path_sample = (1.0 - broadcast_time) * noise + broadcast_time * data
    return nn.functional.mse_loss(model(path_sample, time), data - noise)


def distillation_loss(
    pc: nn.Module,
    teacher: nn.Module,
    data: torch.Tensor,
) -> torch.Tensor:
    noise = torch.randn_like(data)
    time = torch.rand((), device=data.device)
    path_sample = (1.0 - time) * noise + time * data
    with torch.no_grad():
        teacher_velocity = teacher(path_sample, time.expand(data.shape[0]))
    pc_velocity = pc.velocity(path_sample.flatten(1), time)
    return nn.functional.mse_loss(pc_velocity, teacher_velocity.flatten(1))


def serializable_config(args: argparse.Namespace) -> dict:
    config = vars(args).copy()
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value.resolve())
    return config


def run_directory(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    return Path("outputs") / "runs" / args.task / timestamp


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    run_dir = run_directory(args)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = serializable_config(args)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    loader = mnist_loader(
        args.data_dir,
        train=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    batches = iter(loader)

    if args.task == "flow":
        model_config = {
            "base_channels": args.base_channels,
            "time_dim": args.time_dim,
        }
        model = UNet(**model_config).to(device)
        loss_function = partial(flow_matching_loss, model)
        lr = args.lr or 2e-4
    else:
        teacher = load_teacher(args.teacher_checkpoint, device)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        if args.student == "gaussian":
            model_config = {
                "name": "gaussian",
                "num_components": args.num_components,
                "dim": MNIST_DIM,
                "time_order": args.time_order,
            }
        else:
            model_config = {
                "name": "region",
                "height": 28,
                "width": 28,
                "num_channels": args.num_channels,
                "time_order": args.time_order,
            }
        model = build_student(model_config).to(device)
        loss_function = partial(distillation_loss, model, teacher)
        lr = args.lr or 1e-3

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    model.train()
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow(("step", "loss"))
        for step in range(1, args.steps + 1):
            try:
                images, _ = next(batches)
            except StopIteration:
                batches = iter(loader)
                images, _ = next(batches)
            images = images.to(device, non_blocking=True)
            loss = loss_function(images)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            writer.writerow((step, loss.item()))
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                print(f"step {step}/{args.steps} loss={loss.item():.6f}", flush=True)

    checkpoint = {
        "model_type": args.task,
        "model_config": model_config,
        "state_dict": model.state_dict(),
        "step": args.steps,
        "training_config": config,
    }
    torch.save(checkpoint, checkpoint_dir / "last.pt")
    print(f"run saved to {run_dir.resolve()}")


if __name__ == "__main__":
    main()
