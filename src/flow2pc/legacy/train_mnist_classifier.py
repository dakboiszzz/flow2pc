import argparse
import json
import os
from pathlib import Path

import torch
import wandb
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from flow2pc.legacy.mnist_classifier import MNIST_MEAN, MNIST_STD, MNISTClassifier
from flow2pc.legacy.train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation/mnist-classifier-pytorch-example-seed1"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--wandb-project", default="flow2pc")
    parser.add_argument("--wandb-entity")
    parser.add_argument(
        "--wandb-run-name",
        default="eval-feature-extractor-pytorch-mnist-seed1",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.test_batch_size) < 1:
        parser.error("epochs and batch sizes must be positive")
    return args


@torch.no_grad()
def evaluate(
    model: MNISTClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        output = model(images)
        total_loss += F.nll_loss(output, labels, reduction="sum").item()
        correct += output.argmax(dim=1).eq(labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = args.output_dir.resolve()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )
    train_dataset = datasets.MNIST(
        args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        args.data_dir,
        train=False,
        download=True,
        transform=transform,
    )
    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_options,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        **loader_options,
    )
    model = MNISTClassifier().to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    config = {
        **vars(args),
        "data_dir": str(args.data_dir.resolve()),
        "output_dir": str(output_dir),
        "architecture_source": "pytorch/examples/mnist/main.py",
        "feature_layer": "128-dimensional activation after fc1 ReLU",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    wandb_data_dir = output_dir / ".wandb-data"
    wandb_data_dir.mkdir(exist_ok=True)
    os.environ.setdefault("WANDB_DATA_DIR", str(wandb_data_dir))

    with wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        group="evaluation-support",
        job_type="feature-extractor-train",
        tags=("mnist", "evaluation", "classifier", "pytorch-example"),
        config=config,
        mode=args.wandb_mode,
        dir=str(output_dir),
    ) as run:
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("test/*", step_metric="epoch")
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = 0.0
            seen = 0
            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = F.nll_loss(model(images), labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(images)
                seen += len(images)
            test_loss, test_accuracy = evaluate(model, test_loader, device)
            metrics = {
                "epoch": epoch,
                "train/nll": loss_sum / seen,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "test/nll": test_loss,
                "test/accuracy": test_accuracy,
            }
            run.log(metrics)
            print(
                f"epoch {epoch}/{args.epochs} "
                f"train_nll={metrics['train/nll']:.4f} "
                f"test_nll={test_loss:.4f} test_acc={test_accuracy:.4%}",
                flush=True,
            )
            scheduler.step()

        checkpoint_path = checkpoint_dir / "last.pt"
        checkpoint = {
            "model_type": "mnist_classifier",
            "state_dict": model.state_dict(),
            "test_nll": test_loss,
            "test_accuracy": test_accuracy,
            "training_config": config,
        }
        torch.save(checkpoint, checkpoint_path)
        run.summary["test/nll"] = test_loss
        run.summary["test/accuracy"] = test_accuracy
        artifact = wandb.Artifact(
            "mnist-evaluation-classifier",
            type="model",
            metadata={"test_accuracy": test_accuracy},
        )
        artifact.add_file(str(checkpoint_path))
        run.log_artifact(artifact, aliases=["latest", f"seed-{args.seed}"])
    print(f"classifier saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
