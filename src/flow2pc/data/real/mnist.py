from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST

MNIST_DIM = 28 * 28
MNIST_SHAPE = (1, 28, 28)


def quantize_mnist(images):
    """Clamp and quantize continuous [-1, 1] samples to MNIST's 256 bins."""
    pixels = ((images.clamp(-1.0, 1.0) + 1.0) * 0.5 * 255.0).round() / 255.0
    return pixels * 2.0 - 1.0


def mnist_loader(
    root: Path,
    train: bool,
    batch_size: int,
    num_workers: int,
    validation: bool = False,
    validation_size: int = 0,
    split_seed: int = 0,
) -> DataLoader:
    if validation and not train:
        raise ValueError("validation data must be drawn from the training split")
    if validation_size < 0 or validation_size >= 60_000:
        raise ValueError("validation_size must be in [0, 60000)")
    if validation and validation_size == 0:
        raise ValueError("validation=True requires a positive validation_size")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
    )
    dataset = MNIST(root, train=train, transform=transform, download=True)
    if train and validation_size:
        generator = torch.Generator().manual_seed(split_seed)
        indices = torch.randperm(len(dataset), generator=generator).tolist()
        if validation:
            indices = indices[:validation_size]
        else:
            indices = indices[validation_size:]
        dataset = Subset(dataset, indices)
    shuffle = train and not validation
    loader_generator = torch.Generator().manual_seed(split_seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
        generator=loader_generator,
    )
