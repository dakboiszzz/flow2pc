from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

MNIST_DIM = 28 * 28
MNIST_SHAPE = (1, 28, 28)


def mnist_loader(
    root: Path,
    train: bool,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
    )
    dataset = MNIST(root, train=train, transform=transform, download=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )
