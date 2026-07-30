"""MNIST data utilities. Images are scaled to [-1, 1]."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TFM = transforms.Compose(
    [
        transforms.ToTensor(),  # [0,1]
        transforms.Normalize(0.5, 0.5),  # [-1,1]
    ]
)


def get_mnist(root, train=True):
    return datasets.MNIST(root, train=train, download=True, transform=TFM)


def get_fashion_mnist(root, train=False):
    """OOD dataset for the UQ demo."""
    return datasets.FashionMNIST(root, train=train, download=True, transform=TFM)


def get_loader(dataset, batch_size, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=shuffle,
    )


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def flatten(x):
    """(B,1,28,28) -> (B,784)"""
    return x.reshape(x.shape[0], -1)


def unflatten(z):
    """(B,784) -> (B,1,28,28)"""
    return z.reshape(z.shape[0], 1, 28, 28)


@torch.no_grad()
def marginal_path_sample(x1, t):
    """Sample Z_t from the flow-matching marginal (OT/linear) path.

    z_t = (1-t) z0 + t x1,  z0 ~ N(0, I).
    Also returns the conditional target velocity x1 - z0 (used only for
    flow pretraining, not for PC distillation).
    """
    z0 = torch.randn_like(x1)
    while t.dim() < x1.dim():
        t = t.unsqueeze(-1)
    zt = (1 - t) * z0 + t * x1
    return zt, x1 - z0
