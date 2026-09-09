from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
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


def generate_knot_distribution(
    K: int = 60,
    samples_per_marginal: int = 10,
    std: float = 0.1,
    seed: int | None = None,
) -> torch.Tensor:
    """Generate synthetic Knot distribution from ALI-CFM Appendix B.

    Returns a torch.Tensor of shape [K, samples_per_marginal, 2].
    """
    if K <= 0 or K % 3 != 0:
        raise ValueError(f"K (number of marginals) must be a positive multiple of 3, got {K}")

    t = np.linspace(0.0, 3.0, K)
    # \tilde{t} = 3 * (t / t_max) - 1.5 where t_max = 3
    t_tilde = t - 1.5

    seg_size = K // 3
    t1 = t_tilde[:seg_size]
    t2 = t_tilde[seg_size : 2 * seg_size]
    t3 = t_tilde[2 * seg_size :]

    # \mu_X(t)
    mu_x1 = 3.0 * (t1 + 0.5)
    mu_x2 = np.cos(2.0 * np.pi * (t2 - 0.75))
    mu_x3 = 3.0 * (t3 - 0.5)
    mu_x = np.concatenate([mu_x1, mu_x2, mu_x3])

    # \mu_Y(t)
    mu_y1 = -0.5 * np.tanh(5.0 * (t1 + 1.0)) + 0.5
    mu_y2 = np.sin(2.0 * np.pi * (t2 - 0.75))
    mu_y3 = 0.5 * np.tanh(5.0 * (t3 - 1.0)) + 0.5
    mu_y = np.concatenate([mu_y1, mu_y2, mu_y3])

    rng = np.random.default_rng(seed)
    x_samples = mu_x[:, None] + rng.normal(0.0, std, size=(K, samples_per_marginal))
    y_samples = mu_y[:, None] + rng.normal(0.0, std, size=(K, samples_per_marginal))

    samples = np.stack([x_samples, y_samples], axis=-1)
    return torch.from_numpy(samples).float()


class MultiMarginalDataset(Dataset):
    """Dataset wrapping multi-marginal snapshots [K, N, D].

    Flattened across marginals while retaining the temporal k_index (0 to K-1).
    """

    def __init__(self, data: torch.Tensor):
        if data.ndim != 3:
            raise ValueError(f"expected 3D tensor [K, N, D], got shape {tuple(data.shape)}")
        self.data = data
        self.K = data.shape[0]
        self.N = data.shape[1]
        self.D = data.shape[2]

    def __len__(self) -> int:
        return self.K * self.N

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        k_index = index // self.N
        n_index = index % self.N
        return self.data[k_index, n_index], torch.tensor(k_index, dtype=torch.long)


def multimarginal_loader(
    dataset_name: str,
    batch_size: int,
    K: int = 60,
    samples_per_marginal: int = 10,
    shuffle: bool = True,
    num_workers: int = 0,
    data_dir: Path | str = "data",
    seed: int | None = None,
) -> DataLoader:
    """Create a DataLoader for multi-marginal snapshot datasets."""
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower == "knot":
        data = generate_knot_distribution(
            K=K,
            samples_per_marginal=samples_per_marginal,
            seed=seed,
        )
    elif dataset_name_lower == "ht206b1":
        data_path = Path(data_dir)
        candidates = [
            data_path / "ht206b1.npy",
            data_path / "ht206b1.pt",
            data_path / "HT206B1.npy",
            data_path / "HT206B1.pt",
            data_path / "st" / "ht206b1.npy",
            data_path / "ST" / "ht206b1.npy",
        ]
        loaded = None
        for candidate in candidates:
            if candidate.is_file():
                if candidate.suffix == ".npy":
                    loaded = torch.from_numpy(np.load(candidate)).float()
                else:
                    loaded = torch.load(candidate, weights_only=True).float()
                break
        if loaded is None:
            raise FileNotFoundError(
                f"could not find ht206b1 dataset file in {data_path}. Checked: "
                f"{[str(c) for c in candidates]}"
            )
        data = loaded
        if data.ndim == 2:
            if data.shape[0] % K == 0:
                data = data.reshape(K, -1, data.shape[-1])
            else:
                raise ValueError(
                    f"cannot reshape tensor of shape {tuple(data.shape)} into K={K} marginals"
                )
    else:
        raise ValueError(f"unknown multi-marginal dataset: {dataset_name}")

    dataset = MultiMarginalDataset(data)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=shuffle,
    )

