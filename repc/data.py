from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import MNIST

from torchcfm.optimal_transport import OTPlanSampler

MNIST_DIM = 28 * 28
MNIST_SHAPE = (1, 28, 28)

def precompute_ot_couplings(data: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute exact Optimal Transport plan pairings for all K-1 transitions."""
    K, N, D = data.shape
    ot_sampler = OTPlanSampler(method="exact")
    coupled_x0 = []
    coupled_x1 = []

    print(f"Precomputing exact OT plans across {K - 1} marginal transitions ({N} samples each)...")
    for k in range(K - 1):
        m0 = data[k]
        m1 = data[k + 1]
        p0, p1 = ot_sampler.sample_plan(m0, m1)
        coupled_x0.append(p0)
        coupled_x1.append(p1)

    # Shape: [K-1, N, D]
    x0_all = torch.stack(coupled_x0, dim=0).to(device)
    x1_all = torch.stack(coupled_x1, dim=0).to(device)
    return x0_all, x1_all

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


def generate_spiral_distribution(
    K: int = 60,
    samples_per_marginal: int = 10,
    std: float = 0.1,
    seed: int | None = None,
) -> torch.Tensor:
    """Generate synthetic Spiral distribution.

    Returns a torch.Tensor of shape [K, samples_per_marginal, 2].
    """
    if K <= 0:
        raise ValueError(f"K (number of marginals) must be positive, got {K}")

    t = np.linspace(0.0, 3.0, K)
    
    # \mu_X(t) and \mu_Y(t) for an expanding spiral
    mu_x = (1.0 + t) * np.cos(2.0 * np.pi * t)
    mu_y = (1.0 + t) * np.sin(2.0 * np.pi * t)

    rng = np.random.default_rng(seed)
    x_samples = mu_x[:, None] + rng.normal(0.0, std, size=(K, samples_per_marginal))
    y_samples = mu_y[:, None] + rng.normal(0.0, std, size=(K, samples_per_marginal))

    samples = np.stack([x_samples, y_samples], axis=-1)
    return torch.from_numpy(samples).float()


class OTCoupledDataset(Dataset):
    """Dataset yielding (x_k, x_{k+1}, k) from precomputed multi-marginal OT couplings."""

    def __init__(self, x0_all: torch.Tensor, x1_all: torch.Tensor):
        if x0_all.ndim != 3 or x1_all.ndim != 3:
            raise ValueError("expected 3D tensors [K-1, N, D]")
        self.x0 = x0_all
        self.x1 = x1_all
        self.num_transitions = x0_all.shape[0]
        self.N = x0_all.shape[1]

    def __len__(self) -> int:
        return self.num_transitions * self.N

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_index = index // self.N
        n_index = index % self.N
        return self.x0[k_index, n_index], self.x1[k_index, n_index], torch.tensor(k_index, dtype=torch.long)


def multimarginal_loader(
    dataset_name: str,
    batch_size: int,
    K: int = 60,
    samples_per_marginal: int = 10,
    shuffle: bool = True,
    num_workers: int = 0,
    data_dir: Path | str = "data",
    seed: int | None = None,
    time_sampling: str = "uniform_interval",
    time_beta_alpha: float = 5.0,
    time_beta_beta: float = 5.0,
) -> DataLoader:
    """Create a DataLoader for multi-marginal snapshot datasets.

    ``middle_beta`` changes which OT transition is sampled, while retaining
    the paired ``(x_k, x_{k+1})`` examples needed to construct a valid local
    flow-matching path.  The transition midpoint is weighted by the density
    of ``Beta(time_beta_alpha, time_beta_beta)``; the default is uniform.
    """
    if time_sampling not in ("uniform_global", "uniform_interval", "middle_beta"):
        raise ValueError(f"unknown multimarginal time sampling mode: {time_sampling}")
    if time_beta_alpha <= 0 or time_beta_beta <= 0:
        raise ValueError("multimarginal Beta concentrations must be positive")
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower == "knot":
        data = generate_knot_distribution(
            K=K,
            samples_per_marginal=samples_per_marginal,
            seed=seed,
        )
    elif dataset_name_lower == "spiral":
        data = generate_spiral_distribution(
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

    # Precompute OT plans for all K-1 marginal transitions
    x0_all, x1_all = precompute_ot_couplings(data, device=torch.device('cpu'))

    dataset = OTCoupledDataset(x0_all, x1_all)
    sampler = None
    if time_sampling == "middle_beta":
        # Every transition has N examples.  Weighting by transition midpoint
        # gives a discrete approximation to a global Beta time distribution
        # without pairing x_k with data from a different transition.
        transition_midpoints = (
            torch.arange(dataset.num_transitions, dtype=torch.float64) + 0.5
        ) / dataset.num_transitions
        weights = transition_midpoints.pow(time_beta_alpha - 1.0) * (
            1.0 - transition_midpoints
        ).pow(time_beta_beta - 1.0)
        weights = weights.repeat_interleave(dataset.N).double()
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=shuffle,
    )
