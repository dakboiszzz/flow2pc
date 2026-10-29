import math

import numpy as np
import torch
from scipy import linalg


def frechet_distance(real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
    """Standard Fréchet distance between Gaussian feature approximations."""
    real = real_features.double().numpy()
    fake = fake_features.double().numpy()
    real_mean = np.mean(real, axis=0)
    fake_mean = np.mean(fake, axis=0)
    real_covariance = np.cov(real, rowvar=False)
    fake_covariance = np.cov(fake, rowvar=False)
    difference = real_mean - fake_mean
    covariance_mean = linalg.sqrtm(real_covariance @ fake_covariance)
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(real_covariance.shape[0]) * 1e-6
        covariance_mean = linalg.sqrtm(
            (real_covariance + offset) @ (fake_covariance + offset)
        )
    if np.iscomplexobj(covariance_mean):
        if not np.allclose(np.diag(covariance_mean).imag, 0.0, atol=1e-3):
            raise ValueError("FID covariance square root has a complex component")
        covariance_mean = covariance_mean.real
    score = (
        difference @ difference
        + np.trace(real_covariance)
        + np.trace(fake_covariance)
        - 2.0 * np.trace(covariance_mean)
    )
    return float(max(score, 0.0))


def polynomial_mmd(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
) -> torch.Tensor:
    """Unbiased squared MMD with the standard cubic KID kernel."""
    sample_count = real_features.shape[0]
    feature_dim = real_features.shape[1]
    real_kernel = (real_features @ real_features.T / feature_dim + 1.0).pow(3)
    fake_kernel = (fake_features @ fake_features.T / feature_dim + 1.0).pow(3)
    cross_kernel = (real_features @ fake_features.T / feature_dim + 1.0).pow(3)
    real_sum = real_kernel.sum() - real_kernel.diagonal().sum()
    fake_sum = fake_kernel.sum() - fake_kernel.diagonal().sum()
    return (real_sum + fake_sum) / (
        sample_count * (sample_count - 1)
    ) - 2.0 * cross_kernel.mean()


@torch.no_grad()
def kernel_distance(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    device: torch.device,
    subsets: int = 100,
    subset_size: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """KID mean/std over deterministic random subsets (TorchMetrics defaults)."""
    subset_size = min(subset_size, len(real_features), len(fake_features))
    if subset_size < 2:
        raise ValueError("KID needs at least two samples")
    generator = torch.Generator().manual_seed(seed)
    scores = []
    for _ in range(subsets):
        real_indices = torch.randperm(len(real_features), generator=generator)[
            :subset_size
        ]
        fake_indices = torch.randperm(len(fake_features), generator=generator)[
            :subset_size
        ]
        real_subset = real_features[real_indices].to(device)
        fake_subset = fake_features[fake_indices].to(device)
        scores.append(polynomial_mmd(real_subset, fake_subset).cpu())
    values = torch.stack(scores)
    return values.mean().item(), values.std(unbiased=False).item()


def _manifold_radii(
    features: torch.Tensor,
    nearest_neighbors: int,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    reference = features.to(device)
    squared_norm = reference.square().sum(dim=1)
    radii = []
    for start in range(0, len(features), chunk_size):
        query = reference[start : start + chunk_size]
        distances = (
            query.square().sum(dim=1, keepdim=True)
            + squared_norm[None]
            - 2.0 * query @ reference.T
        ).clamp_min_(0.0)
        radii.append(distances.kthvalue(nearest_neighbors + 1, dim=1).values.cpu())
    return torch.cat(radii)


def _fraction_inside_manifold(
    query_features: torch.Tensor,
    reference_features: torch.Tensor,
    reference_radii: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> float:
    reference = reference_features.to(device)
    radii = reference_radii.to(device)
    squared_norm = reference.square().sum(dim=1)
    inside = 0
    for start in range(0, len(query_features), chunk_size):
        query = query_features[start : start + chunk_size].to(device)
        distances = (
            query.square().sum(dim=1, keepdim=True)
            + squared_norm[None]
            - 2.0 * query @ reference.T
        ).clamp_min_(0.0)
        inside += (distances <= radii[None]).any(dim=1).sum().item()
    return inside / len(query_features)


@torch.no_grad()
def improved_precision_recall(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    device: torch.device,
    nearest_neighbors: int = 3,
    chunk_size: int = 512,
) -> tuple[float, float]:
    """Kynkäänniemi et al. improved precision and recall."""
    real_radii = _manifold_radii(
        real_features,
        nearest_neighbors,
        device,
        chunk_size,
    )
    fake_radii = _manifold_radii(
        fake_features,
        nearest_neighbors,
        device,
        chunk_size,
    )
    precision = _fraction_inside_manifold(
        fake_features,
        real_features,
        real_radii,
        device,
        chunk_size,
    )
    recall = _fraction_inside_manifold(
        real_features,
        fake_features,
        fake_radii,
        device,
        chunk_size,
    )
    return precision, recall


def inception_score(
    probabilities: torch.Tensor,
    splits: int = 10,
) -> tuple[float, float]:
    scores = []
    for split_probabilities in probabilities.tensor_split(splits):
        marginal = split_probabilities.mean(dim=0, keepdim=True)
        kl = split_probabilities * (
            split_probabilities.clamp_min(1e-12).log() - marginal.clamp_min(1e-12).log()
        )
        scores.append(kl.sum(dim=1).mean().exp())
    values = torch.stack(scores)
    return values.mean().item(), values.std(unbiased=False).item()


def class_distribution_metrics(
    probabilities: torch.Tensor,
    real_labels: torch.Tensor,
) -> dict[str, float | list[float]]:
    predicted = probabilities.argmax(dim=1)
    fake_distribution = torch.bincount(predicted, minlength=10).float()
    fake_distribution /= fake_distribution.sum()
    real_distribution = torch.bincount(real_labels, minlength=10).float()
    real_distribution /= real_distribution.sum()
    midpoint = 0.5 * (fake_distribution + real_distribution)

    def kl(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        mask = left > 0
        return (left[mask] * (left[mask] / right[mask]).log()).sum()

    entropy = -(
        fake_distribution
        * fake_distribution.clamp_min(torch.finfo(fake_distribution.dtype).tiny).log()
    ).sum()
    return {
        "classifier_confidence": probabilities.max(dim=1).values.mean().item(),
        "class_entropy_normalized": (entropy / math.log(10.0)).item(),
        "class_jsd_to_test": (
            0.5 * kl(fake_distribution, midpoint)
            + 0.5 * kl(real_distribution, midpoint)
        ).item(),
        "class_distribution": fake_distribution.tolist(),
    }


@torch.no_grad()
def nearest_real_indices(
    fake_features: torch.Tensor,
    real_features: torch.Tensor,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    fake = fake_features[:count].to(device)
    real = real_features.to(device)
    distances = (
        fake.square().sum(dim=1, keepdim=True)
        + real.square().sum(dim=1)[None]
        - 2.0 * fake @ real.T
    )
    return distances.argmin(dim=1).cpu()
