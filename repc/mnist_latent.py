"""Frozen MNIST EQ-VAE representation and continuous latent data contract."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import norm, rankdata
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from repc.hclt import parents_from_mutual_information


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def write_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_eqvae(root, checkpoint_path, device, expected_sha256=None):
    root, checkpoint_path = Path(root).resolve(), Path(checkpoint_path).resolve()
    if expected_sha256 and sha256_file(checkpoint_path) != expected_sha256:
        raise ValueError("EQ-VAE checkpoint differs from the exported representation")
    source = root / "train_eqvae" / "train_mnist.py"
    # Use a unique module name; do not collide with Flow2PC's training module.
    for directory in (root, source.parent):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("flow2pc_frozen_mnist_eqvae", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    vae = module.MNISTEQVAE(config["channels"], config["latent_channels"])
    vae.load_state_dict(checkpoint["state_dict"], strict=True)
    return vae.to(device).eval().requires_grad_(False)


def normalize_codes(train_z, validation_z, test_z):
    """Fit only on training posterior means; never clip or quantize codes."""
    mean = train_z.mean(dim=0, keepdim=True)
    raw_scale = train_z.std(dim=0, unbiased=False, keepdim=True)
    scale = raw_scale.clamp_min(1e-6)
    values = [(z - mean) / scale for z in (train_z, validation_z, test_z)]
    if not all(torch.isfinite(z).all() for z in values):
        raise ValueError("non-finite encoded MNIST values")
    return values, mean, scale, raw_scale


def latent_dimension(manifest):
    """Validate the CHW representation for the original 28x28 MNIST images."""
    shape = manifest.get("latent_shape")
    if (not isinstance(shape, list) or len(shape) != 3
            or any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in shape)
            or shape[1:] != [7, 7]
            or manifest.get("dim") != math.prod(shape)
            or manifest.get("encode_mode") != "posterior_mean"
            or manifest.get("flatten_order") != "CHW"):
        raise ValueError("Unsupported latent representation: expected consistent [channels, 7, 7] posterior means")
    return math.prod(shape)


@torch.no_grad()
def export_latents(args, device):
    output = args.bundle_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new bundle directory; {output} is not empty")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.vae_checkpoint.resolve()
    checkpoint_hash = sha256_file(checkpoint_path)
    vae = load_eqvae(args.eqvae_root, checkpoint_path, device, checkpoint_hash)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    datasets = [MNIST(args.data_dir, train=train, transform=transform, download=args.download)
                for train in (True, False)]
    encoded = []
    latent_shape = None
    for dataset in datasets:
        batches = []
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
        for images, _ in loader:
            z = vae.encode(images.to(device)).mode()
            shape = list(z.shape[1:])
            if len(shape) != 3 or shape[1:] != [7, 7]:
                raise ValueError(f"Expected [channels, 7, 7] EQ-VAE codes, got {shape}")
            if latent_shape is not None and shape != latent_shape:
                raise ValueError("EQ-VAE latent shape changed between batches")
            latent_shape = shape
            batches.append(z.flatten(1).cpu())
        encoded.append(torch.cat(batches))
    train_subset, validation_subset = random_split(
        datasets[0], [55000, 5000], generator=torch.Generator().manual_seed(0)
    )
    indices = [torch.tensor(train_subset.indices), torch.tensor(validation_subset.indices),
               torch.arange(len(datasets[1]))]
    values, mean, scale, raw_scale = normalize_codes(
        encoded[0][indices[0]], encoded[0][indices[1]], encoded[1]
    )
    codes = {}
    for split, y, index, dataset in zip(
        ("train", "validation", "test"), values, indices,
        (datasets[0], datasets[0], datasets[1]), strict=True
    ):
        codes[split] = {"values": y, "labels": dataset.targets[index].clone(), "indices": index}
    normalizer = {"mean": mean, "scale": scale, "latent_shape": latent_shape, "flatten_order": "CHW"}
    atomic_save(codes, output / "codes.pt")
    atomic_save(normalizer, output / "normalizer.pt")
    dim = math.prod(latent_shape)
    manifest = {
        "format_version": 1, "dataset": "mnist-latent", "dim": dim,
        "encoder_checkpoint": str(checkpoint_path), "encoder_sha256": checkpoint_hash,
        "eqvae_root": str(args.eqvae_root.resolve()), "encode_mode": "posterior_mean",
        "latent_shape": latent_shape, "flatten_order": "CHW",
        "preprocessing": "original MNIST; ToTensor; Normalize(mean=0.5,std=0.5); no augmentation",
        "split": {"method": "torch.utils.data.random_split", "seed": 0,
                  "lengths": [55000, 5000, 10000], "indices_file": "codes.pt",
                  "indices_sha256": [hashlib.sha256(index.numpy().tobytes()).hexdigest() for index in indices],
                  "test": "official held-out test split; only final evaluation"},
        "normalization": {"fit": "train_only", "std_unbiased": False, "minimum_scale": 1e-6,
                          "raw_scale_min": raw_scale.min().item(),
                          "small_scale_coordinates": (raw_scale < 1e-6).sum().item(),
                          "train_abs_max": values[0].abs().max().item(),
                          "train_abs_q999": torch.quantile(values[0].abs().flatten(), 0.999).item()},
        "codes_sha256": sha256_file(output / "codes.pt"),
        "normalizer_sha256": sha256_file(output / "normalizer.pt"),
        "time_convention": f"t=0: N(0,I_{dim}); t=1: normalized encoded original MNIST",
    }
    write_json(manifest, output / "manifest.json")
    return manifest


def load_latent_bundle(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, key in (("codes.pt", "codes_sha256"), ("normalizer.pt", "normalizer_sha256")):
        if sha256_file(directory / name) != manifest[key]:
            raise ValueError(f"Latent bundle checksum mismatch: {name}")
    dim = latent_dimension(manifest)
    codes = torch.load(directory / "codes.pt", map_location="cpu", weights_only=True)
    normalizer = torch.load(directory / "normalizer.pt", map_location="cpu", weights_only=True)
    if (normalizer.get("latent_shape") != manifest["latent_shape"] or normalizer.get("flatten_order") != "CHW"
            or normalizer["mean"].shape != (1, dim) or normalizer["scale"].shape != (1, dim)
            or not torch.isfinite(normalizer["mean"]).all()
            or not torch.isfinite(normalizer["scale"]).all() or (normalizer["scale"] <= 0).any()):
        raise ValueError("Invalid latent normalization")
    for split, count in zip(("train", "validation", "test"), (55000, 5000, 10000), strict=True):
        item = codes[split]
        if item["values"].shape != (count, dim) or not torch.isfinite(item["values"]).all():
            raise ValueError(f"Invalid {split} codes")
        if len(item["indices"]) != count or len(item["labels"]) != count:
            raise ValueError(f"Invalid {split} labels/indices")
    contract = {"bundle_dir": str(directory), "manifest": manifest,
                "manifest_sha256": sha256_file(directory / "manifest.json"), **normalizer}
    return codes, contract


def check_contract(saved, current):
    if saved["manifest_sha256"] != current["manifest_sha256"]:
        raise ValueError("Teacher/student and data bundle use different latent representations")
    for key in ("mean", "scale"):
        if not torch.equal(saved[key].cpu(), current[key].cpu()):
            raise ValueError(f"Checkpoint {key} differs from the latent bundle")


@torch.no_grad()
def decode_codes(vae, values, contract):
    mean, scale = (contract[key].to(values) for key in ("mean", "scale"))
    z = values * scale + mean
    return vae.decode(z.reshape(-1, *contract["latent_shape"]))


def learn_latent_topology(values, num_samples, seed):
    """Gaussian-copula MI on continuous codes; no pixel histogram/range settings.

    Ranks are transformed to normal scores. Pairwise Gaussian MI approximates
    dependence and is invariant to coordinate scales and robust to outliers.
    Only structure learning uses these scores; PC leaves see original codes.
    """
    indices = torch.randperm(len(values), generator=torch.Generator().manual_seed(seed))[:num_samples]
    data = values[indices].numpy()
    ranks = rankdata(data, axis=0, method="average")
    scores = norm.ppf((ranks - 0.5) / len(data))
    scores -= scores.mean(axis=0, keepdims=True)
    scales = scores.std(axis=0, keepdims=True)
    scores /= np.maximum(scales, 1e-12)
    correlation = scores.T @ scores / len(scores)
    mi = -0.5 * np.log1p(-np.minimum(correlation**2, 1 - 1e-7))
    np.fill_diagonal(mi, 0)
    parents = parents_from_mutual_information(torch.from_numpy(mi))
    return parents, {"estimator": "pairwise_gaussian_copula_mutual_information",
                     "num_samples": len(data), "seed": seed,
                     "fit_split": "train", "constant_coordinates": int((scales < 1e-12).sum())}
