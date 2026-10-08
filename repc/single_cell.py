"""Single-cell checkpoint/data contract and reproducible ALI-CFM sampling.

This module does not import the external ALI repository. H5ADs are read in
place, and all transformations come from the supplied combined checkpoint.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import h5py
import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import torch

from repc.flow import ALICFMGeneratorWrapper, ALICFMTeacherWrapper
from repc.hclt_pc import TimeDependentHCLTPC


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER = ROOT.parent / "ali-cfm/runs/single_cell/cite_5D/wandb/run-20261003_030825-rrcutrtl/files/checkpoints/t=1_seed=42_ali_cfm.pth"
DEFAULT_DATA_DIR = ROOT.parent / "ali-cfm/data/single_cell"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@dataclass
class SingleCellData:
    clouds: list[torch.Tensor]
    train: list[torch.Tensor]
    validation: list[torch.Tensor]
    contract: dict
    identity: dict
    heldout: int
    observed: list[int]
    minimum: torch.Tensor
    maximum: torch.Tensor

    @property
    def dim(self) -> int:
        return self.clouds[0].shape[1]

    def metric_coordinates(self, values: torch.Tensor) -> torch.Tensor:
        """Undo min/max only; 5D paper metrics retain standardization."""
        if not self.contract["preprocessing"].get("normalise", False):
            return values
        return values * (self.maximum - self.minimum).to(values) + self.minimum.to(values)


def load_single_cell_data(
    checkpoint_path: Path,
    data_dir: Path,
    validation_fraction: float = 0.05,
    split_seed: int = 2026,
) -> SingleCellData:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    contract = checkpoint["data_contract"]
    preprocessing = contract["preprocessing"]
    manifest_path = data_dir / "data_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_hash != contract["manifest_sha256"]:
        raise ValueError("data manifest differs from the teacher's training manifest")
    source = contract["source"]
    manifest = json.loads(manifest_bytes)
    entry = next(x for x in manifest["datasets"] if x["dataset"] == source["dataset"])
    if entry["sha256"] != source["sha256"]:
        raise ValueError("checkpoint and manifest identify different source data")
    input_path = data_dir / Path(source["file"]).name
    if input_path.stat().st_size != source["file_size_bytes"]:
        raise ValueError("source H5AD size differs from the checkpoint")
    if sha256_file(input_path) != source["sha256"]:
        raise ValueError("source H5AD checksum differs from the checkpoint")
    dim = int(preprocessing["n_pca_components_used"])
    with h5py.File(input_path, "r") as h5:
        pca = h5["obsm/X_pca"][:, :dim]
        day_node = h5["obs/day"]
        if isinstance(day_node, h5py.Group):
            codes = day_node["codes"][:]
            if (codes < 0).any():
                raise ValueError("day labels contain missing values")
            days = day_node["categories"][:][codes]
        else:
            days = day_node[:]
    if not np.isfinite(pca).all():
        raise ValueError("non-finite PCA coordinates")
    if preprocessing.get("whiten", False):
        scaler = preprocessing["whitening"]
        # The original StandardScaler applies each operation in place to float32.
        pca = pca.astype(np.float32, copy=True)
        pca -= np.asarray(scaler["mean"])
        pca /= np.asarray(scaler["scale"])
    order = preprocessing["time_order"]
    if order != source["day_order_used_by_loader"]:
        raise ValueError("checkpoint day order differs from its data manifest")
    clouds = [torch.from_numpy(pca[days == day].copy()).float() for day in order]
    if [len(x) for x in clouds] != preprocessing["cells_per_time"]:
        raise ValueError("population counts differ from the checkpoint")
    if any(len(x) < 2 for x in clouds):
        raise ValueError("each day requires at least two cells")
    minimum, maximum = torch.zeros(dim), torch.ones(dim)
    if preprocessing.get("normalise", False):
        minimum = torch.tensor(preprocessing["min_max"]["min"], dtype=torch.float32)
        maximum = torch.tensor(preprocessing["min_max"]["max"], dtype=torch.float32)
        if (maximum <= minimum).any():
            raise ValueError("normalization has a non-positive range")
        clouds = [(x - minimum) / (maximum - minimum) for x in clouds]
    heldout = int(contract["time_index"])
    if not 0 < heldout < len(clouds) - 1:
        raise ValueError("checkpoint must hold out an interior population")
    observed = [i for i in range(len(clouds)) if i != heldout]
    if [order[i] for i in observed] != contract["observed_days"]:
        raise ValueError("observed populations do not match the fold")
    rng = torch.Generator().manual_seed(split_seed)
    train, validation = [], []
    for index, cloud in enumerate(clouds):
        if index == heldout:
            train.append(cloud[:0])
            validation.append(cloud[:0])
        else:
            permutation = torch.randperm(len(cloud), generator=rng)
            n_validation = max(1, min(len(cloud) - 1, round(len(cloud) * validation_fraction)))
            validation.append(cloud[permutation[:n_validation]])
            train.append(cloud[permutation[n_validation:]])
    identity = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "data_file": str(input_path.resolve()),
        "data_sha256": source["sha256"],
        "manifest_sha256": manifest_hash,
        "validation_fraction": validation_fraction,
        "split_seed": split_seed,
        "metric_space": "inverse_minmax_standardized_pca" if preprocessing.get("whiten") else "inverse_minmax_pca",
    }
    return SingleCellData(clouds, train, validation, contract, identity, heldout,
                          observed, minimum, maximum)


def load_single_cell_models(checkpoint: Path, device: torch.device):
    teacher = ALICFMTeacherWrapper(checkpoint, map_to_interval=False).to(device).eval()
    generator = ALICFMGeneratorWrapper(checkpoint).to(device).eval()
    teacher.requires_grad_(False)
    generator.requires_grad_(False)
    return teacher, generator


def draw(pool: torch.Tensor, count: int, rng: torch.Generator) -> torch.Tensor:
    if not len(pool):
        raise ValueError("attempted to sample an excluded population")
    return pool[torch.randint(len(pool), (count,), generator=rng)]


def mmot_endpoints(x0: torch.Tensor, middle: torch.Tensor, x1: torch.Tensor):
    """Exact equal-weight minibatch OT, chained through an observed marginal.

    An equal-size uniform OT plan has an optimal permutation. Indexing both
    endpoint assignments by the middle cells implements the conditional
    factorization pi(x0|middle) pi(x1|middle), without using held-out cells.
    """
    row0, column_middle = linear_sum_assignment(cdist(x0.numpy(), middle.numpy(), "sqeuclidean"))
    row_middle, column1 = linear_sum_assignment(cdist(middle.numpy(), x1.numpy(), "sqeuclidean"))
    source_indices, target_indices = np.empty(len(middle), dtype=int), np.empty(len(middle), dtype=int)
    source_indices[column_middle] = row0
    target_indices[row_middle] = column1
    return x0[source_indices], x1[target_indices]


def sample_paths(data: SingleCellData, pools: list[torch.Tensor], generator,
                 count: int, rng: torch.Generator, device: torch.device):
    interior = [i for i in data.observed if 0 < i < len(data.clouds) - 1]
    index = interior[int(torch.randint(len(interior), (), generator=rng))]
    x0, x1 = mmot_endpoints(draw(pools[0], count, rng), draw(pools[index], count, rng),
                            draw(pools[-1], count, rng))
    times = torch.rand(count, generator=rng).to(device)
    with torch.no_grad():
        samples = generator(x0.to(device), x1.to(device), times)
    return samples, times


def observed_batch(data: SingleCellData, pools: list[torch.Tensor], count: int,
                   rng: torch.Generator, device: torch.device):
    positions = torch.randint(len(data.observed), (count,), generator=rng)
    values = torch.empty(count, data.dim)
    times = torch.empty(count)
    for j, index in enumerate(data.observed):
        selected = positions == j
        values[selected] = draw(pools[index], int(selected.sum()), rng)
        times[selected] = index / (len(data.clouds) - 1)
    return values.to(device), times.to(device)


def learn_topology(data: SingleCellData, num_bins: int = 32) -> list[int]:
    """Histogram MI Chow--Liu tree fitted solely on observed training cells."""
    values = torch.cat([data.train[i] for i in data.observed]).numpy()
    graph = nx.Graph()
    graph.add_nodes_from(range(data.dim))
    for i in range(data.dim):
        for j in range(i):
            counts = np.histogram2d(values[:, i], values[:, j], bins=num_bins)[0]
            joint = counts / counts.sum()
            independent = joint.sum(0)[None, :] * joint.sum(1)[:, None]
            mask = joint > 0
            mi = float((joint[mask] * np.log(joint[mask] / independent[mask])).sum())
            graph.add_edge(i, j, weight=mi)
    parents = [-1] * data.dim
    for parent, child in nx.bfs_edges(nx.maximum_spanning_tree(graph), 0):
        parents[child] = parent
    return parents


def build_single_cell_student(config: dict) -> TimeDependentHCLTPC:
    if config.get("name") != "hclt" or config.get("fourier_zero_center", True):
        raise ValueError("single-cell students require an HCLT with a free source marginal")
    return TimeDependentHCLTPC(
        parents=config["parents"], num_latents=config["num_latents"],
        time_conditioning="fourier", time_embedding_dim=config["time_embedding_dim"],
        time_feature_seed=config["time_feature_seed"], fourier_zero_center=False,
    )


@torch.no_grad()
def initialize_source(student: TimeDependentHCLTPC, source: torch.Tensor,
                      rng: torch.Generator) -> None:
    means = draw(source, student.num_latents, rng).T.to(student.mean_coefficients)
    log_scale = (source.std(0).clamp_min(0.01) * 0.5).log().to(means)
    basis, _ = student._time_basis(torch.tensor(0.0, device=means.device))
    direction = basis / basis.square().sum()
    student.mean_coefficients.copy_(means[..., None] * direction)
    student.log_scale_coefficients.copy_(log_scale[:, None, None] * direction)


@torch.no_grad()
def validation_metrics(student, teacher, generator, data: SingleCellData,
                       device: torch.device, seed: int, batches: int = 8,
                       batch_size: int = 128) -> dict:
    was_training = student.training
    student.eval()
    rng = torch.Generator().manual_seed(seed)
    squared_error = teacher_energy = nll = 0.0
    for _ in range(batches):
        samples, times = sample_paths(data, data.validation, generator, batch_size, rng, device)
        target = teacher(times, samples)
        predicted = student.velocity(samples, times)
        squared_error += (predicted - target).square().mean().item()
        teacher_energy += target.square().mean().item()
        observed, observed_times = observed_batch(data, data.validation, batch_size, rng, device)
        nll -= student.log_prob(observed, observed_times).mean().item() / data.dim
    student.train(was_training)
    return {"velocity_mse": squared_error / batches,
            "relative_velocity_mse": squared_error / max(teacher_energy, 1e-12),
            "observed_nll_nats_per_dim": nll / batches}
