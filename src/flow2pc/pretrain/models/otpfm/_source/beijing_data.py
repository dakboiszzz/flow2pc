"""Original coupling/dataloader code; raw loading stays in flow2pc.data."""

import logging
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)

ALL_TIMES = list(range(13))
DEFAULT_HOLDOUT_TIMES = [2, 5, 8, 11]
DEFAULT_TRAIN_TIMES = [t for t in ALL_TIMES if t not in DEFAULT_HOLDOUT_TIMES]
METRIC_TIMES = [f"t{i}" for i in ALL_TIMES if i != 0] + ["rest"]

def compute_beijing_ot_alignments(
    marginals: list[np.ndarray],
    train_times: list[int],
    method: str = "emd",
    reg: float = 0.01,
) -> dict:
    """Compute OT alignments between consecutive training times."""
    try:
        import ot
    except ImportError:
        logger.warning("POT not installed, skipping OT alignment computation")
        return None

    alignments = {}
    for i in range(len(train_times) - 1):
        t_src, t_tgt = train_times[i], train_times[i + 1]
        logger.info(f"Computing OT alignment from t={t_src} to t={t_tgt}...")

        source = marginals[t_src]
        target = marginals[t_tgt]
        cost = ot.dist(source, target, metric="sqeuclidean")
        a = np.ones(len(source)) / len(source)
        b = np.ones(len(target)) / len(target)

        if method == "emd":
            plan = ot.emd(a, b, cost)
        else:
            plan = ot.sinkhorn(a, b, cost, reg=reg)

        alignments[(t_src, t_tgt)] = np.argmax(plan, axis=1)

    return alignments


class BeijingMultiMarginalDataset(Dataset):
    """PyTorch Dataset for Beijing PM2.5 multi-marginal training."""

    def __init__(
        self,
        marginals: list[np.ndarray],
        train_times: list[int] | None = None,
        ot_alignments: dict | None = None,
    ):
        self.all_times = list(range(len(marginals)))
        self.train_times = train_times or DEFAULT_TRAIN_TIMES
        self.ot_alignments = ot_alignments
        self.use_ot = ot_alignments is not None

        self.data_by_time = {
            t: torch.tensor(marginals[t], dtype=torch.float32) for t in self.train_times
        }

        if self.use_ot:
            self._build_ot_chains()
            self.dataset_size = len(self.data_by_time[self.train_times[0]])
        else:
            self.dataset_size = min(len(d) for d in self.data_by_time.values())
            self.indices = {t: torch.randperm(len(self.data_by_time[t])) for t in self.train_times}

    def _build_ot_chains(self):
        normalized = {(int(k[0]), int(k[1])): v for k, v in self.ot_alignments.items()}
        n_source = len(self.data_by_time[self.train_times[0]])
        n_times = len(self.train_times)
        self.ot_chains = np.zeros((n_source, n_times), dtype=np.int64)
        self.ot_chains[:, 0] = np.arange(n_source)

        for t_idx in range(n_times - 1):
            t_src, t_tgt = self.train_times[t_idx], self.train_times[t_idx + 1]
            mapping = normalized[(t_src, t_tgt)]
            for src_idx in range(n_source):
                prev_idx = self.ot_chains[src_idx, t_idx]
                self.ot_chains[src_idx, t_idx + 1] = mapping[prev_idx]

        self.chain_perm = np.random.permutation(n_source)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        samples = []
        if self.use_ot:
            chain_idx = self.chain_perm[idx % len(self.chain_perm)]
            for t_idx, t in enumerate(self.train_times):
                samples.append(self.data_by_time[t][self.ot_chains[chain_idx, t_idx]])
        else:
            for t in self.train_times:
                samples.append(
                    self.data_by_time[t][self.indices[t][idx % len(self.data_by_time[t])]]
                )
        return samples

    def reshuffle(self):
        if self.use_ot:
            self.chain_perm = np.random.permutation(len(self.chain_perm))
        else:
            self.indices = {t: torch.randperm(len(self.data_by_time[t])) for t in self.train_times}

    def get_ot_aligned_samples(self, n_samples: int = 5) -> Tensor:
        if not self.use_ot:
            raise ValueError("OT coupling not enabled")
        n_times = len(self.train_times)
        n_samples = min(n_samples, len(self.ot_chains))
        dim = self.data_by_time[self.train_times[0]].shape[-1]
        samples = torch.zeros(n_samples, n_times, dim)
        for i in range(n_samples):
            for t_idx, t in enumerate(self.train_times):
                samples[i, t_idx] = self.data_by_time[t][self.ot_chains[i, t_idx]]
        return samples


def create_beijing_dataloaders(
    marginals: list[np.ndarray],
    batch_size: int = 128,
    holdout_times: list[int] | None = None,
    val_split: float = 0.0,
    ot_alignments: dict | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create train/val DataLoaders for Beijing dataset."""
    train_times = [t for t in ALL_TIMES if t not in (holdout_times or DEFAULT_HOLDOUT_TIMES)]
    dataset = BeijingMultiMarginalDataset(
        marginals, train_times=train_times, ot_alignments=ot_alignments
    )

    if val_split > 0:
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
    else:
        train_dataset = dataset
        val_dataset = dataset

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader
