"""Original OTP-FM unpaired/OT tuple dataset; loaders stay in shared data.

Author(s): Raghav Kansal
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class EBMultiMarginalDataset(Dataset):
    """
    PyTorch Dataset for EB multi-marginal training.

    Each sample contains one cell from each training time point,
    either with independent or OT-coupled pairing.
    """

    def __init__(
        self,
        pcs: np.ndarray,
        labels: np.ndarray,
        holdout_times: list[int] | None = None,
        ot_alignments: dict | None = None,
    ):
        self.times = sorted(set(labels))
        self.holdout_times = holdout_times or []
        self.train_times = [t for t in self.times if t not in self.holdout_times]
        self.ot_alignments = ot_alignments
        self.use_ot = ot_alignments is not None

        # Group data by time
        self.data_by_time = {}
        for t in self.train_times:
            mask = labels == t
            self.data_by_time[t] = torch.tensor(pcs[mask], dtype=torch.float32)

        # Build OT chains if using OT coupling
        if self.use_ot:
            self._build_ot_chains()
            self.dataset_size = len(self.data_by_time[self.train_times[0]])
        else:
            self.dataset_size = min(len(d) for d in self.data_by_time.values())
            self.indices = {t: torch.randperm(len(self.data_by_time[t])) for t in self.train_times}

    def _build_ot_chains(self):
        """Build OT coupling chains."""
        n_source = len(self.data_by_time[self.train_times[0]])
        n_times = len(self.train_times)
        self.ot_chains = np.zeros((n_source, n_times), dtype=np.int64)
        self.ot_chains[:, 0] = np.arange(n_source)

        normalized_alignments = {(int(k[0]), int(k[1])): v for k, v in self.ot_alignments.items()}

        for t_idx in range(n_times - 1):
            t_src, t_tgt = self.train_times[t_idx], self.train_times[t_idx + 1]
            mapping = normalized_alignments[(t_src, t_tgt)]
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
                actual_idx = self.ot_chains[chain_idx, t_idx]
                samples.append(self.data_by_time[t][actual_idx])
        else:
            for t in self.train_times:
                actual_idx = self.indices[t][idx % len(self.data_by_time[t])]
                samples.append(self.data_by_time[t][actual_idx])
        return samples

    def reshuffle(self):
        """Reshuffle for new epoch."""
        if self.use_ot:
            self.chain_perm = np.random.permutation(len(self.chain_perm))
        else:
            self.indices = {t: torch.randperm(len(self.data_by_time[t])) for t in self.train_times}

    def get_ot_aligned_samples(self, n_samples: int = 5) -> torch.Tensor:
        """Get OT-aligned samples across all training times."""
        if not self.use_ot:
            raise ValueError("Dataset was created without OT alignments")

        n_times = len(self.train_times)
        n_samples = min(n_samples, len(self.ot_chains))
        dim = self.data_by_time[self.train_times[0]].shape[-1]

        samples = torch.zeros(n_samples, n_times, dim)
        for i in range(n_samples):
            for t_idx, t in enumerate(self.train_times):
                cell_idx = self.ot_chains[i, t_idx]
                samples[i, t_idx] = self.data_by_time[t][cell_idx]

        return samples
