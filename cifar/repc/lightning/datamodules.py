"""Lightning DataModules: MNIST (images) and MiniBooNE (tabular).

Both expose `.dim` (flat data dimension) so model configs never hardcode it.
"""

import os
import urllib.request

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from repc.data import TFM

MINIBOONE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00199/MiniBooNE_PID.txt"
)


class MNISTDataModule(L.LightningDataModule):
    dim = 784
    shape = (1, 28, 28)

    def __init__(self, root, batch_size=384, num_workers=4):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        from torchvision.datasets import MNIST

        MNIST(self.root, train=True, download=True)
        MNIST(self.root, train=False, download=True)

    def setup(self, stage=None):
        from torchvision.datasets import MNIST

        self.train_set = MNIST(self.root, train=True, transform=TFM)
        self.test_set = MNIST(self.root, train=False, transform=TFM)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class MiniBooNEDataModule(L.LightningDataModule):
    shape = None  # tabular

    def __init__(
        self, root, batch_size=1024, num_workers=2, train_frac=0.8, test_frac=0.1
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_frac = train_frac
        self.test_frac = test_frac
        self.dim = None

    def prepare_data(self):
        raw = os.path.join(self.root, "MiniBooNE_PID.txt")
        if not os.path.exists(raw):
            os.makedirs(self.root, exist_ok=True)
            urllib.request.urlretrieve(MINIBOONE_URL, raw)

    def setup(self, stage=None):
        cache = os.path.join(self.root, "miniboone_clean.pt")
        if os.path.exists(cache):
            X = torch.load(cache, weights_only=True)
        else:
            raw = os.path.join(self.root, "MiniBooNE_PID.txt")
            with open(raw) as f:
                n_sig, _ = (int(x) for x in f.readline().split())
                X = np.loadtxt(f, max_rows=n_sig).astype(np.float32)
            X = X[(X > -900).all(axis=1)]
            mu, sd = X.mean(0), X.std(0)
            keep = sd > 1e-6
            X = (X[:, keep] - mu[keep]) / sd[keep]
            rng = np.random.default_rng(0)
            rng.shuffle(X)
            X = torch.from_numpy(X)
            torch.save(X, cache)
        self.dim = X.shape[1]
        n = len(X)
        n_train = int(self.train_frac * n)
        n_test = int(self.test_frac * n)
        self.train_set = TensorDataset(X[:n_train])
        self.test_set = TensorDataset(X[n - n_test :])

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)


class CIFAR10DataModule(L.LightningDataModule):
    dim = 3072
    shape = (3, 32, 32)

    def __init__(self, root, batch_size=256, num_workers=4):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        from torchvision.datasets import CIFAR10

        CIFAR10(self.root, train=True, download=True)
        CIFAR10(self.root, train=False, download=True)

    def setup(self, stage=None):
        from torchvision import transforms
        from torchvision.datasets import CIFAR10

        tfm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
        )
        self.train_set = CIFAR10(self.root, train=True, transform=tfm)
        self.test_set = CIFAR10(self.root, train=False, transform=tfm)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class TinyImageNetDataModule(L.LightningDataModule):
    """Tiny ImageNet-200: 100k train images at 64x64x3."""

    dim = 12288
    shape = (3, 64, 64)
    URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    def __init__(self, root, batch_size=128, num_workers=6):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        import zipfile

        target = os.path.join(self.root, "tiny-imagenet-200")
        if os.path.isdir(target):
            return
        zpath = os.path.join(self.root, "tiny-imagenet-200.zip")
        if not os.path.exists(zpath):
            urllib.request.urlretrieve(self.URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(self.root)

    def setup(self, stage=None):
        from torchvision import transforms
        from torchvision.datasets import ImageFolder

        tfm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
        )
        base = os.path.join(self.root, "tiny-imagenet-200")
        full = ImageFolder(os.path.join(base, "train"), transform=tfm)
        n_val = 10000
        g = torch.Generator().manual_seed(0)
        self.train_set, self.test_set = torch.utils.data.random_split(
            full, [len(full) - n_val, n_val], generator=g
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
