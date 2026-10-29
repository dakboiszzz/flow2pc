import torch
from torch import nn
from torch.nn import functional as F

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class MNISTClassifier(nn.Module):
    """Classifier from the official PyTorch MNIST example.

    The 128-dimensional activation after ``fc1`` is exposed as the fixed
    domain feature representation used by FID-MNIST, KID-MNIST, and
    precision/recall.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def features(self, values: torch.Tensor) -> torch.Tensor:
        values = F.relu(self.conv1(values))
        values = F.relu(self.conv2(values))
        values = F.max_pool2d(values, 2)
        values = self.dropout1(values)
        values = torch.flatten(values, 1)
        return F.relu(self.fc1(values))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = self.features(values)
        features = self.dropout2(features)
        return F.log_softmax(self.fc2(features), dim=1)


def normalize_for_classifier(images: torch.Tensor) -> torch.Tensor:
    """Convert Flow2PC images in [-1, 1] to official MNIST normalization."""
    pixels = (images.clamp(-1.0, 1.0) + 1.0) * 0.5
    return (pixels - MNIST_MEAN) / MNIST_STD
