from torch import nn
from torchcfm.models.unet import UNetModel


def torchcfm_mnist_teacher_config() -> dict:
    """Return the architecture used by TorchCFM's official MNIST example."""
    return {
        "name": "torchcfm_unet",
        "dim": [1, 28, 28],
        "num_channels": 32,
        "num_res_blocks": 1,
    }


def build_teacher(config: dict) -> nn.Module:
    if config.get("name") != "torchcfm_unet":
        raise ValueError(f"unknown teacher architecture: {config.get('name')}")
    return UNetModel(
        dim=tuple(config["dim"]),
        num_channels=config["num_channels"],
        num_res_blocks=config["num_res_blocks"],
    )
