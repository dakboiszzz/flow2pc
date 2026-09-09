from pathlib import Path
import torch
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


class ALICFMTeacherWrapper(nn.Module):
    """Wrapper adapting ALI-CFM velocity networks to the Flow2PC teacher interface.

    Accepts calls in the standard Flow2PC format: `teacher(time, path_sample)`
    where `time` is global in [0, 1] and maps it to the interval I_k expected by ALI-CFM.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | nn.Module,
        num_marginals: int = 60,
    ):
        super().__init__()
        self.num_marginals = num_marginals
        if isinstance(checkpoint_path, nn.Module):
            self.velocity_net = checkpoint_path
            return

        path = Path(checkpoint_path)
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            checkpoint = torch.load(path, map_location="cpu")

        if isinstance(checkpoint, nn.Module):
            self.velocity_net = checkpoint
        elif isinstance(checkpoint, dict):
            # Check for module in common keys
            for key in ("ot_cfm_model", "velocity_net", "model", "net"):
                if key in checkpoint and isinstance(checkpoint[key], nn.Module):
                    self.velocity_net = checkpoint[key]
                    return

            # Check for state dict in common keys
            state_dict = None
            for key in ("ot_cfm_model", "velocity_net", "model", "net", "state_dict"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state_dict = checkpoint[key]
                    break
            if state_dict is None:
                state_dict = checkpoint

            self.velocity_net = self._build_mlp_from_state_dict(state_dict)
            self.velocity_net.load_state_dict(state_dict)
        else:
            raise TypeError(f"unexpected checkpoint type: {type(checkpoint)}")

    @staticmethod
    def _build_mlp_from_state_dict(state_dict: dict) -> nn.Module:
        """Reconstruct MLP architecture from state dict layer weights."""
        weight_keys = [k for k in state_dict if k.endswith(".weight")]
        weight_keys.sort()
        if not weight_keys:
            raise ValueError("could not find weight keys in state dict")

        layers = []
        for i, key in enumerate(weight_keys):
            weight = state_dict[key]
            out_features, in_features = weight.shape
            linear = nn.Linear(in_features, out_features)
            layers.append(linear)
            if i < len(weight_keys) - 1:
                layers.append(nn.SELU())

        # If original state dict had 'net.' prefix, wrap in a module that has 'net' attribute
        has_net_prefix = any(k.startswith("net.") for k in state_dict)
        sequential = nn.Sequential(*layers)
        if has_net_prefix:
            class PrefixedMLP(nn.Module):
                def __init__(self, net):
                    super().__init__()
                    self.net = net
                def forward(self, x):
                    return self.net(x)
            return PrefixedMLP(sequential)
        return sequential

    def forward(self, time: torch.Tensor, path_sample: torch.Tensor) -> torch.Tensor:
        """Evaluate velocity field given global time in [0, 1] and path samples."""
        if time.ndim == 0:
            time = time.unsqueeze(0)
        time_col = time[:, None] if time.ndim == 1 else time

        # Map global time \in [0, 1] to interval I_k
        K = self.num_marginals
        k_indices = torch.floor(time_col * (K - 1)).clamp(0, K - 2)
        t_lower = k_indices / (K - 1)
        t_upper = (k_indices + 1.0) / (K - 1)
        interval_time = (time_col - t_lower) / (t_upper - t_lower)

        flat_sample = path_sample.flatten(1)
        cat_input = torch.cat([flat_sample, interval_time], dim=-1)

        try:
            return self.velocity_net(cat_input)
        except TypeError:
            try:
                return self.velocity_net(interval_time, path_sample)
            except TypeError:
                return self.velocity_net(path_sample, interval_time)


def build_teacher(config: dict) -> nn.Module:
    if config.get("name") == "alicfm":
        return ALICFMTeacherWrapper(
            config["checkpoint_path"],
            num_marginals=config.get("num_marginals", 60),
        )
    if config.get("name") != "torchcfm_unet":
        raise ValueError(f"unknown teacher architecture: {config.get('name')}")
    return UNetModel(
        dim=tuple(config["dim"]),
        num_channels=config["num_channels"],
        num_res_blocks=config["num_res_blocks"],
    )

