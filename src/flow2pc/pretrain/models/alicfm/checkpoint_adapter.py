"""Existing ALI-CFM checkpoint adapters retained from the real-data flow.py.

No MNIST or TorchCFM image-model code is imported here. The wrapper APIs and
checkpoint/time semantics are retained; use saved preprocessing and time metadata.
"""

import re
from pathlib import Path

import torch
from torch import nn


class ALICFMTeacherWrapper(nn.Module):
    """Wrapper adapting ALI-CFM velocity networks to the Flow2PC teacher interface.

    Accepts calls in the standard Flow2PC format: `teacher(time, path_sample)`
    where `time` is global in [0, 1] and maps it to the interval I_k expected by ALI-CFM.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | nn.Module,
        num_marginals: int = 60,
        map_to_interval: bool = True,
    ):
        super().__init__()
        self.num_marginals = num_marginals
        self.map_to_interval = map_to_interval
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
            if (
                checkpoint.get("time_format") == "global"
                or checkpoint.get("model_config", {}).get("time_format") == "global"
                or checkpoint.get("data_contract", {}).get("pseudo_time_convention")
                == "equally_spaced_ordered_marginals_on_[0,1]"
            ):
                self.map_to_interval = False
            # Check for module in common keys
            for key in ("ot_cfm_model", "velocity_net", "model", "net"):
                if key in checkpoint and isinstance(checkpoint[key], nn.Module):
                    self.velocity_net = checkpoint[key]
                    return

            # Check for state dict in common keys
            state_dict = None
            for key in ("velocity_state_dict", "ot_cfm_model", "velocity_net", "model", "net", "state_dict"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state_dict = checkpoint[key]
                    break
            if state_dict is None:
                state_dict = checkpoint

            activation = checkpoint.get("model_config", {}).get(
                "activation", "selu"
            )
            self.velocity_net = self._build_mlp_from_state_dict(
                state_dict, activation=activation
            )
            self.velocity_net.load_state_dict(state_dict)
        else:
            raise TypeError(f"unexpected checkpoint type: {type(checkpoint)}")

    @staticmethod
    def _build_mlp_from_state_dict(
        state_dict: dict,
        activation: str = "selu",
    ) -> nn.Module:
        """Reconstruct MLP architecture from state dict layer weights."""
        weight_keys = [k for k in state_dict if k.endswith(".weight")]
        weight_keys.sort(
            key=lambda key: [int(part) if part.isdigit() else part
                             for part in re.split(r"(\d+)", key)]
        )
        if not weight_keys:
            raise ValueError("could not find weight keys in state dict")

        activation_layers = {
            "elu": nn.ELU,
            "selu": nn.SELU,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }
        if activation not in activation_layers:
            raise ValueError(f"unsupported checkpoint activation: {activation}")
        activation_layer = activation_layers[activation]

        layers = []
        for i, key in enumerate(weight_keys):
            weight = state_dict[key]
            out_features, in_features = weight.shape
            linear = nn.Linear(in_features, out_features)
            layers.append(linear)
            if i < len(weight_keys) - 1:
                layers.append(activation_layer())

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
            time = time.expand(path_sample.shape[0])
        time_col = time[:, None] if time.ndim == 1 else time
        time_col = time_col.to(device=path_sample.device, dtype=path_sample.dtype)

        K = self.num_marginals
        k_indices = torch.floor(time_col * (K - 1)).clamp(0, K - 2)
        t_lower = k_indices / (K - 1)
        t_upper = (k_indices + 1.0) / (K - 1)
        tau = (time_col - t_lower) / (t_upper - t_lower)
        k_norm = k_indices / (K - 1.0)

        flat_sample = path_sample.flatten(1)

        # Inspect expected input dimension if available
        in_features = getattr(self.velocity_net, "in_features", None)
        if in_features is None:
            net_obj = getattr(self.velocity_net, "net", self.velocity_net)
            if isinstance(net_obj, (nn.Sequential, list, tuple)) and len(net_obj) > 0:
                in_features = getattr(net_obj[0], "in_features", None)

        if in_features == flat_sample.shape[-1] + 2:
            # 2 time inputs: local progress tau and normalized discrete interval k_norm
            cat_input = torch.cat([flat_sample, tau, k_norm], dim=-1)
        elif in_features == flat_sample.shape[-1] + 3:
            cat_input = torch.cat([flat_sample, tau, k_norm, time_col], dim=-1)
        elif self.map_to_interval:
            cat_input = torch.cat([flat_sample, tau], dim=-1)
        else:
            cat_input = torch.cat([flat_sample, time_col], dim=-1)

        time_input = tau if self.map_to_interval else time_col
        try:
            return self.velocity_net(cat_input)
        except TypeError:
            try:
                return self.velocity_net(time_input, path_sample)
            except TypeError:
                return self.velocity_net(path_sample, time_input)

class ALICFMGeneratorWrapper(nn.Module):
    def __init__(
        self,
        checkpoint_path: str | Path | nn.Module,
    ):
        super().__init__()
        path = Path(checkpoint_path)
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            checkpoint = torch.load(path, map_location="cpu")

        if isinstance(checkpoint, dict):
            state_dict = None
            for key in ("generator_state_dict", "interpolant", "state_dict"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state_dict = checkpoint[key]
                    break
            if state_dict is None:
                state_dict = checkpoint

            # Standalone ALI-CFM ST generators store the MLP under
            # ``interpolant_net.*``.  The generic Flow2PC wrapper uses a plain
            # sequential MLP, so remove that serialization prefix before
            # reconstructing it.  (Knot checkpoints use unprefixed or
            # ``net.*`` state dictionaries and remain supported.)
            is_interpolant = any(key.startswith("interpolant_net.") for key in state_dict)
            prefixes = ("interpolant_net.", "net.")
            for prefix in prefixes:
                if state_dict and all(
                    key.startswith(prefix) for key in state_dict
                ):
                    state_dict = {
                        key[len(prefix):]: value for key, value in state_dict.items()
                    }
                    break

            activation = checkpoint.get("model_config", {}).get(
                "activation", "elu" if is_interpolant else "selu"
            )
            self.mlp = ALICFMTeacherWrapper._build_mlp_from_state_dict(
                state_dict, activation=activation
            )
            self.mlp.load_state_dict(state_dict)
        else:
            raise TypeError(f"unexpected checkpoint type for generator: {type(checkpoint)}")

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.unsqueeze(0).repeat(x0.shape[0])
        t_col = t[:, None] if t.ndim == 1 else t
        t_col = t_col.to(device=x0.device, dtype=x0.dtype)

        cat_input = torch.cat([x0, x1, t_col], dim=-1)
        f = self.mlp(cat_input)
        
        # x_t = (1-t)x_0 + t x_1 + t(1-t)f_phi(x_0, x_1, t)
        x_t = (1.0 - t_col) * x0 + t_col * x1 + t_col * (1.0 - t_col) * f
        return x_t

def build_teacher(config):
    if config.get("name") != "alicfm":
        raise ValueError("This compatibility factory retains the ALI-CFM adapter only")
    return ALICFMTeacherWrapper(
        config["checkpoint_path"], num_marginals=config.get("num_marginals", 60),
        map_to_interval=config.get("map_to_interval", True),
    )
