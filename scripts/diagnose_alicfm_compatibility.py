"""Check ALI-CFM checkpoint reconstruction and generator/teacher compatibility."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import sys
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path or sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)

from repc.flow import ALICFMGeneratorWrapper, ALICFMTeacherWrapper


def build_mlp(checkpoint: dict, state_key: str) -> nn.Module:
    state = checkpoint[state_key]
    activation_name = checkpoint.get("model_config", {}).get("activation", "selu")
    activations = {
        "elu": nn.ELU,
        "selu": nn.SELU,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }
    activation = activations[activation_name]
    keys = sorted(key for key in state if key.endswith(".weight"))
    layers = []
    for index, key in enumerate(keys):
        out_features, in_features = state[key].shape
        layers.append(nn.Linear(in_features, out_features))
        if index + 1 < len(keys):
            layers.append(activation())
    model = nn.Sequential(*layers)
    model.load_state_dict({key.replace("net.", ""): value for key, value in state.items()})
    return model.eval()


def knot_endpoints(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    z = np.linspace(0.0, 3.0, 60) - 1.5
    split = 20
    mean_x = np.concatenate(
        (
            3.0 * (z[:split] + 0.5),
            np.cos(2.0 * np.pi * (z[split : 2 * split] - 0.75)),
            3.0 * (z[2 * split :] - 0.5),
        )
    )
    mean_y = np.concatenate(
        (
            -0.5 * np.tanh(5.0 * (z[:split] + 1.0)) + 0.5,
            np.sin(2.0 * np.pi * (z[split : 2 * split] - 0.75)) + 1.0,
            0.5 * np.tanh(5.0 * (z[2 * split :] - 1.0)) + 0.5,
        )
    )
    initial = np.stack(
        [mean_x[0] + rng.normal(0.0, 0.1, count), mean_y[0] + rng.normal(0.0, 0.1, count)],
        axis=1,
    )
    final = np.stack(
        [mean_x[-1] + rng.normal(0.0, 0.1, count), mean_y[-1] + rng.normal(0.0, 0.1, count)],
        axis=1,
    )
    return torch.from_numpy(initial).float(), torch.from_numpy(final).float()


def metrics_for_time(
    generator: nn.Module,
    teacher: nn.Module,
    x0: torch.Tensor,
    x1: torch.Tensor,
    value: float,
) -> dict[str, float]:
    times = torch.full((len(x0), 1), value, requires_grad=True)
    samples = generator(x0, x1, times[:, 0])
    derivative = torch.autograd.grad(samples.sum(), times)[0]
    with torch.no_grad():
        teacher_velocity = teacher(times[:, 0], samples.detach())
    mse = (teacher_velocity - derivative).square().mean()
    derivative_energy = derivative.square().mean().clamp_min(1e-12)
    slope = (teacher_velocity * derivative).sum() / derivative.square().sum().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        teacher_velocity, derivative, dim=1
    ).mean()
    return {
        "t": value,
        "generator_derivative_rms": derivative.square().mean().sqrt().item(),
        "teacher_velocity_rms": teacher_velocity.square().mean().sqrt().item(),
        "mse": mse.item(),
        "relative_mse": (mse / derivative_energy).item(),
        "least_squares_slope": slope.item(),
        "cosine": cosine.item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", type=Path, default=Path("checkpoints/generator.pt"))
    parser.add_argument("--teacher", type=Path, default=Path("checkpoints/teacher.pt"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/alicfm_compatibility.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    generator_checkpoint = torch.load(args.generator, map_location="cpu", weights_only=False)
    teacher_checkpoint = torch.load(args.teacher, map_location="cpu", weights_only=False)

    generator = ALICFMGeneratorWrapper(args.generator).eval()
    teacher = ALICFMTeacherWrapper(args.teacher, map_to_interval=False).eval()
    explicit_generator = build_mlp(generator_checkpoint, "generator_state_dict")

    x0, x1 = knot_endpoints(args.batch_size, args.seed)
    zero = torch.zeros(args.batch_size)
    one = torch.ones(args.batch_size)
    with torch.no_grad():
        boundary_zero = (generator(x0, x1, zero) - x0).square().mean().item()
        boundary_one = (generator(x0, x1, one) - x1).square().mean().item()
        explicit_correction = explicit_generator(torch.cat([x0, x1, zero[:, None]], dim=1))
        explicit_zero = (1.0 - zero[:, None]) * x0 + zero[:, None] * x1 + zero[:, None] * (1.0 - zero[:, None]) * explicit_correction

    results = {
        "generator_model_config": generator_checkpoint.get("model_config"),
        "teacher_model_config": teacher_checkpoint.get("model_config"),
        "wrapper_generator_activation": type(generator.mlp.net[1]).__name__,
        "checkpoint_generator_activation": generator_checkpoint.get("model_config", {}).get("activation"),
        "boundary_mse_t0": boundary_zero,
        "boundary_mse_t1": boundary_one,
        "wrapper_vs_explicit_checkpoint_generator_mse_t0": (generator(x0, x1, zero) - explicit_zero).square().mean().item(),
        "times": [
            metrics_for_time(generator, teacher, x0, x1, value)
            for value in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
