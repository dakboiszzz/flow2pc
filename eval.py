import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from repc.data import MNIST_SHAPE, mnist_loader
from train import build_student, resolve_device


def parse_times(value: str) -> list[float]:
    times = [float(item) for item in value.split(",")]
    if not times or any(time < 0.0 or time > 1.0 for time in times):
        raise argparse.ArgumentTypeError(
            "times must be comma-separated values in [0, 1]"
        )
    return times


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--times",
        type=parse_times,
        default=parse_times("0,0.25,0.5,0.75,1"),
    )
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.num_samples < 1 or (
        args.max_examples is not None and args.max_examples < 1
    ):
        parser.error("sample and example counts must be positive")
    return args


def time_label(time: float) -> str:
    return f"{time:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    device = resolve_device(args.device)
    checkpoint = torch.load(
        run_dir / "checkpoints" / "last.pt",
        map_location=device,
        weights_only=True,
    )
    if checkpoint["model_type"] != "distill":
        raise ValueError(f"{run_dir} is not a distillation run")
    pc = build_student(checkpoint["model_config"]).to(device).eval()
    pc.load_state_dict(checkpoint["state_dict"])

    data_dir = args.data_dir
    if data_dir is None:
        data_dir = Path(checkpoint["training_config"]["data_dir"])
    loader = mnist_loader(data_dir, False, batch_size=512, num_workers=0)
    times = torch.tensor(args.times, device=device)
    likelihoods = [[] for _ in args.times]
    seen = 0
    generator = torch.Generator(device=device).manual_seed(0)

    with torch.no_grad():
        for images, _ in loader:
            if args.max_examples is not None:
                images = images[: args.max_examples - seen]
            if len(images) == 0:
                break
            images = images.to(device)
            noise = torch.randn(
                images.shape,
                device=device,
                dtype=images.dtype,
                generator=generator,
            )
            path = torch.stack(
                [(1.0 - time) * noise + time * images for time in times]
            ).flatten(2)
            path_likelihood = pc.path_log_prob(path, times).cpu()
            for index, values in enumerate(path_likelihood):
                likelihoods[index].append(values)
            seen += len(images)
            if args.max_examples is not None and seen >= args.max_examples:
                break

        samples = pc.sample_path(args.num_samples, times).reshape(
            len(times), args.num_samples, *MNIST_SHAPE
        )

    output_dir = run_dir / "eval"
    output_dir.mkdir(exist_ok=True)
    means = []
    standard_deviations = []
    for index, time in enumerate(args.times):
        values = torch.cat(likelihoods[index])
        mean = values.mean().item()
        std = values.std(unbiased=False).item()
        means.append(mean)
        standard_deviations.append(std)
        print(f"t={time:.3f}: exact log-likelihood {mean:.2f} ± {std:.2f} nats")
        save_image(
            samples[index].clamp(-1, 1) * 0.5 + 0.5,
            output_dir / f"samples_t{time_label(time)}.png",
            nrow=8,
        )

    torch.save(
        {
            "times": torch.tensor(args.times),
            "mean_log_likelihood": torch.tensor(means),
            "std_log_likelihood": torch.tensor(standard_deviations),
            "log_likelihoods": [torch.cat(values) for values in likelihoods],
        },
        output_dir / "metrics.pt",
    )
    print(f"evaluation artifacts: {output_dir}")


if __name__ == "__main__":
    main()
