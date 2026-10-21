"""Latent-space validation and decoded MNIST evaluation."""

import csv
from pathlib import Path
import time

from PIL import Image, ImageDraw
import torch
from torchvision.datasets import MNIST
from torchvision.utils import make_grid

from repc.evaluation import (class_distribution_metrics, frechet_distance,
                             improved_precision_recall, inception_score, kernel_distance)
from repc.mnist_classifier import MNISTClassifier, normalize_for_classifier
from repc.mnist_latent import decode_codes, sha256_file


def resolve_classifier_checkpoint(requested=None):
    """Fidelity metrics require a trained classifier; never silently skip them."""
    if requested is not None:
        candidates = [Path(requested)]
    else:
        suffix = "outputs/evaluation/mnist-classifier-pytorch-example-seed1/checkpoints/last.pt"
        root = Path("/data/lthung/flow2pc")
        registry = root / "outputs/evaluation/mnist-latent-classifier.txt"
        candidates = ([Path(registry.read_text().strip())] if registry.is_file() else [])
        candidates.extend([root / suffix,
                           Path(__file__).resolve().parents[1] / suffix])
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "MNIST fidelity evaluation needs a trained classifier. Supply "
        "--classifier-checkpoint or use scripts/evaluate_mnist_latent.sh, "
        "which prepares the fixed MNIST evaluator when missing."
    )


# Direction is recorded beside the raw student-minus-teacher gap in the table.
COMPARISON_METRICS = {
    "fid_mnist": False, "kid_mnist_mean": False,
    "inception_score_mnist_mean": True,
    "precision_mnist": True, "recall_mnist": True,
    "classifier_confidence": True, "class_entropy_normalized": True,
    "class_jsd_to_reference": False, "classes_covered": True,
    "low_confidence_fraction": False,
    "generation_seconds": False, "parameter_count": False,
}


def comparison_tables(metrics):
    rows = []
    for metric, higher_is_better in COMPARISON_METRICS.items():
        teacher, student = (metrics.get(f"{label}/{metric}") for label in ("teacher", "student"))
        if teacher is None:
            continue
        rows.append([metric, higher_is_better, metrics.get(f"real/{metric}"),
                     metrics.get(f"vae_reconstruction/{metric}"), teacher, student,
                     student - teacher if student is not None else None])
    tables = {"teacher_student_comparison": {
        "columns": ["metric", "higher_is_better", "real", "vae_reconstruction", "teacher",
                    "student", "student_minus_teacher"], "data": rows}}
    labels = [label for label in ("real", "vae_reconstruction", "teacher", "student")
              if f"{label}/class_distribution" in metrics]
    if labels:
        tables["class_distribution"] = {
            "columns": ["digit", *labels],
            "data": [[digit, *[metrics[f"{label}/class_distribution"][digit] for label in labels]]
                     for digit in range(10)],
        }
    return tables


def save_comparison_tables(metrics, output):
    for name, table in comparison_tables(metrics).items():
        with (output / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(table["columns"])
            writer.writerows(table["data"])


@torch.no_grad()
def fidelity_metrics(real_features, fake_features, probabilities, real_labels, device,
                     kid_subsets=100, kid_subset_size=1000, is_splits=10,
                     precision_recall_k=3, metric_chunk_size=512, seed=0):
    kid, kid_std = kernel_distance(real_features, fake_features, device,
                                  subsets=kid_subsets, subset_size=kid_subset_size, seed=seed)
    precision, recall = improved_precision_recall(
        real_features, fake_features, device, nearest_neighbors=precision_recall_k,
        chunk_size=metric_chunk_size)
    score, score_std = inception_score(probabilities, splits=min(is_splits, len(probabilities)))
    quality = {"fid_mnist": frechet_distance(real_features, fake_features),
               "kid_mnist_mean": kid, "kid_mnist_std": kid_std,
               "inception_score_mnist_mean": score, "inception_score_mnist_std": score_std,
               "precision_mnist": precision, "recall_mnist": recall,
               "classes_covered": probabilities.argmax(dim=1).unique().numel(),
               "low_confidence_fraction": (probabilities.max(dim=1).values < 0.5).float().mean().item(),
               **class_distribution_metrics(probabilities, real_labels)}
    quality["class_jsd_to_reference"] = quality.pop("class_jsd_to_test")
    return quality


@torch.no_grad()
def teacher_student_metrics(images, features, probabilities, device,
                            kid_subsets=100, kid_subset_size=1000, seed=0):
    """Distribution fidelity and paired fidelity from the same source noise.

    Paired errors describe preservation of teacher behavior, not reconstruction
    of a ground-truth image. Feature FID/KID describe distribution agreement.
    """
    teacher_features, student_features = (features[label] for label in ("teacher", "student"))
    teacher_probs, student_probs = (probabilities[label] for label in ("teacher", "student"))
    teacher_pixels, student_pixels = ((images[label].clamp(-1, 1) + 1) / 2
                                     for label in ("teacher", "student"))
    kid, kid_std = kernel_distance(teacher_features, student_features, device,
                                  subsets=kid_subsets, subset_size=kid_subset_size, seed=seed)
    midpoint = (teacher_probs + student_probs) / 2
    probability_jsd = sum(
        (probs * (probs.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(dim=1)
        for probs in (teacher_probs, student_probs)
    ).mean() / 2
    return {"comparison/student_to_teacher_fid_mnist": frechet_distance(teacher_features, student_features),
            "comparison/student_to_teacher_kid_mnist_mean": kid,
            "comparison/student_to_teacher_kid_mnist_std": kid_std,
            "comparison/paired_pixel_mse_01": (teacher_pixels - student_pixels).square().mean().item(),
            "comparison/paired_pixel_mae_01": (teacher_pixels - student_pixels).abs().mean().item(),
            "comparison/paired_feature_mse": (teacher_features - student_features).square().mean().item(),
            "comparison/paired_class_agreement": (teacher_probs.argmax(1) == student_probs.argmax(1)).float().mean().item(),
            "comparison/paired_probability_jsd": probability_jsd.item()}


@torch.no_grad()
def validation_metrics(teacher, student, values, batch_size, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed + 1000003)
    teacher_error = student_error = nll = 0.0
    for batch in values.split(batch_size):
        batch = batch.to(device)
        noise = torch.randn(batch.shape, device=device, generator=generator)
        times = torch.rand(len(batch), device=device, generator=generator).clamp(1e-5, 1 - 1e-5)
        path = (1 - times[:, None]) * noise + times[:, None] * batch
        velocity = teacher(times, path)
        teacher_error += (velocity - (batch - noise)).square().sum().item()
        if student is not None:
            student_error += (student.velocity(path, times) - velocity).square().sum().item()
            nll -= student.log_prob(batch, torch.tensor(1.0, device=device)).sum().item()
    metrics = {"teacher_flow_mse": teacher_error / values.numel()}
    if student is not None:
        metrics.update(student_velocity_mse=student_error / values.numel(),
                       student_latent_nll_nats=nll / len(values))
    return metrics


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def ode_samples(velocity, count, dim, device, generator, steps=100):
    """Fixed-step RK4 from standard normal source to normalized latent endpoint."""
    if steps < 1:
        raise ValueError("ODE steps must be positive")
    values = torch.randn(count, dim, device=device, generator=generator)
    dt = 1.0 / steps
    for index in range(steps):
        times = values.new_full((count,), index * dt)
        k1 = velocity(times, values)
        k2 = velocity(times + dt / 2, values + dt * k1 / 2)
        k3 = velocity(times + dt / 2, values + dt * k2 / 2)
        k4 = velocity(times + dt, values + dt * k3)
        values = values + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return values


def teacher_samples(teacher, count, dim, device, generator, steps=100):
    return ode_samples(teacher, count, dim, device, generator, steps)


def student_samples(student, count, dim, device, generator, steps=100):
    """Integrate the PC velocity field with the same source and solver as teacher.

    HCLT velocity needs local autograd for circuit flows even during sampling;
    eval mode disables higher-order graphs while ode_samples uses no_grad.
    """
    was_training = student.training
    student.eval()
    try:
        return ode_samples(lambda times, values: student.velocity(values, times),
                           count, dim, device, generator, steps)
    finally:
        student.train(was_training)


@torch.no_grad()
def generate_images(teacher, student, vae, contract, count, batch_size, device, seed, steps):
    images, metrics = {}, {}
    for label in (["teacher", "student"] if student is not None else ["teacher"]):
        generator = torch.Generator(device=device).manual_seed(seed)
        batches = []
        generation_seconds = decode_seconds = 0.0
        for start in range(0, count, batch_size):
            size = min(batch_size, count - start)
            synchronize(device)
            started = time.perf_counter()
            if label == "teacher":
                values = teacher_samples(teacher, size, contract["manifest"]["dim"], device, generator, steps)
            else:
                values = student_samples(student, size, contract["manifest"]["dim"], device, generator, steps)
            synchronize(device)
            generation_seconds += time.perf_counter() - started
            started = time.perf_counter()
            decoded = decode_codes(vae, values, contract)
            synchronize(device)
            decode_seconds += time.perf_counter() - started
            if not torch.isfinite(decoded).all():
                raise FloatingPointError(f"Non-finite {label} decoded samples")
            batches.append(decoded.cpu())
        images[label] = torch.cat(batches)
        metrics[f"{label}/latent_generation_seconds"] = generation_seconds
        metrics[f"{label}/decode_seconds"] = decode_seconds
        metrics[f"{label}/generation_seconds"] = generation_seconds + decode_seconds
        metrics[f"{label}/sampling"] = "velocity_ode"
        metrics[f"{label}/ode_solver"] = "rk4"
        metrics[f"{label}/ode_steps"] = steps
        metrics[f"{label}/ode_nfe_per_sample"] = 4 * steps
        metrics[f"{label}/parameter_count"] = sum(p.numel() for p in (teacher if label == "teacher" else student).parameters())
        metrics[f"{label}/parameter_bytes"] = sum(p.numel() * p.element_size() for p in (teacher if label == "teacher" else student).parameters())
    return images, metrics


def real_images(data_dir, codes, split, count=None):
    dataset = MNIST(data_dir, train=split != "test", download=False)
    indices = codes[split]["indices"]
    if count is not None:
        indices = indices[:count]
    return dataset.data[indices, None].float() / 127.5 - 1


def save_panel(sections, path, per_section=100):
    tiles = []
    for label, images in sections:
        grid = make_grid((images[:per_section].clamp(-1, 1) + 1) / 2, nrow=10, padding=2)
        array = (grid.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
        tile = Image.fromarray(array)
        tiles.append((label, tile))
    width = max(tile.width for _, tile in tiles)
    panel = Image.new("RGB", (width * 2, max(tile.height for _, tile in tiles) * ((len(tiles) + 1) // 2) + 30 * ((len(tiles) + 1) // 2)), "white")
    draw = ImageDraw.Draw(panel)
    cell_height = max(tile.height for _, tile in tiles) + 30
    for index, (label, tile) in enumerate(tiles):
        left, top = (index % 2) * width, (index // 2) * cell_height
        draw.text((left + 5, top + 7), label, fill="black")
        panel.paste(tile, (left, top + 30))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)


@torch.no_grad()
def reconstruction_reference(vae, codes, contract, data_dir, split, device, count=100):
    real = real_images(data_dir, codes, split, count)
    reconstructed = decode_codes(vae, codes[split]["values"][:count].to(device), contract).cpu()
    return real, reconstructed


@torch.no_grad()
def classifier_outputs(classifier, images, batch_size, device):
    features, probabilities = [], []
    for images_batch in images.split(batch_size):
        feature = classifier.features(normalize_for_classifier(images_batch.to(device)))
        features.append(feature.cpu())
        probabilities.append(classifier.fc2(feature).softmax(dim=1).cpu())
    return torch.cat(features), torch.cat(probabilities)


@torch.no_grad()
def decoded_evaluation(args, teacher, student, vae, codes, contract, output, device):
    classifier_path = resolve_classifier_checkpoint(args.classifier_checkpoint)
    checkpoint = torch.load(classifier_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") != "mnist_classifier":
        raise ValueError("Expected a trained Flow2PC MNIST classifier checkpoint")
    classifier = MNISTClassifier().to(device).eval().requires_grad_(False)
    classifier.load_state_dict(checkpoint["state_dict"], strict=True)
    split = args.split
    images, metrics = generate_images(teacher, student, vae, contract, args.num_generated,
                                      args.batch_size, device, args.seed, args.ode_steps)
    reference = real_images(args.data_dir, codes, split)
    reconstruction_batches = []
    for batch in codes[split]["values"].split(args.batch_size):
        reconstruction_batches.append(decode_codes(vae, batch.to(device), contract).cpu())
    reconstructions = torch.cat(reconstruction_batches)
    metrics.update({f"{split}/{key}": value for key, value in validation_metrics(
        teacher, student, codes[split]["values"], args.batch_size, device, args.seed
    ).items()})
    metrics.update({"vae/reconstruction_l1": (reference - reconstructions).abs().mean().item(),
                    "vae/reconstruction_mse": (reference - reconstructions).square().mean().item(),
                    "protocol/split": split, "protocol/num_generated": args.num_generated,
                    "protocol/num_real": len(reference), "protocol/image_metrics_available": True,
                    "protocol/teacher_solver": "rk4", "protocol/teacher_ode_steps": args.ode_steps,
                    "protocol/seed": args.seed,
                    "protocol/feature_extractor": "fixed MNIST classifier; 128-dimensional fc1 ReLU",
                    "protocol/fid_is_convention": "MNIST classifier features/probabilities; not ImageNet Inception-v3",
                    "protocol/is_splits": min(args.is_splits, args.num_generated),
                    "protocol/kid_subsets": args.kid_subsets,
                    "protocol/kid_subset_size": min(args.kid_subset_size, len(reference), args.num_generated),
                    "protocol/precision_recall_k": args.precision_recall_k,
                    "protocol/classifier_preprocessing": "clamp[-1,1]; convert[0,1]; (x-0.1307)/0.3081"})
    if student is not None:
        metrics.update({"protocol/student_sampling": "velocity_ode_t0_to_t1",
                        "protocol/student_solver": "rk4", "protocol/student_ode_steps": args.ode_steps})
        metrics[f"{split}/student_raw_code_nll_nats"] = (
            metrics[f"{split}/student_latent_nll_nats"] + contract["scale"].log().sum().item()
        )
    metrics["protocol/classifier_checkpoint"] = str(classifier_path)
    metrics["protocol/classifier_sha256"] = sha256_file(classifier_path)
    if "test_accuracy" in checkpoint:
        metrics["protocol/classifier_checkpoint_test_accuracy"] = checkpoint["test_accuracy"]
    real_features, real_probs = classifier_outputs(classifier, reference, args.batch_size, device)
    metrics["classifier/reference_accuracy"] = (real_probs.argmax(1) == codes[split]["labels"]).float().mean().item()
    feature_sets, probability_sets = {"real": real_features}, {"real": real_probs}
    for label, samples in {"vae_reconstruction": reconstructions, **images}.items():
        feature_sets[label], probability_sets[label] = classifier_outputs(classifier, samples, args.batch_size, device)
    for label in feature_sets:
        quality = fidelity_metrics(real_features, feature_sets[label], probability_sets[label],
                                   codes[split]["labels"], device, args.kid_subsets, args.kid_subset_size,
                                   args.is_splits, args.precision_recall_k, args.metric_chunk_size, args.seed)
        metrics.update({f"{label}/{key}": value for key, value in quality.items()})
    if student is not None:
        metrics.update(teacher_student_metrics(images, feature_sets, probability_sets, device,
                                               args.kid_subsets, args.kid_subset_size, args.seed))
        metrics["protocol/teacher_student_pairing"] = "same Gaussian source noise, batching, RK4 solver and step count"
        for metric in COMPARISON_METRICS:
            metrics[f"comparison/{metric}_student_minus_teacher"] = metrics[f"student/{metric}"] - metrics[f"teacher/{metric}"]
        metrics["comparison/generation_speedup_teacher_over_student"] = metrics["teacher/generation_seconds"] / metrics["student/generation_seconds"]
        metrics["comparison/parameter_compression_teacher_over_student"] = metrics["teacher/parameter_count"] / metrics["student/parameter_count"]
    save_comparison_tables(metrics, output)
    panel_path = output / "images" / "panel.png"
    sections = [(f"Real MNIST ({split})", reference), ("Frozen VAE reconstructions", reconstructions),
                (f"Teacher: RK4 ODE ({args.ode_steps} steps)", images["teacher"])]
    if student is not None:
        sections.append((f"Student: velocity ODE ({args.ode_steps} steps)", images["student"]))
    save_panel(sections, panel_path)
    return metrics, panel_path
