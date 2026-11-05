"""Native unconditional MMFM objective on minibatch-OT MFM marginal chains."""

import csv
import math
import random
from pathlib import Path

import numpy as np
import torch

from flow2pc.data.metric_profiles import dataset_identity
from flow2pc.data.metric_registry import load_dataset
from flow2pc.evaluation.metric_flow import evaluation_seed
from flow2pc.pretrain.models.mmfm.adapter import build_model, model_velocity
from flow2pc.utils.metric_config import prepare_mmfm as prepare
from flow2pc.pretrain.models.mmfm.sampling import MMFMPathSampler
from flow2pc.utils.mfm_source import mmfm_provenance as provenance
from flow2pc.utils.checkpoint import atomic_json, atomic_path
from flow2pc.utils.seed import seed_everything


def rng_state():
    numpy = np.random.get_state()
    return {
        "torch": torch.get_rng_state(), "python": random.getstate(),
        "numpy": (numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def resume_binding(config, metadata):
    return {
        "metadata": metadata,
        "training": {k: v for k, v in config["training"].items()
                     if k not in ("epochs", "device", "resume_checkpoint")},
        "seed": config["seed"],
    }


def run(config):
    """Run a resolved MMFM pretraining config; prepare() itself is read-only."""
    training = config["training"]
    device_name = training["device"]
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                          else "cpu" if device_name == "auto" else device_name)
    output = Path(config["output_dir"])
    resume = training["resume_checkpoint"]
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Refusing to overwrite nonempty MMFM output: {output}")
    seed_everything(config["seed"])
    data = load_dataset(config["dataset_config"], {"dataset_profile": config["dataset_profile"]})
    if config["dataset_profile"].get("profile") and data.source_info.get("profile") != config["dataset_profile"]["profile"]:
        raise ValueError("Prepared snapshot profile differs from config")
    metadata = {
        "method": "mmfm", "source_root": config["source_root"],
        "source_provenance": provenance(config["source_root"]),
        "model_config": config["model_config"], "matcher": config["matcher"],
        "dataset_config": config["dataset_config"], "dataset_profile": config["dataset_profile"],
        "dataset_identity": dataset_identity(data),
        "observed_count": len(data.observed_indices),
        "query_protocol": "minibatch_ot_chain_v1",
        "time_sampling": "uniform_model_time_with_relative_anchor_margin_1e-5",
    }
    model = build_model(config["model_config"], config["source_root"]).to(device)
    # Same Xavier-normal initialization and gain as native train_mmfm.py.
    for parameter in model.parameters():
        if parameter.ndim > 1:
            torch.nn.init.xavier_normal_(parameter, gain=1.4 if config["model_config"]["activation"] == "LeakyReLU" else 0.75)
    sampler = MMFMPathSampler(config["matcher"], config["source_root"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["lr"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-4)
    binding = resume_binding(config, metadata)
    start, step, best = 0, 0, math.inf
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint.get("model_type") != "mmfm_teacher_v1" or checkpoint.get("binding") != binding:
            raise ValueError("MMFM resume model, source, sampler, data or training settings changed")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start, step, best = checkpoint["epoch"], checkpoint["step"], checkpoint["best_validation_loss"]
        restore_rng(checkpoint["rng"])
    if start >= training["epochs"]:
        raise ValueError("epochs must exceed the resumed epoch")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "resolved_config.json", config)
    atomic_json(output / "manifest.json", {**metadata, "artifact_type": "teacher", "schema_version": 1})
    atomic_json(output / "status.json", {"status": "running"})
    count = training["batch_size"]
    steps_per_epoch = training["steps_per_epoch"] or math.ceil(max(len(data.train_marginals[i]) for i in data.observed_indices) / count)

    def save(name, epoch):
        with atomic_path(output / "checkpoints" / name) as temporary:
            torch.save({
                "model_type": "mmfm_teacher_v1", "metadata": metadata, "binding": binding,
                "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "rng": rng_state(),
                "epoch": epoch, "step": step, "best_validation_loss": best,
            }, temporary)

    tracker = None
    try:
        if config["wandb"]["mode"] != "disabled":
            import wandb
            tracker = wandb.init(
                project=config["wandb"].get("project", "flow2pc"),
                mode=config["wandb"]["mode"], group=config["wandb"].get("group", "mmfm-pretrain"),
                name=config["wandb"].get("run_name"), config=config, dir=str(output),
            )
        log_path = output / "metrics.csv"
        fields = ["epoch", "step", "train_loss", "validation_loss", "learning_rate"]
        if log_path.exists() and log_path.stat().st_size:
            with log_path.open(newline="") as stream:
                if next(csv.reader(stream)) != fields:
                    raise ValueError("Incompatible MMFM pretrain metrics header")
        needs_header = not log_path.exists() or log_path.stat().st_size == 0
        if not resume:
            save("last.pt", 0)
        with log_path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if needs_header:
                writer.writeheader()
            for epoch in range(start + 1, training["epochs"] + 1):
                model.train()
                losses = []
                for _ in range(steps_per_epoch):
                    times, values, target = sampler.sample(data, count, device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = (model_velocity(model, values, times) - target).square().mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite MMFM pretraining loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"], error_if_nonfinite=True)
                    optimizer.step()
                    losses.append(loss.item())
                    step += 1
                scheduler.step()
                model.eval()
                with torch.no_grad(), evaluation_seed(1234, device):
                    scores = []
                    for _ in range(training["validation_batches"]):
                        times, values, target = sampler.sample(data, count, device, validation=True)
                        scores.append((model_velocity(model, values, times) - target).square().mean().item())
                    score = float(np.mean(scores))
                if not math.isfinite(score):
                    raise FloatingPointError("Nonfinite MMFM validation loss")
                if score < best:
                    best = score
                    save("best.pt", epoch)
                save("last.pt", epoch)
                row = dict(epoch=epoch, step=step, train_loss=float(np.mean(losses)),
                           validation_loss=score, learning_rate=optimizer.param_groups[0]["lr"])
                writer.writerow(row)
                stream.flush()
                if tracker:
                    tracker.log(row, step=step)
                print(f"epoch={epoch} loss={row['train_loss']:.6g} validation={score:.6g}", flush=True)
        if config["evaluation"]["enabled"]:
            from flow2pc.evaluation.metric_flow import evaluate_teacher_checkpoint
            # best.pt may belong to the resumed run's original directory.
            selected = output / "checkpoints/best.pt"
            if not selected.is_file() and resume:
                selected = Path(resume).parent / "best.pt"
            evaluate_teacher_checkpoint(selected, output_dir=output / "eval", device=device,
                                        settings=config["evaluation"])
        atomic_json(output / "status.json", {"status": "complete", "best_validation_loss": best, "step": step})
    except BaseException as error:
        atomic_json(output / "status.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        if tracker:
            tracker.finish()
    return output


class MMFM:
    """Registry entry matching the shared pretrain method interface."""

    def validate(self, config):
        return prepare(config)

    def fit(self, dataset, config, output):
        return run(dataset)

    def load(self, checkpoint, device="cpu"):
        from flow2pc.pretrain.models.mmfm.adapter import load_teacher
        return load_teacher({"checkpoint_path": str(checkpoint)}, device)
