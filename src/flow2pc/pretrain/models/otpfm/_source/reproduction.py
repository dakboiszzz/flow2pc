"""Original reproduction callback, arguments and trajectory previews."""

import json
import time
from pathlib import Path
import numpy as np
from . import factory as original

def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class RunLogger:
    """Log new values only, with epoch as the W&B chart axis."""

    def __init__(self, run, folder, dataset, holdout, selection_rule="mean_w2"):
        if selection_rule not in ("mean_w2", "mean_squared_w2"):
            raise ValueError(f"Unknown checkpoint selection: {selection_rule}")
        if selection_rule == "mean_squared_w2" and dataset == "synthetic":
            raise ValueError("Squared holdout selection requires real-data holdout times")
        self.run, self.folder, self.dataset, self.holdout = run, folder, dataset, holdout
        self.selection_rule = selection_rule
        self.best_selection_score = None
        self.best_mean_w2 = None
        self.counts = {}
        self.best = None
        self.started = time.perf_counter()
        run.define_metric("epoch")
        run.define_metric("*", step_metric="epoch")

    def __call__(self, trainer, epoch):
        values = {"epoch": epoch, "global_step": trainer.global_step,
                  "elapsed_seconds": time.perf_counter() - self.started,
                  "train/lr": trainer.optimizers["flow"].param_groups[0]["lr"]}
        for key, history in trainer.losses.items():
            if not history or len(history) <= self.counts.get(key, 0):
                continue
            self.counts[key] = len(history)
            if key == "metric_epochs":
                continue
            if key == "otp_alpha":
                value, name = history[-1][1], "train/otp_alpha"
            else:
                value = history[-1]
                name = {"train_loss": "train/loss", "val_loss": "val/loss"}.get(key, f"eval/{key}")
            if not np.isscalar(value):
                continue
            if not np.isfinite(value):
                raise FloatingPointError(f"Non-finite {key} at epoch {epoch}")
            values[name] = float(value)
        if "eval/w2_rest" in values:
            if self.dataset == "synthetic":
                primary_name, primary = "eval/marginal_w2", values["eval/w2_rest"]
            else:
                keys = [f"eval/w2_t{index}" for index in self.holdout]
                primary_name = "eval/holdout_w2"
                primary = float(np.mean([values[key] for key in keys]))
            values[primary_name] = primary
            selection_score = primary
            if self.selection_rule == "mean_squared_w2":
                selection_score = float(np.mean([values[key] ** 2 for key in keys]))
                values["eval/holdout_mean_squared_w2"] = selection_score
                # Keep the released selection rule as a paired diagnostic, using
                # the same trained weights and evaluations rather than retraining.
                if self.best_mean_w2 is None or primary < self.best_mean_w2["score"]:
                    self.best_mean_w2 = {
                        "epoch": epoch, "score": primary, "metric": primary_name,
                        "metrics": {k: v for k, v in values.items() if k.startswith("eval/")},
                    }
                    trainer.save_checkpoint("best_mean_w2.pt")
                    write_json(self.folder / "best_mean_w2_metrics.json", self.best_mean_w2)
                    self.run.summary.update({"best_mean_w2/epoch": epoch,
                                             "best_mean_w2/score": primary,
                                             **{f"best_mean_w2/{k[5:]}": v for k, v in
                                                self.best_mean_w2["metrics"].items()}})
            if self.best is None or selection_score < self.best_selection_score:
                self.best_selection_score = selection_score
                self.best = {"epoch": epoch, "score": primary, "metric": primary_name,
                             "metrics": {k: v for k, v in values.items() if k.startswith("eval/")}}
                if self.selection_rule == "mean_squared_w2":
                    self.best.update(selection_rule=self.selection_rule,
                                     selection_score=selection_score)
                    self.run.summary.update({"best/selection_rule": self.selection_rule,
                                             "best/selection_score": selection_score})
                trainer.save_checkpoint("best.pt")
                write_json(self.folder / "best_metrics.json", self.best)
                self.run.summary.update({"best/epoch": epoch, "best/score": primary,
                                         "best/metric": primary_name,
                                         **{f"best/{k[5:]}": v for k, v in self.best["metrics"].items()}})
        self.run.log(values)
        with (self.folder / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(values, allow_nan=False) + "\n")


def trainer_kwargs(config, setup, folder, device, dataset):
    kwargs = {
        "model": setup["model"], "train_loader": setup["train_loader"],
        "val_loader": setup["val_loader"], "save_dir": folder, "device": device,
        "lr": config["lr"], "epochs": config["epochs"],
        "optimizer": config.get("optimizer", "adam"),
        "grad_clip": config.get("grad_clip", 0.0),
        "weight_decay": config.get("weight_decay", 0.0),
        "lr_schedule": config.get("lr_schedule"), "do_otp": True,
        "otp_alpha_type": config.get("otp_alpha_type", "sigmoid"),
        "otp_alpha_slope": config.get("otp_alpha_slope", 6.0),
        "otp_alpha_mean_scale": config.get("otp_alpha_mean_scale", 0.5),
        "sampling_steps": config.get("sampling_steps", 50), "potentials": setup["potentials"],
    }
    if "synthetic_args" in setup:
        kwargs.update(setup["synthetic_args"])
    else:
        original_name = "gulfofmexico" if dataset == "gom" else "beijingair"
        kwargs.update(original.get_dataset_trainer_args(original_name, setup, config))
    return kwargs


def log_trajectory_previews(run, trainer, folder, config, best_epoch):
    """Make best/final trajectories visible in W&B as well as downloadable."""
    import matplotlib.pyplot as plt
    import wandb

    plot_dir = folder / "plots"
    plot_dir.mkdir(exist_ok=True)
    for label, epoch in (("best", best_epoch), ("final", config["epochs"])):
        path = folder / "trajectories" / f"trajectories_epoch{epoch:04d}.npz"
        with np.load(path, allow_pickle=False) as data:
            trajectories = data["trajectories"].copy()
            times = data["t_eval"].copy()
        scaler = getattr(trainer, "scaler", None)
        if scaler is not None:
            shape = trajectories.shape
            trajectories = scaler.inverse_transform(trajectories.reshape(-1, shape[-1])).reshape(shape)
        fig, ax = plt.subplots(figsize=(8, 5))
        if trajectories.shape[-1] == 2:
            ax.plot(trajectories[:, :64, 0], trajectories[:, :64, 1], alpha=0.3, linewidth=0.8)
            ax.set(xlabel="Coordinate 1 (original units)", ylabel="Coordinate 2 (original units)")
        else:
            ax.plot(times, trajectories[:, :32, 0], alpha=0.3, linewidth=0.8)
            ax.set(xlabel="Normalized time", ylabel="Value (original units)")
        ax.set_title(f"{label.capitalize()} trajectories — epoch {epoch}")
        fig.tight_layout()
        image_path = plot_dir / f"{label}_trajectories.png"
        try:
            fig.savefig(image_path, dpi=150)
        finally:
            plt.close(fig)
        run.log({f"plots/{label}_trajectories": wandb.Image(str(image_path)),
                 "epoch": config["epochs"]})
