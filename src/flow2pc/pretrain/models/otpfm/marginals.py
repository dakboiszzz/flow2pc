"""OTP-FM on shared marginal datasets, with the existing W1/ODE evaluation.

This is a new dataset adaptation, not a paper-reproduction claim. The original
core, objective, optimizer, EMA update and training loop remain in _source.
"""

import copy
import gc
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from flow2pc.data.base import heldout_indices
from flow2pc.data.synthetic.knot import knot_means
from flow2pc.evaluation.figures import validate_resources
from flow2pc.evaluation.runner import evaluate, population
from flow2pc.utils.checkpoint import atomic_json, save_teacher
from flow2pc.utils.tracking import LocalLogger

from ._source.common_trainer import Trainer
from ._source.factory import create_model, create_potentials
from ._source.singlecell_tuples import EBMultiMarginalDataset
from .adapter import FrozenOTPFM
from .provenance import verify_source


def validate(config):
    params, training = config["model"]["params"], config["training"]
    if set(params) != {"recipe", "native_config"} or params["recipe"] != "W2Inf":
        raise ValueError("Shared-data OTP-FM currently uses the original W2Inf recipe")
    if set(training) - {"device", "steps", "samples_per_marginal", "population_split"} or {"device", "steps"} - set(training):
        raise ValueError("Shared-data OTP-FM training requires device/steps and optional samples_per_marginal")
    if type(training["steps"]) is not int or training["steps"] < 1:
        raise ValueError("training.steps must be a positive integer")
    if torch.device(training["device"]).type not in ("cpu", "cuda"):
        raise ValueError("OTP-FM requires a CPU or CUDA device")
    if training.get("population_split", "all") not in ("all", "train"):
        raise ValueError("training.population_split must explicitly select all or train rows")
    if config["data"]["name"] == "knot" and (
        config["data"]["params"].get("normalise", True)
        or config["data"]["params"].get("whiten", False)
        or config["data"]["params"].get("heldout_index") is not None
    ):
        raise ValueError("Fresh analytic knot training requires every time and raw coordinates")
    evaluation = config["evaluation"]
    if not evaluation["enabled"] or evaluation.get("protocol") != "shared_ode":
        raise ValueError("Shared-data teachers require the existing shared_ode W1 EMD evaluation")
    native = params["native_config"]
    required = {"potential", "ot_coupling", "batch_size", "lr", "hidden_dim",
                "num_hidden_layers", "strength", "width_multiplier", "consistency_loss",
                "lossfn", "solver_type", "ema_decay", "otp_alpha_mean_scale"}
    if not isinstance(native, dict) or required - native.keys():
        raise ValueError("Incomplete shared-data native OTP-FM configuration")
    if native["potential"] != "w2inf" or native["ot_coupling"] is not False:
        raise ValueError("This sweep uses native W2Inf independent tuples; global OT is not substituted or approximated")
    if native["solver_type"] != "direct" or native["lossfn"] != "mse":
        raise ValueError("This sweep retains native direct W2Inf solutions and native MSE consistency loss")
    if native["consistency_loss"] not in ("meanflow", "imf"):
        raise ValueError("Unknown native consistency objective")
    if native.get("x_pred", False) or not 0 < native["ema_decay"] < 1:
        raise ValueError("Frozen velocity export requires native velocity prediction and EMA")
    for key in ("batch_size", "hidden_dim", "num_hidden_layers"):
        if type(native[key]) is not int or native[key] < 1:
            raise ValueError(f"Invalid native {key}")
    for key in ("lr", "strength", "width_multiplier"):
        if type(native[key]) not in (int, float) or not math.isfinite(native[key]) or native[key] <= 0:
            raise ValueError(f"Invalid native {key}")
    if native.get("grad_clip", 0) != 0:
        raise ValueError("Gradient clipping remains disabled; native clipping placement is not changed")
    budget = training.get("samples_per_marginal")
    if budget is not None and (config["data"]["name"] != "cell_tracking" or type(budget) is not int or budget < 1):
        raise ValueError("samples_per_marginal is an explicit fixed tracking observation budget")


class AnalyticKnotTuples(Dataset):
    """Fresh independent noisy observations, never persistent trajectory IDs."""

    def __init__(self, dataset, batch_size):
        self.means, _ = knot_means(dataset.params["size"])
        self.noise = dataset.params["noise_std"]
        self.dataset_size = 100 * batch_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        values = self.means + np.random.standard_normal(self.means.shape) * self.noise
        return list(torch.as_tensor(values, dtype=torch.float32))


class MarginalTrainer(Trainer):
    def on_epoch_start(self, epoch, batch=None):
        if hasattr(self.train_loader.dataset, "reshuffle"):
            self.train_loader.dataset.reshuffle()


class EpochLogger:
    def __init__(self, logger):
        self.logger = logger
        self.counts = {}

    def __call__(self, trainer, epoch):
        values = {"train/epoch": epoch, "train/step": trainer.global_step,
                  "train/lr": trainer.optimizers["flow"].param_groups[0]["lr"]}
        for key in ("train_loss", "val_loss", "otp_alpha"):
            history = trainer.losses[key]
            if len(history) > self.counts.get(key, 0):
                values[f"train/{key}"] = float(history[-1][1] if key == "otp_alpha" else history[-1])
                self.counts[key] = len(history)
        self.logger.log(values)


def fit(dataset, config, run_dir):
    validate(config)
    source = verify_source()
    validate_resources(dataset, config["evaluation"])
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError("Output directory exists; preserve it and use sweep recovery or a fresh directory")
    run_dir.mkdir(parents=True)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    atomic_json(run_dir / "status.json", {"status": "running"})
    logger, trainer = None, None
    try:
        logger = LocalLogger(run_dir, config)
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        native = copy.deepcopy(config["model"]["params"]["native_config"])
        split = config["training"].get("population_split", "all")
        clouds = [population(dataset, i, split if i in dataset.observed else "all") for i in range(len(dataset.metadata.model_times))]
        observed, holdout = list(dataset.observed), heldout_indices(dataset)
        if observed[0] != 0 or observed[-1] != len(clouds) - 1:
            raise ValueError("OTP-FM requires both endpoints in the observed training set")
        subset_indices = None
        budget = config["training"].get("samples_per_marginal")
        if budget is not None:
            if any(len(clouds[i]) < budget for i in observed):
                raise ValueError("Tracking training budget exceeds an observed marginal")
            # Match the ALI-CFM experiment's seeded, fixed, without-replacement
            # ten-point sample, chosen before initializing model parameters.
            subset_indices = {str(i): np.random.choice(np.arange(len(clouds[i])), budget, replace=False).tolist() for i in observed}
            clouds = [cloud[subset_indices[str(i)]] if i in observed else cloud for i, cloud in enumerate(clouds)]
        if dataset.metadata.name == "knot":
            tuples = AnalyticKnotTuples(dataset, native["batch_size"])
        else:
            pcs = np.concatenate([cloud.numpy() for cloud in clouds])
            labels = np.concatenate([np.full(len(cloud), i) for i, cloud in enumerate(clouds)])
            tuples = EBMultiMarginalDataset(pcs, labels, holdout_times=holdout)
        loader = DataLoader(tuples, batch_size=native["batch_size"], shuffle=True, num_workers=0,
                            pin_memory=torch.device(config["training"]["device"]).type == "cuda")
        native["epochs"] = math.ceil(config["training"]["steps"] / len(loader))
        native["tks"] = [dataset.metadata.model_times[i] for i in observed[1:-1]]
        spacing = float(np.median(np.diff([dataset.metadata.model_times[i] for i in observed])))
        native["width"] = native["width_multiplier"] * spacing
        native.update(train_times=observed, holdout_times=holdout, seed=config["seed"],
                      data_dir=config["data"]["root"], val_split=0.0, normalize=False)
        atomic_json(run_dir / "native_config.json", native)
        sampling = {
            "observed_indices": observed, "heldout_indices": holdout,
            "model_times": list(dataset.metadata.model_times), "physical_times": list(dataset.metadata.physical_times),
            "potential_times": native["tks"], "training_rows": split,
            "fixed_subset_indices": subset_indices, "has_observed_trajectory_ids": False,
            "coupling": "native independent marginal permutations; reshuffled each epoch",
            "knot_training": "fresh independent analytic noise at all knots" if dataset.metadata.name == "knot" else None,
            "requested_updates": config["training"]["steps"], "updates_per_epoch": len(loader),
            "actual_updates": native["epochs"] * len(loader),
            "selection": "final EMA teacher; sweep ranks configurations by the specified W1 EMD",
            "validation_loss": "empty native validation loader; zero is not a validation score",
        }
        atomic_json(run_dir / "sampling_contract.json", sampling)
        potentials = create_potentials(native)
        model = create_model(native, dataset.dim, potentials, config["training"]["device"])
        trainer = MarginalTrainer(
            model=model, train_loader=loader, val_loader=[], save_dir=run_dir,
            lr=native["lr"], epochs=native["epochs"], optimizer="adam", grad_clip=0.0,
            weight_decay=native.get("weight_decay", 0.0), lr_schedule=native.get("lr_schedule"),
            do_otp=True, otp_alpha_type="sigmoid", otp_alpha_slope=native.get("otp_alpha_slope", 6.0),
            otp_alpha_mean_scale=native["otp_alpha_mean_scale"], potentials=potentials,
            device=config["training"]["device"], traj_skips=native["epochs"], save_skips=native["epochs"],
        )
        trainer.metrics_callback = EpochLogger(logger)
        trainer.train()
        verify_source()
        manifest = {
            "method": "otpfm", "method_version": source["upstream_revision"],
            "model_config": {"dim": dataset.dim, "native_config": native, "velocity_weights": "ema"},
            "training_config": {**config["training"], "actual_updates": trainer.global_step},
            "dataset": dataset.contract(), "seed": config["seed"], "source_provenance": source,
            "velocity_contract": "frozen EMA v_func_ema(x,t,t); dx/dmodel_time in saved model coordinates",
            "selected_checkpoint": {"rule": "final_epoch", "epoch": native["epochs"], "global_step": trainer.global_step},
            "evaluation_protocol": "shared_ode", "sampling_contract": sampling,
            "distillation_sampler": "otpfm_conditional",
            "distillation_integration": "Native conditional-tuple PC sampling is not yet integrated",
        }
        checkpoint = save_teacher(run_dir, {"model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}, manifest, dataset.preprocessing)
        # Training is finished before freezing; never freeze the live model
        # while its optimizer still has updates to perform.
        import json

        bound = json.loads((run_dir / "manifest.json").read_text())
        teacher = FrozenOTPFM(model, bound, dataset.preprocessing)
        result = evaluate(teacher, dataset, config["evaluation"], run_dir / "eval", logger=logger)
        atomic_json(run_dir / "result.json", result)
        trainer.plot_losses(show=False)
        trainer.save_losses_csv()
        if logger.run is not None:
            import wandb

            with logger.remote("OTP-FM teacher artifact"):
                artifact = wandb.Artifact(f"{logger.run.id}-teacher", type="model")
                for relative in ("checkpoints/last.pt", "manifest.json", "preprocessing.json", "resolved_config.yaml", "native_config.json", "sampling_contract.json"):
                    artifact.add_file(str(run_dir / relative))
                logger.run.log_artifact(artifact)
        logger.close()
        logger = None
        atomic_json(run_dir / "status.json", {"status": "complete", "evaluation_status": "complete"})
        return checkpoint
    except BaseException as error:
        atomic_json(run_dir / "status.json", {"status": "failed", "error": str(error)})
        if logger is not None:
            logger.close(failed=True)
        raise
    finally:
        gc.enable()
        if trainer is not None:
            for handler in list(logging.getLogger().handlers):
                if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == trainer.log_file:
                    logging.getLogger().removeHandler(handler)
                    handler.close()
