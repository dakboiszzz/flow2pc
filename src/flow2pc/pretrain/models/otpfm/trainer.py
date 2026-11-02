"""Common OTP-FM method orchestration; original loops belong to _source."""

import copy
import gc
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from flow2pc.utils.checkpoint import atomic_json, save_teacher, sha256
from flow2pc.utils.tracking import LocalLogger

from .profiles import MarginalProfile, get_profile, setup_native


class OTPFMMethod:
    def validate(self, config):
        if isinstance(get_profile(config["data"]["name"]), MarginalProfile):
            from .marginals import validate

            return validate(config)
        if set(config["model"]["params"]) != {"recipe", "native_config"}:
            raise ValueError("OTP-FM model.params requires recipe and the complete native_config mapping")
        get_profile(config["data"]["name"])
        if set(config["training"]) != {"device"}:
            raise ValueError("OTP-FM training requires device; native optimizer/epoch/loss options belong to model.params.native_config")
        if torch.device(config["training"]["device"]).type not in ("cpu", "cuda"):
            raise ValueError("OTP-FM device must be CPU or CUDA")
        if config["evaluation"].get("protocol") != "native_otpfm" or not config["evaluation"]["enabled"]:
            raise ValueError("OTP-FM parity training requires enabled native_otpfm evaluation, including native epoch callbacks")
        native = config["model"]["params"]["native_config"]
        required = {"epochs", "lr", "batch_size", "potential", "tks", "strength",
                    "train_times", "holdout_times", "normalize", "ot_coupling", "val_split"}
        if not isinstance(native, dict) or required - native.keys():
            raise ValueError(f"Incomplete original OTP-FM config; missing {sorted(required - native.keys()) if isinstance(native, dict) else sorted(required)}")
        for name in ("epochs", "batch_size"):
            if type(native[name]) is not int or native[name] < 1:
                raise ValueError(f"OTP-FM {name} must be a positive integer")
        if native["normalize"] is not True or native["val_split"] != 0:
            raise ValueError("Initial native data profiles use all-population StandardScaler and val_split=0; another protocol needs its own data profile")
        if native.get("ema_decay", 0.99) <= 0 or native.get("x_pred", False):
            raise ValueError("Current frozen instantaneous teacher export requires EMA and native velocity prediction")
        if native.get("checkpoint_selection", "mean_w2") not in ("mean_w2", "mean_squared_w2"):
            raise ValueError("Unknown native checkpoint selection rule")

    def fit(self, dataset, config, run_dir):
        self.validate(config)
        if isinstance(get_profile(config["data"]["name"]), MarginalProfile):
            from .marginals import fit

            return fit(dataset, config, run_dir)
        from .provenance import verify_source

        provenance = verify_source()
        from ._source import common_evaluation, common_trainer  # noqa: F401
        from ._source.reproduction import trainer_kwargs
        from .tracking import NativeObserver
        from .evaluation import write_results

        run_dir = Path(run_dir)
        if run_dir.exists():
            raise FileExistsError("Output directory exists; preserve it and select a new OTP-FM run directory")
        native = copy.deepcopy(config["model"]["params"]["native_config"])
        native.update(seed=config["seed"], data_dir=str(dataset.root / get_profile(dataset.metadata.name).source_subdir))
        if native["train_times"] != dataset.observed or native["holdout_times"] != dataset.heldout_indices:
            raise ValueError("OTP-FM observed/holdout times differ from the shared data contract")
        run_dir.mkdir(parents=True)
        (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        atomic_json(run_dir / "native_config.json", native)
        atomic_json(run_dir / "status.json", {"status": "running"})
        logger, trainer = None, None
        try:
            logger = LocalLogger(run_dir, config)
            # Exactly the native reproduction's seed order, after W&B startup.
            random.seed(config["seed"])
            np.random.seed(config["seed"])
            torch.manual_seed(config["seed"])
            setup = setup_native(dataset, native, config["training"]["device"])
            trainer = setup["trainer_class"](**trainer_kwargs(native, setup, run_dir, config["training"]["device"], dataset.metadata.name))
            # Native global master CSV is an unsafe shared output under parallel
            # jobs. Redirect this I/O only; all training/evaluation methods stay native.
            append_native = trainer.append_to_master_csv
            trainer.append_to_master_csv = lambda: append_native(master_csv_path=run_dir / "native_master_results.csv")
            observer = NativeObserver(logger, run_dir, dataset.metadata.name, native)
            trainer.metrics_callback = observer
            tuples = setup["train_loader"].dataset
            if hasattr(tuples, "ot_chains"):
                np.save(run_dir / "ot_chain_indices.npy", tuples.ot_chains, allow_pickle=False)
            atomic_json(run_dir / "sampling_contract.json", {
                "observed_indices": dataset.observed, "heldout_indices": dataset.heldout_indices,
                "potential_times": native["tks"], "ot_coupling": native["ot_coupling"],
                "chain_indices": "ot_chain_indices.npy" if hasattr(tuples, "ot_chains") else None,
                "has_observed_trajectory_ids": False,
            })
            trainer.train()
            trainer.post_training(show=False, create_animation=False)
            if observer.best is None or observer.best_state is None:
                raise RuntimeError("Native OTP-FM training produced no selected model")
            verify_source()
            manifest = {
                "method": "otpfm", "method_version": provenance["upstream_revision"],
                "model_config": {"dim": dataset.dim, "native_config": native, "velocity_weights": "ema"},
                "training_config": {"device": config["training"]["device"], "native": native},
                "dataset": dataset.contract(), "seed": config["seed"], "source_provenance": provenance,
                "selected_checkpoint": {**observer.best, "global_step": observer.best_step, "sampling_alpha": observer.best_alpha,
                                        "native_path": "models/best.pt", "native_sha256": sha256(run_dir / "models/best.pt")},
                "velocity_contract": "frozen EMA v_func_ema(x, t, t); normalized state and global model time",
                "native_sampling_contract": "EMA finite-interval model.sample; not diagonal ODE evaluation",
                "distillation_sampler": "otpfm_conditional",
                "distillation_integration": "Native conditional-tuple PC sampler is not yet integrated; no ALI generator fallback",
            }
            checkpoint = save_teacher(run_dir, {"model_state_dict": observer.best_state}, manifest, dataset.preprocessing, checkpoint_name="best")
            result = write_results(trainer, dataset, native, run_dir, observer.best, logger=logger)
            atomic_json(run_dir / "result.json", result)
            if logger.run is not None:
                import wandb

                with logger.remote("OTP-FM teacher artifact"):
                    artifact = wandb.Artifact(f"{logger.run.id}-teacher", type="model")
                    for name in ("checkpoints/best.pt", "manifest.json", "preprocessing.json", "resolved_config.yaml", "native_config.json", "sampling_contract.json"):
                        artifact.add_file(str(run_dir / name))
                    logger.run.log_artifact(artifact)
            logger.close()
            logger = None
            atomic_json(run_dir / "status.json", {"status": "complete"})
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

    def load(self, checkpoint, device="cpu"):
        from .adapter import load

        return load(checkpoint, device)
