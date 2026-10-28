"""Method-owned ALI pretraining, adversarial learning, then CFM regression.

This orchestrator calls the migrated source loops; it does not replace their
losses with a generic flow matcher. It supports one held-out CITE/Multi fold per run.
"""

import json
import math
import subprocess
from pathlib import Path

import torch
import yaml

from flow2pc.pretrain.models.alicfm.model import (
    MLP,
    Discriminator,
    TrainableInterpolant,
)
from flow2pc.utils.checkpoint import atomic_json, atomic_path, save_teacher, sha256
from flow2pc.utils.seed import seed_everything

MODEL_KEYS = {
    "cfm_dim",
    "interpolant_dim",
    "disc_dim",
    "t_smooth",
    "pretain_ot",
    "interpolant_ot",
    "cfm_ot",
    "correct_coeff",
    "reg_metric",
    "land_gamma",
    "land_t_gamma",
    "land_epsilon",
    "reg_term_type",
    "num_t_steps",
    "finite_difference_h",
    "gan_loss",
    "r3gan_gamma",
}
TRAIN_KEYS = {
    "device",
    "batch_size",
    "ali_batch_size",
    "num_ali_pretrain_steps",
    "num_ali_train_steps",
    "num_cfm_train_steps",
    "pretrain_lr",
    "interpolant_gen_lr",
    "interpolant_disc_lr",
    "cfm_lr",
}


class MethodConfig(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None


class LocalLogger:
    def __init__(self, directory, config):
        self.handle = (directory / "metrics.jsonl").open("w")
        self.run = None
        try:
            if config["logging"]["mode"] != "disabled":
                import wandb

                self.run = wandb.init(
                    **config["logging"], config=config, dir=str(directory)
                )
        except BaseException:
            self.handle.close()
            raise

    def log(self, row):
        if any(
            isinstance(v, (int, float)) and not math.isfinite(v) for v in row.values()
        ):
            raise ValueError("Non-finite ALI-CFM training metric")
        self.handle.write(json.dumps(row, allow_nan=False) + "\n")
        self.handle.flush()
        if self.run is not None:
            self.run.log(row)

    def close(self, failed=False):
        self.handle.close()
        if self.run is not None:
            self.run.finish(exit_code=1 if failed else 0)


class ALICFM:
    def validate(self, config):
        model, train = config["model"]["params"], config["training"]
        if (
            set(model) != MODEL_KEYS
            or set(train) - TRAIN_KEYS
            or TRAIN_KEYS - {"ali_batch_size"} - set(train)
        ):
            raise ValueError(
                f"ALI-CFM model missing={sorted(MODEL_KEYS - set(model))}, "
                f"unknown={sorted(set(model) - MODEL_KEYS)}; training "
                f"missing={sorted(TRAIN_KEYS - {'ali_batch_size'} - set(train))}, "
                f"unknown={sorted(set(train) - TRAIN_KEYS)}"
            )
        for key in ("cfm_dim", "interpolant_dim", "disc_dim", "num_t_steps"):
            if type(model[key]) is not int or model[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("batch_size", "num_ali_train_steps", "num_cfm_train_steps"):
            if type(train[key]) is not int or train[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if (
            type(train["num_ali_pretrain_steps"]) is not int
            or train["num_ali_pretrain_steps"] < 0
        ):
            raise ValueError("num_ali_pretrain_steps must be a nonnegative integer")
        if "ali_batch_size" in train and (
            type(train["ali_batch_size"]) is not int or train["ali_batch_size"] < 1
        ):
            raise ValueError("ali_batch_size must be a positive integer")
        for key in (
            "pretrain_lr",
            "interpolant_gen_lr",
            "interpolant_disc_lr",
            "cfm_lr",
        ):
            if (
                type(train[key]) not in (int, float)
                or not math.isfinite(train[key])
                or train[key] <= 0
            ):
                raise ValueError(f"{key} must be finite and positive")
        for key in (
            "land_gamma",
            "land_t_gamma",
            "land_epsilon",
            "finite_difference_h",
        ):
            if (
                type(model[key]) not in (int, float)
                or not math.isfinite(model[key])
                or model[key] <= 0
            ):
                raise ValueError(f"{key} must be finite and positive")
        for key in ("correct_coeff", "t_smooth", "r3gan_gamma"):
            if (
                type(model[key]) not in (int, float)
                or not math.isfinite(model[key])
                or model[key] < 0
            ):
                raise ValueError(f"{key} must be finite and nonnegative")
        if any(
            model[key] not in ("none", "border", "full", "mmot")
            for key in ("pretain_ot", "interpolant_ot", "cfm_ot")
        ):
            raise ValueError("Unsupported ALI-CFM coupling")
        if model["reg_term_type"] not in ("2nd_derivative", "linear") or model[
            "reg_metric"
        ] not in ("l2", "land"):
            raise ValueError(
                "Initial migration supports linear/2nd_derivative with l2/land only"
            )
        if model["gan_loss"] not in ("RpGAN", "R3GAN", "GAN"):
            raise ValueError("Unsupported GAN loss")
        device = torch.device(train["device"])
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Training device must be cpu or cuda")
        if config["data"]["name"] not in ("cite", "multi"):
            raise ValueError("This ALI-CFM trainer supports CITE/Multi only")

    def fit(self, dataset, config, run_dir):
        self.validate(config)
        # Import optional method dependencies only for an actual training call.
        from torchcfm.conditional_flow_matching import OTPlanSampler

        from flow2pc.pretrain.models.alicfm._source import training_funcs as stages

        run_dir = Path(run_dir)
        if run_dir.exists():
            raise FileExistsError(
                "Output directory already exists; use a new run directory"
            )
        cfg = MethodConfig(**config["model"]["params"], **config["training"])
        cfg.dim = dataset.dim
        clouds = [dataset.get_marginal(i, "train").materialize() for i in range(4)]
        if any(len(clouds[i]) == 0 for i in dataset.observed):
            raise ValueError("Empty observed training marginal")
        seed_everything(config["seed"])
        device = cfg.device
        interpolant = TrainableInterpolant(
            cfg.dim, cfg.interpolant_dim, cfg.t_smooth
        ).to(device)
        discriminator = Discriminator(
            cfg.dim + 1, cfg.disc_dim, apply_sigmoid=False
        ).to(device)
        field = MLP(cfg.dim, w=cfg.cfm_dim, time_varying=True).to(device)
        optimizers = {
            "pretrain": torch.optim.Adam(interpolant.parameters(), lr=cfg.pretrain_lr),
            "generator": torch.optim.Adam(
                interpolant.parameters(), lr=cfg.interpolant_gen_lr
            ),
            "discriminator": torch.optim.Adam(
                discriminator.parameters(), lr=cfg.interpolant_disc_lr
            ),
            "cfm": torch.optim.Adam(field.parameters(), lr=cfg.cfm_lr),
        }
        ot_sampler = OTPlanSampler("exact", reg=0.1)
        # Original optimizer algorithms and all three staged loops are preserved.
        run_dir.mkdir(parents=True, exist_ok=False)
        with atomic_path(run_dir / "resolved_config.yaml") as temporary:
            temporary.write_text(yaml.safe_dump(config, sort_keys=False))
        atomic_json(run_dir / "status.json", {"status": "running"})
        logger = None
        try:
            logger = LocalLogger(run_dir, config)
            stages.pretain_interpolant(
                interpolant,
                optimizers["pretrain"],
                ot_sampler,
                clouds,
                dataset.observed,
                "ali/pretrain",
                cfg,
                logger=logger,
            )
            stages.train_interpolant_with_gan(
                interpolant,
                discriminator,
                ot_sampler,
                clouds,
                optimizers["generator"],
                optimizers["discriminator"],
                "ali/gan",
                dataset.observed,
                config["seed"],
                None,
                cfg,
                plot_freaquency=0,
                logger=logger,
            )
            stages.train_ot_cfm(
                field,
                optimizers["cfm"],
                interpolant,
                ot_sampler,
                clouds,
                "ali/cfm",
                cfg,
                dataset.observed,
                None,
                logger=logger,
            )
            provenance_path = Path(__file__).with_name("SOURCE_PROVENANCE.json")
            provenance = json.loads(provenance_path.read_text())
            for file in provenance["files"]:
                if (
                    sha256(Path(__file__).parent / "_source" / file["migrated_file"])
                    != file["migrated_sha256"]
                ):
                    raise ValueError(
                        "Migrated ALI-CFM source differs from recorded provenance"
                    )
            try:
                revision = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=Path(__file__).parents[5],
                ).stdout.strip()
            except (OSError, subprocess.CalledProcessError):
                revision = None
            model_config = dict(config["model"]["params"], dim=dataset.dim)
            manifest = {
                "method": "alicfm",
                "method_version": provenance["upstream_revision"],
                "model_config": model_config,
                "training_config": config["training"],
                "dataset": dataset.contract(),
                "seed": config["seed"],
                "code_revision": revision,
                "source_provenance": provenance,
                "training_protocol": "one_fold; train_only_affine_scaling; native_ALI_then_CFM",
                "distillation_compatibility": "Requires valid query sampler and free-source PC; not asserted by teacher training",
            }

            def state(model):
                return {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                }

            payload = {
                "cfm_model": state(field),
                "interpolant": state(interpolant),
                "discriminator": state(discriminator),
                "optimizer_states": {k: v.state_dict() for k, v in optimizers.items()},
                "torch_rng_state": torch.get_rng_state(),
            }
            checkpoint = save_teacher(run_dir, payload, manifest, dataset.preprocessing)
            atomic_json(run_dir / "status.json", {"status": "complete"})
        except BaseException as error:
            atomic_json(
                run_dir / "status.json", {"status": "failed", "error": str(error)}
            )
            if logger is not None:
                logger.close(failed=True)
            raise
        else:
            logger.close()
        return checkpoint

    def load(self, checkpoint, device="cpu"):
        from flow2pc.pretrain.models.alicfm.adapter import load

        return load(checkpoint, device)
