"""Hydra + Lightning entry point for Replica-Flow2PC Benchmark 0.

Examples:
  python train.py world=A model_variant=pair_path data.rho_e=0.5
  python train.py world=B model_variant=marginal_path_vel data.d=16

Model variants (the benchmark's compared models):
  endpoint            (1) one-replica, tau=1 only, density
  marginal_path       (2) one-replica, full path, density
  marginal_path_vel   (3) one-replica path + exact velocity matching
  pair_endpoint       (4) pair law at tau=1 only
  pair_path           (5) full pair-PC path (density + velocity)
"""

import os

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from src.lit import Bench0DataModule, ReplicaPCModule
from src.worlds import MatchedMarginalWorlds

VARIANTS = {
    "endpoint": dict(use_pair=False, use_velocity=False, endpoint_only=True),
    "marginal_path": dict(use_pair=False, use_velocity=False, endpoint_only=False),
    "marginal_path_vel": dict(use_pair=False, use_velocity=True, endpoint_only=False),
    "pair_endpoint": dict(use_pair=True, use_velocity=False, endpoint_only=True),
    "pair_path": dict(use_pair=True, use_velocity=True, endpoint_only=False),
}


def load_env(path=".env"):
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k, v)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    load_env(os.path.join(hydra.utils.get_original_cwd(), ".env"))
    L.seed_everything(cfg.seed, workers=True)

    flags = VARIANTS[cfg.model_variant]
    dm = Bench0DataModule(
        world=cfg.world, endpoint_only=flags["endpoint_only"], **cfg.data
    )
    worlds = MatchedMarginalWorlds(
        d=cfg.data.d,
        rho_e=cfg.data.rho_e,
        total_var=cfg.data.total_var,
        rho_ci=cfg.data.rho_ci,
        flow_seed=cfg.data.flow_seed,
    )
    oracle = worlds.oracle_moments(tau=1.0, world=cfg.world, n=cfg.oracle_samples)

    lit = ReplicaPCModule(
        d=cfg.data.d,
        M=cfg.model.M,
        order=cfg.model.order,
        time_mlp=cfg.model.time_mlp,
        lr=cfg.lr,
        max_steps=cfg.trainer.max_steps,
        oracle=oracle,
        **flags,
        **cfg.loss,
    )

    loggers = [hydra.utils.instantiate(lg) for lg in cfg.logger.values()]
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(cfg.paths.run_dir, "checkpoints"),
            save_last=True,
            every_n_train_steps=2000,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    trainer = hydra.utils.instantiate(cfg.trainer, logger=loggers, callbacks=callbacks)
    trainer.fit(lit, datamodule=dm)

    # final decisive numbers -> stdout + summary file
    m = {k: v.cpu() for k, v in lit.pc.moments(1.0).items()}
    oracle = {k: v.cpu() for k, v in oracle.items()}
    eps = 1e-8
    rel = lambda a, b: float(torch.linalg.norm(a - b) / (torch.linalg.norm(b) + eps))  # noqa: E731
    summary = {
        "world": cfg.world,
        "variant": cfg.model_variant,
        "d": cfg.data.d,
        "rho_e": cfg.data.rho_e,
        "rho_ci": cfg.data.rho_ci,
        "err_cross": rel(m["cov_epi"], oracle["cov_cross"]),
        "err_ale": rel(m["cov_ale"], oracle["cov_ale"]),
        "err_total": rel(m["cov_total"], oracle["cov_total"]),
        "epi_trace_model": float(m["cov_epi"].trace()),
        "epi_trace_true": float(oracle["cov_cross"].trace()),
    }
    print("FINAL", summary)
    OmegaConf.save(
        OmegaConf.create(summary), os.path.join(cfg.paths.run_dir, "summary.yaml")
    )


if __name__ == "__main__":
    main()
