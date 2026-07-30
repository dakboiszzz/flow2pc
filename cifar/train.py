"""Hydra + Lightning entry point.

Examples:
  # pretrain flows
  python train.py task=flow data=mnist model=flow_unet
  python train.py task=flow data=miniboone model=flow_mlp

  # recompile (distill) into a PC — single velocity-matching loss
  python train.py task=distill data=mnist model=pc_deep \\
      task.flow_ckpt=outputs/runs/flow/<run>/checkpoints/last.ckpt
  python train.py task=distill data=miniboone model=pc_general \\
      task.flow_ckpt=... task.flow_model._target_=repc.flow.MLPFlow
"""

import os

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf


def load_env(path=".env"):
    """Minimal .env loader (no external dependency). Never logs values."""
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k, v)


def strip_prefix(state_dict, prefix):
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def build_net(model_cfg, dim):
    """Instantiate a model config, injecting the data dimension where the
    config declares `D: null` (avoids hardcoded dims)."""
    cfg = OmegaConf.to_container(model_cfg, resolve=True)
    if "D" in cfg and cfg["D"] is None:
        cfg["D"] = dim
    target = cfg.pop("_target_")
    return (
        hydra.utils.get_class(target)(**cfg)
        if False
        else hydra.utils.instantiate({"_target_": target, **cfg})
    )


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    load_env(os.path.join(hydra.utils.get_original_cwd(), ".env"))
    L.seed_everything(cfg.seed, workers=True)

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup()

    if cfg.task.name == "flow":
        net = build_net(cfg.model, datamodule.dim)
        lit = hydra.utils.get_class(cfg.task.lit_target)(
            net=net,
            lr=cfg.task.lr,
            weight_decay=cfg.task.get("weight_decay", 0.0),
            max_steps=cfg.trainer.max_steps,
        )
    elif cfg.task.name == "distill":
        flow = build_net(cfg.task.flow_model, datamodule.dim)
        if cfg.task.flow_ckpt != "hub":
            ckpt = torch.load(cfg.task.flow_ckpt, map_location="cpu", weights_only=True)
            flow.load_state_dict(strip_prefix(ckpt["state_dict"], "net."))
        model_cfg = cfg.model
        if model_cfg.get("structure") == "clt":
            from repc.clt_structure import learn_patch_clt

            idx = torch.randperm(len(datamodule.train_set))[:2000]
            x = torch.stack([datamodule.train_set[i][0] for i in idx.tolist()]).flatten(
                1
            )
            edges = learn_patch_clt(
                x, model_cfg.H, model_cfg.W, model_cfg.patch, model_cfg.channels
            )
            model_cfg = OmegaConf.create(
                {
                    **OmegaConf.to_container(model_cfg, resolve=True),
                    "clt_edges": [list(e) for e in edges],
                }
            )
            torch.save(
                [list(e) for e in edges],
                os.path.join(cfg.paths.run_dir, "clt_edges.pt"),
            )
        pc = build_net(model_cfg, datamodule.dim)
        if cfg.task.get("data_init") and hasattr(pc, "init_from_data"):
            n_comp = pc.B.shape[0]
            idx = torch.randperm(len(datamodule.train_set))[:n_comp]
            x = torch.stack([datamodule.train_set[i][0] for i in idx.tolist()]).flatten(
                1
            )
            pc.init_from_data(x, sigma1=cfg.task.get("sigma1_init", 0.3))
        if cfg.task.warm_start:
            wck = torch.load(cfg.task.warm_start, map_location="cpu", weights_only=True)
            pc.load_state_dict(strip_prefix(wck["state_dict"], "pc."))
        lit = hydra.utils.get_class(cfg.task.lit_target)(
            pc=pc, flow=flow, lr=cfg.task.lr, max_steps=cfg.trainer.max_steps
        )
    else:
        raise ValueError(f"unknown task {cfg.task.name}")

    loggers = [hydra.utils.instantiate(lg) for lg in cfg.logger.values()]
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(cfg.paths.run_dir, "checkpoints"),
            every_n_train_steps=2000,
            save_last=True,
            save_top_k=1,
            monitor="train/velocity_mse"
            if cfg.task.name == "distill"
            else "train/loss",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    trainer = hydra.utils.instantiate(cfg.trainer, logger=loggers, callbacks=callbacks)
    trainer.fit(lit, datamodule=datamodule)


if __name__ == "__main__":
    main()
