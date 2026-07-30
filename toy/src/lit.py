"""Lightning module + datamodule for Benchmark 0.

Loss (the discussion's spec, staged):
  L = l1d * NLL1 + l2d * NLL2 + l1v * velMSE1 + l2v * velMSE2 + lb * NLL1(tau=0)

Model variants of the benchmark are expressed by flags:
  endpoint_only  -> tau == 1 always            (models 1 and 4)
  use_pair       -> pair terms active          (models 4 and 5)
  use_velocity   -> velocity terms active      (models 3 and 5)

Stages (fractions of max_steps): marginal density -> + marginal velocity ->
+ pair density -> + pair velocity -> joint. Terms whose flags are off stay off.
"""

import lightning as L
import torch
from torch.utils.data import DataLoader, IterableDataset

from src.pair_pc import PairPC
from src.worlds import MatchedMarginalWorlds


class PairBatches(IterableDataset):
    def __init__(self, world_cfg, world, batch_size, endpoint_only, seed=0):
        self.worlds = MatchedMarginalWorlds(**world_cfg)
        self.world = world
        self.batch_size = batch_size
        self.endpoint_only = endpoint_only
        self.seed = seed

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        while True:
            tau = 1.0 if self.endpoint_only else float(torch.rand(1, generator=g))
            x1, x2, u1, u2 = self.worlds.sample_pair(
                self.batch_size, tau, self.world, g
            )
            yield x1, x2, u1, u2, torch.tensor(tau)


class Bench0DataModule(L.LightningDataModule):
    def __init__(
        self,
        d,
        rho_e,
        rho_ci=0.0,
        total_var=1.0,
        flow_seed=0,
        world="A",
        batch_size=512,
        endpoint_only=False,
        val_n=4096,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.world_cfg = dict(
            d=d, rho_e=rho_e, total_var=total_var, rho_ci=rho_ci, flow_seed=flow_seed
        )

    def setup(self, stage=None):
        h = self.hparams
        self.worlds = MatchedMarginalWorlds(**self.world_cfg)
        g = torch.Generator().manual_seed(9999)
        self.val_batches = [
            self.worlds.sample_pair(h.val_n, tau, h.world, g) + (torch.tensor(tau),)
            for tau in (0.0, 0.5, 1.0)
        ]

    def train_dataloader(self):
        h = self.hparams
        ds = PairBatches(self.world_cfg, h.world, h.batch_size, h.endpoint_only)
        return DataLoader(ds, batch_size=None, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_batches, batch_size=None, num_workers=0)


class ReplicaPCModule(L.LightningModule):
    def __init__(
        self,
        d,
        M=16,
        order=3,
        time_mlp=False,
        lr=1e-3,
        max_steps=20000,
        use_pair=True,
        use_velocity=True,
        endpoint_only=False,
        lambda_1d=1.0,
        lambda_2d=1.0,
        lambda_1v=1.0,
        lambda_2v=1.0,
        lambda_b=0.2,
        oracle=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["oracle"])
        self.pc = PairPC(d, M=M, order=order, time_mlp=time_mlp)
        self.oracle = oracle  # dict of tensors from MatchedMarginalWorlds

    def _stage_weights(self):
        """Staged curriculum per the discussion; respects variant flags."""
        h = self.hparams
        f = self.global_step / max(1, h.max_steps)
        w = {"1d": h.lambda_1d, "2d": 0.0, "1v": 0.0, "2v": 0.0, "b": h.lambda_b}
        if f > 0.1 and h.use_velocity:
            w["1v"] = h.lambda_1v
        if f > 0.25 and h.use_pair:
            w["2d"] = h.lambda_2d
        if f > 0.4 and h.use_pair and h.use_velocity:
            w["2v"] = h.lambda_2v
        return w

    def _losses(self, batch):
        x1, x2, u1, u2, tau = batch
        out = {}
        out["nll1"] = -self.pc.log_prob1(x1, tau).mean()
        out["nll2"] = -self.pc.log_prob2(x1, x2, tau).mean()
        if self.hparams.use_velocity:
            v1 = self.pc.velocity1(x1, tau)
            out["vel1"] = (v1 - u1).pow(2).sum(-1).mean()
            w1, w2 = self.pc.velocity2(x1, x2, tau)
            out["vel2"] = 0.5 * ((w1 - u1).pow(2) + (w2 - u2).pow(2)).sum(-1).mean()
        else:
            out["vel1"] = out["vel2"] = torch.zeros((), device=x1.device)
        out["nll1_b"] = (
            -self.pc.log_prob1(x1, torch.zeros_like(tau)).mean()
            if not self.hparams.endpoint_only
            else torch.zeros((), device=x1.device)
        )
        return out

    def training_step(self, batch, _):
        w = self._stage_weights()
        ls = self._losses(batch)
        loss = (
            w["1d"] * ls["nll1"]
            + w["2d"] * ls["nll2"]
            + w["1v"] * ls["vel1"]
            + w["2v"] * ls["vel2"]
            + w["b"] * ls["nll1_b"]
        )
        self.log_dict({f"train/{k}": v for k, v in ls.items()}, on_step=True)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, idx):
        ls = self._losses(batch)
        tau = float(batch[-1])
        self.log_dict({f"val/{k}_t{tau:g}": v for k, v in ls.items()})
        if tau == 1.0:
            self._log_uq_errors()

    def _log_uq_errors(self):
        if self.oracle is None:
            return
        m = self.pc.moments(1.0)
        o = {k: v.to(self.device) for k, v in self.oracle.items()}
        eps = 1e-8

        def rel(a, b):
            return (torch.linalg.norm(a - b) / (torch.linalg.norm(b) + eps)).item()

        self.log("val/err_cross", rel(m["cov_epi"], o["cov_cross"]))
        self.log("val/err_ale", rel(m["cov_ale"], o["cov_ale"]))
        self.log("val/err_total", rel(m["cov_total"], o["cov_total"]))

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.pc.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.hparams.max_steps)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }
