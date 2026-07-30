"""Distillation LightningModule: recompile a frozen flow into a PC.

The ONLY loss is the velocity match E || u_root(Z_t) - u_theta(Z_t) ||^2
with Z_t on the marginal path. Works with every PC family in repc.*
(they all expose .velocity(z_flat, t) and .log_prob(z_flat, t)).
"""

import lightning as L
import torch
from torch import nn


class DistillModule(L.LightningModule):
    def __init__(self, pc, flow, lr=1e-3, max_steps=30000):
        super().__init__()
        self.save_hyperparameters(ignore=["pc", "flow"])
        self.pc = pc
        self.flow = flow
        self.flow.eval()
        for p in self.flow.parameters():
            p.requires_grad_(False)

    def _batch_x(self, batch):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return x

    def _velocity_loss(self, x1, t):
        if hasattr(self.flow, "marginal_sample"):
            zt = self.flow.marginal_sample(x1, t)
        else:
            z0 = torch.randn_like(x1)
            tb = t.reshape(-1, *([1] * (x1.dim() - 1))) if t.dim() else t
            zt = (1 - tb) * z0 + tb * x1
        with torch.no_grad():
            u_theta = self.flow(zt, t.expand(x1.shape[0]) if t.dim() == 0 else t)
        u_pc = self.pc.velocity(zt.flatten(1), t)
        return nn.functional.mse_loss(u_pc, u_theta.flatten(1))

    def training_step(self, batch, _):
        x1 = self._batch_x(batch)
        # one t per batch keeps leaf params (K,D) instead of (B,K,D)
        t = torch.rand((), device=x1.device)
        loss = self._velocity_loss(x1, t)
        self.log("train/velocity_mse", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x1 = self._batch_x(batch)
        # fixed t grid for comparable validation numbers
        for tv in (0.25, 0.5, 0.75, 0.95):
            t = torch.tensor(tv, device=x1.device)
            loss = self._velocity_loss(x1, t)
            self.log(f"val/velocity_mse_t{tv}", loss, sync_dist=True)
        if batch_idx == 0:
            with torch.no_grad():
                t1 = torch.tensor(1.0, device=x1.device)
                ll = self.pc.log_prob(x1.flatten(1), t1).mean()
            self.log("val/exact_ll_t1", ll, sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.pc.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.max_steps
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }
