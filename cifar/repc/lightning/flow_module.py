"""Flow-matching LightningModule (works for image and tabular nets)."""

import lightning as L
import torch
from torch import nn


class FlowMatchingModule(L.LightningModule):
    def __init__(self, net, lr=2e-4, weight_decay=0.0, max_steps=30000):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net

    def _batch_x(self, batch):
        return batch[0] if isinstance(batch, (list, tuple)) else batch

    def _loss(self, x1):
        z0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device)
        tb = t.reshape(-1, *([1] * (x1.dim() - 1)))
        zt = (1 - tb) * z0 + tb * x1
        return nn.functional.mse_loss(self.net(zt, t), x1 - z0)

    def training_step(self, batch, _):
        loss = self._loss(self._batch_x(batch))
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        self.log("val/loss", self._loss(self._batch_x(batch)), sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.max_steps
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }
