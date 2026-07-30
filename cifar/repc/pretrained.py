"""Off-the-shelf pretrained diffusion models as recompilation teachers.

No teacher training: a public DDPM checkpoint is wrapped to expose exactly
what distillation needs, in the flow-time convention (t=0 noise, t=1 data):

  forward(z, t)        probability-flow ODE velocity  dz/dt
  marginal_sample(x,t) a draw from the teacher's own path marginal p_t(.|x)

VP path: x_s = sqrt(abar_s) x0 + sqrt(1-abar_s) eps, s = (1-t)(S-1).
PF-ODE in s: dx/ds = -1/2 beta_s (x - eps_hat/sigma_s); chain rule with
ds/dt = -(S-1) gives the flow-time velocity. At t=0, p ~ N(0, I) — the PC
boundary condition is unchanged.
"""

import torch
from torch import nn


class DDPMTeacher(nn.Module):
    def __init__(self, model_id="google/ddpm-cifar10-32"):
        super().__init__()
        from diffusers import DDPMScheduler, UNet2DModel

        self.unet = UNet2DModel.from_pretrained(model_id)
        sched = DDPMScheduler.from_pretrained(model_id)
        self.register_buffer("abar", sched.alphas_cumprod.float(), persistent=False)
        self.register_buffer("betas", sched.betas.float(), persistent=False)
        self.S = len(sched.betas)

    def _sidx(self, t):
        return ((1.0 - t) * (self.S - 1)).round().long().clamp(0, self.S - 1)

    def forward(self, z, t):
        """Flow-time velocity of the probability-flow ODE."""
        s = self._sidx(t)
        eps = self.unet(z, s).sample
        beta = self.betas[s].view(-1, 1, 1, 1)
        sigma = (1.0 - self.abar[s]).sqrt().clamp_min(1e-4).view(-1, 1, 1, 1)
        return (self.S - 1) * 0.5 * beta * (z - eps / sigma)

    @torch.no_grad()
    def marginal_sample(self, x1, t):
        """Draw z_t from the teacher's own (VP) path marginal."""
        tt = t.expand(x1.shape[0]) if t.dim() == 0 else t
        s = self._sidx(tt)
        a = self.abar[s].sqrt().view(-1, 1, 1, 1)
        sig = (1.0 - self.abar[s]).sqrt().view(-1, 1, 1, 1)
        return a * x1 + sig * torch.randn_like(x1)
