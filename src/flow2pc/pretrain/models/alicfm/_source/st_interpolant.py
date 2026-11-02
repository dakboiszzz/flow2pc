"""ST-specific methods recovered from ALI-CFM 501e9db; see source provenance."""

import torch
from torch.func import jacrev, vmap

from .nets import TrainableInterpolant


class STInterpolant(TrainableInterpolant):
    def compute_piecewise_reg_term(self, xts, metric='l2'):
        """Match the global interpolant to linear segments of the OT-coupled path."""
        if metric != 'l2':
            raise ValueError("The spatial-transcriptomics piecewise objective uses metric='l2'")

        times = sorted(xts)
        path = torch.stack([xts[t] for t in times], dim=1)
        batch_size, n_times, _ = path.shape
        device = path.device
        segment = torch.randint(n_times - 1, (batch_size,), device=device)
        times_tensor = torch.as_tensor(times, dtype=path.dtype, device=device)
        t0 = times_tensor[segment].unsqueeze(-1)
        t1 = times_tensor[segment + 1].unsqueeze(-1)
        t = t0 + torch.rand_like(t0) * (t1 - t0)

        rows = torch.arange(batch_size, device=device)
        left = path[rows, segment]
        right = path[rows, segment + 1]
        alpha = (t - t0) / (t1 - t0)
        target = (1 - alpha) * left + alpha * right

        prediction = self(path[:, 0], path[:, -1], t)
        return (prediction - target).pow(2).mean()

    def dI_dt(self, x0, x1, t):
        t = t[..., None] if t.ndim == 1 else t

        def _interpolnet(x0i, x1i, ti):
            input_ = torch.cat([x0i, x1i, ti], dim=0).unsqueeze(0)
            out = self.interpolant_net(input_)
            return out, out

        (corr_jac, corr_output) = vmap(
            jacrev(_interpolnet, argnums=2, has_aux=True))(x0, x1, t)
        return (
            (x1 - x0) + (1 - 2 * t) * corr_output.squeeze() + 
            t * (1 - t) * corr_jac.squeeze()
        )
