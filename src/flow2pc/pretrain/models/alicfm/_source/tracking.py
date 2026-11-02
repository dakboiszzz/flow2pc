import torch
import wandb
from tqdm.auto import trange
from torch.func import jacrev, vmap
from .nets import TrainableInterpolant
from . import training_utils as utils

def denormalize(x, min_max):
    return x if min_max is None else x * (min_max[1].to(x)-min_max[0].to(x)) + min_max[0].to(x)

def denormalize_gradfield(x, min_max):
    return x if min_max is None else x * (min_max[1].to(x)-min_max[0].to(x))


class AuthorTrackingInterpolant(TrainableInterpolant):
    """Keep the upstream derivative without changing single-cell experiments.

    dI_dt is copied from mmacosha/adversarially-learned-interpolants,
    commit b5936c7437276390c13edd18369d2d5e89aaa68d, ali_cfm/nets.py.
    Inherited initialization, forward map, and linear regularizer match that
    source. State-dict names are unchanged, including for existing checkpoints.
    """

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

def train_ot_cfm(
        ot_cfm_model, ot_cfm_optimizer, interpolant,
         ot_sampler, train_data, batch_size, min_max,
         n_ot_cfm_epochs,
         metric_prefix="", ot='border', device='cpu', times=(0, -1),
         normalized_coordinates=False,
    logger=None,
    ):
    logger = wandb if logger is None else logger
    for step in trange(n_ot_cfm_epochs,
                       desc="Training OT CFM Interpolant", leave=False):
        ot_cfm_optimizer.zero_grad()

        if ot == 'mmot':
            t_max = max(times)
            batch = utils.sample_gan_batch(train_data, batch_size, divisor=t_max, ot_sampler=ot_sampler, ot=ot, times=times)
            x0, x1, xt, _ = (x.to(device) for x in batch)
        else:
            x0, x1 = utils.sample_x0_x1(train_data, batch_size, device=device)
            if ot == 'border' or ot == 'full':
                x0, x1 = ot_sampler.sample_plan(x0, x1)

        t = torch.rand(x0.shape[0], 1, device=device)

        xt = interpolant(x0, x1, t, training=False).detach()
        ut = interpolant.dI_dt(x0, x1, t).detach()

        # The visually strong changed_files checkpoint trained CFM in the
        # normalized coordinate system. Pixel-space remains the default recipe.
        if not normalized_coordinates:
            ut = denormalize_gradfield(ut, min_max)
            xt = denormalize(xt, min_max)
        ut = ut.float()
        xt = xt.float()

        vt = ot_cfm_model(torch.cat([xt, t], dim=-1))

        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        ot_cfm_optimizer.step()

        logger.log({
            f"{metric_prefix}/cfm_loss": loss.item(),
            f"{metric_prefix}_step": step
        })
