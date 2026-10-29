"""Matching ST pretrain/GAN loops from ALI-CFM 501e9db. Logging is injectable."""

import torch
import torch.nn.functional as F
import wandb
from tqdm.auto import trange

from . import training_utils as utils


def pretain_interpolant(
    interpolant, pretrain_optimizer_G, ot_sampler, 
    train_data, train_timesteps, metric_prefix, cfg,
    logger=None,
):
    logger = wandb if logger is None else logger
    if cfg.num_ali_pretrain_steps == 0:
        return
    
    for epoch in trange(cfg.num_ali_pretrain_steps, 
                        desc="Pretraining Interpolant", leave=False):
        pretrain_optimizer_G.zero_grad()
        batch = utils.sample_gan_batch(
            train_data, cfg.batch_size, 
            divisor=max(train_timesteps), 
            ot_sampler=ot_sampler, 
            ot=cfg.pretain_ot, 
            times=train_timesteps
        )
        x0, x1, xt, t = (x.to(cfg.device) for x in batch)
        xt_fake = interpolant(x0, x1, t)

        # if mnist:
        #     loss = (xt_fake - xt).abs().mean()
        
        loss = (xt_fake - xt).pow(2).mean()
        loss.backward()

        pretrain_optimizer_G.step()
        logger.log({
            f"{metric_prefix}/pretrain_loss": loss.item(), 
            f"{metric_prefix}_step": epoch}
        )


def train_interpolant_with_gan(
    interpolant, discriminator, ot_sampler, data,
    gan_optimizer_G, gan_optimizer_D, metric_prefix,
    train_timesteps, seed, min_max, cfg,
    plot_freaquency=5000,
    plot_fn=utils.sc_plot_fn,
    logger=None,
):
    logger = wandb if logger is None else logger
    t_max = max(train_timesteps)
    for epoch in trange(cfg.num_ali_train_steps,
                        desc="Training GAN Interpolant", leave=False):
        xts, t = utils.sample_full_batch(
            data, cfg.batch_size,
            ot_sampler=ot_sampler,
            divisor=t_max,
            ot=cfg.interpolant_ot,
            times=train_timesteps,
        )
        xts = {t: x.to(cfg.device) for t, x in xts.items()}
        x0, x1, xt = xts[0], xts[1], xts[t]
        t = torch.ones(cfg.batch_size, 1, device=cfg.device) * t

        xt_fake = interpolant(x0, x1, t)

        real_inputs = torch.cat([xt, t], dim=-1)
        fake_inputs = torch.cat([xt_fake.detach(), t], dim=-1)
        
        real_proba = discriminator(real_inputs)
        fake_proba = discriminator(fake_inputs)

        # Train discriminator
        gan_optimizer_D.zero_grad()
        if (cfg.gan_loss == "RpGAN") or (cfg.gan_loss == "R3GAN"):
            # Using the relativistic pairing GAN loss
            r1 = 0
            r2 = 0
            if cfg.gan_loss == "R3GAN":
                xt_fake_ = torch.cat([xt_fake, t], dim=-1).detach().requires_grad_(True)
                xt_ = torch.cat([xt, t], dim=-1).detach().requires_grad_(True)
                disc_score_fake = discriminator(xt_fake_)
                disc_score_real = discriminator(xt_)
                grad_D = torch.autograd.grad(
                    outputs=disc_score_real.sum(), inputs=xt_, create_graph=True
                )[0]
                grad_G = torch.autograd.grad(
                    outputs=disc_score_fake.sum(), inputs=xt_fake_, create_graph=True
                )[0]
                r1 = 0.5 * ((grad_D.view(cfg.batch_size, -1).norm(2, dim=1) - 0) ** 2)
                r2 = 0.5 * ((grad_G.view(cfg.batch_size, -1).norm(2, dim=1) - 0) ** 2)
            d_loss = F.softplus((fake_proba - real_proba) + r1 + r2).mean()

            d_loss.backward()
            gan_optimizer_D.step()

            # Train generator
            gan_optimizer_G.zero_grad()
            real_inputs = torch.cat([xt, t], dim=-1)
            fake_inputs = torch.cat([xt_fake, t], dim=-1)

            real_proba = discriminator(real_inputs)
            fake_proba = discriminator(fake_inputs)
            g_loss_ = F.softplus((real_proba - fake_proba)).mean()
        else:
            d_real_loss = F.softplus(-real_proba).mean()
            d_fake_loss = F.softplus(fake_proba).mean()
            d_loss = d_real_loss + d_fake_loss

            d_loss.backward()
            gan_optimizer_D.step()

            # Train generator
            gan_optimizer_G.zero_grad()
            fake_inputs = torch.cat([xt_fake, t], dim=-1)
            fake_proba = discriminator(fake_inputs)

            g_loss_ = F.softplus(-fake_proba).mean()
        
        if cfg.reg_term_type == "linear":
            reg_weight_loss = interpolant.compute_linear_reg_term(
                xt_fake, xts, t,
                metric=cfg.reg_metric,
                land_gamma=getattr(cfg, "land_gamma", 0.2),
                land_t_gamma=getattr(cfg, "land_t_gamma", 0.2),
            )
        elif cfg.reg_term_type in {"piecewise", "picewise_oskar_version"}:
            reg_weight_loss = interpolant.compute_piecewise_reg_term(
                xts, metric=cfg.reg_metric
            )
        elif cfg.reg_term_type == "2nd_derivative":
            reg_weight_loss = interpolant.compute_second_derivative_reg_term(
                xts,
                metric=cfg.reg_metric,
                land_gamma=getattr(cfg, "land_gamma", 0.2),
                land_t_gamma=getattr(cfg, "land_t_gamma", 0.2),
                num_t_samples=getattr(cfg, "num_t_steps", 3),
            )
        else:
            raise ValueError(f"Unknown regularizer type: {cfg.reg_term_type}")
        g_loss = g_loss_ + cfg.correct_coeff * reg_weight_loss
        g_loss.backward()
        gan_optimizer_G.step()

        logger.log({
            f"{metric_prefix}/d_loss": d_loss.item(),
            f"{metric_prefix}/g_loss": g_loss.item(),
            f"{metric_prefix}/g_loss_": g_loss_.item(),
            f"{metric_prefix}/reg_weight_loss": reg_weight_loss.item(),
            f"{metric_prefix}/fake_proba": torch.sigmoid(fake_proba).mean().item(),
            f"{metric_prefix}/real_proba": torch.sigmoid(real_proba).mean().item(),
            f"{metric_prefix}_step": epoch
        })

        if plot_freaquency and plot_freaquency > 0 and epoch % plot_freaquency == 0:
            plot_fn(
                interpolant, epoch, seed, t_max, data, ot_sampler, cfg.device,
                metric_prefix, train_timesteps, wandb, min_max
            )
