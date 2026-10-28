import random
import wandb
import torch
import torch.nn.functional as F
from tqdm.auto import trange


from . import training_utils as utils


def pretain_interpolant(
    interpolant, pretrain_optimizer_G, ot_sampler,
    train_data, train_timesteps, metric_prefix, cfg,
    *, logger=None,
):
    logger = wandb if logger is None else logger
    if cfg.num_ali_pretrain_steps == 0:
        return
    batch_size = int(getattr(cfg, "ali_batch_size", cfg.batch_size))

    for epoch in trange(cfg.num_ali_pretrain_steps,
                        desc="Pretraining Interpolant", leave=False):
        pretrain_optimizer_G.zero_grad()
        xts, time = utils.sample_full_batch(
            train_data, batch_size, divisor=max(train_timesteps),
            ot_sampler=ot_sampler, ot=cfg.pretain_ot, times=train_timesteps,
        )
        x0, x1, xt = (xts[key].to(cfg.device) for key in [0, 1, time])
        t = torch.full((batch_size, 1), time, device=cfg.device)
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
    plot_fn=utils.sc_plot_fn, logger=None,
):
    logger = wandb if logger is None else logger
    t_max = max(train_timesteps)
    batch_size = int(getattr(cfg, "ali_batch_size", cfg.batch_size))
    reference_data = None
    if cfg.reg_term_type == '2nd_derivative' and cfg.reg_metric == 'land':
        # Eq. 20 uses the observed dataset, not only this step's OT minibatch.
        samples = torch.cat([data[time] for time in train_timesteps]).to(
            cfg.device, dtype=torch.float64,
        )
        sample_times = torch.cat([
            torch.full((len(data[time]), 1), time / t_max, dtype=torch.float64)
            for time in train_timesteps
        ]).to(cfg.device)
        reference_data = (samples, sample_times)
    for epoch in trange(cfg.num_ali_train_steps,
                        desc="Training GAN Interpolant", leave=False):

        xts, t = utils.sample_full_batch(
            data, batch_size,
            ot_sampler=ot_sampler,
            divisor=t_max,
            ot=cfg.interpolant_ot,
            times=train_timesteps,
        )
        xts = {t: x.to(cfg.device) for t, x in xts.items()}
        x0, x1, xt = xts[0], xts[1], xts[t]
        t = torch.ones(batch_size, 1, device=cfg.device) * t

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
                xt_fake_ = xt_fake.detach().requires_grad_(True)
                xt_ = xt.detach().requires_grad_(True)
                disc_score_fake = discriminator(torch.cat([xt_fake_, t], dim=-1))
                disc_score_real = discriminator(torch.cat([xt_, t], dim=-1))
                grad_D = torch.autograd.grad(
                    outputs=disc_score_real.sum(), inputs=xt_, create_graph=True
                )[0]
                grad_G = torch.autograd.grad(
                    outputs=disc_score_fake.sum(), inputs=xt_fake_, create_graph=True
                )[0]
                r1 = 0.5 * ((grad_D.view(batch_size, -1).norm(2, dim=1) - 0) ** 2)
                r2 = 0.5 * ((grad_G.view(batch_size, -1).norm(2, dim=1) - 0) ** 2)
            d_loss = F.softplus(fake_proba - real_proba).mean()
            if cfg.gan_loss == "R3GAN":
                d_loss = d_loss + cfg.get("r3gan_gamma", 1.0) * (r1 + r2).mean()

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

        match cfg.reg_term_type:
            case "st":
                t = torch.rand(x0.shape[0], 1, device=cfg.device)
                xt_fake = interpolant(x0, x1, t, training=False)

                coupled_x = torch.stack([x0, xt, x1], dim=1) # (B, 3, D)
                bs, K, d = coupled_x.shape
                times = torch.tensor(train_timesteps, device=cfg.device) / t_max

                idx = torch.bucketize(t, times) - 1
                idx = idx.clamp(0, K - 2)  # keep in valid range

                t0 = times[idx]  # (bs,)
                t1 = times[idx + 1]  # (bs,)
                denom = (t1 - t0)  # (bs,)

                a = (t1 - t) / denom
                b = (t - t0) / denom

                # Select endpoints for each batch element
                x0 = coupled_x[torch.arange(bs), idx]  # (bs, d)
                x1 = coupled_x[torch.arange(bs), idx + 1]  # (bs, d)

                xhat_t = a.unsqueeze(-1) * x0 + b.unsqueeze(-1) * x1
                reg_weight_loss = ((xt_fake - xhat_t) ** 2).mean()

            case "picewise_oskar_version":
                t = torch.rand(x0.shape[0], 1, device=cfg.device)
                xt_fake = interpolant(x0, x1, t, training=False)

                coupled_x = torch.stack([x0, xt, x1], dim=1)
                bs, K, d = coupled_x.shape
                times = torch.tensor(train_timesteps, device=cfg.device) / t_max
                # One segment index per sample; [B, 1] indices broadcast
                # against [B] row indices and incorrectly create [B, B, D].
                idx = (torch.bucketize(t.squeeze(-1), times) - 1).clamp(0, K - 2)
                t0 = times[idx].unsqueeze(-1)
                t1 = times[idx + 1].unsqueeze(-1)
                a = (t1 - t) / (t1 - t0)
                b = (t - t0) / (t1 - t0)
                rows = torch.arange(bs, device=cfg.device)
                left = coupled_x[rows, idx]
                right = coupled_x[rows, idx + 1]
                xhat_t = a * left + b * right
                diff = xt_fake - xhat_t

                if cfg.reg_metric == 'l2':
                    reg_weight_loss = (diff ** 2).mean()
                else:
                    raise ValueError(f"Unknown metric {cfg.reg_metric}")

            case 'linear':
                reg_weight_loss = \
                    interpolant.compute_linear_reg_term(
                        xt_fake, xts, t,
                        cfg.reg_metric, getattr(cfg, "land_gamma", 0.2), getattr(cfg, "land_t_gamma", 0.2)
                    )

            case 'piecewise':
                t1_idx = random.randint(0, len(train_timesteps) - 2)
                t1, t2 = train_timesteps[t1_idx], train_timesteps[t1_idx + 1]
                t1, t2 = t1 / t_max, t2 / t_max

                reg_weight_loss = \
                    interpolant.compute_piecewise_reg_term(
                        xts, t1, t2,
                        cfg.reg_metric, getattr(cfg, "land_gamma", 0.2), getattr(cfg, "land_t_gamma", 0.2)
                    )

            case '2nd_derivative':
                reg_weight_loss = \
                    interpolant.compute_length_reg_term(
                        xts,
                        cfg.reg_metric, cfg.land_gamma, cfg.land_t_gamma,
                        cfg.num_t_steps, h=cfg.get("finite_difference_h", 0.001),
                        reference_data=reference_data,
                        land_epsilon=cfg.get("land_epsilon", 1e-6),
                    )

            case _:
                raise ValueError(f"Unknown regularization term {cfg.reg_term_type}")

        g_loss = g_loss_ + cfg.correct_coeff * reg_weight_loss
        g_loss.backward()
        gan_optimizer_G.step()

        logger.log({
            f"{metric_prefix}/d_loss": d_loss.item(),
            f"{metric_prefix}/g_loss": g_loss.item(),
            f"{metric_prefix}/g_loss_": g_loss_.item(),
            f"{metric_prefix}/reg_weight_loss": reg_weight_loss.item(),
            f"{metric_prefix}/fake_proba": F.sigmoid(fake_proba).mean().item(),
            f"{metric_prefix}/real_proba": F.sigmoid(real_proba).mean().item(),
            f"{metric_prefix}_step": epoch
        })

        if plot_freaquency > 0 and epoch % plot_freaquency == 0:
            plot_fn(
                interpolant, epoch, seed, t_max, data, ot_sampler, cfg.device,
                metric_prefix, train_timesteps, wandb, min_max
            )


def train_ot_cfm(
    ot_cfm_model, ot_cfm_optimizer, interpolant, ot_sampler,
    train_data, metric_prefix, cfg, train_timesteps, min_max,
    *, logger=None,
):
    logger = wandb if logger is None else logger
    # Algorithm 2 optimises only the vector field; the learned path is fixed.
    interpolant.requires_grad_(False)
    for step in trange(cfg.num_cfm_train_steps,
                       desc="Training OT CFM Interpolant", leave=False):
        ot_cfm_optimizer.zero_grad()

        if cfg.cfm_ot == 'mmot':
            xts, _ = utils.sample_full_batch(
                train_data, cfg.batch_size, divisor=max(train_timesteps),
                ot_sampler=ot_sampler, ot=cfg.cfm_ot, times=train_timesteps,
            )
            x0, x1 = xts[0].to(cfg.device), xts[1].to(cfg.device)
        else:
            x0, x1 = utils.sample_x0_x1(
                train_data, cfg.batch_size, device=cfg.device
            )
            if cfg.cfm_ot in {'border', 'full'}:
                x0, x1 = ot_sampler.sample_plan(x0, x1)

        t = torch.rand(x0.shape[0], 1, device=cfg.device)

        xt = interpolant(x0, x1, t, training=False).detach()
        ut = interpolant.dI_dt(x0, x1, t).detach()

        vt = ot_cfm_model(torch.cat([xt, t], dim=-1))

        loss = (vt - ut).square().sum(-1).mean()
        loss.backward()
        ot_cfm_optimizer.step()

        logger.log({
            f"{metric_prefix}/cfm_loss": loss.item(),
            f"{metric_prefix}_step": step
        })
