"""Explicit two-observation adaptation using the existing ALI regularizer/CFM.

There is no interior adversarial observation or generator regression target.
This is recorded as a regularizer-only generator stage, not a full GAN run.
"""

import torch


class ScalarCadence:
    """Bound scalar I/O without changing any optimizer updates or native losses."""

    def __init__(self, logger, every):
        self.logger, self.every = logger, every

    def log(self, row):
        steps = [value for key, value in row.items() if key.endswith("_step")]
        if not steps or any(int(step) % self.every == 0 for step in steps):
            self.logger.log(row)


def train(generator, field, optimizers, sampler, clouds, dataset, cfg, logger):
    from ._source import training_funcs
    from ._source.training_utils import sample_x0_x1

    if len(dataset.observed) != 2:
        raise ValueError("Endpoint adaptation requires exactly two observed times")
    logger.log({"protocol/observed_times": 2, "protocol/interior_gan_active": 0,
                "protocol/generator_regularizer_active": 1})
    cadence = ScalarCadence(logger, cfg.get("scalar_log_every", 100))
    for step in range(cfg.num_ali_train_steps):
        x0, x1 = sample_x0_x1(clouds, cfg.batch_size, device=cfg.device)
        if cfg.interpolant_ot in ("border", "full"):
            x0, x1 = sampler.sample_plan(x0, x1)
        optimizers["generator"].zero_grad(set_to_none=True)
        penalty = generator.compute_length_reg_term(
            {0: x0, 1: x1}, metric=cfg.reg_metric,
            land_gamma=cfg.land_gamma, land_t_gamma=cfg.land_t_gamma,
            num_t_steps=cfg.num_t_steps, h=cfg.finite_difference_h,
            land_epsilon=cfg.land_epsilon,
        )
        loss = cfg.correct_coeff * penalty
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite endpoint interpolant regularizer")
        loss.backward()
        optimizers["generator"].step()
        if step % cadence.every == 0 or step + 1 == cfg.num_ali_train_steps:
            logger.log({"ali/generator_step": step, "ali/generator_regularizer": float(penalty.detach()),
                        "ali/generator_loss": float(loss.detach())})
    # Reuse the actual original field stage. Border/independent coupling avoids
    # sample_full_batch's selection from a nonexistent interior observation.
    training_funcs.train_ot_cfm(
        field, optimizers["cfm"], generator, sampler, clouds, "ali/cfm", cfg,
        dataset.observed, None, logger=cadence,
    )


