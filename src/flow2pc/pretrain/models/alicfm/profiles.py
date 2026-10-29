"""Dataset-specific original ALI-CFM models; no student dependencies."""

import torch

from flow2pc.pretrain.models.alicfm._source.knot import (
    KnotGeneratorMLP,
    KnotVelocityMLP,
    get_dx_dt,
)
from flow2pc.pretrain.models.alicfm.model import (
    MLP,
    Discriminator,
    TrainableInterpolant,
)


def profile_for(name):
    if name in ("cite", "multi", "eb"):
        return "single_cell"
    if name in ("st", "cell_tracking", "knot"):
        return name
    raise ValueError(f"No ALI-CFM profile for dataset {name!r}")


class KnotInterpolant(KnotGeneratorMLP):
    def forward(self, x0, x1, t, training=False):
        return super().forward(x0, x1, t)

    def dI_dt(self, x0, x1, t):
        return get_dx_dt(self, x0, x1, t)


class KnotField(KnotVelocityMLP):
    def forward(self, joined):
        return self.net(joined)


def make_models(config, device):
    profile, dim = config.get("profile", "single_cell"), config["dim"]
    if profile == "knot":
        if dim != 2:
            raise ValueError("The supplied knot teacher is two-dimensional")
        field = KnotField(3, 2, config["cfm_dim"])
        generator = KnotInterpolant(5, 2, config["interpolant_dim"])
        discriminator = Discriminator(dim + 1, config["disc_dim"], apply_sigmoid=False)
    else:
        cls = TrainableInterpolant
        if profile == "st":
            from flow2pc.pretrain.models.alicfm._source.st_interpolant import (
                STInterpolant,
            )

            cls = STInterpolant
        elif profile == "cell_tracking":
            from flow2pc.pretrain.models.alicfm._source.tracking import (
                AuthorTrackingInterpolant,
            )

            cls = AuthorTrackingInterpolant
        generator = cls(
            dim,
            config["interpolant_dim"],
            config["t_smooth"],
            time_varying=profile != "single_cell",
            regulariser="piecewise" if profile == "st" else "linear",
        )
        # Preserve the source scripts' seeded initialization order: G, D, V.
        discriminator = Discriminator(dim + 1, config["disc_dim"], apply_sigmoid=False)
        field = MLP(dim, w=config["cfm_dim"], time_varying=True)
    return generator.to(device), discriminator.to(device), field.to(device)


def train_knot(generator, discriminator, field, optimizers, dataset, cfg, logger):
    """Original local knot GAN + linear-reference MSE + velocity MSE.

    Samples fresh independent noisy analytic marginals, as the supplied
    train_teacher.py did, without inferring observed trajectory identities.
    """
    from flow2pc.data.synthetic.knot import knot_means

    if cfg.num_ali_pretrain_steps != 0 or cfg.gan_loss not in ("GAN", "vanilla"):
        raise ValueError(
            "The supplied knot trainer has no regression pretrain and uses vanilla GAN"
        )
    means, _ = knot_means(dataset.params.get("size", 120))
    means = torch.tensor(means, device=cfg.device, dtype=torch.float32)
    shift, scale = dataset.shift.to(means), dataset.scale.to(means)
    noise = dataset.params.get("noise_std", 0.01)
    observed = dataset.observed

    def draw(index):
        raw = means[index] + noise * torch.randn(cfg.batch_size, 2, device=cfg.device)
        return (raw - shift) / scale

    for step in range(cfg.num_ali_train_steps):
        index = observed[int(torch.randint(1, len(observed) - 1, ()))]
        t = means.new_full((cfg.batch_size, 1), index / (len(means) - 1))
        x0, x1, real = draw(0), draw(len(means) - 1), draw(index)
        optimizers["discriminator"].zero_grad()
        with torch.no_grad():
            fake = generator(x0, x1, t)
        d_loss = (
            torch.nn.functional.softplus(
                -discriminator(torch.cat([real, t], -1))
            ).mean()
            + torch.nn.functional.softplus(
                discriminator(torch.cat([fake, t], -1))
            ).mean()
        )
        d_loss.backward()
        optimizers["discriminator"].step()
        optimizers["generator"].zero_grad()
        fake = generator(x0, x1, t)
        adversarial = torch.nn.functional.softplus(
            -discriminator(torch.cat([fake, t], -1))
        ).mean()
        regularizer = (fake - ((1 - t) * x0 + t * x1)).square().mean()
        g_loss = adversarial + cfg.correct_coeff * regularizer
        g_loss.backward()
        optimizers["generator"].step()
        logger.log(
            {
                "ali/gan/step": step,
                "ali/gan/d_loss": float(d_loss.detach()),
                "ali/gan/g_loss": float(g_loss.detach()),
                "ali/gan/regularizer": float(regularizer.detach()),
            }
        )
    generator.requires_grad_(False)
    for step in range(cfg.num_cfm_train_steps):
        x0, x1 = draw(0), draw(len(means) - 1)
        t = torch.rand(cfg.batch_size, 1, device=cfg.device)
        optimizers["cfm"].zero_grad()
        target = generator.dI_dt(x0, x1, t).detach()
        with torch.no_grad():
            values = generator(x0, x1, t)
        loss = (field(torch.cat([values, t], -1)) - target).square().mean()
        loss.backward()
        optimizers["cfm"].step()
        logger.log({"ali/cfm/step": step, "ali/cfm/loss": float(loss.detach())})
