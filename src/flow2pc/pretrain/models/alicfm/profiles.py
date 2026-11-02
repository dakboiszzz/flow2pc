"""Dataset-specific original ALI-CFM models; no student dependencies."""

import torch

from flow2pc.pretrain.models.alicfm._source.knot import (
    KnotGeneratorMLP,
    KnotVelocityMLP,
    get_dx_dt,
)
from flow2pc.pretrain.models.alicfm._source.knot import (
    Discriminator as KnotDiscriminator,
)
from flow2pc.pretrain.models.alicfm.model import (
    MLP,
    Discriminator,
    TrainableInterpolant,
)


def profile_for(name):
    if name in ("cite", "multi", "eb", "gom", "beijing"):
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
    def forward(self, joined, t=None):
        if t is not None:
            return super().forward(joined, t)
        return self.net(joined)


def make_models(config, device, *, training=None):
    profile, dim = config.get("profile", "single_cell"), config["dim"]
    if profile == "knot":
        if dim != 2:
            raise ValueError("The supplied knot teacher is two-dimensional")
        # Native construction order and classes: V, G, D.
        field = KnotField(3, 2, config["cfm_dim"]).to(device)
        generator = KnotInterpolant(5, 2, config["interpolant_dim"]).to(device)
        discriminator = KnotDiscriminator(dim + 1, config["disc_dim"]).to(device)
        return generator, discriminator, field
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
        if config.get("native_training", False) or training is not None:
            generator = generator.to(device)
        optimizers = {}
        if training is not None:
            optimizers["pretrain"] = torch.optim.Adam(
                generator.parameters(), lr=training["pretrain_lr"]
            )
        # Preserve the source scripts' seeded initialization order: G, D, V.
        discriminator = Discriminator(dim + 1, config["disc_dim"], apply_sigmoid=False)
        if config.get("native_training", False) or training is not None:
            discriminator = discriminator.to(device)
        if training is not None:
            optimizers["generator"] = torch.optim.Adam(
                generator.parameters(), lr=training["interpolant_gen_lr"]
            )
            optimizers["discriminator"] = torch.optim.Adam(
                discriminator.parameters(), lr=training["interpolant_disc_lr"]
            )
        field = MLP(dim, w=config["cfm_dim"], time_varying=True)
        if training is not None:
            field = field.to(device)
            optimizers["cfm"] = torch.optim.Adam(
                field.parameters(), lr=training["cfm_lr"]
            )
            return generator, discriminator, field, optimizers
    return generator.to(device), discriminator.to(device), field.to(device)


def train_knot(generator, discriminator, field, optimizers, dataset, cfg, logger):
    """Run the native full-curve NumPy GAN/CFM loops, without sampler rewrites."""
    from flow2pc.pretrain.models.alicfm._source.knot_training import train

    if cfg.num_ali_pretrain_steps != 0 or cfg.gan_loss not in ("GAN", "vanilla"):
        raise ValueError(
            "The supplied knot trainer has no regression pretrain and uses vanilla GAN"
        )
    if (
        dataset.heldout is not None
        or dataset.preprocessing.get("normalise")
        or dataset.preprocessing.get("whiten")
    ):
        raise ValueError(
            "The native knot trainer uses all analytic times and raw coordinates"
        )
    train(
        field,
        generator,
        discriminator,
        optimizers["generator"],
        optimizers["discriminator"],
        optimizers["cfm"],
        device=cfg.device,
        batch_size=cfg.batch_size,
        n_epochs_interpolant=cfg.num_ali_train_steps,
        n_epochs_flow=cfg.num_cfm_train_steps,
        size=len(dataset.metadata.model_times),
        std=dataset.params.get("noise_std", 0.01),
        correct_coeff=cfg.correct_coeff,
        logger=logger,
    )
