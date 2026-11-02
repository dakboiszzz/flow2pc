"""Original model/potential factories; no algorithm edits."""

from collections import OrderedDict
from .otpfm import OTPFM
from .potentials import KLPotential, MMDRBFPotential, W2InfPotential, W2Potential

def create_potential(config: dict, tk: float, strength: float = None, width: float = None):
    """Create a potential based on config."""
    # Support both "width" and "lambda_width" in config for compatibility
    default_width = config.get("width") or config.get("lambda_width", "auto")

    common_kwargs = {
        "tk": tk,
        "strength": strength if strength is not None else config["strength"],
        "lambda_type": config.get("lambda_type", "gaussian"),
        "width": width if width is not None else default_width,
    }

    potential_type = config.get("potential", "w2inf").lower()

    match potential_type:
        case "w2inf":
            return W2InfPotential(**common_kwargs)
        case "w2":
            return W2Potential(**common_kwargs)
        case "mmd":
            return MMDRBFPotential(
                **common_kwargs,
                sigma=config.get("mmd_bandwidth", [3.0]),
            )
        case "kl":
            # Parse bandwidth
            bandwidth = config.get("kl_bandwidth", 3.0)
            try:
                bandwidth = float(bandwidth)
            except (ValueError, TypeError):
                pass  # Keep as string

            return KLPotential(
                **common_kwargs,
                rho_method=config.get("kl_rho_method", "kde"),
                mu_method=config.get("kl_mu_method"),
                bandwidth=bandwidth,
            )
        case _:
            raise ValueError(f"Unknown potential type: {potential_type}")


def create_potentials(config: dict) -> OrderedDict:
    """Create all potentials from config."""
    potentials = OrderedDict()
    tks = config["tks"]

    # Handle per-potential strengths/widths
    strengths = config.get("strengths") or [config["strength"]] * len(tks)
    default_w = config.get("width") or config.get("lambda_width", 0.2)
    widths = config.get("widths") or [default_w] * len(tks)

    for tk, strength, width in zip(tks, strengths, widths):
        potentials[tk] = create_potential(config, tk, strength=strength, width=width)

    return potentials


def create_model(config: dict, dim: int, potentials: OrderedDict, device: str) -> OTPFM:
    """Create OTPFM model from config."""
    flownet_args = {
        "x_emb_dim": config.get("x_emb_dim", 64),
        "t_emb_dim": config.get("t_emb_dim", 64),
        "hidden_dim": config.get("hidden_dim", 256),
        "num_hidden_layers": config.get("num_hidden_layers", 4),
        "x_hidden_layers": config.get("x_hidden_layers", 0),
        "t_hidden_layers": config.get("t_hidden_layers", 0),
        "activation_fn": config.get("activation", "silu"),
        "layernorm": config.get("layernorm", True),
        "dropout": config.get("dropout", 0.0),
        "residual_every": config.get("residual_every", 2),
    }

    model = OTPFM(
        d=dim,
        tks=list(potentials.keys()),
        potentials=potentials,
        flownet_args=flownet_args,
        consistency_loss=config.get("consistency_loss", "meanflow"),
        lossfn=config.get("lossfn", "adaptive"),
        time_sampler=config.get("time_sampler", "uniform_scaled_marginal"),
        diag_prob=config.get("diag_prob", 0.75),
        euler_steps=config.get("euler_steps", 2),
        ema_decay=config.get("ema_decay", 0.99),
        ema_ot=config.get("ema_ot", True),
        jvp_clamp=config.get("jvp_clamp", 100.0),
        solver_type=config.get("solver_type"),
        picard_steps=config.get("picard_steps", 5),
        adaptive_exp=config.get("adaptive_exp", 1.0),
        x_pred=config.get("x_pred", False),
    ).to(device)

    return model


def get_dataset_trainer_args(dataset: str, setup_result: dict, config: dict) -> dict:
    """Get dataset-specific trainer kwargs.

    Returns only the kwargs needed for the given dataset's trainer,
    avoiding passing unnecessary args across trainers.
    """
    if dataset == "gaussian":
        return {}

    # Real-world datasets: singlecell, gulfofmexico, beijingair
    args = {}
    for key in ("marginals", "scaler", "holdout_times", "train_times"):
        if key in setup_result:
            args[key] = setup_result[key]

    # Only pass if explicitly configured; otherwise use the trainer's per-dataset defaults
    for key in ("eval_n_steps", "eval_num_samples", "eval_metrics", "traj_skips", "save_skips"):
        if key in config:
            args[key] = config[key]

    return args
