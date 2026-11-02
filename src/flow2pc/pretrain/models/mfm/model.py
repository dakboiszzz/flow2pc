"""Construct the original MFM networks and restore their native weights."""

from flow2pc.utils.mfm_source import sample_shape, source_module


def make_network(args, *, geopath=False, repo_path=None):
    if args["data_type"] == "image":
        suffix = "_geopath" if geopath else ""
        kwargs = {
            "dim": sample_shape(args),
            "num_channels": args.get("unet_num_channels" + suffix, 128),
            "num_res_blocks": args.get("unet_num_res_blocks" + suffix, 2),
            "channel_mult": args.get("unet_channel_mult" + suffix),
            "dropout": args.get("unet_dropout" + suffix, 0.0),
        }
        if geopath:
            return source_module(
                "networks.geopath_networks.unet", repo_path
            ).GeoPathUNet(**kwargs)
        kwargs.update(
            resblock_updown=args.get("unet_resblock_updown", False),
            use_new_attention_order=args.get("unet_use_new_attention_order", False),
            attention_resolutions=args.get("unet_attention_resolutions", "16"),
            num_heads=args.get("unet_num_heads", 1),
        )
        return source_module("networks.unet_base", repo_path).UNetModelWrapper(**kwargs)
    if geopath:
        cls = source_module("networks.geopath_networks.mlp", repo_path).GeoPathMLP
        return cls(
            input_dim=int(args["dim"]),
            hidden_dims=args["hidden_dims_geopath"],
            activation=args["activation_geopath"],
            batch_norm=False,
            time_geopath=args.get("time_geopath", False),
        )
    cls = source_module("networks.flow_networks.mlp", repo_path).VelocityNet
    return cls(
        dim=int(args["dim"]),
        hidden_dims=args["hidden_dims_flow"],
        activation=args["activation_flow"],
        batch_norm=False,
    )

def network_from_checkpoint(checkpoint, args, *, geopath=False, repo_path=None):
    state = checkpoint.get("state_dict", {})
    prefixes = (
        ("geopath_net.", "flow_matcher.geopath_net.") if geopath else ("flow_net.",)
    )
    weights = None
    for prefix in prefixes:
        subset = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        if subset:
            weights = subset
            break
    if weights is None:
        raise ValueError(
            f"Checkpoint has no {'geopath_net' if geopath else 'flow_net'} weights"
        )
    model = make_network(args, geopath=geopath, repo_path=repo_path)
    if "num_updates" in weights:
        ema = source_module("flow_matchers.ema", repo_path).EMA
        model = ema(model, decay=args.get("ema_decay") or 0.999)
    model.load_state_dict(weights, strict=True)
    # EMA swaps shadow weights only while parameters still require gradients.
    model.eval()
    model.requires_grad_(False)
    return model
