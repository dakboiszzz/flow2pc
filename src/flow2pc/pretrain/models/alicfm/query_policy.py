"""Recover endpoint coupling from ALI-CFM training metadata."""

# train_ot_cfm_interpolant uses a full marginal chain only for `mmot`.
# Its `full` setting shares the border/endpoints branch, despite that name.
COUPLINGS = {"border": "border_ot", "mmot": "chain_ot", "none": "independent",
             "full": "border_ot"}


def training_sampler(model_config):
    profile = model_config.get("profile", "single_cell")
    if profile == "knot":
        if model_config.get("cfm_ot") not in (None, "none"):
            raise ValueError("Native knot CFM requires independent endpoints")
        return "independent"  # Knot's native trainer rejects non-none OT.
    coupling = model_config.get("cfm_ot")
    if coupling is None:
        return None
    if coupling not in COUPLINGS:
        raise ValueError(f"Unsupported saved ALI-CFM query coupling {coupling!r}")
    if profile == "st" and coupling == "mmot":
        return "independent"  # st.train_ot_cfm only couples border/full.
    if profile == "cell_tracking" and coupling == "mmot":
        return "mmot_middle"  # tracking.train_ot_cfm uses sample_gan_batch.
    return COUPLINGS[coupling]


def resolve_sampler(manifest, settings):
    if settings.get("name") != "teacher_interpolant":
        raise ValueError("ALI-CFM queries require sampler.name: teacher_interpolant")
    params = settings.get("params", {})
    if set(params) - {"coupling"}:
        raise ValueError("ALI-CFM sampler only supports params.coupling")
    # Recorded CFM settings override the historical profile-only sampler tag.
    trained = training_sampler(manifest.get("model_config", {}))
    saved = trained or manifest.get("distillation_sampler")
    requested = params.get("coupling", "teacher")
    if requested != "teacher":
        if requested not in COUPLINGS:
            raise ValueError("coupling must be teacher, border, mmot, full or none")
        requested_sampler = training_sampler({**manifest.get("model_config", {}),
                                             "cfm_ot": requested})
        if trained is not None and requested_sampler != trained:
            raise ValueError("Configured coupling conflicts with this teacher's training settings")
        saved = requested_sampler
    if saved not in (*COUPLINGS.values(), "mmot_middle"):
        raise ValueError("Teacher lacks coupling metadata; specify its original coupling explicitly")
    return saved
