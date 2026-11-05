"""Dataset benchmark conventions, independent of the selected teacher method.

The papers benchmark teachers; PC checkpoint selection is our velocity-MSE
protocol. Never infer a validation split merely from the word 'heldout'.
"""


def dataset_tag(dataset):
    if hasattr(dataset, "metadata"):
        name = dataset.metadata.name
        dim = getattr(dataset.metadata, "dim", getattr(dataset, "dim", None))
    else:
        info = getattr(dataset, "source_info", {})
        name = info.get("profile", info.get("dataset", "unknown"))
        dim = getattr(dataset, "dim", None)
    if name in ("cite", "multi") and dim is not None:
        return f"{name}{dim}"
    return name


def validation_scope(dataset, settings):
    scope = settings.get("validation_scope", "dataset")
    if scope == "dataset":
        # OTP-FM transport experiments have no observation-level validation
        # split. Single-cell retains reserved rows at observed times; the
        # left-out population is used for benchmark evaluation.
        return "heldout_times" if dataset_tag(dataset) in ("gom", "beijing") else "observed_rows"
    if scope not in ("observed_rows", "heldout_times"):
        raise ValueError("validation_scope must be dataset, observed_rows or heldout_times")
    return scope


def benchmark(dataset):
    tag = dataset_tag(dataset)
    transport = tag in ("gom", "beijing")
    single_cell = tag.startswith(("cite", "multi", "eb"))
    return {
        "dataset": tag,
        "reference": "OTP-FM 2606.05327, section 6.4" if transport else "ALI-CFM 2510.01159" if single_cell or tag in ("st", "knot", "cell_tracking") else "MFM 2405.14780",
        "primary_distance": "w2" if transport else "emd_w1",
        "rollout_origin": "preceding_observed" if single_cell or tag == "st" else "earliest",
        "note": "PC velocity-MSE checkpoint selection is a distillation protocol, not a published teacher-selection rule",
    }
