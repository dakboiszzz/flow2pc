from flow2pc.utils.registry import Registry

datasets = Registry("dataset")
for name in ("cite", "multi"):
    datasets.register(
        name,
        "flow2pc.data.real.single_cell:SingleCell",
        "Released PCA coordinates with explicit days and ordinal model times; "
        "choose dimensions with data.params.dim.",
    )
