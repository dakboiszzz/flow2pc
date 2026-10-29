from flow2pc.utils.registry import Registry

datasets = Registry("dataset")
for name in ("cite", "multi"):
    datasets.register(
        name,
        "flow2pc.data.real.single_cell:SingleCell",
        "Released PCA coordinates with explicit days and ordinal model times; "
        "choose dimensions with data.params.dim.",
    )

for name, target, description in (
    (
        "eb",
        "flow2pc.data.temporal:EmbryoidBody",
        "Released EB PHATE 2D and original day-range categories.",
    ),
    (
        "st",
        "flow2pc.data.temporal:SpatialTranscriptomics",
        "Aligned U2-U5 tumor coordinates in saved y,x order.",
    ),
    (
        "cell_tracking",
        "flow2pc.data.temporal:CellTracking",
        "115 cell-4 mask marginals in x,y pixel order.",
    ),
    (
        "knot",
        "flow2pc.data.synthetic.knot:Knot",
        "Actual noisy analytic knot experiment, with explicit generator settings.",
    ),
):
    datasets.register(name, target, description)
