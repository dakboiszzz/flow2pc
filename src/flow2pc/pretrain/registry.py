from flow2pc.utils.registry import Registry

teachers = Registry("teacher")
teachers.register(
    "alicfm",
    "flow2pc.pretrain.models.alicfm.trainer:ALICFM",
    "Original ALI-CFM profiles for CITE/Multi, EB PHATE, ST, knot and cell tracking.",
)
for name in ("mfm", "mmfm", "otpfm"):
    teachers.register(
        name,
        None,
        "Not yet migrated into the framework; requires the authentic method "
        "implementation and its checkpoint adapter.",
    )
