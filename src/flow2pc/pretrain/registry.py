from flow2pc.utils.registry import Registry

teachers = Registry("teacher")
teachers.register(
    "alicfm",
    "flow2pc.pretrain.models.alicfm.trainer:ALICFM",
    "Migrated source training stages for CITE/Multi; other experiments remain legacy.",
)
for name in ("mfm", "mmfm", "otpfm"):
    teachers.register(
        name,
        None,
        "Not yet migrated into the framework; requires the authentic method "
        "implementation and its checkpoint adapter.",
    )
