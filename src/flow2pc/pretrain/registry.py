from flow2pc.utils.registry import Registry

teachers = Registry("teacher")
teachers.register(
    "alicfm",
    "flow2pc.pretrain.models.alicfm.trainer:ALICFM",
    "Original ALI-CFM profiles for CITE/Multi, EB PHATE, ST, knot and cell tracking.",
)
teachers.register(
    "otpfm",
    "flow2pc.pretrain.models.otpfm.trainer:OTPFMMethod",
    "Original OTP-FM core, native GOM/Beijing and shared marginal adaptations; "
    "frozen EMA velocity.",
)
teachers.register(
    "mfm",
    "flow2pc.pretrain.models.mfm.trainer:MFM",
    "Native MFM geopath/flow training and frozen paired-checkpoint adapter.",
)
teachers.register(
    "mmfm",
    None,
    "Not yet migrated into the framework; requires the authentic method "
    "implementation and its checkpoint adapter.",
)
