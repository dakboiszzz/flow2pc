from flow2pc.utils.registry import Registry

teachers = Registry("teacher")
teachers.register(
    "alicfm",
    "flow2pc.pretrain.models.alicfm.trainer:ALICFM",
    "Original ALI-CFM profiles for CITE/Multi, EB PHATE, ST, knot and cell tracking.",
)
teachers.register(
    "otpfm", "flow2pc.pretrain.models.otpfm.trainer:OTPFMMethod",
    "Original OTP-FM core and native GOM/Beijing training; frozen EMA instantaneous velocity.",
)
for name in ("mfm", "mmfm"):
    teachers.register(
        name,
        None,
        "Not yet migrated into the framework; requires the authentic method "
        "implementation and its checkpoint adapter.",
    )
