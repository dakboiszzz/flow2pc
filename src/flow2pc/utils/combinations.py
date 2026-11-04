def validate_combination(dataset_name, teacher_name, *, task="distill", context=None):
    from flow2pc.data.metric_registry import dataset_spec, resolve_factory

    if teacher_name not in ("mfm", "metric_flow", "mmfm"):
        raise ValueError(f"Unsupported MFM teacher: {teacher_name}")
    if teacher_name == "mmfm" and dataset_name not in ("arch", "sphere", "lidar", "mfm_snapshots"):
        raise ValueError("MMFM currently supports arch, sphere and lidar")
    spec = dataset_spec(dataset_name)
    if spec.pipeline != "metric":
        raise ValueError(
            f"Teacher {teacher_name} (metric) cannot use dataset "
            f"{dataset_name} ({spec.pipeline})"
        )
    if task != "distill":
        raise ValueError(
            "Teacher training is not integrated in this checkout"
        )
    if spec.validate:
        resolve_factory(spec.validate)(context or {})

