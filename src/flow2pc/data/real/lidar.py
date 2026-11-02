"""Metric Flow Matching lidar dataset adapter."""


def load(config, context):
    from flow2pc.data.metric_common import load as load_metric

    return load_metric("lidar", config, context)
