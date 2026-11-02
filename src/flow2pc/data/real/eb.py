"""Metric Flow Matching eb dataset adapter."""


def load(config, context):
    from flow2pc.data.metric_common import load as load_metric

    return load_metric("eb", config, context)
