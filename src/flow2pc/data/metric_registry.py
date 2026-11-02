"""Dataset registration and metadata shared by training and evaluation."""


from flow2pc.utils.metric_registry import Registry


from flow2pc.data.metric_base import DatasetSpec, DatasetBundle


_DATASETS = Registry("dataset")


def register_dataset(name, factory, **capabilities):
    _DATASETS.register(name, DatasetSpec(factory, **capabilities))


def dataset_names():
    return _DATASETS.names()


def dataset_spec(name):
    return _DATASETS.get(name.lower())


def resolve_factory(factory):
    if isinstance(factory, str):
        from importlib import import_module

        module, attr = factory.split(":")
        return getattr(import_module(module), attr)
    return factory


def validate_combination(*args, **kwargs):
    from flow2pc.utils.combinations import validate_combination as validate
    return validate(*args, **kwargs)


def load_dataset(config, context=None):
    return _load_dataset(config, context, dataset_spec(config["name"]))


def _load_dataset(config, context, spec):
    import torch
    from torch.utils.data import DataLoader

    result = resolve_factory(spec.factory)(config, context or {})
    if isinstance(result, DatasetBundle):
        return result
    shuffle = config.get("shuffle", True)
    loader = DataLoader(
        result,
        batch_size=config.get("batch_size", 128),
        shuffle=shuffle,
        num_workers=config.get("num_workers", 0),
        drop_last=shuffle,
    )
    info = getattr(result, "source_info", {})
    times = info.get("times")
    if times is None:
        # Snapshot indices retain their positions on the original time grid.
        K = config.get("K", result.K)
        times = torch.linspace(0, 1, K)
    return DatasetBundle(
        loader, (result.D,), spec.pipeline, times=times, source_info=info
    )


for _name in ("afhq", "arch", "cite", "eb", "lidar", "multi", "sphere"):
    register_dataset(_name, f"flow2pc.data.{'synthetic' if _name in ('arch', 'sphere') else 'real'}.{_name}:load", pipeline="metric")
