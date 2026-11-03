"""Adapters around the original MFM losses, optimizers and Lightning steps.

Only endpoint shapes, explicit interval selection, and data delivery differ.
External source is imported, never copied or patched on disk.
"""

import torch

from flow2pc.utils.mfm_source import source_module


def process_intervals(module, x0s, x1s, *, geopath):
    times, locations, velocities = [], [], []
    grid = module.snapshot_times
    if len(x0s) != len(grid) - 1 or len(x1s) != len(x0s):
        raise ValueError("Endpoint batches do not match observed intervals")
    validating = geopath and (module.trainer.validating or module.computing_reference_loss)
    for i, (x0, x1) in enumerate(zip(x0s, x1s, strict=True)):
        if x0.ndim != 2 or x1.shape != x0.shape:
            raise ValueError("MFM endpoint batches must have matching [B,D] shapes")
        if validating:
            x0, x1 = x0.repeat(module.multiply_validation, 1), x1.repeat(module.multiply_validation, 1)
        if module.ot_sampler is not None:
            x0, x1 = module.ot_sampler.sample_plan(x0, x1, replace=True)
        t = module.t_val[i] if validating else None
        t, x, v = module.flow_matcher.sample_location_and_conditional_flow(
            x0, x1, grid[i], grid[i + 1], training_geopath_net=geopath, t=t,
        )
        if x.shape != x0.shape or v.shape != x0.shape:
            raise ValueError("MFM conditional flow changed the feature dimension")
        times.append(t)
        locations.append(x)
        velocities.append(v)
    return times, locations, velocities


def native_classes(source_root):
    matcher = source_module("flow_matchers.models.mfm", source_root).MetricFlowMatcher
    geo = source_module("flow_matchers.geopath_net_train", source_root).GeoPathNetTrain
    flow = source_module("flow_matchers.flow_net_train", source_root).FlowNetTrainBase

    class SnapshotMatcher(matcher):
        @staticmethod
        def doutput_dt_fun(model, x0, x1, t_raw):
            # Upstream squeezes the last axis, broadcasting [B,1] to [B,B] in 1D.
            # Keep precisely the output shape of the native geopath network.
            _, derivative = torch.func.jvp(
                lambda t: model(x0, x1, t.reshape(len(x0), 1)),
                (t_raw,), (torch.ones_like(t_raw),),
            )
            return derivative

    class SnapshotGeo(geo):
        def _process_flow(self, x0s, x1s):
            return process_intervals(self, x0s, x1s, geopath=True)

    class SnapshotFlow(flow):
        def train(self, mode=True):
            super().train(mode)
            self.flow_matcher.geopath_net.eval()
            return self

        def _process_flow(self, x0s, x1s):
            return process_intervals(self, x0s, x1s, geopath=False)

    return SnapshotMatcher, SnapshotGeo, SnapshotFlow


def data_module(data, batch_size):
    from pytorch_lightning import LightningDataModule
    from pytorch_lightning.utilities.combined_loader import CombinedLoader
    from torch.utils.data import DataLoader

    observed = data.observed_indices
    metric_frames = data.metric_marginals
    effective = min(batch_size, *(len(data.train_marginals[i]) for i in observed),
                    *(len(data.validation_marginals[i]) for i in observed))
    if effective < 1:
        raise ValueError("Observed marginals require nonempty train and validation splits")
    metric_batch = min(len(metric_frames[i]) for i in observed)
    if metric_batch < 1:
        raise ValueError("Metric reference pools must be nonempty")

    class SnapshotDataModule(LightningDataModule):
        def __init__(self):
            super().__init__()
            self.batch_size = effective
            self.num_timesteps = len(observed)

        def loaders(self, frames, shuffle):
            return [DataLoader(frames[i], batch_size=effective, shuffle=shuffle,
                               drop_last=True, num_workers=0) for i in observed]

        def combined(self, validation):
            frames = data.validation_marginals if validation else data.train_marginals
            # Match native balanced metric-pool batches, using training rows;
            # LiDAR retains the original full terrain as conditioning geometry.
            metric_loaders = [DataLoader(
                metric_frames[i], batch_size=(len(metric_frames[i]) if data.source_info["profile"] == "lidar" else metric_batch),
                shuffle=data.source_info["profile"] != "lidar", drop_last=False, num_workers=0,
            ) for i in observed]
            loaders = {
                "val_samples" if validation else "train_samples": CombinedLoader(
                    self.loaders(frames, not validation), mode="min_size"),
                "metric_samples": CombinedLoader(metric_loaders, mode="min_size"),
            }
            return CombinedLoader(loaders, mode="max_size_cycle")

        def train_dataloader(self):
            return self.combined(False)

        def val_dataloader(self):
            return self.combined(True)

    return SnapshotDataModule()
