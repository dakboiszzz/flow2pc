"""Native interval adapters and checkpoint-pair regressions; no large data."""

import json
from types import SimpleNamespace

import pytest
import torch

from flow2pc.data.mfm_catalog import FORMAT, digest
from flow2pc.pretrain.models.mfm import snapshot_native, snapshots
from flow2pc.utils.mfm_source import fingerprint


def test_explicit_intervals_keep_nonuniform_times_and_one_dimensional_batches():
    calls = []

    def sample(x0, x1, left, right, **kwargs):
        calls.append((left, right, x0.shape))
        return x0.new_full((len(x0),), (left + right) / 2), x0, x1

    module = SimpleNamespace(snapshot_times=[0., .33, 1.], ot_sampler=None,
                             flow_matcher=SimpleNamespace(sample_location_and_conditional_flow=sample))
    batches = [torch.zeros(1, 1), torch.ones(1, 1)]
    _, x, v = snapshot_native.process_intervals(module, batches, batches, geopath=False)
    assert calls == [(0., .33, (1, 1)), (.33, 1., (1, 1))]
    assert all(t.shape == (1, 1) for t in x + v)


@pytest.mark.parametrize("batch,dim", [(1, 1), (5, 1), (5, 2)])
def test_geopath_time_derivative_never_broadcasts_features(monkeypatch, batch, dim):
    class Base:
        pass

    monkeypatch.setattr(snapshot_native, "source_module", lambda *args: SimpleNamespace(
        MetricFlowMatcher=Base, GeoPathNetTrain=Base, FlowNetTrainBase=Base,
    ))
    matcher, _, _ = snapshot_native.native_classes("unused")
    x = torch.ones(batch, dim)
    derivative = matcher.doutput_dt_fun(lambda a, b, t: a * t.square(), x, x, torch.full((batch, 1), .4))
    assert derivative.shape == (batch, dim)
    torch.testing.assert_close(derivative, torch.full_like(x, .8))


def test_pair_checksum_rejects_a_foreign_geopath_before_constructing_models(tmp_path):
    run = tmp_path / "teacher"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    binding = {"format": FORMAT, "pair_id": "original"}
    files = {}
    for role in ("flow", "geopath"):
        p = checkpoints / f"{role}.pt"
        torch.save({"binding": binding, "role": role, "state_dict": {}}, p)
        files[role] = {"sha256": fingerprint(p)}
    (run / "teacher_pair.json").write_text(json.dumps({
        "binding": binding, "binding_identity": digest(binding), "files": files,
    }))
    foreign = tmp_path / "foreign.pt"
    torch.save({"binding": {**binding, "pair_id": "foreign"}}, foreign)
    with pytest.raises(ValueError, match="pair checksum"):
        snapshots.load_teacher({"checkpoint_path": str(checkpoints / "flow.pt"),
                                "geopath_checkpoint": str(foreign)}, "cpu")


def test_loading_a_bound_pair_freezes_both_networks(tmp_path, monkeypatch):
    from flow2pc.pretrain.models.mfm import model

    run = tmp_path / "teacher"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    provenance = {"files": {}}
    args = {"dim": 1, "sigma": .1, "optimal_transport_method": "None"}
    binding = {"format": FORMAT, "pair_id": "pair", "args": args,
               "source": provenance, "snapshot_identity": "data"}
    files = {}
    for role in ("flow", "geopath"):
        checkpoint = checkpoints / f"{role}.pt"
        torch.save({"binding": binding, "role": role, "state_dict": {}}, checkpoint)
        files[role] = {"sha256": fingerprint(checkpoint)}
    (run / "teacher_pair.json").write_text(json.dumps({
        "binding": binding, "binding_identity": digest(binding), "files": files,
    }))
    monkeypatch.setattr(snapshots, "source_identity", lambda *args: provenance)
    networks = []

    def restored(*args, **kwargs):
        network = torch.nn.Linear(1, 1).eval().requires_grad_(False)
        networks.append(network)
        return network

    monkeypatch.setattr(model, "network_from_checkpoint", restored)
    teacher = snapshots.load_teacher({"checkpoint_path": str(checkpoints / "flow.pt"),
                                      "geopath_checkpoint": str(checkpoints / "geopath.pt")}, "cpu")
    assert not teacher.model.training and not teacher.path_sampler.network.training
    assert len(networks) == 2
    assert all(not p.requires_grad for network in networks for p in network.parameters())


@pytest.mark.parametrize("kind", ["hclt", "gaussian"])
def test_empirical_pc_initialization_keeps_velocity_gradients(kind):
    from flow2pc.distill.sampling import DistillationBatch, velocity_mse
    from flow2pc.distill.student import build_student_from_config, initialize_snapshot_source

    config = ({"name": "hclt", "parents": [-1], "num_latents": 4, "time_embedding_dim": 4, "time_order": 2}
              if kind == "hclt" else {"name": "gaussian", "num_components": 4, "dim": 1, "time_order": 2})
    student = build_student_from_config(config)
    source = torch.linspace(20, 22, 20)[:, None]
    initialize_snapshot_source(student, source, torch.Generator().manual_seed(3))
    teacher = SimpleNamespace(model=torch.nn.Identity().eval())
    values = torch.linspace(20, 22, 4)[:, None]
    batch = DistillationBatch(values, torch.full((4,), .4), target_velocity=torch.ones_like(values))
    loss = velocity_mse(student, teacher, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())
