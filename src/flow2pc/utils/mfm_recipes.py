"""Reviewable configs for the local MFM catalog (no model/data imports)."""

from pathlib import Path

import yaml

from flow2pc.data.mfm_catalog import DATA_ROOT, prepared_path, profile
from flow2pc.pretrain.models.mfm.snapshots import DEFAULTS
from flow2pc.utils.paths import MFM_PRETRAIN_ROOT, REPO_ROOT


def native_recipe(name):
    spec = profile(name)
    family = spec["family"]
    base = REPO_ROOT / "configs/pretrain/mfm/upstream"
    if family in ("arch", "sphere", "lidar"):
        template = base / family / ("ot-mfm.yaml" if family == "arch" else "mfm.yaml")
    elif family in ("cite", "multi"):
        template = base / "single_cell" / f"{spec['dim']}dims" / f"ot-mfm_{family}.yaml"
    else:
        template = base / "single_cell/5dims/ot-mfm_eb.yaml"
    raw = yaml.safe_load(template.read_text())
    # Native parser defaults, then the original dataset-specific recipe.
    native = {**DEFAULTS, "alpha_metric": 1.0, "time_geopath": False}
    native.update({k: v for k, v in raw.items() if k in DEFAULTS})
    native["gamma_current"] = raw.get("gammas", [0.125])[0]
    if raw.get("velocity_metric") == "rbf":
        # The approved new snapshot protocol uses LAND. RBF's negative rho is
        # a different parameterization and must not be copied into LAND.
        native.update(velocity_metric="land", gamma_current=0.125, rho=0.001, alpha_metric=2.0)
    return native, str(template.relative_to(REPO_ROOT))


def configs(name, *, seed=42, data_root=DATA_ROOT, output_root=None, run_name="default", source_root=MFM_PRETRAIN_ROOT):
    profile(name)
    native, template = native_recipe(name)
    output = Path(output_root or REPO_ROOT / "outputs/mfm").resolve() / name / run_name
    prepared = prepared_path(data_root, name, seed)
    pretrain = {
        "stage": "pretrain", "seed": seed,
        "model": {"name": "mfm", "params": {"backend": "snapshots", "source_root": str(source_root), "native": native}},
        "data": {"name": "mfm_snapshots", "root": str(prepared), "params": {"profile": name}},
        "training": {"epochs": 1000, "batch_size": 128, "device": "cuda:0",
                     "patience_geopath": 5, "patience_flow": 5, "validate_every": 1},
        "output": {"dir": str(output / "pretrain")}, "wandb": {"mode": "disabled"},
    }
    # Preserve dataset-specific stopping patience from the supplied native recipe.
    raw = yaml.safe_load((REPO_ROOT / template).read_text())
    pretrain["training"]["patience_geopath"] = raw.get("patience_geopath", 5)
    pretrain["training"]["patience_flow"] = raw.get("patience", 5)
    students = {}
    for student in ("hclt", "gaussian"):
        students[student] = {
            "stage": "distill", "seed": seed,
            "teacher": {"name": "mfm", "checkpoint": str(output / "pretrain/checkpoints/flow.pt"),
                        "params": {"repo_path": str(source_root), "geopath_checkpoint": str(output / "pretrain/checkpoints/geopath.pt")}},
            "data": {"name": "mfm_snapshots", "root": str(prepared)},
            "student": {"name": student, "params": (
                {"num_latents": 64, "time_embedding_dim": 64, "topology": "learned", "topology_samples": 2048, "time_order": 4}
                if student == "hclt" else {"num_components": 64, "time_order": 4})},
            "distillation": {"steps": 20000, "batch_size": 128, "lr": 0.001, "device": "auto",
                             "gradient_clip": 1.0, "log_every": 100, "validate_every": 1000,
                             "checkpoint_every": 5000, "resume_checkpoint": None},
            "output": {"dir": str(output / "distill" / student)},
            "evaluation": {"enabled": False}, "wandb": {"mode": "disabled"},
        }
    return pretrain, students, template
