"""Portable, metadata-only catalog for the approved non-image MFM recipes."""

import hashlib
import json
from pathlib import Path

from flow2pc.utils.paths import MFM_PRETRAIN_ROOT, REPO_ROOT

FORMAT = "mfm_snapshots_v1"
DATA_ROOT = REPO_ROOT / "data/mfm"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


PROFILES = {}
for family in ("cite", "multi"):
    for dim in (5, 50, 100):
        PROFILES[f"{family}{dim}"] = dict(family=family, dim=dim, feature="X_pca", holdouts=[1], standardize=True)
for name, dim, feature in (("eb_pca5", 5, "X_pca"), ("eb_pca50", 50, "X_pca"), ("eb_phate2", 2, "X_phate")):
    PROFILES[name] = dict(family="eb", dim=dim, feature=feature, holdouts=[1], standardize=True)
for name, dim, holdouts, standardize in (
    ("st", 2, [1], True), ("cell_tracking", 2, [], True),
    ("gom", 2, [1, 3, 5, 7], True), ("beijing", 1, [2, 5, 8, 11], True),
    ("knot", 2, [], False), ("arch", 2, [1], False),
    ("sphere", 3, [1], False), ("lidar", 3, [], False),
):
    PROFILES[name] = dict(family=name, dim=dim, holdouts=holdouts, standardize=standardize)
for recipe in ("cli_defaults", "quickstart", "exact_example"):
    PROFILES[f"gaussian_{recipe}"] = dict(family="gaussian", recipe=recipe, dim=1, holdouts=[], standardize=False)


def profile(name):
    if name not in PROFILES:
        raise ValueError(f"Unknown MFM profile {name!r}; choose from {sorted(PROFILES)}")
    return dict(PROFILES[name])


def source_links(bundle=None, native_data=None):
    bundle = Path(bundle or REPO_ROOT / "data").expanduser().resolve()
    native_data = Path(native_data or MFM_PRETRAIN_ROOT / "data").expanduser().resolve()
    return {
        "single_cell": bundle / "single_cell",
        "gom": bundle / "gom", "beijing": bundle / "beijing",
        "spatial_transcriptomics": bundle / "spatial_transcriptomics",
        "cell_tracking": bundle / "cell_tracking",
        "gaussian": bundle / "synthetic",
        "lidar": native_data / "rainier2-thin.las",
    }


def link_sources(root=DATA_ROOT, *, bundle=None, native_data=None):
    """Create links only; never modify or move their targets."""
    root = Path(root).expanduser().resolve()
    links = source_links(bundle, native_data)
    directory = root / "sources"
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise ValueError(f"Source link container must be a real directory: {directory}")
    manifest = {"format": FORMAT, "sources": {k: str(v) for k, v in links.items()}, "profiles": PROFILES}
    path = root / "catalog.json"
    if path.is_symlink() or (path.exists() and json.loads(path.read_text()) != manifest):
        raise ValueError(f"Catalog differs; preserve it and use a new data root: {path}")
    # Check every conflict before creating any link.
    for name, target in links.items():
        if not target.exists():
            raise FileNotFoundError(f"Missing source: {target}")
        dest = directory / name
        if dest.is_symlink():
            if dest.resolve() != target:
                raise ValueError(f"Conflicting source link: {dest}")
        elif dest.exists():
            raise FileExistsError(f"Refusing to replace {dest}")
    directory.mkdir(parents=True, exist_ok=True)
    for name, target in links.items():
        dest = directory / name
        if not dest.is_symlink():
            dest.symlink_to(target, target_is_directory=target.is_dir())
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepared_path(root, name, seed=42):
    profile(name)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    return Path(root).expanduser().resolve() / "prepared" / name / f"seed{seed}"
