"""Shared MFM/MMFM source and representation utilities, without teacher imports."""
import hashlib
import importlib
import subprocess
import sys
from pathlib import Path
from flow2pc.utils.paths import MFM_ROOT as DEFAULT_REPO
from flow2pc.utils.paths import MMFM_ROOT

DATA_TYPES = {
    "afhq": "image",
    "arch": "arch",
    "sphere": "sphere",
    "lidar": "lidar",
    "cite": "scrna",
    "eb": "scrna",
    "multi": "scrna",
}

def source_module(name, repo_path=None, *, method="mfm"):
    """Load an upstream module, retaining MFM as the default backend."""
    if method == "mmfm":
        return _mmfm_source_module(name, repo_path)
    if method != "mfm":
        raise ValueError(f"Unknown source method: {method!r}")
    root = Path(repo_path or DEFAULT_REPO).resolve()
    if not (root / "mfm").is_dir():
        raise FileNotFoundError(f"Metric Flow Matching source not found: {root}")
    loaded = sys.modules.get("mfm." + name)
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(root):
        raise ImportError(
            "Another mfm package is loaded; use a separate process for this source repo"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module("mfm." + name)
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise ImportError(
            f"Expected local MFM source under {root}, got {module.__file__}"
        )
    return module


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_shape(args):
    if args["data_type"] == "image":
        size = int(args.get("image_size", 128))
        if size != 128:
            raise ValueError(
                "AFHQ adapter requires the existing 128-pixel / [4,16,16] latent cache"
            )
        return (4, 16, 16)
    return (int(args["dim"]),)


def validate_dataset(args, name):
    if name not in DATA_TYPES or args["data_type"] != DATA_TYPES[name]:
        raise ValueError(
            f"MFM checkpoint data_type={args['data_type']} does not match {name}"
        )
    if (
        name in ("afhq", "cite", "eb", "multi", "lidar")
        and args.get("data_name") != name
    ):
        raise ValueError(
            f"MFM checkpoint data_name={args.get('data_name')} does not match {name}"
        )
    if name == "arch" and int(args["dim"]) != 2:
        raise ValueError("Arch checkpoint must be 2D")
    if name in ("sphere", "lidar") and int(args["dim"]) != 3:
        raise ValueError(f"{name} checkpoint must be 3D")
    if name == "afhq" and args.get("whiten", False):
        raise ValueError("AFHQ latent checkpoint must not whiten its inputs")


def _mmfm_source_module(name, source_root=None):
    root = Path(source_root or MMFM_ROOT).resolve()
    package = root / "src/mmfm"
    if not (package / "models.py").is_file():
        raise FileNotFoundError(f"MMFM source not found under {root}/src/mmfm")
    for key, module in tuple(sys.modules.items()):
        if key == "mmfm" or key.startswith("mmfm."):
            origin = getattr(module, "__file__", None)
            if origin and not Path(origin).resolve().is_relative_to(package):
                raise ImportError("Another mmfm package is loaded; use a separate process")
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    module = importlib.import_module(f"mmfm.{name}")
    if not Path(module.__file__).resolve().is_relative_to(package):
        raise ImportError(f"MMFM module loaded outside {package}")
    return module


def mmfm_provenance(source_root):
    root = Path(source_root).resolve()
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "sha256": {name: fingerprint(root / name) for name in (
            "src/mmfm/models.py", "src/mmfm/multi_marginal_fm.py", "src/mmfm/utils.py",
        )},
    }
