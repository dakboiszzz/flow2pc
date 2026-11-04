"""Load the explicitly selected upstream MMFM checkout without vendoring it."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

from flow2pc.utils.mfm_source import fingerprint
from flow2pc.utils.paths import PRETRAIN_ROOT

DEFAULT_SOURCE = Path(os.environ.get("FLOW2PC_MMFM_ROOT", PRETRAIN_ROOT / "MMFM"))


def source_module(name, source_root=None):
    root = Path(source_root or DEFAULT_SOURCE).resolve()
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


def provenance(source_root):
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
