"""The cleaned layout must preserve the selected original main implementation."""

import ast
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASELINE = json.loads((ROOT / "docs/BASELINE.json").read_text())


class NormalizeImportLocations(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        node.module = "LOCATION_ONLY"
        node.level = 0
        return node


def algorithm_definitions(source):
    """Stable AST encoding across supported Python versions.

    ast.dump changes empty-field formatting and gains optional fields between
    Python versions. Exclude empty fields instead of hashing its display form.
    """
    def encode(value):
        if value is Ellipsis:
            return {"constant": "ellipsis"}
        if isinstance(value, bytes):
            return {"bytes": value.hex()}
        if isinstance(value, complex):
            return {"complex": [value.real, value.imag]}
        if isinstance(value, ast.AST):
            return {
                "node": type(value).__name__,
                **{key: encode(item) for key, item in ast.iter_fields(value)
                   if item is not None and item != []},
            }
        if isinstance(value, list):
            return [encode(item) for item in value]
        return value

    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            value = encode(NormalizeImportLocations().visit(node))
            result[node.name] = hashlib.sha256(json.dumps(
                value, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
    return result


@pytest.mark.parametrize("entry", BASELINE["source_map"], ids=lambda e:e["baseline_path"])
def test_original_algorithm_definitions_are_preserved(entry):
    paths = [ROOT / entry["canonical_path"]]
    if entry["baseline_path"] == "repc/pc.py":
        paths.append(ROOT / BASELINE["pc_base"])
    actual = {}
    for path in paths:
        actual.update(algorithm_definitions(path.read_text()))
    assert actual == entry["algorithm_definitions"]


@pytest.mark.parametrize("old,new", [
    ("repc.pc", "flow2pc.pc.gaussian_pc"),
    ("repc.hclt_pc", "flow2pc.pc.hclt_pc"),
    ("repc.region_pc", "flow2pc.pc.region_pc"),
    ("repc.hclt", "flow2pc.pc.hclt"),
    ("repc.flow", "flow2pc.pretrain.models.torchcfm_legacy.model"),
    ("train", "flow2pc.legacy.train"),
    ("eval", "flow2pc.legacy.eval"),
])
def test_compatibility_imports_share_one_module(old, new):
    assert importlib.import_module(old) is importlib.import_module(new)


@pytest.mark.parametrize("command", [
    "train.py", "eval.py", "train_endpoint_hclt.py", "train_mnist_classifier.py",
    "summarize_benchmark.py", "summarize_dense_tuning.py",
])
def test_original_entry_points_still_load(command):
    subprocess.run(
        [sys.executable, str(ROOT / command), "--help"],
        check=True, capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def test_no_local_artifacts_or_binary_weights_in_clean_tree():
    for name in ("ali-cfm", "outputs", "checkpoints", "build", "repc",
                 "Flow2PC_Architecture_Docs"):
        assert not (ROOT / name).exists()
    candidates = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, text=True,
    ).splitlines()
    for relative in candidates:
        path = ROOT / relative
        if path.is_file():
            assert path.suffix not in (".pt", ".pth", ".ckpt", ".h5ad", ".npy", ".npz")


def test_original_benchmark_scripts_have_valid_shell_syntax():
    for path in (ROOT / "scripts/legacy").glob("*.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)
