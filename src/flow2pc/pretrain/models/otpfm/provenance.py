"""Source checks without importing Torch, networks, datasets or training code."""

import hashlib
import json
from pathlib import Path


def verify_source(native_root=None):
    root = Path(__file__).parent
    provenance = json.loads((root / "SOURCE_PROVENANCE.json").read_text())
    for record in provenance["files"]:
        path = root / "_source" / record["migrated_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["migrated_sha256"]:
            raise ValueError(f"Pinned OTP-FM source changed: {path}")
        if native_root is not None:
            original = Path(native_root) / record["upstream_file"]
            if hashlib.sha256(original.read_bytes()).hexdigest() != record["upstream_sha256"]:
                raise ValueError(f"Native comparison source changed since migration: {original}")
    return provenance
