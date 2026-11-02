"""Paths independent of the depth of an implementation module."""
import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("FLOW2PC_REPO_ROOT", Path(__file__).resolve().parents[3]))
PRETRAIN_ROOT = REPO_ROOT.parent / "pretrain"
MFM_ROOT = Path(os.environ.get("FLOW2PC_MFM_ROOT", REPO_ROOT.parent / "metric-flow-matching"))
MFM_PRETRAIN_ROOT = Path(
    os.environ.get("FLOW2PC_MFM_PRETRAIN_ROOT", PRETRAIN_ROOT / "metric-flow-matching")
)
