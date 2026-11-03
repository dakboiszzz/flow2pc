"""Explicit data preparation entry point; never trains or evaluates a model."""

import argparse
from pathlib import Path

from flow2pc.data.mfm_catalog import DATA_ROOT, PROFILES
from flow2pc.data.mfm_snapshots import prepare_dataset
from flow2pc.utils.paths import MFM_PRETRAIN_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=PROFILES)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-root", type=Path, default=MFM_PRETRAIN_ROOT)
    args = parser.parse_args()
    print(prepare_dataset(args.profile, args.data_root, args.seed, args.source_root))


if __name__ == "__main__":
    main()
