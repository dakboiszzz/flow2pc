"""Reconstruct the exact dataset representation bound to a trained teacher."""

from pathlib import Path

from flow2pc.data.registry import datasets


def restore_dataset(contract, preprocessing):
    root = contract.get("root")
    if root is None:
        root = Path(contract["source_file"]).parent.parent
    dataset = datasets.get(contract["name"])(
        name=contract["name"],
        root=Path(root),
        seed=contract["split_seed"],
        params=contract["params"],
        preprocessing=preprocessing,
    )
    if dataset.contract() != contract:
        raise ValueError(
            "Dataset source, coordinates, time map, split or saved preprocessing changed"
        )
    return dataset
