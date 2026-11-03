#!/usr/bin/env python3
"""Compare locally produced native/migrated histories and saved model tensors.

No model construction, inference, sampling or training occurs. Native checkpoints
are trusted local reproduction outputs; their NumPy loss histories need the native
pickle loader. Do not point this tool at untrusted downloaded checkpoints.
"""

import argparse
import hashlib
import json
from pathlib import Path


def history(path):
    rows = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            epoch = row["epoch"]
            if epoch in rows:
                raise ValueError(f"Duplicate recorded epoch {epoch}: {path}")
            rows[epoch] = {k: v for k, v in row.items() if k != "elapsed_seconds"}
    return rows


def fingerprint(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/data/lthung/flow2pc/otp-fm/migration-check"))
    parser.add_argument("--datasets", nargs="+", default=["gom", "beijing"])
    parser.add_argument("--methods", nargs="+", default=["W2", "W2Inf"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    reports = []
    for dataset in args.datasets:
        for recipe in args.methods:
            for seed in args.seeds:
                native = args.output_root / "native" / dataset / recipe / f"seed{seed}"
                migrated = args.output_root / "migrated" / dataset / recipe / f"seed{seed}"
                required = [native / "history.jsonl", migrated / "history.jsonl",
                            *[folder / "models" / file for folder in (native, migrated) for file in ("best.pt", "model.pt")],
                            migrated / "checkpoints/best.pt"]
                missing = [str(p) for p in required if not p.is_file()]
                row = {"dataset": dataset, "recipe": recipe, "seed": seed, "missing": missing}
                if missing or args.dry_run:
                    row["status"] = "pending" if missing else "ready"
                    reports.append(row)
                    continue
                import torch

                left, right = history(native / "history.jsonl"), history(migrated / "history.jsonl")
                differences = []
                for epoch in sorted(left.keys() | right.keys()):
                    a, b = left.get(epoch, {}), right.get(epoch, {})
                    for key in sorted(a.keys() | b.keys()):
                        if a.get(key) != b.get(key):
                            differences.append({"epoch": epoch, "key": key, "native": a.get(key), "migrated": b.get(key)})
                row.update(native_epochs=len(left), migrated_epochs=len(right),
                           history_mismatches=len(differences), first_history_mismatches=differences[:10])
                models = {}
                for label, name in (("best", "best.pt"), ("final", "model.pt")):
                    a = torch.load(native / "models" / name, map_location="cpu", weights_only=False)["model_state_dict"]
                    b = torch.load(migrated / "models" / name, map_location="cpu", weights_only=False)["model_state_dict"]
                    fa, fb = fingerprint(a), fingerprint(b)
                    models[label] = {"native_sha256": fa, "migrated_sha256": fb,
                                     "byte_identical": fa == fb, "tensors": len(a)}
                    if label == "best":
                        exported = torch.load(migrated / "checkpoints/best.pt", map_location="cpu", weights_only=True)["model_state_dict"]
                        models[label]["framework_export_identical"] = fingerprint(exported) == fa
                row["models"] = models
                row["status"] = ("exact" if not differences and all(m["byte_identical"] for m in models.values())
                                 and models["best"]["framework_export_identical"] else "mismatch")
                reports.append(row)
    result = {"dry_run": args.dry_run, "comparisons": reports,
              "all_exact": bool(reports) and all(r["status"] == "exact" for r in reports)}
    print(json.dumps(result, indent=2))
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "parity_report.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if args.dry_run or result["all_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
