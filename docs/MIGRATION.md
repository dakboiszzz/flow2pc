# Migration and baseline alignment

The repository is organized around the proposed `src/flow2pc` structure and the
original main-branch implementation at **`3eb65ef67e8643dd844c9a9e819899316ffff542`
— `feat: add literature-aligned flow-to-PC benchmarks`**.

## Implementation ownership

| Original file | Canonical implementation |
| --- | --- |
| `repc/pc.py` | `flow2pc.pc.gaussian_pc` and `flow2pc.pc.base` |
| `repc/hclt.py`, `repc/hclt_pc.py` | `flow2pc.pc.hclt`, `flow2pc.pc.hclt_pc` |
| `repc/region_pc.py`, `repc/published_hclt.py` | corresponding `flow2pc.pc` modules |
| `repc/flow.py` | `flow2pc.pretrain.models.torchcfm_legacy.model` |
| `repc/data.py` | `flow2pc.data.real.mnist` |
| Original evaluation/classifier utilities | `flow2pc.legacy` |
| Root training, evaluation and summary commands | thin wrappers around `flow2pc.legacy` |
| Original benchmark launchers | `scripts/legacy/` |

`src/repc` contains compatibility aliases, not duplicated PC implementations.
Old and new imports resolve to the same class/module objects. The original
baseline tests remain unchanged. `docs/BASELINE.json` records the source map,
original file hashes and normalized algorithm-definition hashes. Tests check
that relocating imports and extracting the shared base class did not change
the baseline definitions. The original default boundary is standard Gaussian;
the later experiment-specific source options and numerical changes are not
silently carried into this baseline.

The original command documentation is in `docs/legacy/README.md`. The six root
compatibility entry points preserve its command names and Python imports.

## ALI-CFM integration

`flow2pc.pretrain.models.alicfm` contains the migrated core from the supplied
`../ali-cfm` checkout: networks, LAND metric, coupling/batch utilities and the
three staged training loops. It does not need the sibling checkout at runtime.
The source's existing numerical repairs are retained; this is not a claim that
all experiments reproduce the paper. Attribution and source hashes are in
`NOTICE.md` and `SOURCE_PROVENANCE.json` beside the method implementation.

The new method-owned trainer supports CITE/Multi, one seed/fold, the original
Adam optimizers, linear/second-derivative regularization with L2/LAND, and valid
native coupling modes. It calls the original pretraining, adversarial and CFM
loop bodies. Edits to those source loops only remove two unused cross-package
imports and add an optional local logger. Source-body and native-update parity
tests check this boundary. Other preserved private legacy branches are not
advertised as validated framework configurations.

The shared runner owns configuration and dispatch. Data code does not import
teachers, and teacher code does not import PCs/distillation. Model discovery is
lazy. The PC namespace has a single implementation owner.

## Commands

```bash
python -m pip install -e '.[dev,alicfm]'
python main.py --list-models
python main.py --config configs/pretrain/alicfm_cite50.yaml --validate-only
python train.py --help
python eval.py --help
```

`--validate-only` resolves method configuration, dataset metadata, train-only
scaling and the source hash. It starts no training, W&B run or output directory.
Removing that option explicitly starts the configured teacher run. No production
training is part of the cleanup or tests.

The example uses `/data/lthung/ali-cfm-data`; override `data.root` or omit it to
use `FLOW2PC_DATA_ROOT`, then `./data`. Explicit configuration wins. Relative
paths are relative to the YAML directory. Missing `${VAR}` references fail.

## Data and checkpoint semantics

- Released PCA is reused. No new PCA or arbitrary cross-time pairing is inferred.
- CITE/Multi physical days `[2,3,4,7]` retain ordinal model times `[0,1/3,2/3,1]`.
  Both grids and an invertible piecewise-linear map are saved. Velocities use
  `dx/dmodel_time`; the adapter does not reset interior intervals to zero.
- New framework runs split first and fit scaling only on observed training
  cells. The old ALI-CFM loader fitted over all days, including the holdout.
  This explicit protocol difference prevents claiming identical old paper scores.
  Legacy pretrained weights must keep their saved preprocessing.
- Source H5ADs stay in place. Row access is lazy; native training explicitly
  materializes its CPU training pools, then transfers minibatches to the device.
- The held-out day is test-only. Point-level validation/test cells are excluded
  from optimizer pools; this does not establish donor-level generalization.
- Completed runs write resolved YAML, a versioned manifest, preprocessing JSON,
  local loss logs, status and `checkpoints/last.pt`. The manifest binds dimensions,
  data hashes, splits, times, initial law and source provenance. Writes are atomic;
  inference loading verifies metadata/hashes and uses `weights_only=True`.
- Optimizer/Torch RNG states are saved, but full training resume is not exposed
  or claimed. Each new run requires a new output directory.

Load a new-format teacher independently:

```python
from flow2pc.pretrain.runner import load_teacher

teacher = load_teacher('/path/to/run/checkpoints/last.pt', device='cpu')
v = teacher.velocity(x_in_saved_model_coordinates, global_model_time)
```

The frozen adapter validates shape, dtype/device, finite values, time and context.
Existing external checkpoints are not rewritten by this cleanup.

## Archived material and remaining work

Post-baseline multi-marginal, single-cell, cell-tracking, OTP-FM and latent-MNIST
experiment scripts/configurations, results, local weights, copied W&B runs and
generated caches were moved outside the repository to a recoverable local cache.
The cache contains a full pre-cleanup source snapshot, hashes, move manifest and
restore notes. External datasets and published run outputs were not moved.
No Git history, branch or remote was rewritten, committed or pushed.

The new framework registry implements ALI-CFM and marks MFM/MMFM/OTP-FM unavailable
until their authentic integrations are migrated. Original MNIST distillation
remains available through the baseline commands. Unified real-data distillation
is separate work: its sampler and source-compatible PC require an explicit design
decision because the preserved baseline PCs assume a standard-Gaussian source.

The architecture specification and proposal describe the full target, not a claim
that all datasets, teacher methods, experiment wrappers or resume paths are done.

## Verification

Use `make test` and `make lint`. Verification includes the unchanged baseline
tests, algorithm-definition parity against the recorded commit, canonical/legacy
class identity, native ALI-CFM stage-update parity, small fixture checkpoint
roundtrips, metadata tampering, lazy registry behavior and the legacy CLIs.
Small H5AD-shaped fixtures are test inputs, not research toy experiments.
