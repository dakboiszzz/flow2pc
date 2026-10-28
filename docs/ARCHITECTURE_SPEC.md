# Flow2PC Framework — Codex Implementation Specification

**Status:** Proposed target architecture; implementation requires teammate/mentor approval.  
**Scope:** Teacher pretraining + shared data + checkpoint-driven Flow2PC distillation; experiment suite excluded for now.  
**Authority:** This document defines *intended* behavior; inspect the repository to distinguish existing functionality from new functionality. The companion `README_PROPOSAL.md` provides the mentor-facing rationale.

## 0. Instructions for implementation agents

1. **Inspect before editing:** read existing `train.py`, `eval.py`, `repc/`, `tests/`, `.gitignore`, `pyproject.toml`, and README. List behavior that must be preserved. Report mismatches with this proposal.
2. **Incremental migration only:** do not delete, rename, or break working entry points before new equivalents are tested. Prefer adapters and small PRs over wholesale rewrites.
3. **Do not invent paper algorithms:** the user/team supplies or ports real MFM, MMFM, OTP-FM, and ALI-CFM code. Until available, create documented registries/interfaces and real smoke tests with an existing teacher or minimal toy adapter. Do not claim a placeholder is a working paper reproduction.
4. **No experiment-suite work:** focus on pretraining, data access, checkpoint contracts, and student distillation. Existing evaluation tooling can remain intact.
5. **No large data or secrets in Git.** Do not download large datasets without explicit instruction; do not assume a source URL or redistribution rights.
6. **Preserve mathematical meaning:** record initial laws, time maps, conditioning, and query distributions. Flag incompatibility rather than replacing a teacher's objective with another flow-matching loss.
7. **Do not perform unrequested commits/pushes or install system-wide dependencies.** Run accessible tests, summarize results and blockers.
8. When asked to implement, **plan one phase and stop after that phase** unless further work is explicitly authorized.

## 1. Current repo: compatibility baseline

As observed from public `main` at the time of this proposal:

- `repc/flow.py` builds a TorchCFM MNIST UNet; `repc/data.py` loads MNIST.
- Root `train.py` supports `flow` and `distill` and includes existing PC training and checkpoint behavior.
- Root `eval.py` imports helpers from `train.py`; breaking them has downstream consequences.
- `repc/pc.py`, `repc/hclt.py`, `repc/hclt_pc.py`, and `repc/region_pc.py` provide existing PC functionality.
- `pyproject.toml` currently packages `repc*` with setuptools, uses Python >=3.11, Ruff line length 88. It must be deliberately updated for `src/flow2pc`, while preserving import compatibility until migration is validated.
- Existing README discusses the `p_0 = N(0,I)` time-boundary assumption for the current student, and noise-to-image paths. This is not automatically valid for physical-time multi-marginal systems.

**Migration rule:** retain the legacy package or provide tested compatibility shims while migrating the PC implementation. Do not copy modules and maintain two diverging sources indefinitely; decide one implementation owner after parity tests.

## 2. Scope, non-goals, invariants

### In scope

- Installable package: `pip install -e '.[dev]'` from repository root.
- Shared model and dataset registries with explicit unknown-name errors.
- Unified command: `python main.py --config <yaml>` and optional `python -m flow2pc.cli --config <yaml>`.
- Four model-zoo namespaces: `mfm`, `mmfm`, `otpfm`, `alicfm` (implement when actual source code is provided).
- Config-driven teacher pretraining with method-specific trainers.
- Dataset loading with support for multi-marginal observed times, deterministic splitting, and large-data access.
- Teacher checkpoints with self-describing metadata and a reproducible reload path.
- Distillation from **frozen teacher checkpoints** to PC students using valid teacher queries.
- Small tests and one minimal end-to-end demonstration on a genuinely implemented model/dataset.

### Out of scope

- Full benchmark/metrics implementation (E1–E5, UQ, OOD, distribution shift, etc.).
- Reimplementing all source papers from scratch solely from method names.
- Assuming all combinations of 4 teachers × 6 datasets work out of the box.
- Shipping datasets/checkpoints in the Python wheel or Git repository.
- Distributed training infrastructure, DVC, complex workflow engines, or a mandatory Hydra dependency for v1.

### Invariants

- No teacher code imports `flow2pc.distill` or `flow2pc.pc`.
- Dataset loaders don't import teacher implementations.
- The student stage never silently trains or changes teacher parameters.
- Teacher-only checkpoint must be independently restorable for inference.
- Every persisted checkpoint points to the exact dataset representation and time convention used in training.
- Train/validation/test data separation and normalization are deterministic and shared across teachers.
- A trained teacher is not automatically a valid Flow2PC target: scientific compatibility checks are required.

## 3. Target file layout

```text
main.py
configs/
  pretrain/*.yaml
  distill/*.yaml
src/flow2pc/
  __init__.py
  cli.py
  pretrain/
    __init__.py
    base.py             # interfaces/protocols
    registry.py         # name -> factory; lazy imports
    runner.py           # orchestration, not algorithm internals
    models/
      mfm/{__init__.py, ...}
      mmfm/{__init__.py, ...}
      otpfm/{__init__.py, ...}
      alicfm/{__init__.py, ...}
      torchcfm_legacy/   # optional preserved existing teacher adapter
  data/
    __init__.py
    base.py             # temporal metadata and access contract
    registry.py
    preprocessing.py
    synthetic/{__init__.py, ...}
    real/{__init__.py, ...}
  distill/
    __init__.py
    runner.py
    sampling.py         # q(x_t, t) construction
    losses.py
  pc/
    __init__.py
    base.py
    gaussian_pc.py
    hclt.py
    hclt_pc.py
    region_pc.py
  utils/{config.py, checkpoint.py, seed.py}
tests/
data/                   # real datasets, not committed
outputs/                # checkpoints/logging, not committed
README_PROPOSAL.md
AGENTS.md
docs/ARCHITECTURE_SPEC.md
```

Create files only as needed; empty packages are not a deliverable. Keep team-owned methods isolated to their folders and avoid a global megaclass.

## 4. Program entry point and config contract

### CLI

```bash
python main.py --config configs/pretrain/mfm_toy.yaml
python main.py --config configs/pretrain/alicfm_cite50.yaml
python main.py --config configs/distill/alicfm_cite50_hclt.yaml
```

These paths illustrate target behavior; only ship configs that can be validated with an actually available method/dataset. The code must parse `stage: pretrain|distill`, validate required keys, reject unknown model/dataset names, and provide actionable errors. Avoid unvalidated hidden defaults that change scientific settings.

### Proposed minimal configuration schema

**Pretrain:**

```yaml
stage: pretrain
seed: 42
model:
  name: alicfm
  params: {}          # original-method-specific hyperparameters

data:
  name: cite50
  root: ${FLOW2PC_DATA_ROOT}
  params: {}          # loader-specific validated options

training:
  epochs: 100
  batch_size: 512
  lr: 0.0001
  num_workers: 4

output:
  dir: outputs/pretrain/alicfm/cite50
```

**Distill:**

```yaml
stage: distill
seed: 42
teacher:
  checkpoint: /actual/path/to/checkpoints/last.pt
student:
  name: hclt
  params:
    num_latents: 64
    time_order: 4
distillation:
  sampler:
    name: teacher_path
    params: {}
  steps: 20000
  batch_size: 256
  lr: 0.001
output:
  dir: outputs/distill/alicfm/cite50/hclt
```

Hyperparameters are examples only. The loader must implement `${FLOW2PC_DATA_ROOT}` expansion explicitly (or omit the syntax and document passing normal paths). Precedence: explicit `data.root` in config > environment `FLOW2PC_DATA_ROOT` > repository `data/` for toy/local use. An unset env reference must error; do not expand silently to an empty path. The distillation stage obtains dataset metadata from the teacher manifest; it should reject incompatible overrides unless explicitly handled and documented.

Use standard `argparse` and a simple YAML parser for v1; optional Hydra-style composition can be added only if config proliferation becomes a problem. `main.py` should be a thin shim to `flow2pc.cli.main`.

## 5. Shared dataset contract

### Goals

Represent unpaired samples at observed physical times, support irregular observation intervals, and allow optional observed trajectories without making them mandatory. Separate **metadata** from **batches** to avoid loading large data during registry lookup.

### Conceptual types (adapt exact names/types to existing code)

```python
from dataclasses import dataclass
from typing import Protocol
import torch

@dataclass(frozen=True)
class TemporalMetadata:
    name: str
    dim: int
    physical_times: tuple[float, ...]
    model_times: tuple[float, ...]  # transformation explicitly recorded
    has_trajectory_ids: bool
    preprocessing_id: str

class TemporalDataset(Protocol):
    @property
    def metadata(self) -> TemporalMetadata: ...

    def get_marginal(self, time_index: int, split: str):
        """Return a torch Dataset / lazy indexable view, not all samples in RAM."""
        ...
```

**Required semantics:**

- Physical times are strictly increasing and correspond to data provenance.
- Model-time conversion is deterministic, invertible over the relevant interval, recorded in artifacts, and applied exactly once.
- Data typically shaped `[N_t, D]` per time, with potentially different `N_t` at each observation time.
- `get_marginal(time_index, split)` must never invent sample correspondences across different times.
- Optional trajectories/pairings have separate explicit accessors and flags when supported.
- Fit normalization **only on training data**, save its parameters, reuse on val/test and on inference/student queries. Include inverse-transform when necessary.
- Deterministic split generation: seed and scheme saved; if identities exist, design leakage-safe entity-level splits.
- Device transfer occurs during batching, not by putting the entire dataset on the GPU.

### Large-data handling

- Keep actual data under configurable root: `data/raw/` + `data/processed/` (ignored by Git).
- Suitable storage: `.npy` plus `np.load(..., mmap_mode='r')` for array datasets; HDF5 or Zarr for bigger/chunked collections; PyTorch `Dataset`/`DataLoader` for lazy access.
- Do not mistake compressed `.npz` for efficient mmap-backed random access.
- Keep large files on persistent shared server storage if the team shares a machine; paths must be configurable, not hardcoded.
- Do not have import-time downloads, full data scans, GPU allocation, or preprocessing.
- Report source URL/DOI, license/access restrictions, checksums/version, preprocessing command, and expected disk footprint in the dataset's documentation.

### Initial dataset shortlist (candidate, subject to confirmation)

- Synthetic: Gaussian mixture temporal toy; moving Gaussian temporal toy; high-dimensional synthetic temporal toy.
- Real: CITE 50D; CITE 100D; cell tracking.

If CITE 50D/100D are two representations of one raw source, implement a common loader with a dimension option. Only implement data sources actually available and licensed. Initial acceptance test uses one deterministic synthetic dataset with 3+ timepoints (including an irregular-time example).

## 6. Teacher method contracts

**Separate pretraining behavior from frozen inference behavior.** One generic fit loop is not required and may be scientifically invalid. Keep model-specific objective/coupling/interpolant/optimizer behavior in each model package.

### Conceptual interfaces

```python
from pathlib import Path
from typing import Protocol
import torch

class FrozenTeacher(Protocol):
    def velocity(
        self,
        x: torch.Tensor,                 # [B, D]
        t: torch.Tensor,                 # scalar or [B], documented
        *,
        context: dict | None = None,
    ) -> torch.Tensor:                  # [B, D]
        ...

class TeacherMethod(Protocol):
    def fit(self, dataset, config: dict, run_dir: Path) -> Path:
        """Use original algorithm; return restorable checkpoint path."""
        ...

    def load(self, checkpoint: Path, device: str) -> FrozenTeacher:
        """Return frozen teacher adapter in evaluation mode."""
        ...
```

**API validation:**

- For `[B, D]` input, output shape `[B, D]`; finite values for tested valid states/times.
- Correct device/dtype handling. Clarify accepted time tensor shape and broadcasting.
- If model requires path/context information beyond `(x,t)`, validate it explicitly and expose `context` requirements and a compatible query sampler. Do not claim unconditional access when that is false.
- No teacher parameter changes or gradients during distillation. Gradients **must** still flow into PC parameters.
- A `velocity` method must be mathematically consistent with the teacher's trained time variable and dynamics. If no equivalent query is available, mark the teacher **not yet distillation-compatible** rather than fabricate one.

**Method directory ownership example:**

```text
pretrain/models/alicfm/
  __init__.py
  model.py       # architecture(s)
  objective.py   # method-specific losses
  trainer.py     # its optimizers/schedulers/training loop
  adapter.py     # FrozenTeacher adapter
```

Model registry names `mfm`, `mmfm`, `otpfm`, `alicfm` should use lazy imports when algorithm-specific dependencies differ. Prefer package extras for heavyweight optional dependencies when practical; don't import all four implementations at CLI startup just to list choices.

## 7. Checkpoints and provenance contract

### Output convention

```text
outputs/pretrain/<method>/<dataset>/<run_id>/
  resolved_config.yaml
  manifest.json
  checkpoints/last.pt
  logs/                   # optional

outputs/distill/<method>/<dataset>/<student>/<run_id>/
  resolved_config.yaml
  manifest.json
  checkpoints/last.pt
  logs/                   # optional
```

### Minimum teacher manifest

```json
{
  "schema_version": 1,
  "artifact_type": "teacher",
  "method": "alicfm",
  "method_version": "commit-or-package-version",
  "model_config": {},
  "training_config": {},
  "dataset": {
    "name": "cite50",
    "source_version": "source-or-checksum",
    "dim": 50,
    "physical_times": [],
    "model_times": [],
    "time_transform": {},
    "preprocessing_id": "hash-or-version",
    "split_seed": 42
  },
  "seed": 42,
  "code_revision": "git-sha-if-available",
  "checkpoint_relpath": "checkpoints/last.pt"
}
```

`physical_times`, `model_times`, and the other fields are **required populated values in real artifacts**; empty lists in this example are schema placeholders, not valid finished metadata for time-series data. Store full preprocessing statistics via a referenced artifact rather than embedding giant arrays in JSON.

Checkpoint must allow exact model construction + `state_dict` load. Use `torch.load(..., map_location=..., weights_only=True)` where the format supports it and avoid arbitrary code deserialization. A resume checkpoint additionally needs optimizer, scheduler, and RNG state where applicable. Do not confuse inference checkpoint loading with full training resume. Record the frozen teacher checkpoint path/provenance in the student artifact.

Prefer atomic writes (temporary path then replace) for manifests and checkpoints to reduce partially written runs. Log the exact resolved config. W&B may be optional; local runs must work without login.

## 8. Distillation contract and scientific validity checks

### Pipeline

1. Read teacher checkpoint and manifest; resolve registered method and reconstruct frozen teacher.
2. Validate that the teacher supports velocity inference for this data and that method-specific context is satisfied.
3. Resolve original data metadata/preprocessing; confirm dimension and time mapping match student and sampler.
4. Choose a `DistillationSampler` defining `q(x_t,t)` with documented source and reproducibility.
5. Sample teacher query batches; call `teacher.velocity` in evaluation/no-grad mode.
6. Construct PC student, compute `student.velocity(x_t,t)` and default squared error, update **student only**.
7. Save student checkpoint with sampler, teacher, dataset, and PC boundary-condition metadata.

### Conceptual distillation batch

```python
from dataclasses import dataclass
import torch

@dataclass
class DistillationBatch:
    x_t: torch.Tensor             # [B, D]
    model_t: torch.Tensor         # [B] (preferred)
    context: dict | None = None
```

Baseline loss, when applicable:

```python
with torch.no_grad():
    target = teacher.velocity(batch.x_t, batch.model_t, context=batch.context)
loss = torch.nn.functional.mse_loss(
    student.velocity(batch.x_t, batch.model_t), target
)
```

### Hard scientific checks

- **Do not assume `x_t = (1-t)z_0 + t x_1`** for every teacher. Multi-marginal datasets represent distributions at multiple times and may use learned/stochastic interpolants or method-specific couplings.
- Distinguish unpaired marginals from observed trajectories. Never invent ID matching, path continuity, or ground-truth couplings.
- Real observation times and normalized model times are distinct; query the teacher at precisely the time convention in its checkpoint.
- The current Gaussian-initial PC parameterization may not represent the real marginal at earliest observed time. Before claiming distillation compatibility, document and either (a) ensure a legitimate transformation/initial law, or (b) extend the PC initialization. Do not silently re-label physical initial state as standard Gaussian.
- Use the same data scaling in teachers and students. A mismatch invalidates velocity matching.
- The student PC must expose `velocity`; keep `sample` and `log_prob` behavior where supported but do not rewrite PC mathematics under a packaging task.
- Flow-specific path quantities, e.g. a coupling or tangent velocity, can require more than a simple `(x,t)` network; review each paper/codebase and document the adapter semantics.

For the first framework smoke test, a toy teacher with an explicitly valid path and a PC matching that path is acceptable. **Do not claim it validates the research result for all four teachers.**

## 9. Implementation phases and acceptance criteria

### Phase 0 — Read-only audit and contract approval

- Inventory entry points, package imports, tests, configs, outputs, and how checkpoint loading works today.
- Make migration map from existing `repc` source files to proposed `flow2pc` modules.
- Confirm names for the first model and synthetic dataset; decide how physical/model time is represented.
- Produce a short plan and record unresolved scientific decisions; make no implementation changes if only planning requested.

**Exit criterion:** mentor/team agree on CLI schema, teacher adapter, dataset representation, and checkpoint metadata.

### Phase 1 — Minimal runnable skeleton (recommended first Codex implementation)

- Add `src/flow2pc`, package configuration, `main.py`, config validation, registries, shared seed utility.
- Include one deterministic temporal toy dataset and one **real** working training/inference method available from existing source (possibly legacy TorchCFM with an appropriate compatible dataset; otherwise use a documented toy velocity model only for an infrastructure smoke test).
- Implement checkpoint save/load and manifest; errors on unknown stage/model/dataset.
- Add `pytest` dev dependency if needed; add small tests. Preserve legacy scripts.

**Exit criterion:** `pip install -e '.[dev]'`, registry lookup, toy data loading, one train/save/load/infer roundtrip, and existing smoke tests pass. Do not claim the four paper models are done.

### Phase 2 — Initial paper-teacher integration

- Integrate supplied ALI-CFM and/or OTP-FM source into owned subpackages.
- Preserve exact optimizer/training logic; add adapter and provenance.
- Add only available real-world loaders and matching configs.

**Exit criterion per method:** one forward/optimizer step; checkpoint roundtrip; finite properly shaped velocity on valid test inputs; clear note whether it supports Flow2PC teacher querying.

### Phase 3 — Parallel integrations from teammate

- Teammate merges MFM and MMFM on the same contracts and adds model-specific tests.
- Avoid unrelated changes to shared runner/data internals; resolve a method's differences via adapter or an explicitly documented contract extension.

**Exit criterion:** four registered namespaces, each correctly classified as `implemented`, `unavailable`, or `incompatible` for supported datasets, and genuine validation for any claimed working method.

### Phase 4 — Flow2PC student integration

- Adapt existing PC implementations without silently changing their dynamics.
- Implement sampler registry and frozen-teacher target loss.
- Produce at least one reproducible end-to-end pretrain -> reload -> distill smoke test where initial conditions and time semantics are valid.

**Exit criterion:** student updates, teacher parameters unchanged, matching metadata, reloadable student artifact, and a test of the chosen query distribution.

### Phase 5 — Documentation and contributor hygiene

- Document setup, data acquisition, model owners, checkpoint schema, supported combinations matrix, known scientific limitations, and licensing/source acknowledgements.
- Keep README accurate and avoid representing roadmap items as implemented functionality.

## 10. Tests and quality gates

**Data tests**

- Deterministic metadata and splits; 3+ timepoints including irregular spacing.
- Distinct physical vs model time; correct feature shape/dtype.
- Loader never assumes paired samples across marginal times.
- Normalization uses only train samples and round-trips (when invertible).
- Lazy dataset path supports batching without reading the full source at construction.

**Registry/CLI tests**

- Known model/dataset resolution, useful unknown-key errors.
- Pretrain/distill config validation and environment-variable behavior.
- Lazy model import for unavailable optional deps.
- Root CLI entry point dispatches correct stage.

**Teacher/checkpoint tests**

- One optimizer step with finite loss for each implemented teacher.
- Checkpoint save/load preserves predictions to expected numerical precision.
- Manifest includes dimensions/time/preprocess metadata; mismatches fail early.
- Frozen adapter returns finite `[B, D]`; honors conditioning and time shapes.

**Distillation tests**

- Student gets gradients and changes after a step; frozen teacher does not.
- Explicit valid sampler used; no implicit random image/noise interpolation.
- Mismatched time mapping, normalization or feature dimension raises error.
- Existing `repc` training/evaluation tests continue to pass until migration complete.

**Lightweight local checks:**

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check .
```

If a command is unavailable before adding a dependency, document the blocker; do not report tests as passed without running them. Avoid actual large training jobs in CI.

## 11. Team workflow and merge discipline

- Agree on contracts/registry names in a small foundational PR first.
- Contributor A owns `pretrain/models/alicfm/` and `otpfm/`.
- Contributor B owns `pretrain/models/mfm/` and `mmfm/`.
- Shared code (`data/`, `pretrain/base.py`, `registry.py`, `utils/checkpoint.py`) has one designated review owner per change.
- Each model PR includes model code, method adapter, config for a verified dataset, tests, README source attribution, and dependency notes.
- Do not silently vendor external code without checking license, attribution, and dependency compatibility.
- Keep model-specific patches scoped; avoid copy-pasting all four repos and duplicating shared data loaders.

## 12. Definition of success and known unknowns

The **first successful milestone** is *not* an experimental leaderboard. It is a command-driven, reproducible run on one valid synthetic temporal dataset where code can load data, train one actual teacher implementation, save/reload, and run a compatible PC distillation step without changing the teacher weights.

Open questions for mentor/team review:

1. Confirm the precise 5–6 dataset versions, physical time grids, availability, and licenses.
2. Verify which teacher APIs genuinely support a velocity query and what context each requires.
3. Decide if PC initial law/boundary needs to change for observed earliest-time marginals.
4. Define scientifically valid distillation query distributions for each teacher.
5. Decide whether CITE 50D and 100D are data configurations or distinct sources.
6. Decide who owns shared code review and how pretrained checkpoints are shared externally.

## 13. Copy-paste Codex startup prompt

> Read `AGENTS.md` and `docs/ARCHITECTURE_SPEC.md`, then inspect the current repository before modifying code. I want **Phase 1 (minimal runnable skeleton) only**. First report the proposed diff, the existing APIs/checkpoints/tests that need to remain working, and any science-compatibility issues. Implement an installable `src/flow2pc` package, `python main.py --config ...` CLI, registries, one deterministic temporal toy loader, checkpoint manifest utilities, and tests. Keep legacy `repc/`, `train.py`, and `eval.py` working; do not implement fake MFM/MMFM/OTP-FM/ALI-CFM algorithms, rewrite the PC math, download large datasets, or begin later phases. Run available tests and report exact results and remaining work.

