# Flow2PC — Proposed Research Library Architecture

> **Status:** Architecture proposal for discussion with the mentor and contributors; **not a description of an already implemented system**.  
> **Repository:** https://github.com/dakboiszzz/flow2pc  
> **Primary goal:** A shared, reproducible pretraining library for four flow-matching teachers, followed by a separate Flow2PC probabilistic-circuit distillation pipeline.

## 1. Executive summary

We propose **one installable Python package** consisting of three distinct layers:

1. **Shared data library:** reproducible loaders for approximately 5–6 synthetic and real-world datasets, independent of the teacher implementation.
2. **Teacher model zoo (`pretrain/`):** the original **MFM, MMFM, OTP-FM, and ALI-CFM** algorithms, with their own training objectives and trainers, exposed through shared *execution and checkpoint* conventions.
3. **Flow2PC student (`distill/` + `pc/`):** a separate pipeline that loads an already trained teacher, obtains valid teacher-velocity queries, and optimizes a time-dependent probabilistic circuit (PC).

The mentor's preferred interface is retained:

```bash
python main.py --config configs/pretrain/alicfm_cite50.yaml
python main.py --config configs/distill/alicfm_cite50_hclt.yaml
```

These commands describe the **target interface**; they will work only after the corresponding adapters, data loaders, and configs are implemented.

### Architectural diagram

```text
                         configs/*.yaml
                               |
                           main.py
                               |
                      stage dispatcher
                        /           \
                       /             \
             PRETRAIN /               \ DISTILL
                     v                 v
       +-----------------------+   +---------------------------+
       | Shared data registry  |   | Load frozen teacher      |
       | synthetic / real      |   | from saved checkpoint     |
       +-----------+-----------+   +-------------+-------------+
                   |                             |
                   v                             v
       +-----------------------+   +---------------------------+
       | Teacher model zoo     |   | Query-state/time sampler  |
       | MFM / MMFM            |   | + teacher velocity calls  |
       | OTP-FM / ALI-CFM      |   +-------------+-------------+
       +-----------+-----------+                 |
                   |                             v
                   v                 +---------------------------+
       teacher checkpoint +          | PC student                 |
       metadata + data provenance    | HCLT / region / Gaussian  |
                                     +-------------+-------------+
                                                   |
                                                   v
                                     student checkpoint + metadata
```

**Dependency rule:** teacher pretraining must not import or depend on `distill/` or `pc/`. The distillation pipeline may import the trained teacher adapter.

## 2. Why this structure?

The existing public repository is organized around `repc/`, a single MNIST/TorchCFM flow teacher, and a root `train.py` which handles `flow` and `distill` tasks. Its PC code is valuable and should be preserved. The new project requires multiple teacher methods and multi-marginal datasets; simply placing four separate upstream repositories side-by-side would duplicate data preprocessing, configuration, run orchestration, and checkpoint loading.

**What is shared:** CLI, configuration conventions, dataset access, model registration, checkpoint conventions, basic reproducibility and logging.  
**What stays method-specific:** model architectures, coupling/interpolants, loss functions, training loops/optimizers, required batches, solver details, and algorithm-specific dependencies.  
**What is unique to Flow2PC:** PC student models, teacher-query distribution, velocity-matching/distillation logic, and student checkpoints.

## 3. Proposed source tree

```text
flow2pc/
├── main.py                       # Small unified CLI wrapper
├── pyproject.toml                # Installable src-layout Python package
├── README.md                     # Current public README; migrate carefully
├── README_PROPOSAL.md            # This proposal, until approved
├── AGENTS.md                     # Codex instructions
├── docs/
│   └── ARCHITECTURE_SPEC.md      # Implementation contract and checklist
├── configs/
│   ├── pretrain/
│   │   ├── alicfm_cite50.yaml
│   │   ├── otpfm_cite50.yaml
│   │   └── mfm_toy.yaml
│   └── distill/
│       └── alicfm_cite50_hclt.yaml
├── src/
│   └── flow2pc/
│       ├── cli.py
│       ├── pretrain/
│       │   ├── registry.py
│       │   ├── base.py
│       │   ├── runner.py
│       │   └── models/
│       │       ├── mfm/
│       │       ├── mmfm/
│       │       ├── otpfm/
│       │       └── alicfm/
│       ├── data/
│       │   ├── base.py
│       │   ├── registry.py
│       │   ├── preprocessing.py
│       │   ├── synthetic/
│       │   └── real/
│       ├── distill/
│       │   ├── runner.py
│       │   ├── sampling.py
│       │   └── losses.py
│       ├── pc/
│       │   ├── base.py
│       │   ├── gaussian_pc.py
│       │   ├── hclt.py
│       │   ├── hclt_pc.py
│       │   └── region_pc.py
│       └── utils/
│           ├── config.py
│           ├── checkpoint.py
│           └── seed.py
├── tests/
│   ├── test_data_contract.py
│   ├── test_teacher_contract.py
│   ├── test_checkpoint_roundtrip.py
│   └── test_distillation_smoke.py
├── data/                         # Actual data; Git-ignored
│   ├── raw/
│   └── processed/
└── outputs/                      # Checkpoints/logs; Git-ignored
```

The tree is a **target layout**, not a requirement to create empty placeholder files for every feature immediately. Each teacher's folder may contain `model.py`, `objective.py`, `trainer.py`, and `adapter.py` as needed, without forcing all methods into the same trainer.

## 4. Teacher model zoo: one entry point, four independent methods

Each registered teacher implementation must support:

- **Fit:** train on a dataset using its faithful original training algorithm.
- **Checkpoint:** save a restorable artifact with method, architecture, dataset, time normalization, and preprocessing metadata.
- **Load:** reconstruct a frozen model that can expose a suitable velocity query function.

The **inference adapter** presents a common minimum contract:

```python
velocity(x_t, t, *, context=None) -> tensor  # input and output: [batch, dim]
```

`context` is optional **only if the underlying model is unconditional**. Some methods may need time grids, conditioning, coupling, or other information. The adapter must expose these requirements explicitly and **must not silently discard them**. At integration time, verify that each of the four teachers genuinely provides the velocity field needed for the chosen Flow2PC objective.

Do **not** replace MFM/MMFM/OTP-FM/ALI-CFM training with the existing TorchCFM loss simply to make the code uniform. Standardize their boundaries, not their mathematics.

## 5. Dataset ownership and large-data policy

The library stores **dataset loaders and preprocessing code**, not full real-world dataset files. The root `data/` directory is ignored by Git, while `src/flow2pc/data/` contains versioned code.

A dataset should expose:

- Feature dimensionality and dtype.
- **Physical observation times** and the normalized model-time map, if used.
- Observed marginal samples at each time point; optional trajectory IDs/pairings **only if actually available**.
- Deterministic train/validation/test splits and normalization fitted on training data only.
- Efficient batch access without requiring every file to fit in RAM.

**Initial candidate coverage (to confirm with mentor):** Gaussian mixture toy, moving Gaussians / time-varying toy, a high-dimensional synthetic example, CITE 50D, CITE 100D, and a cell-tracking dataset. CITE 50D/100D should share a loader when they derive from the same source. Do not claim that every candidate is already available or compatible with every teacher.

### Storage and loading

```text
repo code                 shared/local storage
src/flow2pc/data/  --->  /data/flow2pc/raw/...
configs/data paths  --->  /data/flow2pc/processed/...
```

Use a configurable `data.root` or `FLOW2PC_DATA_ROOT` rather than a hardcoded machine path. For larger arrays use lazy batched access (`.npy` memory maps, HDF5, or Zarr where appropriate). **`np.load(..., mmap_mode='r')` works for `.npy`, not as a true memory-mapping strategy for compressed `.npz`.** Downloading, preprocessing, and storage must be separately documented. Avoid training-time downloads unless explicitly enabled.

Do not commit raw datasets, processed large arrays, checkpoints, W&B files, or local credentials. Provide download/preparation instructions and a small generated toy dataset for smoke tests.

## 6. Teacher pretraining pipeline

```text
YAML config
  -> validate selected model and dataset
  -> load dataset metadata, split, normalization, observation times
  -> construct method-specific batches
  -> instantiate registered teacher method
  -> run method's original optimization loop
  -> save checkpoint + metadata + resolved config
```

### Example configuration (illustrative values, not validated hyperparameters)

```yaml
stage: pretrain
seed: 42
model:
  name: alicfm
  params:
    hidden_dim: 256
data:
  name: cite50
  root: ${FLOW2PC_DATA_ROOT}
  params:
    normalize: true
training:
  epochs: 100
  batch_size: 512
  lr: 0.0001
output:
  dir: outputs/pretrain/alicfm/cite50
```

`${FLOW2PC_DATA_ROOT}` here means **documented environment-variable substitution that the config loader must implement**, not YAML built-in behavior. Alternatively use a normal absolute/relative `root` value. A missing path must produce a clear error.

A training run should produce something like:

```text
outputs/pretrain/alicfm/cite50/<run_id>/
├── resolved_config.yaml
├── manifest.json
├── checkpoints/
│   ├── last.pt
│   └── best.pt                  # Optional if validation is defined
└── logs/
```

At minimum, the manifest should identify teacher method/version, model config, dataset source/version, dimension, time mapping, preprocessing/split parameters, code revision when available, seed, and compatible checkpoint path.

## 7. Separate Flow2PC distillation pipeline

Distillation takes a **trained teacher checkpoint**, not a new teacher training job.

```text
teacher checkpoint -> frozen teacher adapter
                 + shared data metadata and time convention
                 + valid teacher-query distribution q(x_t, t)
                 -> teacher velocity targets (no teacher gradient)
                 -> PC student velocity prediction
                 -> loss and PC updates
                 -> student checkpoint
```

### Example configuration (illustrative)

```yaml
stage: distill
seed: 42
teacher:
  checkpoint: outputs/pretrain/alicfm/cite50/<run_id>/checkpoints/last.pt
student:
  name: hclt
  params:
    num_latents: 64
    time_order: 4
distillation:
  sampler:
    name: teacher_path
  steps: 20000
  batch_size: 256
  lr: 0.001
output:
  dir: outputs/distill/alicfm/cite50/hclt
```

The `<run_id>` value is a path placeholder that must be replaced with a real checkpoint path.

A basic velocity loss is:

```text
L = E_(x_t,t ~ q) [ ||v_PC(x_t,t) - v_teacher(x_t,t)||² ]
```

**Scientific guardrails:** (1) The query distribution `q` must be meaningful for each trained teacher. Do not universally reuse the existing Gaussian-noise-to-MNIST straight-line path. (2) Time in real datasets can be irregular; keep physical times and normalized model times distinguishable. (3) The current PC code assumes a standard-Gaussian distribution at its initial time; real multi-marginal models may need a different initial condition. Confirm this before treating every method/dataset pairing as compatible.

## 8. How we collaborate and merge

| Workstream | Primary owner | Output |
|---|---|---|
| Framework contracts, CLI, data registry, checkpoint schema | Agree jointly; one owner merges | Stable base skeleton |
| ALI-CFM + OTP-FM | Contributor A | Two method folders + adapters + smoke tests |
| MFM + MMFM | Contributor B | Two method folders + adapters + smoke tests |
| Flow2PC distillation and PC integration | Flow2PC maintainers | Checkpoint-driven student training |

Suggested feature branches: `feat/teacher-alicfm-otpfm`, `feat/teacher-mfm-mmfm`, and (separately) `feat/framework` or a small foundational PR merged first. Avoid both people editing a global runner at the same time. Model-specific source lives within each model's folder; only add a registry entry and a config when integrating it.

### Definition of done for a teacher integration

1. Import and register the model without forcing unrelated optional dependencies.
2. Load one supported dataset and run at least one actual optimizer step.
3. Save, reload, and run inference from a checkpoint.
4. Validate velocity output shape, finite values, device, time convention, and any required conditioning.
5. Run a minimal teacher-to-PC distillation smoke test **only when the chosen query distribution and PC boundary condition are scientifically valid**.
6. Document dependencies, data assumptions, and source/paper attribution.

## 9. Recommended migration sequence

**Phase 0 — Approve contracts.** Agree on dataset/time representation, velocity API, checkpoint metadata, model names, and the first genuinely supported toy dataset. Document unresolved research choices.

**Phase 1 — Build the minimum framework.** Add a `src/flow2pc` package, simple YAML CLI, registries, toy loader, and checkpoints. Preserve the existing `train.py`, `eval.py`, and `repc/` commands while migrating.

**Phase 2 — Integrate the first teacher.** Add ALI-CFM or whichever existing code is easiest to smoke-test on toy data. Keep its training loop faithful to the source method. Test checkpoint loading.

**Phase 3 — Parallel teacher additions.** Contributors independently merge OTP-FM, MFM, and MMFM using the agreed contracts; validate datasets separately.

**Phase 4 — Distillation integration.** Reuse existing PC components; introduce teacher-adapter-based target queries and a configurable sampling strategy. Demonstrate one reproducible pretrain-to-distill run.

**Out of scope for this first library milestone:** the full five-axis experiment program, benchmark dashboards, broad automated hyperparameter search, and guaranteeing all 4 × 6 method–dataset combinations.

## 10. Decisions to confirm together

| Question | Proposed default |
|---|---|
| First teacher/dataset smoke test? | One existing teacher on an inexpensive synthetic temporal dataset |
| Exactly which 5–6 datasets? | Confirm sources, licensing, time structure, and actual availability |
| Uniform training implementation? | **No**; method-specific trainers behind shared registry |
| Checkpoint convention? | Shared manifest + method-specific saved weights/state |
| Source of large data? | Local or shared server storage, never Git |
| Which physical-time and model-time convention? | Explicit per dataset and saved in metadata |
| PC initial density constraint? | Verify/extend before real multi-marginal distillation |
| Should existing MNIST tooling disappear? | **No**; preserve as legacy until parity tests pass |

## 11. Files to give Codex

- **`AGENTS.md`** — short repository-level instructions automatically discoverable by Codex.
- **`docs/ARCHITECTURE_SPEC.md`** — authoritative implementation contract, interfaces, tests, and phase-by-phase acceptance criteria.

Ask Codex to first inspect the existing code and implement **Phase 1 only**; do not let it fabricate the four paper implementations or silently change the PC mathematics. See the spec for an exact starter prompt.
