# Flow2PC

Flow2PC distills frozen flow-matching velocity fields into time-dependent
probabilistic circuits. ALI-CFM and MFM use teacher-velocity MSE.

## Retained code

```text
main.py
configs/pretrain/
src/flow2pc/
  pc/hclt_pc.py                 HCLT student and its original checkpoint API
  pc/base.py                    shared circuit interface
  pc/gaussian_pc.py             helpers imported by the HCLT implementation
  pretrain/models/alicfm/       ALI-CFM networks, training and frozen adapters
  pretrain/models/otpfm/        original OTP-FM, dataset profiles and frozen adapter
  data/                        real-data loaders and time metadata
  utils/                       configuration, checkpoints and provenance
  distill/                     unified distillation integration boundary
src/repc/                      retained model/ALI-adapter compatibility imports
tests/                         HCLT mathematics and shared framework contracts
docs/                          architecture and migration notes
```

The HCLT implementation retains the constructor, free-source Fourier option
and state-dict keys used in the successful real-data runs. Its `log_scale`
clamp has been removed, so values outside the former range can change velocity
and likelihood results for existing checkpoints.
Its Gaussian/base helper files are retained because they are actual imports.
The ALI-CFM classes from the real-data version of `flow.py` are kept in
`pretrain/models/alicfm/checkpoint_adapter.py`, with `repc.flow` compatibility.

MNIST trainers, classifier, image metrics, endpoint/benchmark commands, loader,
TorchCFM MNIST model and their compatibility wrappers have been archived outside
the repository. The unused region-graph benchmark and MNIST topology modules were
also archived. See [the cleanup and migration notes](docs/MIGRATION.md).

## Setup

```bash
python -m pip install -e '.[dev,alicfm]'
```

For MFM, install the optional dependencies with
`python -m pip install -e '.[metric]'` and keep the original MFM source checkout and its datasets at the
paths selected by the configs. No MFM data or weights are downloaded.

## MFM integration

For all locally available datasets except AFHQ, the new snapshot workflow has
20 profiles, including the H5AD EB PCA/PHATE representations, transport data and
the three Gaussian exports. It supports paired MFM pretraining followed by HCLT
or Gaussian PC distillation:

```bash
python scripts/run_mfm_data.py list
python scripts/run_mfm_data.py both all --student both --dry-run
python scripts/run_mfm_data.py both cite50 --student both --device cuda:0 --run-name exp1
```

The last command prepares snapshots and starts training. Data sources are linked
under `data/mfm/sources/`; outputs go under `outputs/mfm/`. New recipes use
train-only normalization and exclude temporal test holdouts from checkpoint
selection. See [the complete catalog, commands and protocol](docs/MFM_LOCAL_DATA.md).
Runtime and numerical behavior remain unverified; no model jobs or tests were
run during implementation. The older workflow below remains available.

MFM pretraining delegates to the original two-phase geopath/flow trainer:

```bash
PYTHONPATH=src python main.py --config configs/pretrain/mfm/arch.yaml --validate-only
PYTHONPATH=src python main.py --config configs/pretrain/mfm/arch.yaml
```

The 17 runnable MFM pretraining configs are under `configs/pretrain/mfm/`.
Its 22 original MFM/CFM YAML files are preserved under
`configs/pretrain/mfm/upstream/`; five CFM files are reference-only.
The trainer writes native Lightning checkpoints. Point a distillation config
at matching flow and geopath checkpoints, then inspect it with:

```bash
PYTHONPATH=src python main.py --config configs/distill/mfm/arch/hclt.yaml --validate-only
```

MFM distillation supports HCLT and Gaussian students with velocity MSE only.
The HCLT `log_scale` is not clamped. MFM's physical observation labels,
normalized model times and preprocessing metadata are kept separate. See
[MFM integration notes](docs/MFM_INTEGRATION.md) for datasets and limitations.

MMFM and the MNIST latent pipeline are not integrated in this checkout.

For the unified local bundle in `data/`, use the launcher below. It overrides the
historical `/data/lthung/...` paths, pairs the ALI-CFM teacher/student checkpoints,
and writes results under `outputs/local/`:

```bash
python scripts/run_local_data.py both cite50 --dry-run
python scripts/run_local_data.py both cite50 --device cuda:0 --run-name experiment1
```

The second command starts training and the recipes' evaluation. See
[local dataset commands and current limits](docs/LOCAL_DATA.md) for individual
stages, OTP-FM pretraining and unsupported distillation combinations. Data and
results under the repository remain Git-ignored. Existing historical configs
still accept explicit `data.root` overrides; their explicit roots take precedence
over `FLOW2PC_DATA_ROOT`.

Prepared ALI-CFM teacher and student commands:

```bash
bash scripts/run_alicfm.sh pretrain cite50
bash scripts/run_alicfm.sh distill cite50
```

These commands start training only when explicitly executed. The student loads
the teacher checkpoint produced by the first command and optimizes velocity MSE
only. Configs cover CITE/Multi 5D, 50D and 100D, EB PHATE-2D, spatial
transcriptomics, knot and cell tracking. The ST teacher method is recovered from
the matching historical ALI-CFM implementation, rather than changing its
regularizer. See [dataset support, commands and limitations](docs/ALICFM_DATASETS.md),
including how to use existing checkpoints with their original preprocessing.

To check the complete teacher-to-student workflow, with reference and comparison
images logged to W&B project `flow2pc`:

```bash
bash scripts/check_alicfm_pipeline.sh cite50
```

This uses the full configured training budgets. Outputs go to
`/data/lthung/flow2pc/framework/migration-check/cite50/{teacher,student}/`.
Each run's `eval/media.json` lists its figures. Set `FLOW2PC_DATA_ROOT` and
`FLOW2PC_CHECK_ROOT` for different directories; the check script forwards them
as explicit config overrides. Both stages automatically evaluate after training.
The check script reuses an existing teacher and resumes its student.

Standalone evaluation uses existing migrated checkpoints and performs no training:

```bash
bash scripts/run_alicfm.sh evaluate-teacher cite50
bash scripts/run_alicfm.sh evaluate-student cite50
```

These two commands use the original example-config output paths; pass
`--set output.dir=/actual/run/directory` (and, for a student,
`--set teacher.checkpoint=/actual/teacher/checkpoints/last.pt`) when evaluating
a different run. See [the recovered evaluation workflow](EVALUATION_WORKFLOW.md)
for image names and evaluation conventions.

To check all ten datasets concurrently, the local recipe is
`local_runs/check_all_alicfm.sh`. That folder is ignored by Git; the reusable
launcher is `scripts/run_alicfm_checks.py`. It reuses the previous sweep's fair
GPU queue, warm-up memory reservations and bounded workers. See
[parallel launch instructions](docs/PARALLEL_CHECKS.md) for preview, GPU selection
and output paths.

For the new Gulf of Mexico and Beijing teacher search, use the
[1,024-run ALI-CFM transport sweep](docs/ALICFM_TRANSPORT_SWEEP.md):

```bash
bash scripts/run_alicfm_transport_sweep_8gpu.sh --dry-run
bash scripts/run_alicfm_transport_sweep_8gpu.sh
```

It prepares 512 configs per dataset, uses eight GPUs with bounded parallel workers,
logs to W&B `flow2pc`, and saves per-run figures plus dataset-specific rankings
under `/data/lthung/flow2pc/alicfm-transport/sweep-512`. These are new dataset
adaptations using the existing ALI-CFM stages; numerical behavior is unverified
until training is run.

OTP-FM uses a shared model/trainer/adapter with dataset profiles; GOM and Beijing
are the first supported datasets. To prepare and run native-versus-migrated
teacher checks across eight GPUs:

```bash
bash scripts/run_otpfm_checks_8gpu.sh --backend native --dry-run
bash scripts/run_otpfm_checks_8gpu.sh --backend native
bash scripts/run_otpfm_checks_8gpu.sh --backend migrated
python scripts/compare_otpfm_runs.py
```

Each batch has eight teachers: two datasets, W2/W2Inf and seeds 42/43. Results and
best/final figures go under `/data/lthung/flow2pc/otp-fm/migration-check`, with
W&B project `flow2pc`. Numerical parity awaits these runs. OTP-FM's conditional
PC sampler is a separate, pending integration. See the
[OTP-FM migration notes](docs/OTP_FM_MIGRATION.md) for configuration, outputs,
native evaluation semantics and dataset extension points.

For OTP-FM on the ALI-CFM experiment datasets, prepare the
[100-config-per-profile sweep](docs/OTP_FM_ALI_DATA_SWEEP.md):

```bash
bash scripts/run_otpfm_ali_data_sweep_8gpu.sh --dry-run
bash scripts/run_otpfm_ali_data_sweep_8gpu.sh
```

Standalone OTP-FM teacher configs live in `configs/pretrain/`. For example:

```bash
python main.py --config configs/pretrain/otpfm_cite50.yaml
```

The sweep prepares 1,000 teachers across ten dataset/dimension profiles and defaults
to two workers per GPU across eight GPUs. It inherits each experiment's data and
W1 EMD settings, with the six README-specified frames for tracking, final figures,
W&B `flow2pc`, and disk-safe resumable orchestration. These are new dataset
adaptations; training/evaluation of this sweep has not been run by the assistant.

## Development

`make test` and `make lint` are available for explicit use. No training,
distillation, evaluation or tests were executed during this cleanup, at the
user's request. The migration changes have received static checks only; their
runtime correctness and numerical parity remain unverified.

The [architecture specification](docs/ARCHITECTURE_SPEC.md) and
[proposal](README_PROPOSAL.md) describe the broader target. References there to
old MNIST compatibility are historical and superseded by the current scope.
