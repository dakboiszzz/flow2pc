# Flow2PC

Flow2PC distills a frozen flow-matching velocity field into a time-dependent
probabilistic circuit. The student training objective is teacher-velocity MSE.

## Retained code

```text
main.py
configs/pretrain/
src/flow2pc/
  pc/hclt_pc.py                 HCLT student and its original checkpoint API
  pc/base.py                    shared circuit interface
  pc/gaussian_pc.py             helpers imported by the HCLT implementation
  pretrain/models/alicfm/       ALI-CFM networks, training and frozen adapters
  data/                        real-data loaders and time metadata
  utils/                       configuration, checkpoints and provenance
  distill/                     unified distillation integration boundary
src/repc/                      retained model/ALI-adapter compatibility imports
tests/                         HCLT mathematics and shared framework contracts
docs/                          architecture and migration notes
```

The HCLT implementation retains the constructor, free-source Fourier option,
state-dict keys and velocity calculations used in the successful real-data runs.
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

The existing real data lives in `/data/lthung/ali-cfm-data`; configure `data.root`
or `FLOW2PC_DATA_ROOT` on other systems. Data, weights and results remain outside
the code checkout.

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

## Development

`make test` and `make lint` are available for explicit use. No training,
distillation, evaluation or tests were executed during this cleanup, at the
user's request. The migration changes have received static checks only; their
runtime correctness and numerical parity remain unverified.

The [architecture specification](docs/ARCHITECTURE_SPEC.md) and
[proposal](README_PROPOSAL.md) describe the broader target. References there to
old MNIST compatibility are historical and superseded by the current scope.
