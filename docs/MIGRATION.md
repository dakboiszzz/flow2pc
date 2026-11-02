# Student-focused cleanup

The current scope is the HCLT used in real-data velocity distillation, its required
dependencies, ALI-CFM teacher interfaces and the shared library infrastructure.
The user explicitly retired the original MNIST experiment suite.

## Kept

| File or area | Why it is required |
| --- | --- |
| `src/flow2pc/pc/hclt_pc.py` | Student density/message passing, time-conditioned leaves, velocity and saved-state API |
| `src/flow2pc/pc/gaussian_pc.py` | Provides Gaussian helpers imported by HCLT; retained whole to avoid a broken transitive import |
| `src/flow2pc/pc/base.py` | Circuit base API |
| `src/repc/hclt_pc.py`, `src/repc/pc.py` | Existing model import compatibility |
| `pretrain/models/alicfm/model.py`, `_source/` | Migrated genuine ALI-CFM networks, losses, couplings and training functions |
| `pretrain/models/alicfm/adapter.py` | Frozen global-time teacher/interpolant interface for new-format checkpoints |
| `pretrain/models/alicfm/checkpoint_adapter.py`, `src/repc/flow.py` | Original real-data `ALICFMTeacherWrapper` and `ALICFMGeneratorWrapper` interfaces |
| Real-data loaders, registry/config/checkpoint utilities | Data units, time conventions, provenance and checkpoint reconstruction |
| HCLT likelihood, derivative, velocity and time-batching tests | Relevant student mathematical regression coverage |

The complete HCLT implementation was retained from the successful real-data code,
with only its dependency import location adapted. The free initial marginal is
available through `fourier_zero_center=False`; callers must retain the saved model
configuration. The ALI wrapper class bodies were retained from the real-data
`flow.py`. `STUDENT_SOURCE.json` records their source hashes and changes.

`flow.py` had different meanings across commits: the original baseline version
only built the MNIST UNet, while the later real-data version also implemented ALI
adapters. Only the actual MNIST model was retired; the ALI adapter interfaces stay.

## Archived

- The six root MNIST/benchmark compatibility commands and `flow2pc.legacy`.
- MNIST training/evaluation/classifier and benchmark launchers in `scripts/legacy`.
- The MNIST loader and TorchCFM MNIST model namespace.
- Image-metric, categorical endpoint, capacity-benchmark and legacy-alignment tests.
- Region-graph and published MNIST topology modules and their unused aliases.
- Legacy MNIST usage documentation and the now-obsolete baseline layout manifest.

The archive is under `/data/lthung/cache/flow2pc-student-cleanup-20261029T075616Z`.
`snapshot/` preserves the complete pre-cleanup source; `final_retired/` contains the
actual archived paths. `manifest.json` and `RESTORE.md` document recovery. No Git
history, external dataset or trained model checkpoint was deleted or rewritten.

An unfinished CITE example was set aside under `pending_example/` so the cleanup
could be completed before adding new entry points. Subsequent user-authorized
work adds dataset-specific ALI-CFM teacher profiles and the unified velocity-MSE
student runner. See [the dataset migration notes](ALICFM_DATASETS.md) for source
recovery, commands and limits. Old experiment/sweep/evaluation launchers have not
all been migrated, and the new integration has not been runtime-verified.

## Validation scope

The user explicitly requested no execution. This cleanup used static import and
caller inspection and file-content comparisons only. No training, distillation,
evaluation or test suite was run. Earlier test counts describe earlier revisions
and must not be represented as validation of this cleanup.
