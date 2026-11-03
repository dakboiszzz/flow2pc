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

## Development

`make test` and `make lint` are available for explicit use. No training,
distillation, evaluation or tests were executed during this cleanup, at the
user's request. The migration changes have received static checks only; their
runtime correctness and numerical parity remain unverified.

The [architecture specification](docs/ARCHITECTURE_SPEC.md) and
[proposal](README_PROPOSAL.md) describe the broader target. References there to
old MNIST compatibility are historical and superseded by the current scope.
