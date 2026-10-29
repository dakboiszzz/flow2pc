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

ALI-CFM pretraining is available through the existing configuration:

```bash
python main.py --config configs/pretrain/alicfm_cite50.yaml
```

This command starts training only when explicitly executed. Its outputs are
self-describing teacher checkpoints, fitted preprocessing and provenance.
The unified distillation CLI still validates its configuration contract only;
the runnable CITE-50 teacher/student example is separate follow-up work.

## Development

`make test` and `make lint` are available for explicit use. No training,
distillation, evaluation or tests were executed during this cleanup, at the
user's request. The current changes have only received static dependency review.

The [architecture specification](docs/ARCHITECTURE_SPEC.md) and
[proposal](README_PROPOSAL.md) describe the broader target. References there to
old MNIST compatibility are historical and superseded by the current scope.
