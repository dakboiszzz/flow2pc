# Flow2PC

Flow2PC reconfigures pretrained flow-matching velocity fields into time-dependent
probabilistic circuits. Teacher pretraining and PC distillation have separate
ownership, with shared configuration, data and checkpoint contracts.

The original PC and MNIST implementation is preserved from **`3eb65ef` —
`feat: add literature-aligned flow-to-PC benchmarks`**. The package layout changes
its ownership and imports, while retaining the baseline mathematical definitions.
See [the baseline source map](docs/BASELINE.json) and
[migration guide](docs/MIGRATION.md).

## Installation

Use Python 3.11 or newer in a project environment:

```bash
python -m pip install -e '.[dev,alicfm]'
```

## Layout

```text
main.py                          configuration-driven entry point
configs/pretrain/                verified method configuration
src/flow2pc/
  cli.py                         stage dispatch
  data/                          temporal metadata and shared loaders
  pretrain/models/alicfm/         migrated original ALI-CFM stages
  pretrain/models/torchcfm_legacy/ original MNIST teacher architecture
  pc/                            original Gaussian, HCLT and region PCs
  distill/                       boundary for unified distillation integration
  utils/                         configuration, registries and checkpoints
  legacy/                        original training/evaluation commands
src/repc/                        compatibility aliases, without duplicate models
tests/                           baseline and migration regression tests
scripts/legacy/                  original benchmark launchers
docs/                            architecture, provenance and legacy usage
```

Root `train.py`, `eval.py`, endpoint/classifier trainers and summarizers are small
compatibility entry points for the original commands. Their implementations live
under `flow2pc.legacy`. No datasets, checkpoints, experiment results or caches are
required in the code checkout.

## Framework commands

```bash
python main.py --list-models
python -m flow2pc.cli --list-datasets
python main.py --config configs/pretrain/alicfm_cite50.yaml --validate-only
```

The CITE-50 example reads the existing real data from
`/data/lthung/ali-cfm-data`. Set `data.root` in your configuration, or use
`FLOW2PC_DATA_ROOT`. `--validate-only` reads and validates data/configuration
without training, W&B initialization or run creation.

To explicitly start the configured teacher training:

```bash
python main.py --config configs/pretrain/alicfm_cite50.yaml
```

The migrated ALI-CFM trainer currently supports one CITE/Multi fold per run,
with its original staged losses and Adam optimizers. New runs fit preprocessing
on observed training cells only; existing checkpoints retain their original saved
preprocessing. The manifest records this protocol difference.

MFM, MMFM and OTP-FM are marked unavailable in the **new framework registry**
until their authentic implementations are migrated. Unified framework distillation
execution is deferred. The preserved original MNIST commands remain usable:

```bash
python train.py --help
python eval.py --help
```

See [original MNIST usage](docs/legacy/README.md) for its complete commands.
The original PCs enforce a standard-Gaussian source; a real-data teacher needs an
explicit source-compatible student and query sampler before unified distillation.

## Development

```bash
make test
make lint
```

Tests cover the original baseline, import/class compatibility, source-body parity,
ALI-CFM staged updates and checkpoint roundtrips. Small generated fixtures are
test inputs; no research training run is launched by these checks.

[Architecture specification](docs/ARCHITECTURE_SPEC.md) and
[proposal](README_PROPOSAL.md) describe the intended full framework. Planned
capabilities are distinct from the integrations currently available.
