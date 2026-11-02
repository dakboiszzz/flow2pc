# MFM integration

Method-owned code lives under `src/flow2pc/pretrain/models/mfm/`, following
the same organization as ALI-CFM:

- `trainer.py`: registry method and original pretraining command preparation/execution.
- `model.py`: original network construction and native weight restoration.
- `adapter.py`: frozen velocity interface and geopath sampling.
- `checkpoint_adapter.py`: paired checkpoint loading, metadata validation and freezing.
- `base.py`: loaded-teacher result contract.

The former `pretrain/mfm_native.py` is now `models/mfm/trainer.py`;
the MFM-only `pretrain/metric_backends.py` registry was removed in favor of
the method's checkpoint adapter. CLI/config callers use the new module paths.
Shared data, distillation and configuration helpers remain in their own layers.
This layout does not vendor the upstream trainer into the repository.

Use `scripts/run_mfm.sh` to select a stage and dataset:

```bash
bash scripts/run_mfm.sh pretrain arch
bash scripts/run_mfm.sh pretrain cite           # 100D OT-MFM, as in the source launcher
bash scripts/run_mfm.sh pretrain cite5 i-mfm
bash scripts/run_mfm.sh distill cite            # all available HCLT/Gaussian configs
bash scripts/run_mfm.sh distill eb hclt_t2       # one checkpoint/holdout variant
bash scripts/run_mfm.sh distill arch --dry-run  # print commands only
```

Pretraining defaults mirror the original `pretrain/metric-flow-matching/scripts`.
Distillation enumerates the same student/holdout configs as the source
`scripts/metric/run_dataset.py`, in sequence, and stops on the first failure.
It does not include that launcher's GPU scheduler, automatic resume/skip or
post-training export step. Evaluation configured inside the distillation job
still follows its YAML settings. Use `FLOW2PC_PYTHON` to select the environment
and `CUDA_VISIBLE_DEVICES` to restrict GPUs. `--validate-only` and `--set`
are passed to the shared CLI; `--dry-run` only prints commands and checks that
the selected config files exist.

Data, logging, output paths and flow/geopath checkpoints are read from the
existing configs. Distillation does not discover newly pretrained checkpoints;
set `teacher.checkpoint` and `teacher.params.geopath_checkpoint` to the intended
pair. Overrides apply to every selected config, so select a single variant when
overriding `output.dir` or a specific checkpoint pair to avoid mixing runs.

The MFM method is isolated from ALI-CFM's manifest-backed runner. Pretraining
calls the original code in `../pretrain/metric-flow-matching` (override with
`FLOW2PC_MFM_PRETRAIN_ROOT` or `model.params.source_root`). Distillation loads
frozen flow and geopath checkpoints with the adapter copied from
`flow2pc-multi-marginal`; its default source is sibling
`../metric-flow-matching` (override with `FLOW2PC_MFM_ROOT` or
`teacher.params.repo_path`). No source algorithm was replaced with TorchCFM.

Configs cover Arch, Sphere, CITE, Multi, EB, LiDAR and AFHQ. The 22 upstream
YAML files include five CFM reference configs; the 17 MFM wrappers use the
original MFM objective. Pretraining writes the original Lightning checkpoint
format under `output.dir/checkpoints/`. The distillation adapter checks the
flow/geopath pair, records their hashes, and freezes both networks.

MFM data loading retains the source's ordinal unit-interval model times. It
records original observation labels separately as physical times where numeric,
plus the split indices, normalization and source file hashes. AFHQ image classes
are labels, not physical times. The MFM student loss is velocity MSE plus
`source_nll_weight` times earliest-marginal NLL. ALI-CFM continues to use its
existing frozen-teacher velocity MSE objective.

Existing whitened MFM checkpoints were trained with the upstream scaler fitted
on all source frames, including temporal holdouts. The adapter records
`fit_scope: all_source_frames_as_in_upstream` and reuses those coordinates to
match the frozen teacher. This is a known exception to the proposed
training-only normalization rule; changing it requires retraining the teacher.

Only MFM is integrated in this phase. MMFM, `ConditionalGaussianPC`, MNIST
latent code and source benchmark launchers were not copied. The migrated MFM
code requires its original external source and data/checkpoint files; the
metadata-only CLI can be checked without them except for pretraining source
and data path existence. Runtime numerical parity and checkpoint reload still
need verification when test execution is authorized.
