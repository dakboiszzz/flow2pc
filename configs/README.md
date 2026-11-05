# Configuration layout

Runnable configurations use the same filenames across methods:

```text
configs/
  pretrain/
    alicfm/alicfm_<dataset>.yaml
    otpfm/otpfm_<dataset>.yaml
    mfm/mfm_<dataset>.yaml
    mmfm/mmfm_<dataset>.yaml
  distill/
    alicfm/alicfm_<dataset>_hclt.yaml
    mfm/mfm_<dataset>_{hclt,gaussian}.yaml
    mmfm/mmfm_<dataset>_{hclt,gaussian}.yaml
  evaluate/
    mmfm/mmfm_<dataset>_{teacher,hclt,gaussian}.yaml
```

Pretraining creates a teacher, so its filename has no student suffix. Student
filenames explicitly distinguish HCLT and Gaussian PCs. Method variants use a
suffix on the dataset/recipe, such as `otpfm_gom_w2.yaml` or
`mfm_eb_t2_hclt.yaml`. Names describe existing supported configurations; this
layout does not imply every method supports every dataset or student type.

MFM additionally retains its original experiment wrappers under
`pretrain/mfm/native/` and their original-format checkpoint student recipes under
`distill/mfm/native/`. The primary MFM files use the shared prepared snapshots
and bound flow/geopath checkpoint pairs. The two paths have different data and
checkpoint contracts. Native parameter snapshots in `pretrain/mfm/upstream/`
retain their original filenames and contents because they are source reference
files, not Flow2PC stage configurations.

The organization is shared; method-specific YAML parameters, loss functions and
training loops remain method-specific. Existing launchers resolve these paths.

## Selected sweep teachers

The ALI-CFM GOM/Beijing presets and ten OTP-FM single-cell/knot/ST/tracking
presets contain the exact scientific settings from the best completed sweep
runs. [Selected teacher results](../docs/SELECTED_TEACHERS.md) lists their scores;
[the index](../docs/selected_teachers.json) records original checkpoints, hashes,
resolved configurations and ranking scope.

Running a pretrain preset trains a new teacher in its configured output directory.
It does not reuse the selected checkpoint automatically. The ALI-CFM GOM/Beijing
distill presets already point to the existing selected checkpoint:

```bash
PYTHONPATH=src python main.py --config configs/pretrain/alicfm/alicfm_gom.yaml
PYTHONPATH=src python main.py --config configs/distill/alicfm/alicfm_gom_hclt.yaml
```

These are commands for the user to execute. Data and outputs remain external;
override `data.root`, `output.dir` or `teacher.checkpoint` through `--set` as needed.
The selected teacher presets log to W&B project `flow2pc` and retain final
evaluation/figures. Existing trained checkpoints and sweep outputs are preserved.

OTP-FM teacher pretraining/evaluation is integrated. Its conditional-tuple student
sampler is not implemented in the current organized distillation runner, so no
OTP-FM distillation preset is advertised as runnable. Adding that sampler is a
separate implementation task; ALI's generator sampler is not a valid substitute.
