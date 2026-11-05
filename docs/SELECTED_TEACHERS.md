# Selected ALI-CFM and OTP-FM teachers

Selection snapshot: 2026-11-05. Each preset copies the winning run’s seed,
data/preprocessing, model parameters, training budget and evaluation settings.
Only the output directory and W&B run name change for a new pretraining run.
Selected existing checkpoint paths, original resolved-config paths, hashes,
media manifests and per-dataset completed counts are in
[selected_teachers.json](selected_teachers.json). No experiments were run to
make this selection.

| Teacher | Dataset | Saved selection metric | EMD ↓ | Completed / planned | Pretrain preset |
| --- | --- | --- | ---: | ---: | --- |
| alicfm | beijing | Held-out-time W1 | 12.581962 | 512 / 512 | [YAML](../configs/pretrain/alicfm/alicfm_beijing.yaml) |
| alicfm | gom | Held-out-time W1 | 0.133903 | 512 / 512 | [YAML](../configs/pretrain/alicfm/alicfm_gom.yaml) |
| otpfm | cell_tracking | Observed-time W1 | 34.825586 | 28 / 100 | [YAML](../configs/pretrain/otpfm/otpfm_cell_tracking.yaml) |
| otpfm | cite100 | Held-out-time W1 | 48.923553 | 32 / 100 | [YAML](../configs/pretrain/otpfm/otpfm_cite100.yaml) |
| otpfm | cite5 | Held-out-time W1 | 0.867000 | 33 / 100 | [YAML](../configs/pretrain/otpfm/otpfm_cite5.yaml) |
| otpfm | cite50 | Held-out-time W1 | 43.121042 | 32 / 100 | [YAML](../configs/pretrain/otpfm/otpfm_cite50.yaml) |
| otpfm | eb | Held-out-time W1 | 0.424751 | 32 / 100 | [YAML](../configs/pretrain/otpfm/otpfm_eb.yaml) |
| otpfm | knot | Observed-time W1 | 0.015208 | 32 / 100 | [YAML](../configs/pretrain/otpfm/otpfm_knot.yaml) |
| otpfm | multi100 | Held-out-time W1 | 52.916254 | 31 / 100 | [YAML](../configs/pretrain/otpfm/otpfm_multi100.yaml) |
| otpfm | multi5 | Held-out-time W1 | 0.912809 | 32 / 100 | [YAML](../configs/pretrain/otpfm/otpfm_multi5.yaml) |
| otpfm | multi50 | Held-out-time W1 | 46.237329 | 32 / 100 | [YAML](../configs/pretrain/otpfm/otpfm_multi50.yaml) |
| otpfm | st | Held-out-time W1 | 117.998783 | 31 / 100 | [YAML](../configs/pretrain/otpfm/otpfm_st.yaml) |

## Selection scope

ALI-CFM GOM/Beijing were selected from all 512 completed configurations per
dataset, minimizing mean raw-coordinate W1 EMD across the four saved temporal
holdouts. Those holdouts are tuning/validation populations, not an untouched
test set. Beijing remains in its original representation; no log transform or
prediction clipping was introduced.

OTP-FM was selected from the 315 completed runs in its 1,000-run sweep.
The remaining 685 runs are marked failed in the saved summary; these presets
are best among the completed models, not evidence that all 100 settings per
dataset finished. Single-cell includes CITE and Multi at 5D, 50D and 100D,
plus EB. Their left-out-time W1 is the saved ranking rule, as it is for ST.
Knot and cell tracking use observed-time reconstruction W1 because their
protocols have no held-out times. Tracking evaluates the six README frames:
1, 25, 50, 75, 100 and 115. Metrics retain each sweep’s coordinate system,
population initialization, sample budget and solver; scores across datasets
or different evaluation protocols are not interchangeable.

The selected checkpoint files were verified against the SHA-256 recorded in
their manifests. Saved metrics and ranking files agree with the selected
rows. No checkpoint, data array, running sweep or existing output was changed.

## Using the selected teachers

Running a pretrain preset starts fresh training with the selected settings.
Its output is `/data/lthung/flow2pc/pretrain/METHOD/DATASET/best-config-seed42`.
All selected presets retain final evaluation/images and W&B project `flow2pc`.
An existing output directory is not automatically overwritten.

The two ALI-CFM student presets already load the existing selected teacher:

- [GOM HCLT](../configs/distill/alicfm/alicfm_gom_hclt.yaml)
- [Beijing HCLT](../configs/distill/alicfm/alicfm_beijing_hclt.yaml)

They optimize velocity MSE and select the student using the implemented
held-out-time velocity-MSE validation. Population metrics remain diagnostics.
Their output directories are under `/data/lthung/flow2pc/distill/alicfm/`.
If retraining a teacher, explicitly point `teacher.checkpoint` to the new
teacher rather than continuing to reference the archived sweep winner.

OTP-FM teacher pretraining and frozen velocity export are integrated, but the
organized student runner still lacks its native conditional-tuple sampler.
This config update does not implement that sampler or advertise an OTP-FM
student recipe as runnable. Selected OTP-FM teachers remain available through
the checkpoint paths in the index.

The Arch/Sphere/LiDAR sweep settings and its endpoint-only interpretation
remain as documented in [MFM_TEACHER_SWEEP.md](MFM_TEACHER_SWEEP.md).

All config moves preserve other presets’ scientific contents. Static syntax,
path and saved-setting comparisons are separate from runtime validation.
Regression cases are provided in `tests/test_config_layout.py`; no tests,
training, distillation or evaluation were run during this change.
