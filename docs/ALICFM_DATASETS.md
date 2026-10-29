# ALI-CFM teachers and velocity-MSE distillation

The organized package includes teacher profiles and HCLT distillation entry
points for the datasets below. This migration has received static source review;
training, checkpoint loading, evaluation and tests have **not** been executed.

| Config suffix | Data representation | Teacher implementation |
| --- | --- | --- |
| `cite5`, `cite50`, `cite100` | Saved CITE PCA, first 5/50/100 components | Corresponding supplied single-cell configs and native staged loops |
| `multi5`, `multi50`, `multi100` | Saved Multi PCA, first 5/50/100 components | Corresponding supplied single-cell configs and native staged loops |
| `eb` | EB PHATE, 2 dimensions, original day-range categories | Native single-cell model/loops; example hyperparameters adapted from CITE-5D, not an established EB reproduction |
| `st` | U2–U5 aligned tumor coordinates, stored `(y,x)` | ST method/config recovered from ALI-CFM `501e9db` |
| `cell_tracking` | Cell-4 mask pixels `(x,y)`, 115 frames | Supplied tracking architecture and training definitions; normalized-coordinate CFM example |
| `knot` | Original noisy analytic knot, 120 time points | Original two-hidden-layer knot models and GAN/CFM objectives |

The EB PCA experiment expects `eb_velocity_v5.npz`, which is absent from the
shared data root. The EB entry therefore explicitly uses the available upstream
`eb-2d` representation; it does not silently substitute PCA. Day-range labels
remain recorded as labels, with an ordinal model clock. ST slide indices also
describe an ordered series, not measured elapsed time.

## Commands prepared for user execution

Activate your existing CUDA environment or set `FLOW2PC_PYTHON` to its interpreter.
Run from this repository. For example:

```bash
bash scripts/run_alicfm.sh pretrain cite50
bash scripts/run_alicfm.sh distill cite50
```

The second command consumes the first command's checkpoint. It never starts
teacher training. Replace `cite50` with another config suffix from the table.
Defaults read `/data/lthung/ali-cfm-data` and write under
`/data/lthung/flow2pc/framework/{pretrain,distill}/alicfm/`. Logging uses W&B
project `flow2pc`, with the dataset in each run name. Choose separate GPUs using
`--set training.device=cuda:1` for teachers or
`--set distillation.device=cuda:1` for students. Independent commands can run on
different GPUs; these entry points do not queue a sweep automatically.

Overrides use `--set dotted.key=value`. When changing a teacher seed/fold or
student hyperparameters, also choose a distinct `output.dir`, logging name and
student `teacher.checkpoint`. Existing teacher output directories are rejected;
student resume requires the same teacher, data, model and training settings.

## Recovered ST teacher method

The matching network method and trainer were found together in ALI-CFM revision
`501e9db6a3bcb1b96eb893882b62a56f78fe1823`. Later commit `80c8673` replaced both
pieces during the single-cell refactor, leaving the ST config selecting
`piecewise` with an incompatible call.

The restored ST profile uses:

- `STInterpolant.compute_piecewise_reg_term(xts, metric="l2")` from `501e9db`.
  It samples a segment of the OT-coupled observed populations, samples time
  uniformly within that segment, and penalizes the mean squared difference
  between the learned interpolant and that segment's linear path.
- The matching pretraining/GAN loops from the same revision, with logging
  injection only. It does **not** select `picewise_oskar_version` from the newer
  trainer. The original ST derivative calculation is also retained.
- Original G/D width 64, CFM width 128; batches 256 for pretraining/GAN and 128
  for CFM; 1,000 / 70,000 / 10,000 updates; learning rates 0.001 for pretraining,
  0.00001 for G/D and 0.001 for CFM; correction coefficient 10, vanilla GAN,
  full OT, and time smoothing 0.001 during interpolant training.

These objectives belong to **teacher pretraining**. The student optimizes only:

```python
mean((student.velocity(x_t, t) - frozen_teacher.velocity(x_t, t)) ** 2)
```

The earlier Flow2PC launchers selected U3/seed45 and U4/seed42 teacher bundles.
Those bundles were not found at their recorded `outputs/ST/...` paths on this
host. The historical source and results CSV establish the earlier ST code, but
do not establish the exact revision that produced an absent checkpoint. No
claim of matching those weights or their EMD is made.

New teacher runs use the framework's deterministic training/validation/test
splits and fit scaling on observed training rows only. The old ST script fitted
normalization before excluding a slide and used all observed rows. Consequently
this recovers its training method and settings, **not** a bitwise replay of that
data protocol. New artifacts save the fitted transform, split and source hashes.

## Distillation behavior retained

The HCLT model is the retained real-data implementation, with dense Gaussian
leaves and free-source Fourier time conditioning (`fourier_zero_center=False`).
Earliest-marginal training samples initialize leaf means/scales; initialization
does not fit a source likelihood. All student parameters remain trainable.
Optimization uses AdamW, weight decay 0.01, cosine scheduling and gradient
clipping at 1.0.

Query distributions follow the prior dataset-specific code: source/middle/final
OT matching for single-cell, endpoint OT for tracking, and independent endpoint
draws for ST/knot, followed by the frozen learned teacher interpolant. These are
training query distributions, not claims of observed cell correspondences.
The target is always the frozen teacher's **velocity field**. There is no
direct-PC-sampling, NLL, endpoint, or interpolant-derivative student objective.
Tracking additionally retains late/final teacher-trajectory velocity validation
for checkpoint selection; it contributes no extra gradient loss.

New outputs include resolved configs, source/data/time/preprocessing bindings,
metrics, and best/last student checkpoints with optimizer, scheduler and RNG
state. The examples prepare training only; old overlay/evaluation/sweep tools
have not been restored as part of this migration.

## Existing checkpoints

Framework artifacts contain the manifest and preprocessing next to the weights.
Historical checkpoints use an explicit metadata file to avoid guessing their
time inputs, scaling or coordinate conventions. For an old ST teacher, the
metadata has this shape:

```yaml
data:
  name: st
  root: /data/lthung/ali-cfm-data
  split_seed: 2026
  params:
    heldout_index: 1
    normalise: true
    whiten: false
time_input: global
velocity_time_units: model
cfm_coordinates: model
field_activation: selu
generator_activation: elu
distillation_sampler: independent
# Supply this teacher's ACTUAL saved transform. Never refit a new one.
preprocessing_file: /absolute/path/to/teacher/preprocessing.json
```

The preprocessing JSON must contain its affine `shift` and positive `scale`
vectors, with raw coordinates mapped as `(raw - shift) / scale`; it may instead
be embedded under `preprocessing:` in the YAML. For old combined single-cell
or tracking checkpoints, the importer can recover the saved transform from
`data_contract.preprocessing` or `min_max`. Missing scaling fails explicitly.
It does not fit a replacement. Preserve the actual original axis order.

Then invoke the student with the actual checkpoint and metadata paths:

```bash
bash scripts/run_alicfm.sh distill st \
  --set teacher.format=legacy \
  --set teacher.checkpoint=/absolute/path/to/teacher.pth \
  --set teacher.legacy_metadata=/absolute/path/to/metadata.yaml \
  --set teacher.generator_checkpoint=/absolute/path/to/generator.pth \
  --set output.dir=/data/lthung/flow2pc/framework/distill/alicfm/st/existing-u3
```

Omit `generator_checkpoint` if the combined checkpoint includes `interpolant`.
Legacy loading is for trusted local checkpoints; it permits their original
NumPy-containing format. Legacy time layout and velocity units are explicit:
`global`, `local`, `local_interval`, or `local_interval_global` inputs, and model
time velocity. The importer does not infer a velocity rescaling from input width.

Source revisions, extracted definitions and file hashes are recorded in
`src/flow2pc/pretrain/models/alicfm/SOURCE_PROVENANCE.json`. Student recovery
details are recorded in `docs/DISTILLATION_SOURCE.json`.
