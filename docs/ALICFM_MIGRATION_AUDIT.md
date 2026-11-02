# ALI-CFM migration audit — 2026-10-30

The teacher migration is **not yet established as a faithful reproduction** of
the supplied ALI-CFM training pipelines. The migrated core losses and networks
are substantially preserved, but the orchestration changes data exposure,
preprocessing, random sampling, some experiment recipes and evaluation. A run
finishing, or a student having a similar EMD, does not validate that migration.

Restoration update: the knot training rewrite described in finding 5 was replaced
on 2026-10-30 by the original NumPy full-curve sampler and native update loops.
See [ALICFM_DATASETS.md](ALICFM_DATASETS.md#restored-native-knot-training-path-2026-10-30).
The findings below describe the code that produced the already-saved runs.
No retraining, numerical parity test or solver comparison has been executed;
restoration does not retroactively change those results or establish their cause.

Further restoration: the user requested native CITE/Multi/ST/tracking behavior
and selected the repository's raw-pixel tracking config as the baseline.
[ALICFM_NATIVE_TEACHERS.md](ALICFM_NATIVE_TEACHERS.md) records the explicit new
protocol, corrected preprocessing/population exposure/callbacks, source review,
comparison limits and teacher-only eight-GPU launcher. The older data and recipe
findings below remain the diagnosis of the original migration-check runs.

This audit compares the live working tree at HEAD
`bcca40373d40a0615c02c013c68555aa7a086153` with:

- `../ali-cfm`, revision `a35963ef226e1bbdc6344a944c50e0f42ed67335`;
- its matching historical ST implementation at
  `501e9db6a3bcb1b96eb893882b62a56f78fe1823`;
- the selected teacher documentation under `../ali-cfm/best_models` and
  `../ali-cfm/docs/CELL_TRACKING_TEACHER_README.md`;
- the successful real-data Flow2PC sources at `f2c6942`, also preserved under
  `/data/lthung/cache/flow2pc-cleanup-20261028T172848Z`.

Only source, YAML, H5AD label metadata and existing run artifacts were inspected.
No training, distillation, evaluation, tests or new figure generation was run
for this audit. Existing jobs were launched earlier by the user.

## Execution hold

Following the user's instruction to verify teachers before further distillation,
only automatic queue controller PID **3095660** was suspended with `SIGSTOP`.
Process inspection confirmed its state was `T`. Its independently launched
cell-tracking teacher PID **3250754** remained running (`Rsl`). No teacher was
terminated, and no checkpoint or training configuration was changed.

The suspended controller cannot enqueue/start the next student. Its JSON status
will remain stale while suspended, and it still holds its launcher locks.
Resuming that same controller would restore its existing teacher-to-student
sequence; it must not be resumed as a teacher-only launcher.

## Confirmed teacher differences

### 1. Observed training populations were reduced

Native `ali_cfm/data_utils.py:get_dataset` loads every cell/spot of each marginal.
The native experiment excludes a held-out *time* when constructing training
batches; it does not additionally remove validation/test rows from observed
times.

The migrated `pretrain/models/alicfm/trainer.py:282` requests only the `train`
split. CITE/Multi/ST/EB example configs remove 5% validation and 5% test rows
from observed populations. Tracking removes 10% and 5% before choosing its ten
teacher training points per frame.

Consequences extend beyond smaller minibatch pools: LAND's reference population
in `_source/training_funcs.py:58` is built from these reduced arrays, so the
regularization geometry changes as well. Tracking's seeded ten-point selection
also selects from a different population and ordering.

Affected: all real-data teacher examples. Knot teacher training draws fresh
analytic samples and does not use the stored training pools in this way.

### 2. Normalization scope and arithmetic changed

Native preprocessing computes whitening/min-max on all loaded populations before
holding out an observation time. This includes the held-out time. Saved teacher
READMEs explicitly document that original convention.

The new loaders fit only observed training rows:
`data/real/single_cell.py:122` and `data/temporal.py:94`. Thus the new experiments
use different coordinate transforms. At 5D, the historical EMD also retains
whitening, so changing whitening statistics changes the metric's coordinate
units, not just network inputs.

There is an additional precision difference: the new loader combines whitening
and min-max into one affine transformation, with double-precision fitting and
float32 stored constants. Native loading performs its scaler transform and
Torch min-max separately. The selected CITE-5 README specifically documents the
effective float32 scaler arithmetic needed to replay its original inputs.
`pretrain/models/alicfm/legacy.py:68` also folds an imported teacher's original
transforms into one affine map; it does not preserve that sequence bit-for-bit.

Train-only fitting is a possible new experimental protocol. It was not an
appropriate silent substitution while verifying an unchanged migration. Native
all-time preprocessing is transductive; that exposure must be recorded explicitly
if reproducing the native experiment. This audit does not authorize changing the
protocol in either direction.

### 3. Disabling training plots changes the seeded training sequence

Native single-cell GAN training invokes `sc_plot_fn` at epoch 0 and then every
5,000 updates. That function calls `sample_gan_batch`, consuming NumPy/Python
random draws and OT sampling. Native ST's `Plotter.plot_fn` similarly performs
`np.random.choice` and `sample_plan`.

The new orchestrator forces `plot_freaquency=0`
(`pretrain/models/alicfm/trainer.py:488`). Even with identical data, seed,
initialization and losses, subsequent minibatches no longer follow the native
random-number sequence. This is a reproducibility difference, not evidence that
the mathematical GAN loss changed or that plots improve training quality.

Tracking's supplied trainer already disables these plots; this particular
difference should not be attributed to tracking. Root knot training has a
separate sampling rewrite described below.

### 4. Cell-tracking mixes two different teacher recipes

The new config follows much of the *reconstructed, unverified* upstream
`configs/cell_tracking/ali_cell_tracking.yaml`, but switches CFM to normalized
coordinates. It does not reproduce the selected, previously distilled root
`changed_files/ali_cfm_model.pth` documented in the teacher README.

| Setting | Selected root checkpoint | New tracking example |
| --- | --- | --- |
| Time smoothing | 0.01 | 0.001 |
| GAN updates | 10,001 | 70,000 |
| CFM updates | 10,001 | 50,000 |
| Linear regularizer strength | 10 with coordinate-mean MSE | 1 with coordinate-summed MSE |
| CFM coordinates | Normalized | Normalized |
| Teacher point selection | Ten points from each full frame | Ten points from each reduced training pool |

The README explains the loss reduction conversion: selected-checkpoint
coefficient 10 corresponds to **5**, not 1, in the coordinate-summed penalty.
The current regularizer is therefore five times weaker than that particular
selected recipe. This is a recipe mismatch, not a universal correction to every
tracking checkpoint. The archived `latest-run` checkpoint has yet another
configuration and must not be treated as the selected root checkpoint.

Relative to the reconstructed upstream default, selecting normalized instead of
raw pixel CFM changes the training coordinate system and objective weighting.
Both variants are supported by the source; they are different experiments.

### 5. Knot training was rewritten rather than replayed

Native `../ali-cfm/train_teacher.py:train` generates complete noisy 120-time
curves with NumPy for every batch, selects the GAN's interior time using NumPy,
then converts samples to float32. The migrated
`pretrain/models/alicfm/profiles.py:73` selects times using Torch and generates
only the required marginals with device-side Torch noise, adding noise directly
to float32 means. It also explicitly seeds a run, whereas the supplied root
trainer has no seed call inside `train()`.

The analytic means, noise standard deviation 0.01, independent-noise marginal
law, model widths, Adam learning rates, 50,000/30,000 update budgets and written
GAN/CFM loss formulas match the root recipe. Drawing unused intermediate noise
is not mathematically required for those marginal laws. Nevertheless the random
stream and numerical operations differ, so this is not an exact replay. The
observed failure cannot be assigned to this rewrite without a controlled native
comparison.

### 6. EB is an adapted experiment, not a verified supplied recipe

The new EB config explicitly uses the available 2D PHATE H5AD and CITE-5-derived
hyperparameters because no released EB-specific training config was supplied.
The upstream PCA/NPZ `eb` experiment is a different representation from its
`eb-2d` branch. The latter also standardizes the full PHATE population before
normalization, rather than using the new train-only preprocessing.

The actual H5AD categories inspected are already chronological:
`Day 00-03`, `Day 06-09`, `Day 12-15`, `Day 18-21`, `Day 24-27`.
Sorting them is not an observed time-order bug for this file. However, an EB run
completing does not establish native EB reproduction.

### 7. Fold seeding and stage reuse no longer follow the original driver

Native CITE/Multi/ST drivers seed once per seed and then train both folds in
sequence. A new one-fold command reseeds each fold independently. Fold 2's
initialization/minibatches therefore differ from native fold 2, which inherits
the random state after fold 1. This does not explain a fold-1-only discrepancy.
The common seed helper also omits the native cuDNN deterministic settings;
these MLPs do not by themselves establish a material cuDNN-related difference.

The new teacher runner also always executes its configured stages and saves the
combined teacher only after all stages. It does not retain the original
single-cell stage-load flags or tracking's ALI checkpoint plus complete random
state handoff for running CFM later. This is a workflow/resume regression,
distinct from changing the fresh-run loss formula.

### 8. Teacher evaluation no longer measures exactly the original experiment

Native single-cell/ST paper-style CFM evaluation pushes the full preceding
observed marginal to the held-out marginal. New shared evaluation always starts
at observation index 0, evaluates the complete time grid, and usually caps metric
populations at 2,048 or 512 particles. For fold 1 the starting population happens
to coincide with the native source; for fold 2 it does not.

Native ST uses a TorchDyn Dopri5 rollout; the new ST teacher config selects
classical RK4. New common tolerance/batch settings also differ from native
evaluation and from prior Flow2PC references. Even when the mathematical field
acts on cells independently, an adaptive solver's error norm and accepted steps
can depend on batch composition.

There is a further CITE/Multi subset mismatch:
`evaluation/runner.py:21` builds `all` by concatenating train/val/test, whereas
the old loader preserves source H5AD order. `H5Marginal.materialize` also sorts
each split. Reusing seed 2026 therefore does not recover the same selected cells.
This explains why an initial-time subset EMD can differ before integrating any
velocity; it does not by itself explain the full downstream score difference.

The EMD's intended mathematical cost has **not** changed from Euclidean W1 to
squared-distance W2. Native code takes `sqrt` of POT's default squared distance;
new code constructs Euclidean distances directly, then calls `emd2` with uniform
masses. The `2` in that API name means transport cost, not automatically W2.
Native float32 distance calculations versus new double-precision SciPy distances
and inverse-transform arithmetic can introduce numerical differences. Population
selection, preprocessing units, integration and start times are the larger
confirmed protocol differences; their individual causal effects remain unmeasured.

## What source inspection did verify

- All **nine** migrated source files match their recorded SHA-256 values.
  The core native `nets.py`, `land_metric.py` and `training_utils.py` are copied
  intact. These hashes do not cover every orchestration decision above.
- For all **six CITE/Multi examples**, model widths, smoothing, OT modes, loss
  settings/coefficient, LAND settings, batch size, learning rates and update
  budgets match the selected teachers' `run_config.yaml` files. Whitening flags
  also match. Their changed training rows/scaler are separate issues.
- ST's recovered piecewise regularizer and derivative match revision `501e9db`
  by AST comparison. Its pretrain/GAN function bodies match after removing the
  logger injection. ST's original widths, stage budgets, GAN/pretrain batch 256
  and CFM batch 128 are retained. No alternate piecewise loss was substituted.
- Current model factory initialization order matches the supplied drivers:
  G/D/V for real-data profiles and V/G/D for root knot. This establishes current
  source behavior, not the precise dirty working-tree contents of every older
  already-started process.
- Current frozen velocity adapters preserve declared global model time. Raw
  tracking velocities are transformed by the coordinate-scale chain rule.
  No silent interval remapping was found in the new-format global-time path.

The staged-update parity test in `tests/test_framework_migration.py:315` gives
both implementations the **same new loader's arrays** and disables plots on both
sides. It can check those function updates while missing native data/scaler and
callback differences. No test was run during this audit. Its existence or past
results do not validate the complete native training pipeline.

The artifact records a Git HEAD and hashes migrated source cores, but not a
complete hash of dirty data/trainer/profile/evaluation code. Therefore current
artifacts do not fully pin all migration behavior by that HEAD alone.

## Existing artifacts: what can and cannot be concluded

At approximately 01:17 UTC, before suspending the launcher, eight dataset
pipelines were marked complete. CITE50's teacher was complete but its pipeline
had an earlier preparation failure and no student. Tracking's teacher was still
running. These are execution states, not scientific acceptance states.

The new CITE50 teacher's held-out EMD is **42.5226756550**. The selected original
teacher's prior Flow2PC 2,048-particle reference is **41.8294353963**; its native
full-population number is **41.0524368286**. These are different checkpoints
and/or protocols. Neither closeness nor the gap isolates a migration effect.

The existing knot artifact is a clear reason to withhold acceptance:

| Saved knot diagnostic | Value |
| --- | ---: |
| Teacher mean observation EMD | 19.62876 |
| Teacher final observation EMD | 99.79487 |
| Student final observation EMD | 0.34280 |
| Student–teacher final population EMD | 99.53078 |
| Student–teacher final paired position RMSE | 109.57476 |

These values come from the existing JSON, not a new evaluation. They show large
rollout disagreement and a poor teacher under this saved protocol. A lower
student-to-data EMD cannot validate faithful teacher migration or faithful
distillation here. Whether native training also fails, the migrated training
changed its result, or the evaluation setup is responsible remains unresolved.

Artifact locations:

- `/data/lthung/flow2pc/framework/migration-check/cite50/teacher/eval/metrics.json`
- `/data/lthung/flow2pc/outputs/single_cell/best_ali_cfm_sweeps_t1/reference/cite50d-t1-teacher42/metrics.json`
- `/data/lthung/flow2pc/framework/migration-check/knot/student/eval/metrics.json`

## Teacher-only acceptance sequence — proposed, not executed

1. Preserve the selected native recipe and data/preprocessing protocol for each
   dataset. Explicitly distinguish the selected tracking checkpoint from the
   reconstructed tracking baseline, and EB adaptation from an original recipe.
   A faithful migration must not silently introduce a new benchmark protocol.
2. Compare native and migrated preprocessed arrays, observation order, held-out
   indices, source rows, coordinate units, seeded initial weights and random
   states before training. Include callback sampling in the comparison.
3. With matched inputs/random states, compare sampled batches and OT endpoints,
   interpolant outputs/derivatives, individual loss terms, gradients and optimizer
   updates at every teacher stage. Cover dataset-specific ST/tracking/knot paths.
4. Evaluate the **same existing teacher weights** through native and migrated
   adapters using identical cells, times, preprocessing/inverse transforms,
   solver settings and EMD implementation. This separates inference/evaluation
   migration from training stochasticity without retraining a teacher.
5. Only after these gates pass, compare full teacher training under the intended
   native recipe. Finite end-to-end metrics alone are insufficient evidence.
6. Resume consideration of student distillation only after teacher migration
   and teacher evaluation have been accepted. This report launches nothing.

## Parked student audit findings

These findings are retained for later work; they are not a substitute for the
teacher acceptance sequence above, and no student changes were made here.

- The HCLT definitions in `src/flow2pc/pc/hclt_pc.py` match `f2c6942` by AST
  comparison. The model mathematics were not rewritten. The student loss remains
  frozen-teacher velocity MSE, with AdamW/cosine scheduling and clipping.
- Student splits now inherit the teacher's reduced pools. Old single-cell used
  a 5% validation-only split with Torch seed 2026; old tracking split only endpoint
  rows. New split RNG, query RNG offsets and sampling draw order differ.
- Old knot distillation builds 500 endpoint samples at noise std 0.1 and uses
  60 marginals. The new config inherits a 120-time teacher dataset with only
  10 samples/time at std 0.01 (eight training samples per endpoint).
- Old ST/knot launchers default to interval-conditioned time sampling. New queries
  are globally uniform. For old ST holding out U3, original marginal indices
  `[0,2,3]` produce no middle-interval queries and double weight on the last
  interval. Global uniform sampling changes that behavior; any correction should
  be explicit rather than called an unchanged migration.
- ST/knot gained source-based initialization and validation-selected `best.pt`,
  whereas the old launcher initialized its circuit normally and evaluated
  `last.pt`. ST endpoint probabilities also change from cyclically repeated
  rectangular pools to uniform actual rows.
- Single-cell validation changed from eight batches at fixed seed 2026 to four
  at student-seed + 100000. Tracking retains the worst global/late/final MSE
  selection formula, but changes the actual validation queries/paths and includes
  frame 91 in `time >= 0.8`, versus the old cached late range 92–114.
- New example capacities, seeds and update budgets are not replays of every
  previously selected sweep winner. No successful-sweep reproduction claim is
  justified by using these examples unchanged.
