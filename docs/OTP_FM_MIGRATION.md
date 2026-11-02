# OTP-FM migration: original training, native evaluation, common interfaces

The later [ALI-CFM-data sweep](OTP_FM_ALI_DATA_SWEEP.md) adds shared marginal
profiles for CITE/Multi, EB PHATE-2D, ST, tracking and knot. That extension uses
the existing W1/velocity-ODE experiment metric and is distinct from the native
GOM/Beijing parity phase described here. The original 17 pinned source files
remain unchanged; one original tuple-dataset class is additionally retained.

Implemented first phase: GOM and Beijing teachers behind the common
`python main.py --config ...` CLI, with the original OTP-FM model, losses,
potential solvers, EMA, training/validation loops, data couplings, evaluation
callbacks and plots. Numerical reproduction has **not** been demonstrated yet:
the commands below are prepared for the user to run. No training, inference,
evaluation, unit tests or dependency installation was performed by the assistant.

## Structure and extension points

```text
src/flow2pc/pretrain/models/otpfm/
  trainer.py              common method orchestration
  profiles.py             dataset-profile registry and native setup adapter
  adapter.py              frozen EMA velocity / interval-sampling contract
  evaluation.py           native metrics and final visualizations
  tracking.py             adapter for the original reproduction callback
  provenance.py           metadata-only source verification
  _source/                original core and experiment implementations
  SOURCE_PROVENANCE.json  reviewed revision, per-file hashes and import changes
  LICENSE, NOTICE.md      original Apache-2.0 license and attribution
```

Dataset loading stays in `flow2pc.data`, without teacher imports. OTP coupling
and tuple DataLoader classes belong to the method. The first registered dataset
profiles are `gom` and `beijing`; the core trainer/adapter/launcher are shared.
Further datasets need an audited shared loader and a native dataset profile,
including any dataset-specific training/evaluation hooks. No EB or CITE-seq
profile is claimed implemented by this phase.

Source is the supplied `../otp-fm` working checkout, revision
`ca4fe796930414ae749eba2b299e7c7ec8cebf3d`, with actual per-file hashes recorded.
Seventeen source files are retained. Full core files are unchanged; experiment
imports are rewired to the method namespace, and selected factory/callback/data
definitions are copied unchanged. Static comparison verified all **67 selected
function/class ASTs** against the source. Hash checks run before training and
before checkpoint export. The package does not depend on a sibling repository
or an installed `otpfm` package for migrated training/inference.

The native control uses the original `experiments.reproduce.run_one`. Both paths
redirect only the native shared master-results CSV to a per-run file, avoiding
parallel writers. This changes output plumbing, not training/sampling arithmetic.
No native source files are modified.

The Flow2PC wrapper also guards W&B I/O. W&B 0.30.0 can wrap a service timeout
inside its optional artifact preflight as `AuthenticationError`, escaping the
SDK's own timeout handler. Both OTP-FM paths treat that communication error as a
preflight warning and continue the original artifact enqueue/upload operation;
artifact type/value errors still propagate. The guard is applied to the in-memory
run instance, without editing the installed SDK. A failed preflight is not retried
for every later artifact. The SDK also calls its API after enqueue to attach a
save handle. Communication errors there are guarded too, and later artifacts
are marked deferred instead of repeating failed calls. Scalar/media logging
continues; native source and output files remain locally available. Startup and
API preflight use the common short-lived
lock; training remains parallel. Native final-sync communication errors are
recorded in `tracking_status.json` rather than discarding local completion.
Online initialization communication errors fall back to an offline SDK run;
if even offline startup cannot establish the SDK, native callbacks use a local
sink and migrated logging retains its local files. Native strict-online checking
is updated to the effective fallback mode, recorded separately from the requested
mode. Later scalar/summary communication failures do not abort native training.
OTP-FM online initialization has a 30-second timeout. Configuration errors and
training exceptions still fail the job.
Offline and disabled SDK runs are left untouched by the artifact guard. Online
guard state is attached to the wrapper function, because SDK Run objects reject
new attributes.

## What stays exact

- Full native FlowNet architecture, embeddings, residuals, normalization layers,
  EMA network, potential classes/lambda functions, fixed-point/direct solvers,
  finite-difference/JVP/clamping behavior and consistency objectives.
- Native initialization order: seed Python/NumPy/Torch after W&B startup;
  build coupling/tuple dataset/DataLoaders, potentials, model, then optimizer.
- Original optimizer, curriculum conversion, forward/loss/backward/EMA order,
  validation draws, reshuffling, epoch-0 evaluation, and periodic plotting hooks.
- The original gradient-clipping placement is retained; it is not silently
  moved. Shipped presets disable gradient clipping.
- `W2` means `potential="w2inf"` plus precomputed OT coupling. `W2Inf` disables
  coupling. The potential label is not substituted based on the recipe name.
- GOM uses the supplied MeanFlow preset; Beijing uses its IMF preset. These are
  original teacher-pretraining objectives. The PC student's velocity MSE is
  untouched, and no GAN/ALI interpolant or direct PC sampling is introduced.
- Source `solver` versus `solver_type` behavior and rounded Beijing potential
  times are preserved. Fields ignored by the original factory remain ignored;
  the complete original config is saved rather than reinterpreted.

Model snapshots for export clone selected state tensors on CPU without random
draws. Original native `models/best.pt` and `models/model.pt` are also retained
for direct comparison. The original reproduction checkpoint selection callback
chooses `best.pt`, including an epoch-0 model if that is its best score.

## Data and normalization

Default data root is `/data/lthung/otp-fm`, configurable through `data.root` or
`OTP_FM_DATA_ROOT`. The shared loader consumes safe `marginals.npz` bundles:
GOM has 9 populations of 111 points in 2D, with observed indices 0,2,4,6,8;
Beijing has 13 one-dimensional PM2.5 populations, with observed indices
0,1,3,4,6,7,9,10,12. Their other four interior times are held out of optimization.

StandardScaler fits all populations, including temporal holdouts, as the native
loader does. This is explicit transductive preprocessing. No min-max/log transform,
clipping, positivity constraint or invented cell/particle trajectory is added.
Original row order and float32 scaling operations are retained. Native OT
row-argmax chains may duplicate row indices; they are method-induced couplings,
not ground-truth tracks. Their indices are saved when coupling is enabled.

Native controls require the original prepared GOM NPY or trusted Beijing NPY
cache in those same folders. The launcher rejects missing caches rather than
requesting dataset downloads. Migrated training does not load pickle arrays.

## Native sampling and evaluation are not diagonal ODE evaluation

The frozen instantaneous teacher is `model.v_func_ema(x,t,t)` (`dt=0`), with
normalized states and global model time. `mean_velocity(x,t1,t2)` and
`sample(x0,n_steps)` expose the original EMA finite-interval model separately.
The initial frozen adapter requires EMA and v-prediction; unsupported conditioning
or checkpoint formats fail explicitly. Native core x-prediction code is retained
but is not claimed supported by this instantaneous adapter.

All shipped configs choose `evaluation.protocol: native_otpfm`. Native results
use the finite-interval consistency sampler and the original W2 helper (square
root of OT squared-Euclidean cost), inverse scaling, unequal-population truncation,
and native time selection. GOM's nearby-time holdout minimum is retained and
labeled. These W2 numbers are not the shared evaluator's W1 EMD or a diagonal
ODE metric. They must not be compared as if those protocols were the same.

The final `best_marginals.png` / `final_marginals.png` show raw-coordinate target
and predicted populations at the nearest saved time, and label held-out times.
The figure time rule is stated separately from GOM's nearby-time score rule.
Native original PDFs and selected/final trajectory PNGs are retained too.

## Run the first parallel parity batch

From Flow2PC, preview the migrated jobs:

```bash
bash scripts/run_otpfm_checks_8gpu.sh --dry-run
```

Default is **eight jobs**: GOM/Beijing × W2/W2Inf × seeds 42/43, on GPUs 0–7,
one independent worker per GPU, at least 4 GiB currently free per launch and four
CPU threads per worker. Full native epoch budgets are retained: GOM 2,000,
Beijing 600. No distillation is queued. Output preparation/preview imports no
model or Torch and starts no W&B run or CUDA context.

Run the native controls and migrated teachers as separate batches on the same
selected GPUs/software, letting the first finish before launching the second:

```bash
bash scripts/run_otpfm_checks_8gpu.sh --backend native --dry-run
bash scripts/run_otpfm_checks_8gpu.sh --backend native

bash scripts/run_otpfm_checks_8gpu.sh --backend migrated
```

Every run logs to W&B project `flow2pc`, with dataset/recipe/width/lr/seed in its
name. Controls use the native reproduction logger. Migrated jobs reuse the shared
logger and record pending uploads separately from local numerical completion.
W&B errors do not justify replacing trained weights or altering the algorithm.

To rerun the eight native jobs that failed before training at source-artifact
upload, preserve their logs and use a fresh common comparison root:

```bash
export OTP_FM_CHECK_ROOT=/data/lthung/flow2pc/otp-fm/migration-check-wandbfix
bash scripts/run_otpfm_checks_8gpu.sh --backend native --dry-run
bash scripts/run_otpfm_checks_8gpu.sh --backend native
bash scripts/run_otpfm_checks_8gpu.sh --backend migrated
/data/nvidia/conda_envs/uffm/bin/python scripts/compare_otpfm_runs.py \
  --output-root "$OTP_FM_CHECK_ROOT"
```

The default mode remains online. A native `online_with_warnings` tracking status
means a preflight communication check failed but SDK logging continued;
`pending_sync` means artifact upload or final sync could not be confirmed. Neither is evidence of
successful training by itself: check `result.json`, histories and saved weights.

During a known W&B outage, skip online attempts entirely and use a fresh root:

```bash
export OTP_FM_CHECK_ROOT=/data/lthung/flow2pc/otp-fm/migration-check-offline-v2
bash scripts/run_otpfm_checks_8gpu.sh --backend native --mode offline
bash scripts/run_otpfm_checks_8gpu.sh --backend migrated --mode offline
/data/nvidia/conda_envs/uffm/bin/python scripts/compare_otpfm_runs.py \
  --output-root "$OTP_FM_CHECK_ROOT"
```

Both batches still run eight workers across eight GPUs. Offline SDK runs retain
their W&B history/media/artifacts under each run's `wandb/` directory for later
explicit `wandb sync <offline-run-directory>` when service is restored. Local
checkpoints, metrics and PNG/PDF figures are produced regardless of remote
availability. A `local_only` status means the SDK itself could not start; use the
ordinary local outputs, since that status does not promise a syncable SDK run.

Eight explicit YAML presets also include the native MMD and KL configurations.
For those optional recipes, use `--methods MMD KL --seeds 42`. They retain the
source nonlinear solvers and have not been numerically validated by this port.
Use `--datasets gom` or `--datasets beijing` to narrow a batch. An optional
`--epochs` is a debug budget override: it changes the curriculum and is not a
full reproduction. Use a fresh output root for changed scientific settings.

Environment overrides: `FLOW2PC_PYTHON`, `OTP_FM_DATA_ROOT`, `OTP_FM_CHECK_ROOT`,
`GPUS`, `MAX_PARALLEL`, `CPU_THREADS_PER_RUN`, `MIN_FREE_GB`. Additional options
include `--native-root`, `--mode online|offline|disabled` and `--output-root`.
Fresh output directories are required for incomplete runs. Compatible completed
runs are skipped; existing outputs are never deleted or silently restarted.
Ctrl-C stops only this controller's worker groups.

For a single migrated teacher, the common CLI remains:

```bash
PYTHONPATH=src /data/nvidia/conda_envs/uffm/bin/python main.py \
  --config configs/pretrain/otpfm_gom_w2.yaml
```

That example uses its config's separate base output directory, rather than the
parallel batch directory. Paths are explicit in the resolved YAML.

## Outputs, restore and comparison

Batch root defaults to `/data/lthung/flow2pc/otp-fm/migration-check`:

```text
native/<dataset>/<recipe>/seed<seed>/       original reproduction control
migrated/<dataset>/<recipe>/seed<seed>/
  history.jsonl                           original epoch callback values
  metrics.jsonl                           local W&B scalar sink
  native_config.json                      complete effective native config
  models/best.pt                          original selected native checkpoint
  models/model.pt                         original final native checkpoint
  checkpoints/best.pt                     portable selected framework teacher
  manifest.json, preprocessing.json       source/data/time/weights binding
  sampling_contract.json                 tuple/potential/observation contract
  ot_chain_indices.npy                    when native OT coupling is enabled
  trajectories/*.npz                      original periodic native samples
  eval/metrics.json                       selected native W2, labeled protocol
  eval/media.json                         final comparison-image paths
  eval/best_marginals.png
  eval/final_marginals.png
  plots/{best,final}_trajectories.png
  tracking_status.json                    online/warnings/offline/pending upload state
<backend>/_launcher/{status.json,logs/,configs/}
```

The framework checkpoint contains **selected best** live+EMA weights, not
arbitrarily the last epoch. `load_teacher(path, device)` discovers OTP-FM through
the common registry. Source, preprocessing and checkpoint hashes are verified,
weights load strictly, and the model is frozen. Native final weights remain in
`models/model.pt` for parity comparisons.

After both batches finish, inspect stored results without a new model evaluation:

```bash
/data/nvidia/conda_envs/uffm/bin/python scripts/compare_otpfm_runs.py
```

The report compares all saved epoch values except wall-clock elapsed time,
selected/final native live+EMA tensors, and the exported framework best tensors.
It records exact equality, missing epochs/keys, and byte hashes under
`parity_report.json`. It loads trusted locally produced native checkpoints; do
not use that pickle path for arbitrary downloaded checkpoints. A native W&B
finalization failure does not preclude comparing saved histories and weights.

Explicit evaluation is available when the user requests it, using the native
selected model and original metric function, with outputs kept separate:

```bash
PYTHONPATH=src /data/nvidia/conda_envs/uffm/bin/python main.py \
  --config configs/pretrain/otpfm_gom_w2.yaml --evaluate teacher \
  --set output.dir=/data/lthung/flow2pc/otp-fm/migration-check/migrated/gom/W2/seed42
```

That writes `native_checkpoint_eval/` without overwriting the native training
trajectories/metrics. It is a finite-interval evaluation, not the diagonal ODE.

## Distillation boundary and verification limits

This phase prepares teacher training, native evaluation and frozen velocity
interfaces. The analytic, all-observed-tuple conditional interpolant from
`../otp-fm/README_OTP_FM_TO_FLOW2PC.md` is the next PC integration step. It is
**not** replaced by an ALI endpoint generator, an independent endpoint line, a
Gaussian source, direct PC samples, or conditional velocity as the student target.
The manifest identifies an unsupported `otpfm_conditional` sampler, so the current
distiller fails explicitly instead of silently using an ALI sampling objective.

Static validation performed: all 17 source hashes and 67 selected definition
ASTs match the native implementation after import wiring; both metadata-only
launch previews found eight jobs and eight GPUs without creating output roots.
Python/Bash syntax and whitespace checks were run. Regression definitions cover
source/config contracts, diagonal EMA queries, finite-interval API distinction,
freezing/context rejection, selected checkpoint identity, shared logger alias,
and absence of PC/distill imports in teachers. Unit tests and numerical
reproduction have not been run, and no source equivalence check substitutes for
the user-run native-versus-migrated training comparison.
