# OTP-FM teachers on the ALI-CFM experiment datasets

This is a new shared-data adaptation of the original OTP-FM core, not a claim
that the paper reports these particular experiments. The native GOM/Beijing
profiles and their finite-step W2 parity workflow remain separate. Student/PC
code and the student's velocity-MSE objective are not changed.

## Prepared sweep

Each teacher has a standalone config in `configs/pretrain/`:

```text
otpfm_cite5.yaml       otpfm_cite50.yaml       otpfm_cite100.yaml
otpfm_multi5.yaml      otpfm_multi50.yaml      otpfm_multi100.yaml
otpfm_eb.yaml          otpfm_st.yaml
otpfm_cell_tracking.yaml                      otpfm_knot.yaml
```

These files contain the complete model, data, training, evaluation, logging and
output settings. They work directly with the normal project CLI, independently
of the sweep launcher:

```bash
python main.py --config configs/pretrain/otpfm_cite50.yaml
python main.py --config configs/pretrain/otpfm_cell_tracking.yaml
```

`DEFAULT_SWEEP` in `scripts/run_otpfm_dataset_sweep.py` contains the config
references, ranking choices, sweep seeds/count and parameter grid. The launcher loads the
OTP-FM pretrain presets; it does not build teacher configs from ALI-CFM presets.
Search parameters and explicit CLI seed/budget/path/device/logging overrides
are applied to copies of those presets. Dataset observation budgets, the six
tracking frames and all metric settings live in the pretrain configs.

```bash
bash scripts/run_otpfm_ali_data_sweep_8gpu.sh --dry-run
bash scripts/run_otpfm_ali_data_sweep_8gpu.sh
```

Default: **100 distinct configurations per profile, 1,000 teachers total**.
Jobs rotate across datasets rather than exhausting one dataset first. The queue
uses GPUs 0–7, at most two independent workers per GPU and 16 workers total,
with four CPU threads each. GPU discovery uses `nvidia-smi`; the controller
imports no Torch and initializes no CUDA context. Workers each see `cuda:0` on
their assigned GPU. Each worker is a separate teacher, not DDP.

`--dry-run` reads configuration, source hashes and file-existence metadata only.
It creates no output directory, W&B run or model, and runs no training/evaluation.

Data: `/data/lthung/ali-cfm-data`, configurable with `FLOW2PC_DATA_ROOT` or
`--data-root`. Results: `/data/lthung/flow2pc/otp-fm/ali-data-sweep-100`, configurable
with `OTP_FM_ALI_SWEEP_ROOT` or `--output-root`. No dataset downloads or changes
to the source data are performed.

## Dataset-specific experiment and metric

The standalone `otpfm_<profile>.yaml` files explicitly retain the existing
experiments' data/evaluation settings, including the requested six-frame tracking
selection. They are not replaced by ocean/air-quality settings. The training loss
is original OTP-FM; ALI losses, interpolants and GANs are not imported.

| Profile | Representation and observations | Ranking metric |
| --- | --- | --- |
| cite5, multi5 | Released 5D PCA, ordinal times for days 2/3/4/7, hold out index 1 | Held-out-time W1 EMD in standardized feature coordinates |
| cite50, cite100, multi50, multi100 | Released PCA at the named dimension, same times/fold | Held-out-time W1 EMD in raw PCA coordinates |
| eb | Supplied PHATE-2D, original ordinal day-range categories, hold out index 1 | Held-out-time W1 EMD in standardized PHATE coordinates |
| st | Aligned tumour-only U2–U5 coordinates in saved y,x order, hold out U3 | Held-out-time W1 EMD in aligned spatial coordinates |
| cell_tracking | All 115 cell-4 masks; fixed seeded 10-point training subset per frame | Mean noninitial-frame W1 EMD at the six specified display times, in pixels |
| knot | Lifted middle curve `sin(...)+1`, 120 times, noise std 0.01 | Mean W1 EMD over all 119 noninitial times in raw x,y coordinates |

EMD is the existing **uniform-mass W1 with Euclidean costs in the full feature
dimension**, not W2, squared EMD, sliced Wasserstein, a paired trajectory error,
or the first-two-coordinate visualization. It uses the frozen instantaneous EMA
field `v_func_ema(x,t,t)` integrated by the existing velocity-driven ODE evaluator.

Source/reference split scope, deterministic subsets, metric sample counts,
evaluation seed 2026, solver, tolerances and coordinate conversions are inherited
unchanged. CITE/Multi use 2,048 source/metric samples; EB uses 512. ST/tracking/knot
retain the full source population (`num_samples: 0`) and up to 512 metric samples.
The sampled reference and predicted sets need not have the same number of rows;
the existing exact uniform-mass EMD handles that without truncating them.

Tracking explicitly uses zero-based indices `[0,24,49,74,99,114]`, corresponding
to README frames **1,25,50,75,100,115**, for both scores and overlays. Its final
overlay transports the full initial cell-4 mask. The existing eight-frame default
is retained for old configs; compare aggregate ALI/OTP scores only using the same
frame set. The shared evaluator now accepts `evaluation.time_indices` for this.

Knot and tracking have no held-out time in these experiments. Their scores are
labelled reconstruction scores, not unseen-time interpolation. Other temporal
holdouts are used for hyperparameter selection; they are not untouched test scores.
Rankings are per profile, never pooled across different dimensions or units.

## Original OTP-FM training with a general marginal adapter

`otpfm/marginals.py` subclasses the original base `Trainer` only to reshuffle
unpaired marginal tuples at epoch boundaries. The original network, potential,
direct solver, MeanFlow/iMF objectives, optimizer, forward/backward order, EMA,
curriculum conversion and training loop are retained. The tuple dataset class is
copied unchanged from the original OTP-FM single-cell implementation, with source
hashes and Apache-2.0 attribution. Shared data loaders own the representations.

The first sweep uses **W2Inf with independent marginal tuples** and the native
direct solver. It does not run a global OT precomputation on the large cell
populations or substitute an approximate coupling under the native W2 label.

For the native ALI-data profiles, training sees all observed rows, except EB,
which keeps its existing observed `train`-row scope. Temporal holdouts are excluded
from training tuples and potential centers. Tracking's seeded without-replacement
ten-point indices are selected before model initialization, matching the existing
ALI experiment, and saved in `sampling_contract.json`. Its full mask remains
available to evaluation. Knot uses fresh independent analytic noise during
training and the fixed ten-sample-per-time shared dataset for evaluation. No
cross-time observation IDs or persistent particle/cell pairings are inferred.

Data preprocessing is inherited too. Some presets intentionally fit all-time
statistics; that transductive choice is recorded rather than silently changed.

The search space is in `DEFAULT_SWEEP` in the launcher:

| Parameter | Values |
| --- | --- |
| Hidden width | 128, 256, 512 |
| Hidden layers | 2, 4, 6 |
| Learning rate | 1e-4, 3e-4, 1e-3, 3e-3 |
| Batch size | 64, 128, 256 |
| Potential strength | 0.1, 1, 10, 100, 1000 |
| Temporal width multiplier | 0.25, 0.5, 1, 2 |
| Temporal kernel | Gaussian, box |
| Consistency setting | Original MeanFlow or iMF, both using native MSE |
| Effective curriculum midpoint | 0.25, 0.5 |

The width multiplier multiplies the median gap between observed model times.
Potential centers use the exact shared observed times, excluding both endpoints.
One hundred distinct combinations are sampled deterministically with grid seed
2026 and used across profiles. Teacher/data seed is 42. This is a hyperparameter
search, not 100 seeds. Dropout, weight decay and gradient clipping are disabled;
the original clipping placement is not edited.

The pretrain configs request **10,000 optimizer updates per run** by default;
the sweep inherits that budget unless `--steps` is provided. The original
trainer uses complete epochs, so epochs are resolved as
`ceil(requested_updates / batches_per_epoch)`. The actual update count can exceed
the request by less than one epoch and is recorded. Knot uses 100 batches/epoch.
The exported teacher is the **final EMA teacher**, and the sweep ranks those
final teachers by EMD. It does not claim to search checkpoints within every run.
The empty native validation loader's zero loss is explicitly not a ranking score.

## Outputs, W&B and recovery

```text
<root>/sweep_plan.json
<root>/summary.csv
<root>/<profile>/summary.csv
<root>/<profile>/best_models.json           top 10 completed teachers by EMD
<root>/<profile>/<run>/resolved_config.yaml
<root>/<profile>/<run>/native_config.json  resolved times, widths and epochs
<root>/<profile>/<run>/sampling_contract.json
<root>/<profile>/<run>/metrics.jsonl        original training loss/curriculum
<root>/<profile>/<run>/checkpoints/last.pt  portable final live+EMA weights
<root>/<profile>/<run>/manifest.json
<root>/<profile>/<run>/preprocessing.json
<root>/<profile>/<run>/eval/metrics.json
<root>/<profile>/<run>/eval/media.json
<root>/_configs/<profile>/*.yaml
<root>/_launcher/{status.json,logs/}
<root>/_runtime/{tmp,wandb-cache,wandb-staging}/
```

Final images are `teacher_ode_marginals.png` for CITE/Multi/EB,
`teacher_overlay.png` for ST and tracking, and `teacher_trajectories.png` for knot.
Tracking uses a six-panel microscopy overlay; ST uses clean aligned H&E. No
unrelated fallback image is generated if those source images are missing.

Every run logs to W&B **`flow2pc`**, with dataset/dimension, architecture, learning
rate, strength, width, consistency setting and seed in the run name. Startup
communication failures fall back to offline logging; local losses/checkpoints/
EMD/figures remain authoritative if remote logging is unavailable.

For a known outage, start without contacting W&B:

```bash
bash scripts/run_otpfm_ali_data_sweep_8gpu.sh --mode offline
```

Run fewer profiles or lower concurrency explicitly:

```bash
bash scripts/run_otpfm_ali_data_sweep_8gpu.sh --datasets cite50 st
WORKERS_PER_GPU=1 MAX_PARALLEL=8 bash scripts/run_otpfm_ali_data_sweep_8gpu.sh
```

Reusing the same command skips compatible completed teachers. Valid final
checkpoints with missing evaluation receive evaluation-only recovery. An explicit
`--retry-incomplete` archives partial attempts before retraining; final checkpoints
are never archived/overwritten. Mid-training optimizer resume is not claimed.
OOM/infrastructure failures receive one bounded retry; numerical failures do not.

```bash
bash scripts/run_otpfm_ali_data_sweep_8gpu.sh --dry-run --retry-incomplete
bash scripts/run_otpfm_ali_data_sweep_8gpu.sh --retry-incomplete
bash scripts/run_otpfm_ali_data_sweep_8gpu.sh --summarize-only
```

Disk-full/quota errors while writing reports or preparing launches pause new
work rather than stopping active workers. At least 20 GiB free on the output
filesystem is required for new launches. Temporary files and W&B cache/staging
use that monitored filesystem. GPU admission requires 4 GiB free per new worker,
including warm-up reservations. These checks do not reserve resources against
other users. The controller never stops unrelated jobs. Ctrl-C stops its own
worker process groups. A root lock prevents duplicate controllers.

Environment overrides: `FLOW2PC_PYTHON`, `FLOW2PC_DATA_ROOT`,
`OTP_FM_ALI_SWEEP_ROOT`, `GPUS`, `WORKERS_PER_GPU`, `MAX_PARALLEL`,
`CPU_THREADS_PER_RUN`, `MIN_FREE_GB`, `MIN_DISK_FREE_GB`. CLI also supports
`--runs-per-dataset`, `--steps`, `--seed`, `--grid-seed`, `--worker-retries`,
and `--mode`. Changing scientific settings should use a fresh root.

## Verification boundary

The native GOM/Beijing code previously matched all eight original controls
exactly; its pinned source files are unchanged. This new dataset adapter/sweep
has received syntax, source-equivalence, configuration and metadata-preview
checks only. Regression definitions cover the 1,000-config grid, inherited data/
EMD settings, six tracking frames, unchanged default evaluation behavior, original
training-loop inheritance and evaluation-only recovery. No training, distillation,
model evaluation or unit tests were run for this extension. New-dataset model
quality is unverified until the user executes the prepared commands.
