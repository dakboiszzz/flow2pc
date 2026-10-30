# ALI-CFM on Gulf of Mexico and Beijing

This is a new-data adaptation of the existing ALI-CFM networks and three original
stages, not a reproduction of an ALI-CFM paper result on these datasets. It trains
teachers only. No student, OTP-FM teacher training or direct PC sampling is run.

## Data and selection protocol

Data root defaults to `/data/lthung/otp-fm` and is configurable. Only the prepared
safe `gom/marginals.npz` and `beijing/marginals.npz` are loaded, never object-array
pickles or automatic downloads. Source hashes and exact preprocessing/time/split
metadata are stored with each teacher checkpoint.

| Dataset | Observed times | Held-out times | Coordinates/model time |
| --- | --- | --- | --- |
| GOM | 0,2,4,6,8 | 1,3,5,7 | 9 populations of 111 points in 2D; model time i/8 |
| Beijing | 0,1,3,4,6,7,9,10,12 | 2,5,8,11 | 13 hourly PM2.5 monthly populations in 1D; model time i/12 |

There are no observed trajectory pairings and no extra observation-row split.
Optimization samples only observed-time populations. StandardScaler is fit on
all populations, including temporal holdouts, matching the supplied data READMEs.
This transductive preprocessing is explicit; min-max normalization is disabled.
Beijing physical metadata records calendar-month offsets, not invented day counts;
the GOM physical elapsed-time unit is unknown and recorded as observation index.

All original pretrain/GAN/CFM functions, OT minibatch routines, loss reductions,
finite-difference estimator and Adam optimizers remain unchanged. The new
`training.protocol: marginal` selects full observed populations and the existing
single-cell-style network/stages. ST and earlier native single-cell recipes remain
unchanged. The native single-cell PCA callback is disabled for this adaptation:
it assumes four 2D/PCA panels and is invalid for 1D Beijing and 9/13 timepoints.
Both loss curves and final velocity-driven ODE figures are still logged to W&B.

Evaluation starts from the entire earliest observed population, integrates to
the exact fixed time knots with Dopri5 at tolerance 1e-5, and computes uniform-mass
W1 EMD using Euclidean costs in raw coordinates. The 1,024-point metric cap exceeds
every population in both datasets, so all points are used, including unequal-size
Beijing populations. Rank by `teacher_heldout_emd`, the unweighted mean across
all four holdouts. This is a model-selection/validation score, not an untouched
test score. The source OTP-FM README describes W2, nearby-time selection for GOM
and truncation for Beijing: those are different protocols. Do not compare their
reported W2 numbers directly with this fixed-time W1 ranking.

## Grid: 512 teachers per dataset

| Parameter | Values |
| --- | --- |
| Shared G/D/CFM hidden width | 128, 256 |
| G and D learning rate | 1e-5, 5e-5, 1e-4, 5e-4 |
| CFM learning rate | 1e-4, 1e-3 |
| Time smoothing | 0, 0.01 |
| Curvature regularization coefficient | 0, 0.001, 0.01, 0.1 |
| Coupling for all stages | border, mmot |
| GAN objective | vanilla, R3GAN |

The full Cartesian product is 512 configs; both datasets give 1,024 runs.
The shuffle is deterministic and the queue alternates datasets. A fixed seed 42
is used for the search; it does not establish multi-seed reliability. Curvature
uses the source `2nd_derivative`/`l2` branch with three sampled times and h=0.001.
Batch size is 128. These are teacher-specific ALI objectives; the PC student MSE
objective is untouched.

Each run defaults to 2,000 interpolant-pretrain updates, 30,000 GAN updates and
10,000 CFM updates. Selection compares final teachers at equal budgets; there is
no early pruning or claim of selecting the best intermediate CFM checkpoint.
Use `--pretrain-steps`, `--gan-steps`, `--cfm-steps` to change budgets, with a new
output root for a different experiment. All configs, names and output paths are
deterministic. Names include dataset, widths, learning rates, smoothing, coefficient,
coupling, GAN objective, seed and a config hash.

## Run

From Flow2PC:

```bash
# Source/array-header/config/GPU preview; no outputs or workers.
bash scripts/run_alicfm_transport_sweep_8gpu.sh --dry-run

# Launch only when ready.
bash scripts/run_alicfm_transport_sweep_8gpu.sh
```

Defaults select physical GPUs 0–7, one independent worker per GPU and eight
workers maximum, with four CPU threads per worker. The queue respects inherited
CUDA visibility and existing GPU memory use, checks memory through `nvidia-smi`
without a controller CUDA context, and requires 4 GiB free before each launch.
It spreads workers and reserves admission memory during startup. The threshold
is an admission guard, not a guarantee against allocations by unrelated jobs.
Other jobs are never stopped. Worker failures are isolated and the rest continue.
This is parallel independent training, not DDP that changes the native batch.

For an explicit higher-concurrency experiment after checking available memory:

```bash
WORKERS_PER_GPU=2 MAX_PARALLEL=16 MIN_FREE_GB=4 \
  bash scripts/run_alicfm_transport_sweep_8gpu.sh
```

Environment overrides: `FLOW2PC_PYTHON`, `OTP_FM_DATA_ROOT`, `ALICFM_SWEEP_ROOT`,
`GPUS`, `WORKERS_PER_GPU`, `MAX_PARALLEL`, `CPU_THREADS_PER_RUN`, `MIN_FREE_GB`.
The default Python is `/data/nvidia/conda_envs/uffm/bin/python`. Additional CLI
arguments pass through, including `--datasets gom` or `--datasets beijing`.

## Outputs and recovery

All artifacts default to `/data/lthung/flow2pc/alicfm-transport/sweep-512`:

```text
sweep_plan.json                       full reproducible config plan
summary.csv                           all runs and hyperparameters/status/EMD
_configs/<dataset>/<run>.yaml          generated worker configs
_launcher/status.json                  running worker PIDs/GPUs and queue counts
_launcher/logs/<run>.log               worker stdout/stderr
gom/summary.csv                       dataset-specific results
gom/best_models.json                  top 10 completed teachers
gom/<run>/checkpoints/last.pt          final teacher/interpolant/optimizer states
gom/<run>/eval/teacher_ode_marginals.png
gom/<run>/eval/media.json
beijing/summary.csv
beijing/best_models.json
beijing/<run>/eval/teacher_ode_marginals.png
beijing/<run>/eval/teacher_quantiles.png
```

Project is always W&B `flow2pc`. Losses, raw-coordinate EMD per time, aggregate
scores, final figures, model artifacts and resolved config are logged there.
GOM plots show observed/predicted 2D populations at all nine times. Beijing plots
show distributions at all thirteen months and q10/q50/q90 curves, highlighting
temporal holdouts. All model predictions come from the trained velocity ODE.

Restarting the same command skips compatible completed runs. A saved final
checkpoint with missing/failed evaluation is recovered with `--evaluate teacher`,
without retraining. Without `--retry-incomplete`, incomplete training without a
final checkpoint is preserved and reported as blocked; mid-training optimizer
resume is not implemented. Incomplete attempts are archived only for an explicit
retry or the bounded infrastructure retry described below. No logs or weights
are deleted, and trained teachers are never moved. Scientific config changes
require a fresh root. A root lock, inherited by workers, prevents duplicate
controllers. Ctrl-C terminates this launcher's own worker process groups.

Training/evaluation completion and remote upload state are independent. A W&B
communication timeout during transport training/evaluation is recorded in
`tracking_status.json` as `pending_sync`; local scientific results remain valid.
W&B startup communication failures fall back to the SDK's offline mode, retaining
full histories/artifacts for later sync. If that also fails, scalar JSON and local
artifacts remain authoritative and can be republished by the recovery script.
Programming/serialization errors and numerical failures still propagate. Native
single-cell/ST logging remains strict; their verified numerical training is not
altered by this transport-only policy.

Old upload-only failures with valid metrics, correct four-holdout averaging,
matching protocols and existing figures/checkpoints are included in rankings,
with a separate `tracking_status` CSV column. They do not require retraining.
Recover their full loss histories, final images and model artifacts from disk,
then sync them serially, without model construction or evaluation:

```bash
/data/nvidia/conda_envs/uffm/bin/python scripts/recover_alicfm_transport_tracking.py --dry-run
/data/nvidia/conda_envs/uffm/bin/python scripts/recover_alicfm_transport_tracking.py --sync
```

Without `--sync`, recovery builds offline SDK histories only. Uploads remain
marked pending. The script does not append duplicate rows to `metrics.jsonl`;
it replays the saved scalar step sequence and retains previous failure metadata.
Run it while the controller is stopped so the root lock prevents races.

Restart the interrupted sweep using its existing output root:

```bash
bash scripts/run_alicfm_transport_sweep_8gpu.sh --dry-run --retry-incomplete
bash scripts/run_alicfm_transport_sweep_8gpu.sh --retry-incomplete
```

`--retry-incomplete` archives only attempts without a final checkpoint under
`_attempts/<dataset>/<run>/<timestamp>/` before retraining that configuration.
Trained teachers are never archived or overwritten. This is a fresh attempt,
not mid-training optimizer resume. OOM/startup communication failures in new
workers receive one bounded retry by default (`--worker-retries`), with the prior
attempt preserved and a two-minute GPU cooldown. Other workers continue. There
is no automatic numerical-failure retry or objective/data modification.

CSV/rankings update as workers finish and periodically while running. After the
controller has stopped, rebuild summaries without training/evaluation using:

```bash
bash scripts/run_alicfm_transport_sweep_8gpu.sh --summarize-only
```

## Validation limits

Prepared regression definitions cover multiple held-out times, exact data contract
restoration, mean-of-four ranking, 1D Beijing figures, 1,024 unique configs and
evaluation-only recovery. They have not been executed: the user requested scripts
only. No teacher/student training or evaluation has been launched by the assistant.
Static Python/Bash checks and the metadata-only preview are the available checks;
new-dataset numerical training behavior is unverified until the user runs it.

## Interrupted first batch and narrow fixes

The first user-run batch produced 48 fully completed teachers and 16 additional
teachers with saved weights, finite EMD and figures but W&B publication failures.
Six startup W&B timeouts, one CUDA OOM and fifteen workers stopped by the user
account for 22 incomplete attempts. The controller and its worker groups were
stopped cleanly; no data/checkpoint was deleted and no new training was launched
by the assistant. Metadata-only recovery/queue previews recognize 64 reusable
teachers and 960 remaining configurations with no blocked runs when retries are
explicitly requested.

The original 2-worker/2-GiB admission defaults were reduced to 1 worker and 4 GiB
per GPU. W&B communication failure isolation and the recovery path above address
operational failures. They do not prove that prediction quality is fixed. Current
GOM candidates still miss some early populations, and Beijing candidates smooth
monthly changes and can predict negative PM2.5. At the user's instruction, the
original representation, network, losses, time mapping, batches, update budgets
and raw-unit metrics are retained; no log transform, clipping or positivity
constraint was introduced. No fresh sweep is justified merely to repeat the
already trained 64 configurations.
