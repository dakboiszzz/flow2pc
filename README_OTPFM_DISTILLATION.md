# OTP-FM → Flow2PC: Beijing and Gulf of Mexico sweeps

This implements the conditional-interpolant route in
`README_OTP_FM_TO_FLOW2PC.md`. The **existing pretrained OTP-FM teachers are
loaded and frozen**. No OTP-FM retraining, learned generator, or GAN is involved.
Only the Flow2PC students are optimized.

The default now reuses the working ALI student and optimizer recipe. The first
324-run sweep used the proposed anchored-source variant, which materially
changed that recipe. A controlled 20k-step seed-0 comparison with the teacher
and sampler unchanged obtained GoM ODE W2 0.133605 (teacher 0.133019) and Beijing
15.255944 (teacher 15.417650) using the original ALI recipe. These are diagnostic
single-seed results, not a multi-seed guarantee. Full evidence is under
`/data/lthung/flow2pc/otp-fm/diagnostics/recipe_audit_20261021_164824`.

## Run

```bash
cd /data/pqhung_group/lthung/flow2pc

# Full corrected hyperparameter sweep, 324 students, existing caches reused.
bash scripts/run_otpfm_corrected.sh --dry-run
bash scripts/run_otpfm_corrected.sh

# Check selected checkpoints, GPU pool, names and run counts; starts no jobs.
bash scripts/run_otpfm_sweep.sh --dry-run

# Smaller confirmation plan: the checked h32/f16/lr3e-4/b32 recipe, three seeds
# per dataset, six students total. Remove --dry-run only to launch it.
bash scripts/run_otpfm_sweep.sh --grid configs/otpfm/ali_recipe_check.json --dry-run

# Start the two student sweeps when ready.
bash scripts/run_otpfm_sweep.sh

# Optional explicit logical GPU pool and process count per GPU.
bash scripts/run_otpfm_sweep.sh \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 --workers-per-gpu 2
```

The wrapper uses `/data/nvidia/conda_envs/uffm/bin/python`, overridable through
`FLOW2PC_PYTHON`. It preserves the existing environment and installs nothing.
It requires at least two visible CUDA devices. All visible GPUs are considered;
free-memory checks gate new workers (4 GiB by default). The shared queue
uses `nvidia-smi` memory queries without allocating CUDA contexts in the launcher.
Full/unavailable GPUs are skipped; transient monitoring failures pause launches
while active workers continue and the monitor retries. The shared queue
interleaves datasets, with two independent worker processes per GPU by default.
Preparation for one dataset can overlap student work on the other. This is
parallel hyperparameter search, not DDP for a single student. GPU allocation
failures have one bounded retry. Ctrl-C/SIGTERM stops worker process groups.

`run_otpfm_corrected.sh` runs the full grid: latent states 32/64/128, time features
8/16/32, LR 1e-4/3e-4/1e-3, batch size 32/64, and seeds 0/1/2 on each dataset.
This is 162 students per dataset, 324 total, with 20,000 velocity updates each.
It logs online to W&B project `flow2pc`, evaluates/plots student and teacher ODE
trajectories at the end of each run, and disables direct student sampling. Results go to
`/data/lthung/flow2pc/otp-fm/corrected`; `--reference-root` reuses the existing
`/data/lthung/flow2pc/otp-fm/reference` without rebuilding the teacher caches.
Appending `--grid configs/otpfm/ali_recipe_check.json` instead selects only the
six-student confirmation plan at h32/f16/lr3e-4/b32.

## Teachers, sampling and source policy

The selected existing checkpoints, verified against the handoff manifest, are:

- Beijing: `../otp-fm/outputs/reproduction/main-reproduction/beijing/W2/seed44/models/best.pt`
  (selected epoch 120).
- GoM: `../otp-fm/outputs/reproduction/main-reproduction/gom/W2/seed44/models/best.pt`
  (selected epoch 1876).

Native factories strictly reconstruct all live/EMA weights. Distillation queries
only frozen `v_func_ema(x,t,t)`. Precomputed observed-only OT row-argmax chains
preserve the native coupling, including duplicate indices for unequal Beijing
marginal sizes. Source rows define the deterministic 95%/5% train/validation
split. Intermediate observations may be shared across these sets. Real held-out
observations are never student training inputs or checkpoint-selection targets.

The conditional W2 interpolant uses a single LU factorization and cached forces.
Its alpha is fixed outside the full-strength solve: Beijing
`0.916572766752024`, GoM `0.9997395918739371`. The cache validates its linear
residual and both endpoint identities. Training batches use direct cache queries,
one frozen EMA target query and a student velocity-MSE update. There are **no ODE
rollouts per update**. Conditional path velocity is a diagnostic, not the target.

The scaler exactly replays the native float32 StandardScaler transformation.
As in the pretrained experiment, scaler fitting includes all original marginals;
that convention is recorded explicitly. No ALI min/max normalization is used.
Saved potential knots (including rounded Beijing knots) are preserved. Scoring
uses exact original observation times `i/12` for Beijing and `i/8` for GoM.

By default, `--student-recipe ali` calls the existing single-cell HCLT builder
with uncentered Fourier features and the existing `initialize_source`. All
student parameters remain trainable. There is no source-MLE phase or NLL loss.
As in ALI training, AdamW uses weight decay .01 and cosine learning-rate decay;
scheduler state is saved/restored in checkpoints. The teacher target and cached
sampling protocol are unchanged. Marginal/source NLL remains diagnostic only.

The earlier proposal is retained as `--student-recipe anchored`: source MLE,
frozen source offsets/circuit weights, zero-centered Fourier drift, and
fixed-rate Adam. This is an experimental student variant, not the original ALI
recipe. The source .01 std floor does not constrain future widths. The recipe
is included in configs/checkpoints, and its source boundary is still checked.

## Sweep and selection

`configs/otpfm/sweep.json` contains, for each dataset:

| Parameter | Values |
|---|---|
| Latent states | 32, 64, 128 |
| Fourier features | 8, 16, 32 |
| Learning rate | 0.0001, 0.0003, 0.001 |
| Batch size | 32, 64 |
| Student seed | 0, 1, 2 |
| Velocity steps | 20,000 |
| Validation / checkpoint interval | 500 / 1,000 steps |

This is **162 students per dataset, 324 total**. Gradient clipping is 1; its
logged norm is the norm before clipping. Checkpoints and the best configuration are
selected solely by validation absolute velocity MSE on fixed cached validation
tuple rows and continuous times. Source fitting has its separate NLL-based
selection. Holdout W2 never selects a student. Repeating the same command reuses
immutable references and resumes/skips students. Changed settings, source code
or reference identities require a new output root where the binding changes.

## Results and W&B

The corrected launcher writes results to **`/data/lthung/flow2pc/otp-fm/corrected`**,
outside the code checkout, and reuses teacher references under the parent
directory. Override `--output-root` for a separate experiment root.

```text
otp-fm/
  reference/{gom,beijing}/
    cache.pt                 # normalized clouds, chain indices, forces, scaler, splits
    metrics.json             # immutable measured references and their hashes
    data_contract.json
    marginal_comparison.png
    w2_comparison.png
  sweeps/{gom,beijing}/
    configs/<configuration>.json
    runs/<configuration>/
      source_fit.json
      source_metrics.csv
      metrics.csv
      checkpoints/{best,last}.pt
      eval/
        metrics.json
        per_time_metrics.csv
        predictions.pt
        marginal_comparison.png
        w2_comparison.png
    selected_student.json
  logs/
  summary.csv
  sweep_manifest.json
  sweep_status.json
```

All production W&B runs use project **`flow2pc`**, with groups
`otpfm-beijing-w2-teacher44` and `otpfm-gom-w2-teacher44`. Student names include
dataset and settings, e.g. `otpfm-gom-h64-f16-lr0.001-b64-s0-…`.
Source fit, training/validation curves, per-time W2/NLL, reference identities,
configuration, checkpoints, numerical metrics and comparison plots are logged
or uploaded through the existing W&B initialization/retry/file handling.
The general launcher defaults to `/data/lthung/flow2pc/otp-fm/ali-recipe`; the
dedicated corrected launcher selects the `corrected` directory above. The old
324 anchored students, their local run logs and matching W&B runs were deleted
at the user's request. Teacher references, successful diagnostic checkpoints
and small diagnostic summaries remain. The deletion receipt is
`/data/lthung/flow2pc/otp-fm/diagnostics/cleanup_20261021_175524/cleanup_result.json`.
Changed source/config hashes require a new run directory.

Each completed student is evaluated and visualized at every observation time:

1. Real observations.
2. Finite-step EMA teacher, with steps additionally split at exact observation
   events. This common protocol is labeled separately from the original score.
3. Direct cached conditional interpolant.
4. Optional direct student density samples, only with `--direct-sampling`.
   They are disabled by default and never enter the velocity objective.
5. Teacher and student diagonal-ODE predictions from the same empirical source
   (enabled by default, computed once for the reference and once per completed
   student). Use `--no-ode-comparison` to disable these final diagnostics.

Metrics use exact unequal-size empirical **W2** in physical coordinates: squared
Euclidean transport cost followed by a square root. All real observations are
retained. The four holdout scores and observed-time scores are averaged separately;
observed-time means exclude the source. Physical NLL applies the scaler Jacobian.
The original recorded teacher finite-step scores remain separately labeled
legacy provenance; their truncation/nearby-time-minimum protocols are not reused
as common-protocol scores. These teachers were previously selected on benchmark
holdouts; the new student validation split does not make those holdouts untouched.

## Focused validation

```bash
/data/nvidia/conda_envs/uffm/bin/python -m unittest \
  tests.test_otpfm tests.test_otpfm_sweep
```

These check native direct-solver and OT-chain parity, full-vector interpolant
derivatives, EMA selection, exact source anchoring, unequal-size W2, normalized
data arithmetic, GPU scheduling limits, failure isolation, and dry-run behavior.
An end-to-end CPU check can prepare a reference with `--wandb-mode disabled`
and train a temporary four-latent student for a few updates using
`scripts/train_otpfm.py --steps 3 --source-steps 25`; this is a validation artifact,
not a sweep or a performance result.
