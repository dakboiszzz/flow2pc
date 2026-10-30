# Native ALI-CFM teacher runs on eight GPUs

Completed native controls now verify exact training-loss and final model-weight
parity for all six single-cell recipes and ST, seed 42 / first fold. See the
[completed comparison and native failure diagnosis](ALICFM_NATIVE_PARITY_RESULTS.md).
The static-only notes below describe preparation before those user-run controls.

This prepares eight independent Flow2PC teacher jobs: CITE5/50/100,
Multi5/50/100, ST and cell-tracking. Each new worker is assigned a different GPU;
there is no distributed-loss rewrite and no automatic student stage.

## Source review and corrections

| Teachers | Native reference | Restored behavior |
| --- | --- | --- |
| CITE/Multi | Supplied `../ali-cfm` working checkout, HEAD `a35963e`, with recorded file hashes | Full observed training populations; all-time StandardScaler/min-max; original staged loops and native plotting callbacks |
| ST | Matching ST implementation at `501e9db` in that repository | Original piecewise method, 256/128 batches, staged loops and original H&E training Plotter |
| Cell-tracking | Current repository's `configs/cell_tracking/ali_cell_tracking.yaml` and author trainer | Full-mask global bounds; seeded ten-point subsets per frame; original derivative and raw-pixel CFM |

The supplied native checkout has local changes (including tracking support in
shared training functions). Source provenance distinguishes its actual file
hashes from committed-file hashes; current single-cell/tracking comparisons use
the actual checkout, while ST uses its matching historical revision.

The eight configs explicitly select `training.protocol: native` and
`data.params.preprocessing_protocol: native_all`. The latter preserves the native
sequence: fit StandardScaler on the complete supplied float32 PCA array where
configured, transform in float32, then apply Torch min-max. Statistics include
the held-out time, as the native scripts do. This is a documented transductive
preprocessing protocol, not a claim of fitting only training observations.

The teacher trains on all **observed-time** rows (or the native tracking subset),
while the held-out time is excluded from its optimization batches. Dataset row
splits remain available for later student work. Existing artifacts without the
new protocol flags continue using their saved affine preprocessing; old weights
and output directories are not rewritten.

Native model/optimizer construction order and cuDNN settings are retained. The
framework's later W&B initialization restores the post-model Python/NumPy/Torch
RNG states so startup logging cannot shift the native sampling stream.
Single-cell/ST GAN callbacks run at epoch 0 and every 5,000 updates, consuming
their original random draws. Tracking's native trainer disables these callbacks,
so tracking keeps them disabled. Loss formulas/reductions and OT routines are the
pinned native source; no new student objective is involved.

The ST reference is deliberately `501e9db`: the later upstream refactor leaves
the ST piecewise call incompatible with the newer network method. This recovery
does not substitute a different regularizer or claim the broken HEAD entry point
can reproduce the matching historical implementation unchanged.

Cell-tracking uses the repository recipe selected by the user, **not** the earlier
successful checkpoint: normalized ALI coordinates, raw-pixel CFM, smoothing
0.001, coefficient 1, 70,000 GAN and 50,000 CFM updates. CFM position and velocity
targets are converted to pixels exactly as in that trainer. The frozen adapter
applies the corresponding chain rule when queried in dataset coordinates.

Each single-cell/ST job uses the first native fold with teacher seed 42. The
native multi-fold driver seeds once and then runs folds in sequence; a separately
seeded fold 2 would not reproduce that driver's inherited fold-1 RNG state.
For native controls, use the same seed, fold, GPU/software and CPU thread settings.

## Run

From the Flow2PC checkout:

```bash
bash scripts/run_alicfm_teachers_8gpu.sh --dry-run
bash scripts/run_alicfm_teachers_8gpu.sh
```

The preview queries `nvidia-smi`, resolves configs and checks existing output
metadata. It creates no run directories and starts no worker, W&B run or model.

Defaults:

- GPUs: physical indices `0,1,2,3,4,5,6,7`, one new worker per GPU, eight workers
  maximum. Parent `CUDA_VISIBLE_DEVICES` allocations are respected; missing GPUs
  cause a clear error rather than a single-GPU fallback.
- Interpreter: `/data/nvidia/conda_envs/uffm/bin/python`.
- Data: `/data/lthung/ali-cfm-data`.
- Outputs: `/data/lthung/flow2pc/framework/native-teachers/<dataset>/teacher`.
- W&B: project `flow2pc`, names including the dataset/native protocol/seed.
- CPU threads: four per worker.
- Minimum currently free GPU memory: 2 GiB. Eligible GPUs launch concurrently;
  the queue waits for other selected GPUs to have enough free memory. Memory is
  queried with `nvidia-smi`; the controller does not initialize Torch/CUDA.

Override machine settings through `GPUS`, `FLOW2PC_PYTHON`,
`FLOW2PC_DATA_ROOT`, `FLOW2PC_TEACHER_ROOT`, `CPU_THREADS_PER_RUN` and
`TEACHER_MIN_FREE_GB`. For example, choose a fresh output root for a new experiment:

```bash
FLOW2PC_TEACHER_ROOT=/data/lthung/flow2pc/framework/native-teachers-repeat \
  bash scripts/run_alicfm_teachers_8gpu.sh
```

Completed, compatible teacher/evaluation outputs are reused. Partial teacher
outputs without a checkpoint are preserved and reported; choose a fresh root
for those runs. The launcher never deletes or restarts completed training to
resolve an output collision. Failed datasets are reported individually.

## Results and comparison limits

Per-run weights are `checkpoints/last.pt`; logs are `metrics.jsonl`, with run
configuration, source provenance, data splits and preprocessing saved alongside.
`eval/media.json` lists the final images: single-cell marginal panels, ST H&E
overlay, or cell-tracking microscopy overlay. `_launcher/status.json`,
`_launcher/summary.csv` and `_launcher/logs/` track the parallel workers.

Post-training images/EMD still use the explicitly configured shared evaluation
protocol. It is not automatically the paper's full-population evaluation:
single-cell metrics cap populations at 2,048 and start at the earliest marginal;
ST uses the shared RK4 grid; tracking uses the normalized adapter's integration
coordinates. Native full-population sample sets, solvers, tolerances and coordinate
error norms must be matched before claiming identical EMD. Restore/evaluate the
same weights through both paths first when investigating a numerical mismatch.

The assistant performed static source/config/syntax review only. Numerical
regression checks are prepared in `tests/test_alicfm_native_protocol.py` and
`tests/test_parallel_checks.py`; no tests, training or evaluation were run by the
assistant. Source equality and config matching are not numerical reproduction
results. The user's later native and migrated runs are the comparison evidence.

Validation performed during preparation: all six single-cell scalar recipes and
the raw-pixel tracking recipe matched the supplied configs; ST scalar settings
matched `501e9db`. The seven selected stage functions and three ST plotting
methods matched their reference ASTs after logger normalization. Python parsing,
Bash syntax and whitespace checks passed. The metadata-only launcher preview
selected eight GPUs and eight teacher tasks with no preparation errors, and
created no training output root.

## Native repository controls

`scripts/run_native_alicfm_control.py` executes the native driver without importing
Flow2PC. Single-cell imports the supplied `../ali-cfm` working checkout; ST extracts
`ali_cfm/` and its config from `501e9db` into the control output directory and imports
that isolated historical package. The supplied repository is never modified.

The wrapper restricts the driver and its result tables to the first held-out fold
(`t=1`) and seed 42, and redirects data/output/W&B settings. It leaves the native
training functions, sampling, losses, callbacks and native post-training evaluation
intact. All four original marginal/time indices are retained. Scientific scalar
settings come directly from the same source YAMLs used for the migrated recipes.
The output includes `native_config.yaml`, `control_manifest.json` with source hashes,
and `native_driver_first_fold.py` showing the fold-selection adaptation. Native W&B
run directories contain checkpoints, including `t=1_seed=42_ali_cfm.pth`.

From either repository, run one control (GPU 0 shown):

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /data/nvidia/conda_envs/uffm/bin/python \
  /data/pqhung_group/lthung/flow2pc/scripts/run_native_alicfm_control.py \
  --dataset cite50
```

Replace `cite50` with `cite5`, `cite100`, `multi5`, `multi50`, `multi100` or `st`.
Append `--dry-run` for source/path validation without importing the native models.

For all seven native controls concurrently:

```bash
bash scripts/run_native_alicfm_controls_7gpu.sh --dry-run
bash scripts/run_native_alicfm_controls_7gpu.sh
```

The launcher uses seven distinct selected GPUs, default 0–6 (or an inherited
`CUDA_VISIBLE_DEVICES` list), four CPU threads per worker and project `flow2pc`.
Use available GPUs: this native-control launcher starts its workers immediately
after preflight and does not wait for free GPU memory. Avoid running both sets
concurrently on the same occupied GPUs if memory is insufficient.

Data defaults to `/data/lthung/ali-cfm-data`; controls go under
`/data/lthung/flow2pc/alicfm-native-controls/<dataset>/seed42-t1`, with stdout/stderr
in `_launcher/logs/<dataset>.log`. Set `ALI_CFM_ROOT`, `FLOW2PC_DATA_ROOT`,
`ALI_CFM_CONTROL_ROOT`, `GPUS`, `FLOW2PC_PYTHON` or `CPU_THREADS_PER_RUN` for the shell
launcher; use `--native-root`, `--data-root` and `--output-root` for a single control.
Existing run directories are refused; choose a fresh root to repeat.

These controls retain native EMD evaluation. Compare training loss traces and
checkpoint fields first; native EMD and shared Flow2PC evaluation EMD differ in
sample sets/integration settings as described above. Same hyperparameters/source
do not establish numerical reproduction without running that comparison. With a
single seed, native sample-standard-deviation summaries may be undefined (NaN).

## Inspection of the 2026-10-30 native-teachers batch

The saved launcher reported five completed runs (CITE50, CITE100, Multi50,
Multi100 and ST), two failed workers (CITE5 and Multi5), and cell-tracking still
running. Tracking was near GAN update 11,000 / 70,000 at inspection; it had not
started CFM. This is an inspection snapshot, not a numerical reproduction claim.

Both failed workers completed all 2,000 pretrain, 70,000 GAN and 2,001 CFM
updates. They saved `checkpoints/last.pt`, `eval/teacher_predictions.pt`,
`eval/metrics.json` and `eval/marginal_metrics.csv` before the failure. Saved
held-out EMD was 0.9650360072 for CITE5 and 1.0432501927 for Multi5, in the
configured standardized feature coordinates.

The failure is in final marginal figure generation: the panel concatenates
train/validation/test splits, but the held-out time deliberately has zero
training and validation rows. These two 5D recipes enable whitening, so the new
native preprocessing passed a `(0, 5)` array into `StandardScaler.transform`,
which requires at least one sample. The other four single-cell recipes disable
whitening and did not encounter that sklearn validation. Both top-level and
evaluation statuses were therefore marked failed even though training and EMD
had finished. This is an empty-split integration bug, not evidence of failed
teacher optimization or GPU OOM.

`data/preprocessing.py:transform` now preserves empty two-dimensional native
splits before calling sklearn. Nonempty transforms, fitting statistics, loss
functions and saved checkpoints are unchanged. A regression for the actual
whitened single-cell final figure is prepared in `tests/test_evaluation_contract.py`.
No test or recovery evaluation was executed by the assistant.

Recover using the existing evaluation-only entry point, for example from Flow2PC:

```bash
for dataset in cite5 multi5; do
  FLOW2PC_PYTHON=/data/nvidia/conda_envs/uffm/bin/python \
    bash scripts/run_alicfm.sh evaluate-teacher "$dataset" \
    --set "output.dir=/data/lthung/flow2pc/framework/native-teachers/$dataset/teacher" \
    --set training.device=cpu &
done
wait
```

This loads the saved checkpoint, reuses a matching cached teacher trajectory,
and regenerates metrics/figures and W&B evaluation artifacts. It performs no
training and does not replace model weights. Run-level statuses become complete
on success; the already-running launcher's in-memory failure list may still show
the original failure until its next invocation. Leave the ongoing tracking worker
alone; do not start a second controller for the same output root while it runs.
