# ALI-CFM and OTP-FM on prepared MFM data

The standalone teacher configs use the existing **Arch, Sphere, and LiDAR**
arrays under `prepared/PROFILE/seed42`. No data is regenerated, standardized
again, or resplit. The shared temporal adapter reuses MFM's saved array hashes,
coordinates, time grid, row indices, and LiDAR scaler/terrain metadata.

| Dataset | Model times | Training populations | Validation | Temporal test |
| --- | --- | --- | --- | --- |
| Arch, 2D | 0, 0.5, 1 | Saved 90% rows at 0 and 1 | Saved 10% rows at 0 and 1 | Entire middle population |
| Sphere, 3D | 0, 0.5, 1 | Saved 90% rows at 0 and 1 | Saved 10% rows at 0 and 1 | Entire middle population |
| LiDAR, 3D | 0, 1 | Saved 90% endpoint rows | Saved 10% endpoint rows | No temporal test population |

Time labels are synthetic transport parameters, not measured physical times.
LiDAR's original full-terrain scaler remains an explicitly recorded conditioning
exception. Terrain points are not relabeled as time-indexed observations.

## What the methods train

These are **two-observation adaptations**, with explicit inactive components:

- ALI-CFM retains its existing generator and velocity networks. Its generator
  minimizes the existing L2 curvature regularizer, then the original CFM field
  stage learns the generator's conditional velocities. Regression pretraining and
  the interior GAN term are absent because there is no observed interior time.
  No held-out middle population is passed to those losses. Coupling is independent
  or endpoint minibatch OT, as configured. Scalar logging is sampled every 100
  updates without changing optimizer updates.
- OTP-FM retains its native network, EMA, optimizer, and MeanFlow/IMF consistency
  objective. Its native `do_otp=False` switch disables potential corrections when
  the interior-potential collection is empty. The original source code is not
  patched. It still trains a velocity/mean-velocity model between the endpoints.

Neither adaptation adds MFM's manifold-metric objective or a new terrain force.
In particular, low endpoint validation EMD does **not** establish that either
teacher learned the curved Arch/Sphere paths or avoided LiDAR terrain. The
held-out metrics and geometry figures expose that distinction. These runs are
not claimed to reproduce a published ALI/OTP synthetic benchmark.

## Configs and individual runs

Configs follow the existing pretrain organization:

```text
configs/pretrain/ali_cfm/alicfm_{arch,sphere,lidar}.yaml
configs/pretrain/otpfm_{arch,sphere,lidar}.yaml
```

For example, commands for the user to execute:

```bash
export FLOW2PC_MFM_DATA_ROOT=/data/pqhung_group/pdtung/flow2pc/data/mfm
export FLOW2PC_OUTPUT_ROOT=/data/lthung/flow2pc
export PYTHONPATH=src

/data/nvidia/conda_envs/uffm/bin/python main.py \
  --config configs/pretrain/ali_cfm/alicfm_arch.yaml

/data/nvidia/conda_envs/uffm/bin/python main.py \
  --config configs/pretrain/otpfm_arch.yaml
```

Use a fresh `output.dir` for each teacher. Dataset files and outputs remain
outside Git. The prepared files suffice; these teachers do not require importing
the external MFM trainer or reading/generating its original datasets.

## Intensive eight-GPU sweep

```bash
# Metadata/hash/config inspection only: no model execution, GPU query or writes.
bash scripts/run_mfm_teacher_sweep_8gpu.sh --dry-run

# Launch only when ready: 3 datasets × 2 methods × 256 configs = 1,536 teachers.
bash scripts/run_mfm_teacher_sweep_8gpu.sh
```

The launcher uses the existing `ParallelSweep` queue. It interleaves all six
method/dataset groups and allows **16 concurrent workers across GPUs 0–7**,
up to two per GPU. Admission uses `nvidia-smi` free-memory checks; the controller
does not create a CUDA context. Use `WORKERS_PER_GPU=1 MAX_PARALLEL=8` for one
worker per GPU. `GPUS`, `MIN_FREE_GB`, `MIN_DISK_FREE_GB`, `CPU_THREADS_PER_RUN`,
`FLOW2PC_PYTHON`, `FLOW2PC_MFM_DATA_ROOT`, and `MFM_TEACHER_SWEEP_ROOT` override
the machine defaults. New launches pause when disk headroom is insufficient.

ALI sweeps width `{64,128,256,512}`, generator LR `{1e-5,5e-5,1e-4,5e-4}`,
field LR `{1e-4,3e-4,1e-3,3e-3}`, batch size `{64,128,256,512}`, and
independent/border coupling. Each run uses 10,000 generator-regularizer updates
and 30,000 CFM updates by default.

OTP sweeps width `{64,128,256,512}`, depth `{2,4,6,8}`, LR
`{1e-4,3e-4,1e-3,3e-3}`, batch size `{64,128,256,512}`, MeanFlow/IMF,
and EMA decay `{0.99,0.999}`. Its default budget requests 30,000 updates; the
native trainer rounds up to complete epochs, with actual updates recorded.

The controller selects 256 distinct configurations from each active grid with
fixed grid seed 2026. It does not sweep unused GAN variants, potential strengths,
potential widths, or lambda kernels. `--steps` overrides the ALI field/OTP update
budgets; `--generator-steps` overrides the ALI regularizer budget.

## Selection, images, logging and recovery

All runs log to W&B project **`flow2pc`**, with dataset/method/hyperparameters in
the run name. Communication failures fall back to offline/local tracking;
`--mode offline` selects offline explicitly. The controller and each worker keep
all outputs under the chosen sweep root, default:

```text
/data/lthung/flow2pc/mfm-teachers/sweep-256/
  summary.csv
  sweep_plan.json
  _launcher/{status.json,logs/}
  METHOD/DATASET/
    summary.csv
    best_models.json
    RUN/
      checkpoints/last.pt
      resolved_config.yaml
      manifest.json
      preprocessing.json
      metrics.jsonl
      tracking_status.json
      eval/
        metrics.json
        marginal_metrics.csv
        media.json
        teacher_ode_marginals.png
        teacher_geometry.png
        geometry_trajectories.npz
        selection.json
        validation/{metrics.json,status.json,...}
```

`validation_emd` ranks final trained teachers across configurations. It is exact
full-dimensional W1 from validation source cells to observed-time validation
populations, excluding the source-time score. Arch/Sphere's middle-time EMD is
reported as `test_heldout_emd` and **never ranks configurations**. LiDAR has no
middle-time test score. `historical` metric coordinates retain the saved LiDAR
whitening; Arch/Sphere use identity coordinates. Figures use raw coordinates.

`teacher_geometry.png` includes observed/predicted populations plus 101-time ODE
curves. Sphere and LiDAR are shown in 3D; the LiDAR trajectory panel also shows
the original terrain. Those figures are saved and logged after every successful
run. A failed numerical evaluation stays visibly failed and unranked.

Rerun the same command to skip verified complete runs. Saved final checkpoints
with missing evaluation outputs are reevaluated through the normal CLI rather
than retrained. Interrupted pre-checkpoint runs require `--retry-incomplete`,
which preserves the incomplete directory under `_attempts` before restarting.
An output-root lock prevents duplicate controllers or orphaned workers from
training the same sweep twice. To refresh summaries without launching jobs:

```bash
bash scripts/run_mfm_teacher_sweep_8gpu.sh --summarize-only
```

No training, evaluation, distillation or tests were executed during implementation.
The metadata-only preview, syntax checks, and input-hash inspection are separate
from runtime verification. Regression cases are in `tests/test_mfm_teacher_sweep.py`.
