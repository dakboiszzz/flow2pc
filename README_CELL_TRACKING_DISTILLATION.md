# Cell-tracking velocity distillation

The pretrained teacher is `../ali-cfm/changed_files/ali_cfm_model.pth`, pinned to
SHA-256 `d2c071a2b2fea1931c5c515d4dbc35e8f436043f58322fa0c8083b3503049129`.
The archived `latest-run` model is not substituted. No teacher or interpolant
is retrained. Read `CELL_TRACKING_TEACHER_README.md` for its provenance.

## Launch

```bash
cd /data/pqhung_group/lthung/flow2pc
bash scripts/run_cell_tracking_sweep.sh --dry-run
bash scripts/run_cell_tracking_sweep.sh
```

The scripts are prepared for explicit launch; preparing them does not start
teacher evaluation or any production student training. The dry run hashes
inputs and prints the plan without loading the checkpoint or creating runs.

All visible GPUs are considered, with at least two required. Each student is
an independent process, with two workers per GPU by default (16 on eight
GPUs). `nvidia-smi` reports memory without allocating scheduler CUDA contexts;
full GPUs are skipped. A temporary monitoring failure pauses new launches
while current workers continue. OOM GPUs get a 60-second cooldown; OOM and
transient tracking failures have two bounded retries. Override, for example:

```bash
bash scripts/run_cell_tracking_sweep.sh \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 --workers-per-gpu 2
```

The grid has **108 students**, including the previously successful settings:

| Parameter | Values |
|---|---|
| Latent states | 32, 64, 128, 256 |
| Fourier time features | 16, 32, 64 |
| Initial LR | 1e-4, 3e-4, 1e-3 |
| Batch size | 64 (fixed) |
| Updates | 50,000 (fixed) |
| Student seeds | 0, 1, 2 |

Use `--grid path.json` for a custom grid. `--steps N` overrides/deduplicates
the update-count axis. The standard PC has uncentered Fourier features,
source initialization only, all parameters trainable, AdamW with weight decay
.01, cosine LR decay, and gradient clipping 1. Source MLE and frozen source
offsets are not used. Saturated scale derivatives are masked consistently
with the existing log-scale clamp; unsaturated student behavior is unchanged.

## Teacher/data contract and loss

The adapter strictly loads `cfm_model` into the native time-varying 2D SELU
MLP and passes global `t` unchanged. Frame i has time i/114. Both training and
ODEs use the checkpoint's normalized coordinates. Pixel outputs use its saved
min-max transform and preserve `(x,y)` ordering; bounds must match all masks.

The frozen saved interpolant supplies positions with `training=False` (no
time jitter). Endpoint batches use uniform, squared-cost border OT and sample
matched edges with replacement. Endpoint pools are the full masks rather than
the teacher trainer's original ten subsampled points. This is explicitly a
full-mask student-fidelity experiment. Source and final endpoint pixels are
split 90/10 using a fixed seed; validation endpoints do not appear in training
endpoint pools. No real intermediate mask observations are used in updates.

```text
x_t = frozen_interpolant(x0, x1, t)
target = frozen_CFM(concat(x_t, global_t))
loss = mean((PC.velocity(x_t,t) - target)^2)
```

There is no PC direct sampling or NLL loss. The source split measures student
fidelity, not generalization to cells unseen by the already-trained teacher.

## Late-time selection and evaluation

Reference preparation happens once, before the parallel student queue. It
propagates every frame-0 Cell4 pixel through the frozen teacher for all 115
frames and caches independent validation inputs/targets. This is a common
teacher/student `dopri5` protocol with rtol=atol=1e-5 and matched chunks of 512,
not a claim of bitwise equality to the legacy full-batch plot's default solver
tolerances. The checkpoint, mask, native code and eight displayed microscopy
images are hashed; reference/cache/source identities are checked on resume.

Each checkpoint-selection score is the worst of global-interpolant, late-time
teacher-ODE-state (frames 93–115), and final-frame teacher-ODE-state velocity
MSE, normalized by the global validation teacher velocity energy. It does not
use real-mask EMD. Each run logs the separate errors so late drift cannot be
hidden by a good global mean. This selection rule does not guarantee a good
student rollout; final rollout metrics measure that separately.

At the end of each completed run the selected student propagates **every
frame-0 pixel**, saves all 115 ODE snapshots, and creates:

- `eval/student_overlay.png`: 2×4 microscopy panels, cyan actual masks and red
  student predictions, matching the existing checkpoint-overlay style.
- `eval/comparison_overlay.png`: teacher and student rows at the same eight
  frames, initialized from the same particles.
- `eval/per_frame_metrics.csv`: paired teacher/student position RMSE at every
  frame, plus EMD at displayed frames; frame 115 is always included.
- `eval/metrics.json`: separate final-frame and late-time errors, teacher
  references, validation errors and function-evaluation counts.

Overlay frame numbers are 1, 17, 33, 49, 66, 82, 98, 115. Overlay particle
counts are full-mask. **EMD is W1 in pixel units on fixed deterministic
subsamples of at most 256 points**, not an exact full-mask EMD. Teacher and
student use identical metric particle indices. This keeps full-mask overlays
while avoiding enormous optimal-transport matrices per student. Increase
`--metric-samples` deliberately to change that reference protocol.

## Outputs and tracking

All results go outside the code checkout to
`/data/lthung/flow2pc/cell_tracking/distillation_sweep`:

```text
reference/{cache.pt,metrics.json,teacher_overlay.png,status.json}
configs/<configuration>.json
runs/<configuration>/checkpoints/{best.pt,last.pt}
runs/<configuration>/eval/{student_overlay.png,comparison_overlay.png,predictions.pt,metrics.json,per_frame_metrics.csv}
logs/<configuration>.log
summary.csv
selected_student.json
sweep_manifest.json
sweep_status.json
```

W&B project is **flow2pc**. Names include `cell-tracking`, latent count, time
features, LR, batch, seed and update count. Every completed run uploads its
student/teacher comparison images, metrics, config, checkpoints and ODE
predictions. Restart the same command to skip complete runs and resume partial
ones, including optimizer, scheduler, RNG and the late-time selection score.
Changed code/data/settings require a new output root. Ctrl-C stops worker
process groups cleanly. A reference-preparation failure prevents all student
launches; one failed student does not cancel the others.
