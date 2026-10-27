# Cell-tracking ALI-CFM teacher

This note records the checkpoint selected for a future Flow2PC distillation
port, its known provenance, available figures, and the adapter/data work that
Flow2PC will need. It documents the saved model; it does not claim a verified
paper reproduction.

## Selected checkpoint

The candidate selected from the user-reviewed `changed_files` model is:

```text
changed_files/ali_cfm_model.pth
SHA-256: d2c071a2b2fea1931c5c515d4dbc35e8f436043f58322fa0c8083b3503049129
```

This is the file directly under `changed_files/`, **outside**
`changed_files/latest-run/`. It is the checkpoint to retain for the planned
distillation handoff.

It is a combined ALI-CFM checkpoint with keys `interpolant`, `discriminator`,
`cfm_model`, `min_max`, and `config`. For velocity distillation, the relevant
weights are `cfm_model`; the CFM MLP has 2D coordinates, one time input, width
256, three SELU hidden layers, and a 2D velocity output. The saved coordinate
bounds are `min=[206, 260]`, `max=[484, 496]` in `(x, y)` pixel order.

The checkpoint's embedded training config is:

| Setting | Value |
|---|---:|
| Seed / samples per frame | 42 / 10 |
| Dataset normalization | global min-max; whitening off |
| Spatial dimension / hidden width | 2 / 256 |
| GAN | vanilla; batch 128; `lr_G=lr_D=1e-4` |
| ALI steps / time smoothing | 10,001 / 0.01 |
| Regularization coefficient | 10 in the saved trainer's loss reduction |
| CFM | batch 128; `lr_CFM=1e-3`; 10,001 steps |
| OT | border for interpolant and CFM |
| ODE evaluation | `dopri5`; 115 frames; 100 internal steps per frame interval |

The checkpoint's CFM was trained and evaluated on **normalized coordinates**.
For normalized position `z`, recover pixels using `x = z * (max-min) + min`.
If physical pixel velocities are needed, scale each coordinate by `(max-min)`;
the Flow2PC teacher target should remain in normalized coordinates when the
student is trained there.

The saved config says `regulariser: piecewise`, but the accompanying trainer
forms its GAN regularization loss directly as `MSE(xt_fake, (1-t)*x0+t*x1)`.
When porting that exact loss to this repository's linear penalty, which sums the
two squared coordinate errors before averaging over the batch, coefficient 5
matches coefficient 10 with the saved trainer's mean over batch and
coordinates. Keep the original coefficient 10 when using the saved friend's
trainer unchanged.

## Checkpoint and overlay files

The root checkpoint above is **not byte-identical** to the checkpoint archived
inside `changed_files/latest-run`:

```text
changed_files/latest-run/files/results/cell_tracking/gan100001_cfm100001_20261023_075825_566915422_cuda0/ali_cfm_model.pth
SHA-256: 76a94de29800838f47b3df6f06ad794b23a222650314c5006ea13736264f2d80
```

The archived copy has a different embedded config: 70,001 ALI steps,
70,001 CFM steps, and `correct_coeff=1`. Its W&B overlays and trajectory plots
belong to that archived run, **not** to the root 10,001-step, coefficient-10
checkpoint. Do not use the two checkpoint files interchangeably.

Available microscopy overlay frames from the archived W&B run:

- [Frame 1](../changed_files/latest-run/files/media/images/eval/overlay_frame_0_2919_dc5752a51bb034b18691.png)
- [Frame 23](../changed_files/latest-run/files/media/images/eval/overlay_frame_22_2920_0ecd7c38a29842ff38bb.png)
- [Frame 46](../changed_files/latest-run/files/media/images/eval/overlay_frame_45_2921_9ec54cc2d40e7807ecc9.png)
- [Frame 69](../changed_files/latest-run/files/media/images/eval/overlay_frame_68_2922_f4a98cf92273397f51ba.png)
- [Frame 92](../changed_files/latest-run/files/media/images/eval/overlay_frame_91_2923_374211f8541d512e626b.png)
- [Frame 115](../changed_files/latest-run/files/media/images/eval/overlay_frame_114_2924_40d5a3420bf90cd5e12c.png)

Available trajectory plots from the same archived run:

- [2D trajectory evaluation](../changed_files/latest-run/files/results/cell_tracking/gan100001_cfm100001_20261023_075825_566915422_cuda0/plots/trajectory_eval_seed42.png)
- [3D space-time trajectory evaluation](../changed_files/latest-run/files/results/cell_tracking/gan100001_cfm100001_20261023_075825_566915422_cuda0/plots/trajectory_eval_3d_seed42.png)

These saved frame overlays show the ten subsampled training points. The
trajectory plots show the archived run's predictions. I searched the checkout
and `/data/lthung/ali-cfm/cell_tracking` but found no saved all-particle,
Figure-4-style overlay tied by hash to the selected root checkpoint. The
earlier image shown in chat is not present as a file there. Sweep overlays are
from other checkpoints and should be treated as comparisons, not as
visualizations of this model.

## Flow2PC handoff

The adjacent `../flow2pc` checkout already has ALI-CFM teacher wrappers and
velocity-distillation infrastructure, but it does **not** yet support this
microscopy checkpoint/data contract directly:

1. `ALICFMTeacherWrapper` currently recognizes velocity weights under keys such
   as `ot_cfm_model` and `velocity_state_dict`; this checkpoint stores them as
   `cfm_model`.
2. Its default time mapping converts global time to an interval-local time.
   This cell-tracking CFM instead uses one continuous global time `t` in
   `[0,1]`; its wrapper must pass `t` through unchanged.
3. Flow2PC's current `train.py` dataset choices are `mnist`, `knot`, `ht206b1`,
   and `st`. There is no microscopy cell-tracking data loader yet. The existing
   single-cell HCLT workflow is for H5AD/PCA observations and expects a
   different teacher bundle.

The future adapter should load `checkpoint["cfm_model"]`, reconstruct the
2D/time-varying width-256 SELU MLP, freeze it, and expose
`teacher(t, x_normalized) -> dx_normalized/dt`. Use the 115-frame Cell4 mask
stack to sample observed positions and frame times `t=i/114`. Preserve the
saved min-max transform and coordinate order. Keep the ALI interpolant as a
separate optional artifact; the CFM vector field is the velocity teacher.

For student selection, log held-out velocity error and evaluate full
trajectories at the paper's displayed frames, including frame 115. Do not
select solely on mean-over-time error: prior runs with better mean scores still
showed late-frame drift. Current Flow2PC distillation code has not been modified
for this dataset.

## Related run records

- The 36-run ALI-CFM sweep outputs are under `/data/lthung/ali-cfm/cell_tracking/sweeps/`.
- The best mean full-mask diagnostic among our sweeps was from
  `20261023_082422_UTC/norm_ts0.02_rpgan_aliB96_lambda0.5_seed42`; it is a
  different model and drifts late in the sequence.
- The user's selected checkpoint remains a qualitative teacher candidate;
  the available logged W&B metrics/figures above are not evidence that the
  selected root checkpoint has been independently reevaluated under the
  current full-mask protocol.

## Prepared implementation update

The dataset-specific adapter, reference preparer, student worker and multi-GPU
launcher are now prepared in Flow2PC. See
[README_CELL_TRACKING_DISTILLATION.md](README_CELL_TRACKING_DISTILLATION.md).
`bash scripts/run_cell_tracking_sweep.sh` is the explicit launch command for
108 configurations; `--dry-run` prints the plan without checkpoint execution.
Preparing this implementation did not evaluate the selected real checkpoint
or launch production students. The older provenance/quality limitations above
still apply until its frozen reference is actually evaluated.
