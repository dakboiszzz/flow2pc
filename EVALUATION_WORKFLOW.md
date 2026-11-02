# Evaluation restored from the previous distillation workflow

The source audit used Flow2PC commit `f2c6942` and its archived files. Previously,
single-cell sweeps evaluated a teacher first, trained a student, loaded `best.pt`,
then ran the student evaluator and uploaded figures to the same W&B run.
Tracking did the evaluation and overlay generation inside its trainer before
marking a run complete. ST launchers called the H&E evaluator after training;
the old knot trainer invoked its trajectory plotter after training.

The organized pipeline now restores those evaluation/illustration steps through
`src/flow2pc/evaluation/`. Teacher training saves its restorable checkpoint before
evaluation. Student training reloads the selected `best.pt` before evaluation.
Both keep a persistent W&B run ID in `wandb_run.json`, so standalone evaluation
and resumed training attach results to the same run in project `flow2pc`.
An evaluation failure leaves the trained checkpoint available for a separate retry.

## Run the complete check

```bash
# Activate the existing CUDA/Python environment, or set FLOW2PC_PYTHON.
bash scripts/check_alicfm_pipeline.sh cite50
```

The defaults are:

```text
data:    /data/lthung/ali-cfm-data
outputs: /data/lthung/flow2pc/framework/migration-check/cite50/
         teacher/checkpoints/last.pt
         teacher/eval/
         student/checkpoints/{best,last}.pt
         student/eval/
```

This is a full-budget teacher/student run, not a short parity test. It starts
teacher training only if its checkpoint is absent. Completed teacher evaluations
and figures are reused; a saved teacher with missing evaluation is evaluated.
Existing students resume or regenerate their final evaluation.
No jobs are launched merely by creating these scripts.

All dataset tags accepted by `run_alicfm.sh` are supported. For example:

```bash
FLOW2PC_DATA_ROOT=/data/lthung/ali-cfm-data \
FLOW2PC_CHECK_ROOT=/data/lthung/flow2pc/framework/migration-check/st \
TEACHER_DEVICE=cuda:0 STUDENT_DEVICE=cuda:1 \
bash scripts/check_alicfm_pipeline.sh st
```

The two stages are dependent: the student starts after its teacher is ready.
This check script does not launch a parallel hyperparameter sweep.

For concurrent checks of all ten datasets, use the recovered shared GPU queue
described in [parallel launch instructions](docs/PARALLEL_CHECKS.md). Its local
recipe lives in the Git-ignored `local_runs/` folder.

## Figures and metrics

Every evaluation writes `metrics.json`, `marginal_metrics.csv`, `media.json`,
cached teacher predictions, and an evaluation status file. Student evaluations
also save teacher/student ODE paths in `predictions.pt`. `media.json` contains
absolute paths to the generated images.

| Dataset | Teacher figure | Student figure | Comparison figure |
| --- | --- | --- | --- |
| CITE/Multi/EB | `teacher_ode_marginals.png` | `student_ode_marginals.png` | `comparison_marginals.png` |
| Cell tracking | `teacher_overlay.png` | `student_overlay.png` | `comparison_overlay.png` |
| ST, one held-out slide | `teacher_overlay.png` | `student_overlay.png` | `st_he_overlay.png` |
| Knot | `teacher_trajectories.png` | `student_trajectories.png` | `comparison_trajectories.png` |

Tracking uses the original display frames: 1, 17, 33, 49, 66, 82, 98 and 115.
ST renders the original `(y,x)` coordinates on clean aligned H&E TIFFs. The data
root supplies backgrounds through these relative paths:

```text
cell_tracking/PhC-C2DH-U373/01/t000.tif ... t114.tif
spatial_transcriptomics/aligned/ref_U5_warped_images/U2_aligned_to_U5.tif ...
```

Override `evaluation.image_dir` for a different image package. Missing images
fail before training rather than producing an unrelated fallback figure.

Evaluation uses **velocity-driven ODE rollouts from observed source cells**.
It does not draw direct PC samples and introduces no extra student training loss.
Teacher and student use identical starting cells, model times and solver settings.
The teacher reference is reused only when its teacher/data/protocol binding
matches. Scalar EMDs, marginal tables, images, prediction arrays and artifacts
are uploaded to the configured W&B run. Local copies remain in `eval/`.

The old single-cell EMD formula is retained: exact uniform-mass W1 with Euclidean
costs in the full feature dimension, computed on deterministic subsets. High
dimensional figures display only the first two metric coordinates. Source and
reference counts, seed, solver, units and split scope are recorded in metrics.
The examples use the previous all-population evaluation scope; set
`evaluation.reference_split=test` for held-out-row metrics. Held-out observations
never enter optimization or checkpoint selection.

For whitened single-cell representations, historical EMD units undo min-max
scaling while retaining the original feature standardization. Newly trained
artifacts save the extra metric-space transform; legacy imports retain their
saved whitening statistics when present. Missing historical statistics cause a
clear error; `--set evaluation.metric_space=raw` explicitly selects raw-feature
EMD instead. Those numbers must not be mixed with standardized EMDs.

The earlier ST evaluator loaded precomputed teacher arrays without verifying
their initial particles/rollout protocol. The new ST comparison integrates both
fields from the same U2 cells, following the old student visualization convention.
This differs from the original ST teacher paper-score calculation initialized
from the observation immediately before the held-out slide. Match protocols
before comparing against the old `st_ali_cfm_emd_results.csv`.

## Existing checkpoints without training

For the paths specified in the example YAMLs:

```bash
bash scripts/run_alicfm.sh evaluate-teacher cite50
bash scripts/run_alicfm.sh evaluate-student cite50
```

For check-script outputs, override the paths explicitly:

```bash
bash scripts/run_alicfm.sh evaluate-teacher cite50 \
  --set output.dir=/data/lthung/flow2pc/framework/migration-check/cite50/teacher
bash scripts/run_alicfm.sh evaluate-student cite50 \
  --set output.dir=/data/lthung/flow2pc/framework/migration-check/cite50/student \
  --set teacher.checkpoint=/data/lthung/flow2pc/framework/migration-check/cite50/teacher/checkpoints/last.pt
```

For a legacy teacher, use its explicit metadata options documented in
`docs/ALICFM_DATASETS.md` with `--evaluate teacher` on the distill command. Its
reference goes in a sibling `*-teacher-reference/` directory. Old student
checkpoint schemas are not automatically converted; student evaluation here
uses the new migrated checkpoint format and verifies its teacher/data binding.

Dataset roots and output directories are configurable. The example YAMLs and
check launcher contain machine-specific defaults, while the loaders receive the
configured root and append dataset-relative paths. Direct example configs have
an explicit `data.root`, so change it with `--set data.root=...`; the environment
fallback applies when that key is omitted. The check launcher explicitly honors
`FLOW2PC_DATA_ROOT` and `FLOW2PC_CHECK_ROOT`.

The added code and commands have received static checks only. No training,
distillation, evaluation, figure generation or tests were executed during this work.
