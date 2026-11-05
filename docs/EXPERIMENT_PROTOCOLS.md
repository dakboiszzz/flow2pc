# Dataset benchmarks and student selection

Dataset conventions are independent of the teacher method. The references are
[ALI-CFM, §4 and Appendix D](https://arxiv.org/html/2510.01159) for single-cell,
ST, knot and tracking, and [OTP-FM, §6.4 and Appendices H–I](https://arxiv.org/html/2606.05327v1)
for GOM/Beijing. Native teacher objectives stay method-specific. These papers
do not define a PC velocity-distillation experiment; the student selection rule
below is our explicitly recorded protocol.

## Selection and diagnostics

The only optimized student loss is student–teacher **velocity MSE**. The best
student minimizes validation velocity MSE; population distances never select it.
Final MFM/MMFM evaluation reloads `checkpoints/best.pt`, records its step/path,
and keeps `last.pt` as the latest resumable training state. Resuming into another
directory copies the matching old `best.pt`; missing selected weights or changed
learning rate/training settings cause an error rather than silently resetting
selection or retaining an unintended optimizer learning rate.

`distillation.validation_scope` accepts:

| Value | Validation queries |
| --- | --- |
| `dataset` (default) | GOM/Beijing: `heldout_times`; other datasets: `observed_rows` |
| `observed_rows` | Reserved rows at observed times, using the teacher's query sampler |
| `heldout_times` | Whole left-out populations queried at their recorded model times |

For temporal selection, a seeded subset of at most
`validation_batches * validation_batch_size` rows is used per held-out time.
Time points receive equal weight. GOM/Beijing need no additional validation-row
split. Their holdout scores are **validation**, not an untouched final test after
checkpoint selection. Observed-row selection leaves single-cell/ST left-out
populations for benchmark evaluation. Existing ALI tracking late/final velocity
checks are retained; their selection score is a normalized maximum velocity MSE.

The saved dataset still owns its split indices and input preprocessing. Neither
selection mode adds held-out cells to gradient updates. A resume cannot silently
change selection policy. `validation_population_metrics: false` turns off the
additional population diagnostics, while preserving velocity-MSE selection.

Both prediction modes are logged under separate names:

- `student_ode_*`: integrate the learned student velocity from observed source cells.
- `direct_pc_*`: sample the student's distribution at the requested time.
- `teacher_ode_*`: the frozen teacher, using the same ODE source/reference protocol.

Direct samples are diagnostic only. Their RNG scope is independent of the ODE
particles and training RNG. A numerical failure is recorded for that mode; it
does not discard valid results from another mode. Programming/contract errors
still fail clearly. Evaluation batches particles (default 128) and bounds ODE
function evaluations (`evaluation.max_nfe`, default 20,000 per batch).

## Population metrics

`emd` means uniform-mass **W1**, with full-dimensional Euclidean costs, exact
POT transport optimization and no square root. Particles may be rematched by
the optimal transport plan. There is no sliced/projection approximation.
`evaluation.projections` is accepted for old config compatibility but is unused.

For GOM/Beijing, `w2` is additionally reported and named as the primary paper
distance: the square root of OT with **squared** Euclidean costs. To match the
released OTP-FM helper, its reference and prediction are truncated to equal
length after seeded subsetting. W1 also remains available for common diagnostics.
W1, W2 and W2-squared must not be compared as interchangeable scores. OTP-FM's
teacher sweep uses average W2-squared selection; our PC selection remains MSE.

`evaluation.metric_space: dataset` uses benchmark whitening for 5D single-cell
representations and raw feature coordinates for higher-dimensional single-cell,
ST, tracking, knot and transport data. The fixed 5D reporting scaler is computed
from the benchmark populations, not fitted to predictions. It is separate from
the teacher's saved input scaler. `raw` and `teacher` explicitly override these
reporting coordinates; the chosen units and policy are saved in `evaluation.json`.
Plots use raw coordinates; distances always use all features, even when a plot
shows only the first two.

ALI-CFM single-cell/ST held-out evaluation pushes the **preceding observed
population** to the left-out time. It is distinct from integrating every particle
from the earliest time. Shared benchmark metrics and the plotted held-out panels
use the former. Complete earliest-time trajectories are also saved as diagnostics.
Transport uses earliest-time integration, reporting held-out and noninitial
observed times separately. Saved per-time entries mark holdouts and source indices.

## Outputs and logging

MFM/MMFM student outputs remain at `output.dir`:

```text
checkpoints/{best,last}.pt
metrics.csv
metrics.jsonl
evaluation.json
trajectories.npz
comparison.png
media.json
tracking_status.json
```

Validation diagnostics are in `metrics.jsonl`/W&B and identify the training step.
Numerical failures retain their messages locally and emit a failure count.
MFM/MMFM student presets enable final evaluation and W&B project `flow2pc`.
Set `wandb.mode: offline|disabled` explicitly when desired. Shared distillation
logging falls back to offline/local storage on W&B communication failures;
missing W&B SDK support also permits local-only logging. Native delegated MFM
teacher training retains the upstream trainer's logging implementation.

ALI outputs retain historical ODE metrics and dataset-specific figures under
`eval/`. The shared metrics for **both** PC and ODE modes are under
`eval/population/evaluation.json`, with `comparison.png` and `media.json` alongside.
Historical earliest-time rollout scores are not relabeled as preceding-time
benchmark scores. Both figure sets are linked from `eval/media.json` and logged
when W&B is available.

## Coupling and external source checkouts

ALI queries use the frozen generator and recover coupling from the checkpoint's
**CFM training settings**, ahead of old profile-only sampler tags:

| Native `cfm_ot` | Distillation endpoints |
| --- | --- |
| `none` | Independent source/target samples |
| `border` or `full` | Endpoint minibatch OT, matching `train_ot_cfm_interpolant` |
| `mmot` | OT chain through the observed marginals |

ST uses endpoint OT for `border`/`full`, otherwise independent endpoints; tracking
uses its three-marginal coupling for `mmot`, and endpoint OT for `border`/`full`.
Knot retains independent endpoints as required by its native trainer. Explicit
`sampler.params.coupling` must agree with known training metadata. Unknown
metadata requires an explicit original coupling; old explicitly recorded
`mmot_middle` artifacts remain readable. Legacy loaders may supply a historical
profile default when original CFM metadata is absent; verify that setting against
the original run and set `sampler.params.coupling` explicitly if necessary.
This compatibility fallback cannot prove the original coupling. MFM queries its frozen geopath; MMFM
reuses its checkpoint's interpolant, coupling and noise settings.

Paths are configurable via `FLOW2PC_MFM_PRETRAIN_ROOT`, `FLOW2PC_MFM_ROOT` and
`FLOW2PC_MMFM_ROOT`, or explicit source-root config parameters. Presets no longer
override those defaults with a nonexistent relative checkout. No source, data
or checkpoint is downloaded. Missing MMFM source modules fail during preparation.

## Limits of alignment

Existing prepared MFM/MMFM snapshot recipes use observed-row splits and train-only
input standardization. Their preparation mathematics has **not** been changed.
The native ALI/OTP dataset pipelines can use different input normalization/split
conventions. Reusing a pretrained teacher requires preserving its saved coordinates;
changing reporting metrics does not establish identical training preprocessing.

The current MMFM presets are unconditional adaptations using native MMFM
interpolants, including adaptive Gaussian noise. ALI-CFM's reported MMFM
single-cell baseline instead uses cubic interpolation without that Gaussian path
and reports 5D results. The current general presets therefore are not a claim of
reproducing that baseline. Teacher-specific hyperparameters, raw dataset protocol
differences and runtime numerical parity require separate verification; do not
interpret shared metric names as proof of paper reproduction.

Regression cases are provided in `tests/test_distill_selection_and_metrics.py`
and the existing validation tests. No training, evaluation or tests were run as
part of these edits; only static checks are authorized in this session.
