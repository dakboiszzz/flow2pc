# Saved ALI-CFM teacher results review — 2026-10-30

## Follow-up: restored native knot run completed

The user's run at
`/data/lthung/flow2pc/framework/pretrain/alicfm/knot/native-restored-seed42`
completed 50,000 GAN and 30,000 CFM updates and its post-training evaluation.
Its saved checkpoint SHA-256 matches the manifest. Read-only inspection of the
saved results finds mean EMD **0.12751303** and final EMD **0.03241096**, compared
with **19.62875760** and **99.79487342** for the earlier migrated run. The saved
evaluation protocol dictionaries, including the source-particle hash, are equal.
The restored figure follows the lifted loop and reaches the target endpoint,
with residual shape differences. This supports the training-path restoration;
the shared RK4 evaluator was not changed between these two runs. No further
training or evaluation was launched to obtain this comparison.

W&B: https://wandb.ai/pqhhust-hanoi-university-of-science-and-technology/flow2pc/runs/ec10b64605bb419f

The original review below concerns the earlier migration-check artifacts.

This review reads the existing results under
`/data/lthung/flow2pc/framework/migration-check`. No model inference, ODE
integration, training, evaluation command or test was executed. All nine completed
teacher figures were visually inspected, together with existing CITE50/Multi50
reference plots. The automatic launcher remains suspended; the existing
cell-tracking teacher continues independently.

**Assessment:** the six CITE/Multi teachers produce broadly comparable held-out
scores to the earlier selected teachers, but several later marginals are worse.
Knot has a clear failed saved rollout. EB misses substantial intermediate
population structure. ST captures several clusters with visible mismatches.
Tracking is unfinished. These results do not establish a faithful migration,
and they do not justify discarding every completed checkpoint either.

## Six single-cell datasets

Both columns below use existing 2,048-particle Flow2PC evaluations of the held-out
day 3. The earlier reference uses the previously selected teacher; every new run
uses teacher seed 42. Lower EMD is better within a row.

| Dataset | Earlier teacher seed | Earlier held-out EMD | New held-out EMD | Relative change |
| --- | ---: | ---: | ---: | ---: |
| CITE5 | 43 | 0.93718 | 0.95248 | +1.63% |
| CITE50 | 42 | 41.82944 | 42.52268 | +1.66% |
| CITE100 | 44 | 48.09459 | 47.86457 | -0.48% |
| Multi5 | 42 | 1.04367 | 1.00846 | -3.37% |
| Multi50 | 43 | 45.97033 | 47.90084 | +4.20% |
| Multi100 | 45 | 52.02818 | 52.94936 | +1.77% |

These are descriptive comparisons, not parity tests: checkpoint seeds differ
in four rows, preprocessing/training exposure changed, the chosen cells differ,
and ODE batch size changed from 256 to 64. At 5D the historical metric retains
whitening, whose fitted statistics also changed. Scores across dimensions are
not directly comparable. Earlier teachers were selected using held-out scores
among seeds 42–46, rather than constituting an unbiased seed-matched control.

Day 3 alone hides later discrepancies:

| Dataset | Earlier day-4 EMD | New day-4 EMD | Relative change |
| --- | ---: | ---: | ---: |
| CITE50 | 40.58992 | 44.82497 | +10.43% |
| Multi50 | 60.58148 | 72.78216 | +20.14% |
| Multi100 | 62.21019 | 70.48201 | +13.30% |

The Multi50/100 plots show the predicted day-4 population displaced toward
positive coordinate 1. The earlier Multi50 plot already shows a displacement;
the saved metrics indicate a larger discrepancy in the new result. The precise
increment cannot be attributed to a particular migration change from these
uncontrolled comparisons. CITE plots broadly follow the changing populations,
but show differences in spread, tails and later-time locations.

The figures display only the first two supplied feature coordinates. EMD uses
all 5/50/100 features; visually similar first-two-coordinate plots cannot certify
full-dimensional agreement.

For reference only, the original native **full-population** day-3 scores for seed
42, read from the original runs' W&B summary JSON, are:

| Dataset | Native seed-42 full-population EMD | New seed-42 2,048-particle EMD |
| --- | ---: | ---: |
| CITE5 | 0.92905 | 0.95248 |
| CITE50 | 41.05244 | 42.52268 |
| CITE100 | 48.55606 | 47.86457 |
| Multi5 | 1.02371 | 1.00846 |
| Multi50 | 45.31132 | 47.90084 |
| Multi100 | 52.48024 | 52.94936 |

These columns intentionally identify their different population protocols. Their
differences are not measurements of migration error.

## ST, EB, knot and tracking

| Dataset | Saved teacher result | Interpretation |
| --- | --- | --- |
| ST | Held-out U3 EMD 124.23619; mean noninitial EMD 110.61031 | Partial spatial agreement; reproduction unverified |
| EB PHATE2D | Held-out EMD 1.03976; mean noninitial EMD 0.71647 | Visible intermediate-population mismatch; adapted recipe |
| Knot | Mean EMD 19.62876; final EMD 99.79487 | Failed saved rollout; cause unresolved |
| Cell-tracking | GAN approximately 21,300 / 70,000 updates at inspection; CFM not started | No final checkpoint/evaluation yet |

ST's U3 overlay places several clusters near the observed tumor regions, including
the large right-hand region. It also has visibly different cluster locations and
mass allocations. EMD is in aligned raw image coordinates. No matched native ST
evaluation was located in the checked output/reference directories, so this is
not evidence that ST became worse by a known amount.

EB's endpoint looks substantially better than its intermediate times: final EMD
is 0.26279, while day 12–15 EMD is 0.95525. The full 2D PHATE figure shows missing
upper-left structure and excess mass on the right at intermediate times. The
small final number must not be used to label the whole time-course successful.
This EB recipe was already identified as an adaptation, not a reproduced released
experiment.

Knot diverges visibly from the target curve. Reading the cached trajectory tensor
`teacher/eval/teacher_predictions.pt`, without running the field again, confirms:

- Ten source particles and 120 saved times;
- Final x range approximately **[-2.39, 131.35]**;
- Final y range approximately **[-146.18, 93.09]**;
- **Nine of ten** final particles outside the saved plot's x=[-5,5], y=[-3,3]
  limits;
- Saved population EMD first exceeds 5 at model time approximately 0.5462.

The fixed plot limits conceal most final particles. The only visible purple
endpoint therefore understates the extent of the failure. No new figure was
generated and no plotting code was changed. The existing RK4 result establishes
failure under that saved protocol; it does not yet distinguish training failure,
off-path velocity behavior or integration sensitivity. That needs a matched
native comparison and an explicit numerical check before any new training.

Follow-up formula check: the intended lifted middle segment is
`y2 = sin(2*pi*(t2 - 0.75)) + 1`. It appears in the supplied root teacher
trainer, the upstream knot distribution, the older Flow2PC data generator and
the current shared knot generator used for both teacher training and evaluation.
The failed checkpoint records data-generator source SHA-256
`01b32393a188bbbe7b3e24a1f6f5b85231e57efc8a0e5fa910219f04501c5512`, matching
the current source containing that lift. A missing `+1` is therefore not the
observed training/reference mismatch.

The archived `artifacts/checkpoints/teacher.pt` and `generator.pt` under the
20261028 cleanup archive are also available as potential controls. Read-only
checkpoint inspection finds the same declared global-time convention and network
layer shapes as the new knot teacher: velocity width 32/two hidden layers and
interpolant width 128/two hidden layers. Their original numerical performance
was not re-evaluated here. The next diagnostic must distinguish a poor learned
interpolant, CFM field error/instability away from interpolant queries, and ODE
numerical error; existing scalar logs alone do not identify which caused the
divergence.

## Recorded training behavior

All completed datasets have their configured number of recorded updates. No
nonfinite numeric values were found in the existing teacher metrics JSONL files.
The CFM loss averages over the first and last 100 recorded updates are:

| Dataset | CFM updates | First-100 mean loss | Last-100 mean loss |
| --- | ---: | ---: | ---: |
| CITE5 | 2,001 | 0.07038 | 0.03960 |
| CITE50 | 2,001 | 1.75684 | 0.70132 |
| CITE100 | 2,001 | 3.03357 | 1.73251 |
| Multi5 | 2,001 | 0.06352 | 0.03482 |
| Multi50 | 2,001 | 1.81286 | 0.73796 |
| Multi100 | 2,001 | 3.32246 | 2.00098 |
| ST | 10,000 | 0.01377 | 0.00603 |
| EB | 2,001 | 0.04570 | 0.00578 |
| Knot | 30,000 | 75.72978 | 0.23049 |

These are the original logged training losses, not newly computed validation
losses. Coordinate scales and loss reductions vary by native teacher profile;
their absolute magnitudes should not be compared across datasets. In particular,
knot's large loss reduction accompanies a failed rollout, demonstrating why
training loss alone is insufficient. Finite scalar logs also do not establish
that every gradient or trajectory was stable. ST GAN discriminator/generator
losses near their vanilla-GAN equilibrium values do not, by themselves, prove
either successful population matching or failure.

## Acceptance decision and next verification

Keep every completed checkpoint and its original preprocessing metadata. Do not
relabel the completed experiments as faithful native reproductions. The primary
priorities are knot's failure and the later-time Multi50/100 discrepancies;
ST/EB need appropriately matched reference evaluations rather than acceptance
based on completion status or a visually plausible endpoint.

The next useful check is the same *existing native checkpoint* through native
and migrated inference/evaluation with identical preprocessing, particles, time
grid and numerical settings. It separates evaluation/adapter issues from training
changes without first repeating days of training. Then compare the training
protocol and stage updates as described in
[ALICFM_MIGRATION_AUDIT.md](ALICFM_MIGRATION_AUDIT.md). No such evaluation was run
as part of this read-only review, and no new student work was started.

## Existing figures and source artifacts

For each CITE/Multi/EB example, the inspected figure is
`/data/lthung/flow2pc/framework/migration-check/<dataset>/teacher/eval/teacher_ode_marginals.png`.
The corresponding `metrics.json` and `teacher/metrics.jsonl` contain the scores
and training records above.

- ST overlay:
  `/data/lthung/flow2pc/framework/migration-check/st/teacher/eval/teacher_overlay.png`
- Knot teacher trajectory:
  `/data/lthung/flow2pc/framework/migration-check/knot/teacher/eval/teacher_trajectories.png`
- Earlier single-cell 2,048-particle references:
  `/data/lthung/flow2pc/outputs/single_cell/best_ali_cfm_sweeps_t1/reference/<dataset>d-t1-teacher<seed>/metrics.json`
- Original selected-teacher metadata and source-run locations:
  `/data/lthung/ali-cfm/best_models/single_cell/<dataset>_<dim>D/t=1/metadata.json`

This review changes only this documentation file. Existing jobs, configs,
checkpoints, results and plotting implementations were left untouched.
