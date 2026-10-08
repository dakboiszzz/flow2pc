# Flow2PC

Flow2PC distills a pretrained flow-matching model into a time-indexed
probabilistic circuit (PC). The PC provides direct samples and exact
log-likelihoods at every time along the learned path.

## Method

For a data sample `x`, Gaussian noise `z0` (independently or minibatch-OT
paired), and time `t`:

```text
z_t = (1 - t) z0 + t x
L = E ||u_pc(z_t, t) - u_flow(z_t, t)||²
```

The PC uses Gaussian leaves whose means and log-scales are polynomial in time.
Their parameterization gives `p_0 = N(0, I)`. Time-independent sum weights
combine the leaf velocities into a valid PC velocity field.

Three command-selectable structures share the same API:

| `--pc-structure` | Structure |
|---|---|
| `hclt` | Hidden Chow--Liu tree; `--pc-layer dense` or `monarch` |
| `pd` | Poon--Domingos-style hierarchical image region graph |
| `gaussian` | shallow mixture of Gaussian product nodes (control) |

`hclt --pc-layer dense` is the primary MNIST baseline. The published width 256
and topology settings (`x=20000`, 64 bins, `sigma=0.02`, chunk size 32) come
from the strongest HCLT configuration in the official
[PyJuice MNIST benchmark](https://github.com/liuanji/pyjuice-benchmarks/blob/main/configs/model/hclt_256.yaml).
The topology is learned once with PyJuice's published Chow--Liu estimator and
cached under `data/MNIST/hclt/`. `hclt --pc-layer monarch` uses the official
three block-diagonal sum layers (a composition of two Monarch matrices) from
[MonarchCircuits](https://github.com/wangben88/MonarchCircuits); its published
default width is 1024. The flow-path adaptation is isolated to the emissions:
categorical image leaves become time-conditioned Gaussian leaves, while each
paper's hidden tree and sum-layer connectivity are retained.

Student runs default to `--capacity-profile standard`. The comparison uses only
power-of-two widths from conventional PC capacity grids. Each student must
exceed the teacher, and no width is selected from downstream performance:

| Student | Standard width | Parameters | Difference from teacher |
|---|---:|---:|---:|
| Dense HCLT | 64 latents | 3,608,640 | +235.59% |
| Monarch HCLT | 64 latents | 1,604,160 | +49.18% |
| Spatial PD | 8 channels | 1,570,824 | +46.08% |
| Gaussian control | 256 components | 1,605,888 | +49.34% |

Widths 8, 64, and 256 are all standard powers of two. Dense HCLT-32 has
1,002,528 parameters, slightly fewer than the 1,075,361-parameter teacher, so
the next literature-grid width is HCLT-64. Its 3.36x capacity is reported as an
explicit exception; using 33 or 40 would create a tuned, nonstandard width.
Monarch-64, PD-8, and Gaussian-256 remain in a tight 1.46--1.49x band.

Use `--capacity-profile published` to recover dense HCLT-256,
Monarch-HCLT-1024, PD-64, or Gaussian-256 defaults. An explicit
`--num-latents`, `--num-channels`, or `--num-components` overrides either
profile. W&B records both parameter counts and their ratio.

### Required `t=1` endpoint sanity check

Before evaluating a continuous PC path, reproduce the standard categorical
MNIST HCLT endpoint:

```bash
bash scripts/run_published_endpoint_hclt.sh
```

This follows Table 2 and Appendix B of [Liu et al., ICLR
2022](https://starai.cs.ucla.edu/papers/LiuICLR22.pdf): `M=16`, categorical
256-way leaves, a Chow--Liu tree learned on all 60,000 training images after
retaining their three most-significant bits, and parameter learning on the
original 8-bit pixels. Training uses batch size 1024, 100 mini-batch EM epochs
with a linear `0.15 -> 0.05` step-size schedule, then 20 exact full-batch EM
epochs. No endpoint oversampling, early stopping, or cosine-restart schedule is
used. The run logs the likelihood curve, test diagnostics, throughput, memory,
samples, configuration, and final checkpoint to W&B.

The implementation contains 3,411,728 parameter tensors at `M=16` (the paper
reports 3.3M) and targets 1.26 theoretical test bpd. `M=24` (5.1M, 1.22 bpd) and
`M=32` (7.0M, 1.20 bpd) are capacity ablations, not defaults. The categorical
M=16 model is close in total size to the 3,608,640-parameter continuous HCLT-64
student because categorical emissions require 256 probabilities per latent
state. This endpoint model does not define a continuous velocity away from
`t=1`.

```python
pc.sample(num_samples, time)
pc.log_prob(values, time)
pc.sample_path(num_samples, times)
pc.path_log_prob(path_values, times)
```

`sample_path` keeps the circuit choices and Gaussian noise fixed across time,
producing a coupled trajectory through the PC marginals.

The teacher follows the official
[TorchCFM MNIST example](https://github.com/atong01/conditional-flow-matching/blob/main/examples/images/mnist_example.ipynb):
its `UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1)`, `sigma=0`,
Adam optimizer, batch size 128, and three training epochs. `--flow-matcher`
selects either the notebook's vanilla `ConditionalFlowMatcher` or
`ExactOptimalTransportConditionalFlowMatcher`:

| Value | Endpoint coupling |
|---|---|
| `vanilla` | independently sampled noise and MNIST images |
| `ot` | exact minibatch-OT pairing; default and active notebook configuration |

## Usage

### CITE-5D single-cell distillation

The single-cell workflow uses the existing ALI-CFM checkpoint and H5AD data
in place. It imports no code from the neighboring repository and writes only
inside Flow2PC. The default teacher is the combined day-3-held-out, seed-42
checkpoint under
`../ali-cfm/runs/single_cell/cite_5D/wandb/run-20261003_030825-rrcutrtl/files/checkpoints/t=1_seed=42_ali_cfm.pth`.
The data directory is `../ali-cfm/data/single_cell`.

Install the optional reader/metric dependencies in your environment:

```bash
python -m pip install -e ".[single-cell]"
```

For the six newly exported best teachers (CITE/Multi at 5D/50D/100D), use
the shared multi-GPU launcher below. It reads the bundle index and selects
only **t=1**, prepares a fresh Flow2PC teacher reference per dataset, and
interleaves all six student sweeps across the visible GPUs:

```bash
# Preview the six teachers, GPU pool, run counts, and W&B names; starts no jobs.
bash scripts/run_single_cell_all_sweeps.sh --dry-run

# Launch all six sweeps when ready. Use all visible GPUs, two workers per GPU.
bash scripts/run_single_cell_all_sweeps.sh

# Optionally restrict the logical GPU pool or increase concurrency.
bash scripts/run_single_cell_all_sweeps.sh \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 --workers-per-gpu 3
```

The wrapper uses `/data/nvidia/conda_envs/uffm/bin/python` unless
`FLOW2PC_PYTHON` is set. It reads
`/data/lthung/ali-cfm/best_models/single_cell` and the existing H5ADs in
`/data/lthung/single_cell`. It preserves each bundle's saved architecture,
global time, min/max normalization, and effective float32 5D whitening
constants. Teacher retraining regularizers are already embodied in the
frozen weights; they are not student loss terms.

The neutral `configs/single_cell/intensive_sweep.json` retains the existing
288-configuration grid: 32/64/128/256 latents, 16/32/64 time features,
four learning rates, batch sizes 64/128, three student seeds, and 50,000
steps. Six sweeps therefore contain **1,728 student runs**. Override
`--grid` or `--steps` before the first launch if desired. W&B names begin
with the dataset and dimension, such as `cite5d-h128-f32-lr0.001-b64-s0-...`;
groups also include fold and teacher seed. Existing W&B runs retain their
saved names on resume.

Outputs default to
`/data/lthung/flow2pc/outputs/single_cell/best_ali_cfm_sweeps_t1`, with
`reference/<dataset>-t1-teacher<seed>/metrics.json`,
`sweeps/<dataset>-t1-teacher<seed>/runs/`, per-sweep summaries and selections,
and `all_sweeps_status.json`. The default evaluation protocol uses 2,048
cells, seed 2026, batch size 256, and Dopri5 tolerance 1e-5 for both teacher
and student. Original ALI selection EMD values are not reused as references.

The scheduler requires at least two visible CUDA devices and has no CPU
fallback. It checks free memory before launching each worker (8 GiB by
default, adjustable with `--min-free-gb`), reserves space during worker
startup, and fills slots fairly across datasets. Increasing
`--workers-per-gpu` allows independent student processes to share a GPU;
actual utilization depends on available memory and other GPU jobs. GPU
allocation failures get one bounded retry by default. Rerunning the same
command skips completed students and resumes interrupted ones from saved
checkpoints. Ctrl-C or SIGTERM stops the launcher and its worker process
groups while preserving checkpoints. Change the output root when changing
teachers or evaluation protocol.

For a single older teacher, the workflow has two separate commands: evaluate the teacher first, then
launch student configurations across your GPUs. The sweep never evaluates
the teacher implicitly. Set the Python environment and authenticate W&B once:

```bash
export FLOW2PC_PYTHON=/home/nvidia/anaconda3/envs/uffm/bin/python
export WANDB_HTTP_TIMEOUT=120
export WANDB__SERVICE_WAIT=300
# Authenticate only if this machine does not already have saved credentials.
"$FLOW2PC_PYTHON" -m wandb login
# Optional: export WANDB_ENTITY=your-team

# 1. Evaluate the default CITE-5D teacher on one GPU and log its EMD.
"$FLOW2PC_PYTHON" scripts/evaluate_single_cell.py \
  --device cuda:0 \
  --output-dir outputs/reference/cite_5D/day3_seed42-wandb \
  --wandb-mode online
```

The evaluator saves EMD and projected marginal plots at all four times. Its
default protocol uses reproducible subsets of 2,048 cells, starts from day 2,
and integrates with Dopri5 at `rtol=atol=1e-5`. Use `--num-samples 0` for the
full populations; those scores are a different evaluation protocol and need
their own matching reference. The measured 2,048-cell teacher reference has
day-3 EMD **1.03575896**. This is a measured checkpoint reference, not the
paper's five-seed aggregate. If the output directory already contains a
matching reference, the evaluator reuses and logs it without recomputing or
overwriting `metrics.json`. This preserves the reference hash used by
student checkpoints. A different protocol requires a new output directory.

Run multiple student configurations concurrently:

```bash
# 2. Run the intensive sweep: 288 jobs, four running concurrently.
bash scripts/run_single_cell_sweep.sh \
  --grid configs/single_cell/cite5d_intensive_sweep.json \
  --reference outputs/reference/cite_5D/day3_seed42-wandb/metrics.json \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 --max-parallel 4 \
  --wandb-mode online \
  --output-dir outputs/single_cell/cite5d-intensive

# Inspect the jobs without evaluating or training:
bash scripts/run_single_cell_sweep.sh \
  --grid configs/single_cell/cite5d_intensive_sweep.json \
  --reference outputs/reference/cite_5D/day3_seed42-wandb/metrics.json \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 --max-parallel 4 --dry-run
```

For two GPUs use `--devices cuda:0,cuda:1 --max-parallel 2`; remaining
configurations queue until a worker finishes. These are independent runs
with one full student per GPU, following the ALI-CFM sweep's
`multi_gpu=false` pattern. `CUDA_VISIBLE_DEVICES` is respected: indices are
relative to your visible allocation. For example, with
`CUDA_VISIBLE_DEVICES=4,6`, request `cuda:0,cuda:1`.

`--devices auto` (the default) uses visible GPUs with at least 8 GiB free,
or falls back to CPU workers (`--devices cpu,cpu` selects CPUs explicitly).
`--threads` controls CPU threads per worker. The supplied reference is
validated against the teacher and data; final student evaluation inherits
its sample count, seed, solver tolerances, and batch size. Training and
final evaluation use the worker's assigned GPU; exact EMD runs on CPU
float64 arrays to avoid mixing GPU tensors with the OT solver.

Each configuration has its own log, checkpoints, and W&B ID. The launcher
resumes partially trained runs, skips completed and evaluated runs,
continues other jobs if one fails, and returns a failure status if any job
failed. Ctrl+C stops worker process groups. A directory lock prevents
two launchers from writing the same runs concurrently; input hashes
prevent a sweep directory from being reused with a different reference.
Use `--output-dir` for an independent experiment. W&B service startup and
initialization timeouts default to 300 seconds, following the ALI-CFM
startup fix; override with `--wandb-service-wait`.
W&B startup requests are serialized across workers, while training remains
concurrent. Fresh runs skip resume lookup; existing runs keep their saved ID.
Transient communication failures retry up to three initialization attempts
with backoff. Supported SDKs also get a separate 60-second GraphQL request
timeout and three GraphQL retries. Credential-verification RPCs get a
separate 120-second timeout. W&B sometimes wraps service-busy and network
timeouts as `AuthenticationError`; those are retried with a fresh worker
service. Rejected credentials and permission failures fail immediately;
logging stays online in `flow2pc` rather than silently switching modes.

The default grid is `configs/single_cell/cite5d_intensive_sweep.json`.
It expands every combination of these parameters:

| Parameter | Values |
| --- | --- |
| HCLT latent states | 32, 64, 128, 256 |
| Fourier embedding dimension | 16, 32, 64 |
| Learning rate | 1e-4, 3e-4, 5e-4, 1e-3 |
| Minibatch size | 64, 128 |
| Training seed | 0, 1, 2 |

This is **96 hyperparameter combinations, each repeated with three seeds:
288 runs**. Each run has 50,000 velocity-MSE distillation steps,
validation every 1,000 steps, and direct EMD
every 5,000 steps on 512 fixed cells. Final direct and ODE EMD inherit the
separate teacher reference's protocol. The full sweep has 14.4 million
distillation updates; four GPUs execute four jobs at a time,
equivalent to 72 groups of four jobs. Runtime depends on capacity, batch size,
GPU speed, and evaluation time. The queue varies all four capacities before
moving to another embedding dimension, learning rate, or batch size.

The original four-run grid remains available with
`--grid configs/single_cell/cite5d_sweep.json`. It uses 20,000 distillation
steps and is useful for a smaller experiment. Both formats are supported:
`common` plus an explicit `runs` list, or `common` plus a `parameters` object
whose lists form a Cartesian product. Edit the JSON or pass `--grid another.json`.
CLI overrides take precedence; duplicate configurations caused by an override
collapsing an axis are queued only once. Run names include the swept values
and a configuration fingerprint. Each training seed is a separate W&B run.
The intensive group's name is `cite5d-intensive` in project `flow2pc`.

For a bounded pipeline check, create a separate 128-cell teacher reference
first so the final evaluations use the same protocol:

```bash
"$FLOW2PC_PYTHON" scripts/evaluate_single_cell.py \
  --device cpu --num-samples 128 --wandb-mode disabled \
  --output-dir outputs/reference/cite_5D/smoke-128
bash scripts/run_single_cell_sweep.sh \
  --grid configs/single_cell/cite5d_sweep.json \
  --reference outputs/reference/cite_5D/smoke-128/metrics.json \
  --devices cpu,cpu --steps 30 --emd-every 0 \
  --wandb-mode disabled --output-dir outputs/single_cell/cite5d-smoke
```

The combined checkpoint supplies both networks: an ELU interpolant and a
SELU velocity teacher evaluated in global time. Inputs use the first five
released PCA columns, the saved StandardScaler parameters, and saved min/max
normalization; the loader validates source and manifest SHA-256 hashes.
Days `[2,3,4,7]` remain at pseudo-times `[0,1/3,2/3,1]`. Metric coordinates
reverse min/max scaling while retaining the standardization used by the
original 5D evaluator. Teacher preprocessing was fitted on all days,
including the held-out day; the saved metadata retains that convention.

Students have a free initial marginal, a Chow--Liu topology learned only from
observed training cells, and frozen teacher/generator targets. Equal-weight
exact minibatch OT couples the endpoints through an observed intermediate
population. Continuous times are uniform globally. The distillation objective
is **velocity MSE only**. Observed-marginal NLL is logged as a diagnostic and
does not contribute gradients or checkpoint selection. Training starts with
velocity MSE at step 1. Initial Gaussian means and scales are set directly
from observed day-2 training cells without optimizer updates.
Continuous-density NLL can be negative, whereas
the distillation MSE is nonnegative. The held-out day is excluded from updates.
Five percent of each observed population is reserved for student validation.
The best checkpoint and configuration minimize validation velocity MSE;
NLL and held-out EMD are never used for selection. Checkpoints record the
objective and reject resuming an earlier NLL-regularized training run.

Outputs under your sweep directory include one training log per job,
`runs/<config>/checkpoints/{best,last}.pt`, `metrics.csv`, `emd.csv`,
`monitor/step_*.json`, per-run `eval/metrics.json`, marginal plots,
`sweep_manifest.json` (expanded job settings), `summary.csv` (including
capacity, embedding dimension, learning rate, batch size, seed, and W&B links), and
`selected_student.json`. Student evaluation reports direct ancestral samples
separately from ODE predictions starting at the same empirical day-2 cells
as the teacher.

W&B defaults to **online** logging in project `flow2pc`. The intensive grid
uses group `cite5d-intensive`; the four-run grid uses `cite5d-distillation`.
The teacher has its own run. Each student's final
evaluator resumes the same online W&B ID as its trainer, so its losses,
EMD curves, and final scores appear together. W&B receives:

- Distillation loss, velocity MSE, observed NLL, gradient norm,
  learning rate, elapsed time, and validation scores.
- Teacher EMD for every day under `teacher/day_<day>/ode_emd`, plus
  `teacher/heldout_ode_emd`. The baseline is also recorded in every student
  run under `reference/*`.
- Direct-sampling EMD and NLL for every day every 5,000 steps, with
  `monitor/heldout_direct_emd` as the held-out curve. This uses 512 fixed
  cells by default; change `--emd-every` and `--emd-samples`, or set
  `--emd-every 0` to disable monitoring. Monitoring never affects selection.
- Final `eval/heldout_direct_emd` and `eval/heldout_ode_emd`, per-day EMD
  and NLL, teacher-relative differences/ratios, ODE evaluations, runtimes,
  marginal plots, and a CSV table of all populations. Final EMD uses the
  teacher reference's protocol (2,048 cells by default), separately from
  the smaller monitoring protocol.
- Saved configuration, data/checkpoint hashes, model checkpoints, metrics
  files, and plots as attachments in the run's **Files** tab; H5AD data
  remains in place.

EMD scalars are published before media. Tables are uploaded as CSV files
and checkpoints/metrics use `run.save()`: this avoids the extra public-API
credential verification triggered by W&B 0.30 table/artifact uploads, which
can fail with "the service process is busy" after a run initializes.

Logging uses [W&B custom step axes](https://docs.wandb.ai/ref/python/experiments/run/)
for training and final evaluation, so evaluating a best checkpoint
from an earlier training step does not discard its EMD. Use
`--wandb-project`, `--wandb-entity`, and `--wandb-group` on both commands
to choose the destination. `--wandb-mode offline` writes local W&B sessions
for later syncing (training/evaluation may be separate offline sessions);
`disabled` keeps only local metrics.

Install with Python 3.11 or newer:

```bash
python -m pip install -e ".[dev]"
```

Shell launchers use `python` by default. Set `FLOW2PC_PYTHON=/path/to/python`
to run them in a specific environment.

Train both teacher variants into separate run directories:

```bash
python train.py flow \
  --flow-matcher ot \
  --epochs 3 \
  --batch-size 128 \
  --output-dir outputs/runs/flow/torchcfm-ot-mnist

python train.py flow \
  --flow-matcher vanilla \
  --epochs 3 \
  --batch-size 128 \
  --output-dir outputs/runs/flow/torchcfm-vanilla-mnist
```

All training commands log online to the single W&B project `flow2pc` by
default. Use groups for experiment stages and configuration-specific run names
for comparison:

```bash
python train.py flow \
  --flow-matcher vanilla \
  --epochs 3 \
  --batch-size 128 \
  --wandb-group teacher-pretraining \
  --wandb-run-name teacher-vanilla-torchcfm-seed0 \
  --wandb-tags baseline \
  --output-dir outputs/runs/flow/teacher-vanilla-torchcfm-seed0
```

The trainer logs loss, learning rate, gradient norm, epoch progress, throughput,
and examples seen. W&B also records system metrics, the run configuration, a
source snapshot, and the final self-describing checkpoint as a versioned model
artifact. Use `--wandb-mode offline` for deferred syncing or
`--wandb-mode disabled` for local-only debugging.

Distill the pretrained vanilla teacher into the standard-width dense-HCLT
configuration. This first experiment deliberately has no endpoint `t=1` EM
pretraining:

```bash
python train.py distill \
  --teacher-checkpoint outputs/runs/flow/teacher-vanilla-torchcfm-seed0/checkpoints/last.pt \
  --pc-structure hclt \
  --pc-layer dense \
  --capacity-profile standard \
  --endpoint-pretraining none \
  --steps 20000 \
  --batch-size 64 \
  --checkpoint-every 5000 \
  --wandb-group student-distillation \
  --time-order 4 \
  --lr 0.001 \
  --optimizer adamw \
  --scheduler cosine \
  --weight-decay 0.01 \
  --gradient-clip 1.0 \
  --wandb-run-name hclt-dense-h64-vanilla-noem-step20000-seed0 \
  --wandb-tags baseline,hclt,standard-powers2,no-endpoint-em \
    --output-dir outputs/runs/distill/hclt-dense-h64-vanilla-noem-step20000-seed0
```

For the selected spatial-transcriptomics ALI-CFM folds, run both U3-held-out
(`t=1`, seed 45) and U4-held-out (`t=2`, seed 42) distillations with:

```bash
bash scripts/run_st_distill.sh
```

The launcher reads the normalized U2--U5 clouds and the bundled global-time
teacher/generator checkpoints from `outputs/ST`, excludes the fold's held-out
slide from the student minibatches, and writes Flow2PC checkpoints under
`outputs/runs/ST/`. `ST_STEPS`, `ST_NUM_LATENTS`, `ST_LR`, and `FLOW2PC_GPU`
can override its defaults. After each fold, it samples the student at all four
slide times, writes a three-panel overlay on the clean H&E TIFFs from
`outputs/ST/he_visualization_package/images/he_stains/`, and logs held-out EMD,
moment errors, endpoint NLL, and the figure to the same W&B run. The renderer
uses the package's Plotter convention: raw coordinate column 1 is horizontal
and column 0 is vertical.
The ST launcher also adds a `0.1` observed-marginal NLL anchor by default;
override it with `ST_MARGINAL_NLL_WEIGHT=...` if running a pure velocity-only
ablation. The evaluation reports relative velocity MSE in addition to raw MSE.

For the full intensive grid over both folds, use:

```bash
bash scripts/run_st_distill_intensive_sweep.sh
```

This runs 640 configurations total: 5 latent widths, 4 learning rates, 4
Fourier dimensions, 4 marginal-NLL weights, and 2 held-out folds. Use
`ST_LIMIT_RUNS` for a bounded pilot.

Architecture and endpoint coupling are independent command options. For
example, the same HCLT student can use an OT teacher checkpoint, and the
Monarch and spatial PD baselines require no code changes:

```bash
# Standard-width Monarch-HCLT (64 latent states, 8x8 blocks)
python train.py distill \
  --teacher-checkpoint /path/to/teacher/checkpoints/last.pt \
  --pc-structure hclt --pc-layer monarch \
  --capacity-profile standard \
  --endpoint-pretraining none --batch-size 64

# Poon--Domingos spatial baseline
python train.py distill \
  --teacher-checkpoint /path/to/teacher/checkpoints/last.pt \
  --pc-structure pd --capacity-profile standard \
  --endpoint-pretraining none --batch-size 64
```

Run the complete equal-budget vanilla/no-EM architecture benchmark (dense
HCLT-64, Monarch-HCLT-64, spatial PD-8, and Gaussian-256) sequentially on one
GPU:

```bash
bash scripts/run_standard_vanilla_noem.sh
```

The script waits until at least 12 GiB is free, evaluates the vanilla flow
teacher, then trains and fully evaluates each architecture. It selects the
first safe GPU automatically; set
`FLOW2PC_GPU=1` to pin a device, or `FLOW2PC_MIN_FREE_MB` only when the target
GPU's safe memory budget is known.

After those final evaluations, evaluate the round intermediate checkpoints and
build the consolidated CSV/Markdown Pareto table:

```bash
FLOW2PC_GPU=1 ./scripts/evaluate_vanilla_noem_checkpoints.sh
```

## Evaluation protocol

All comparable student runs use exactly **20,000 optimization steps**, batch
size 64, fourth-order time emissions, AdamW at `1e-3`, weight decay `0.01`,
cosine decay, and gradient clipping at `1.0`. Every training example receives
an independently sampled continuous time, matching standard flow-matching
training. These adaptation-specific choices are fixed round values because no
published optimizer recipe exists for velocity-distilling a flow into a PC.
The
trainer saves checkpoints at the round 5k, 10k, 15k, and 20k budgets so the
time--quality Pareto curve can be evaluated without selecting an irregular
stopping point. The earlier 30k dense-HCLT run is a pilot and is not included
in equal-budget architecture tables.

### Dense HCLT time-conditioning tuning (no endpoint training)

The follow-up dense-HCLT sweep uses the standard 32- and 64-state capacities
while keeping batch size 64, AdamW, cosine schedule, and the 20k-step budget
fixed. It compares the original fourth-order polynomial path with fixed
Gaussian Fourier feature layers of dimensions 8 and 16, then checks the
standard learning-rate grid
`{3e-4, 1e-3, 3e-3}` around Fourier-16. The Fourier construction follows the
sine/cosine feature form used by probabilistic integral circuits; centering its
cosine features at zero preserves the required exact standard-normal boundary
at `t=0`.

All candidates use `--endpoint-pretraining none`. There is no endpoint EM, NLL
fine-tuning, endpoint loss, or extra endpoint sampling. Hyperparameters and the
checkpoint among the round 5k, 10k, 15k, and 20k budgets are selected using a
fixed 3,000-image (5%) subset of the MNIST training split. The test set is only
reported after training and is not used for selection.

Run the consecutive sweep and selected full evaluation with:

```bash
bash scripts/run_dense_hclt_time_tuning_noem.sh
```

The live and final validation table is written to
`outputs/benchmark/dense-hclt-time-tuning/metrics.md`, and every train,
validation, likelihood, sample-quality, and performance metric is logged to the
single W&B project `flow2pc`.

### Data-endpoint timestep sampling ablation

Distillation uses `t=0` for Gaussian noise and `t=1` for MNIST data. The
default `--timestep-sampling uniform` retains the standard flow-matching
objective. To allocate more updates near the data endpoint without endpoint EM
or an endpoint likelihood loss, select a Beta distribution explicitly:

```bash
python train.py distill \
  --teacher-checkpoint outputs/runs/flow/teacher-vanilla-torchcfm-seed0/checkpoints/last.pt \
  --pc-structure hclt --pc-layer dense --num-latents 64 \
  --time-conditioning fourier --time-embedding-dim 8 \
  --timestep-sampling beta \
  --timestep-beta-alpha 1.0 \
  --timestep-beta-beta 0.8
```

`Beta(1.0, 0.8)` is the time-reversed form of the data-endpoint-biased
`Beta(0.8, 1.0)` training distribution reported by B-TTDM, whose convention
places data at `t=0`. The trainer records timestep statistics and histograms so
the realized allocation is auditable. Run the fixed 20k-step ablation and its
full evaluation with:

```bash
bash scripts/run_beta_timestep_ablation.sh
```

### Two-stage endpoint EM and flow distillation

Endpoint pretraining is independently controlled by
`--endpoint-pretraining {none,em}`. The `em` setting first fits the dense HCLT
at `t=1` with exact full-batch soft EM, writes
`endpoint_em_metrics.csv`, and saves `checkpoints/endpoint_em.pt`; the normal
velocity-distillation loop then starts from that model. Root and transition
probabilities use exact posterior circuit counts, while Gaussian means and
variances use their posterior sufficient statistics. The fixed time basis
continues to enforce `p_0 = N(0, I)`.

The defaults are 20 full-batch epochs and a 0.1 pseudocount, matching the
full-batch phase of the official
[PyJuice benchmark configuration](https://github.com/liuanji/pyjuice-benchmarks/tree/main/configs/optim).
The Gaussian standard-deviation floor is 0.01. Every setting is explicit and
command-configurable:

```bash
python train.py distill \
  --teacher-checkpoint outputs/runs/flow/teacher-vanilla-torchcfm-seed0/checkpoints/last.pt \
  --pc-structure hclt --pc-layer dense --num-latents 64 \
  --time-conditioning fourier --time-embedding-dim 8 \
  --endpoint-pretraining em \
  --endpoint-em-epochs 20 \
  --endpoint-em-pseudocount 0.1 \
  --endpoint-em-min-std 0.01
```

Run the controlled H64/Fourier-8 experiment and its standard full evaluation:

```bash
FLOW2PC_GPU=0 bash scripts/run_endpoint_em_then_distill.sh
```

Set `ENDPOINT_PRETRAINING=none` to disable the first stage without changing
code. The launcher intentionally keeps post-EM timestep sampling uniform so
endpoint EM is the only methodological difference from the no-EM baseline.

The primary PC endpoint metric is exact test bits per dimension (bpd) over all
10,000 MNIST test images. Gaussian leaf densities are integrated over each of
the 256 pixel quantization bins, with the edge bins absorbing the tails; this
produces a normalized discrete likelihood comparable to the bpd protocol used
by [Lossless Compression with Probabilistic Circuits](https://arxiv.org/abs/2111.11632)
and [Probabilistic Integral Circuits](https://proceedings.mlr.press/v238/gala24a.html).
Continuous log density is retained as a path diagnostic, but is never presented
as discrete image bpd.

Sample metrics compare 10,000 generated endpoint samples with the full 10,000
image test set. Following established domain-specific MNIST evaluation, the
fixed feature extractor is the
[official PyTorch MNIST classifier](https://github.com/pytorch/examples/blob/main/mnist/main.py),
trained with its published 14-epoch Adadelta recipe. Its 128-dimensional
penultimate activation defines FID-MNIST, KID-MNIST, and improved
precision/recall. The explicit `-MNIST` suffix prevents these numbers from
being confused with ImageNet-Inception metrics. The suite also reports
IS-MNIST, classifier confidence, class entropy/JSD, sampling and likelihood
throughput, parameter count, checkpoint size, training time, and peak
evaluation memory.

PC endpoint samples are one-shot ancestral samples from the learned marginal
at `t=1`; they do not use an ODE solver or a temporal discretization. The
continuous draws are clamped and quantized to MNIST's 256 pixel bins before
sample-quality evaluation. The same full 10k protocol is also run after
integrating the analytically derived PC velocity with 20-step Euler, 20-step
RK4, and adaptive Dopri5 (`rtol=atol=1e-4`). An ancestral coupled `sample_path`
panel is included as a separate qualitative trajectory; its endpoint has the
same marginal law as the direct draw. Every numerical sampler reports
throughput and function evaluations. Flow-teacher samples use adaptive
`dopri5` integration.

Qualitative artifacts include a fixed-seed 8x8 endpoint grid, coupled
trajectories at `t=0,0.25,0.5,0.75,1`, and generated/training-nearest-neighbor
pairs in the same fixed feature space. Every metric and image is logged to the
single W&B project `flow2pc` under the `standard-evaluation` group.

Train the fixed evaluator once:

```bash
python train_mnist_classifier.py \
  --data-dir data \
  --epochs 14 \
  --wandb-project flow2pc \
  --wandb-run-name eval-feature-extractor-pytorch-mnist-seed1
```

Run the complete standard evaluation:

```bash
python eval.py \
  --run-dir outputs/runs/distill/hclt-dense-h64-vanilla-noem-step20000-seed0 \
  --times 0,0.25,0.5,0.75,1 \
  --num-generated 10000
```

Each run contains `config.json`, detailed `metrics.csv`, and a self-describing checkpoint.
Evaluation writes JSON metrics, raw feature/sample tensors, and qualitative
figures to `<run>/eval/<checkpoint>/`. The checkpoint carries the selected
matcher configuration through distillation and evaluation, preventing
accidental mixing of vanilla and OT paths.

## Layout

```text
train.py           flow training and PC distillation
eval.py            path sampling and exact likelihood evaluation
train_mnist_classifier.py  fixed standard MNIST feature extractor
repc/flow.py       flow teacher
repc/pc.py         shared path API and Gaussian PC
repc/hclt.py       official HCLT topology learning and caching
repc/hclt_pc.py    dense and Monarch time-dependent HCLTs
repc/region_pc.py  Poon--Domingos-style spatial PC
repc/evaluation.py FID/KID, precision/recall, and class metrics
repc/mnist_classifier.py official PyTorch MNIST classifier architecture
repc/data.py       MNIST loader
```
