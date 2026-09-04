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
