# Flow2PC

Flow2PC compiles a frozen flow-matching model into a time-indexed
probabilistic circuit (PC). The PC supports exact likelihoods, marginal and
conditional queries, direct sampling, and trajectory sampling.

## Method

For endpoint data `x`, Gaussian noise `z0`, and time `t`:

```text
z_t = (1 - t) z0 + t x
L_vel = E ||u_pc(z_t, t) - u_flow(z_t, t)||²
L = L_vel + lambda_den L_den
```

The PC has time-conditioned Gaussian leaves and satisfies
`p_0 = N(0, I)`. Its common interface is:

```python
pc.log_prob(values, time)
pc.sample(num_samples, time)
pc.path_log_prob(path_values, times)
pc.sample_path(num_samples, times)
```

## Installation

Python 3.11 or newer is required.

```bash
python -m pip install -e ".[dev]"
```

Launchers use `python` by default. Override it when needed:

```bash
FLOW2PC_PYTHON=/mnt/disk1/aiotlab/envs/uffm/bin/python bash <script>
```

## Reference configurations

### MNIST teacher

The teacher follows the official
[TorchCFM MNIST example](https://github.com/atong01/conditional-flow-matching/blob/main/examples/images/mnist_example.ipynb).

| Setting | Value |
|---|---|
| Model | `UNetModel`, 32 channels, 1 residual block |
| Flow noise | `sigma=0` |
| Optimizer | Adam |
| Batch size | 128 |
| Training | 3 epochs |
| Coupling | `vanilla` or exact minibatch `ot` |

### PC students

| CLI configuration | Standard width | Parameters |
|---|---:|---:|
| `--pc-structure hclt --pc-layer dense` | 64 latents | 3,608,640 |
| `--pc-structure hclt --pc-layer monarch` | 64 latents | 1,604,160 |
| `--pc-structure pd` | 8 channels | 1,570,824 |
| `--pc-structure gaussian` | 256 components | 1,605,888 |

The teacher has 1,075,361 parameters. Standard student widths are powers of
two and are not selected using evaluation performance. Use
`--capacity-profile published` for dense HCLT-256, Monarch-HCLT-1024, PD-64,
or Gaussian-256.

The dense HCLT topology follows the
[PyJuice HCLT benchmark](https://github.com/liuanji/pyjuice-benchmarks/blob/main/configs/model/hclt_256.yaml).
The Monarch variant follows
[MonarchCircuits](https://github.com/wangben88/MonarchCircuits).

### Standard student training

| Setting | Value |
|---|---:|
| Steps | 20,000 |
| Batch size | 64 |
| Time order | 4 |
| Optimizer | AdamW |
| Learning rate | `1e-3` |
| Weight decay | `0.01` |
| Scheduler | cosine |
| Gradient clipping | `1.0` |
| Checkpoints | 5k, 10k, 15k, 20k |

## MNIST workflow

### 1. Train a teacher

```bash
python train.py flow \
  --flow-matcher vanilla \
  --epochs 3 \
  --batch-size 128 \
  --wandb-group teacher-pretraining \
  --wandb-run-name teacher-vanilla-torchcfm-seed0 \
  --output-dir outputs/runs/flow/teacher-vanilla-torchcfm-seed0
```

Use `--flow-matcher ot` and a separate output directory for the OT version.

### 2. Distill into a PC

This is the standard dense-HCLT baseline without endpoint EM:

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
  --time-order 4 \
  --lr 0.001 \
  --optimizer adamw \
  --scheduler cosine \
  --weight-decay 0.01 \
  --gradient-clip 1.0 \
  --wandb-group student-distillation \
  --wandb-run-name hclt-dense-h64-vanilla-noem-step20000-seed0 \
  --output-dir outputs/runs/distill/hclt-dense-h64-vanilla-noem-step20000-seed0
```

Change `--pc-structure`, `--pc-layer`, or `--capacity-profile` to switch
architectures without changing code.

### 3. Evaluate

Train the fixed MNIST feature extractor once:

```bash
python train_mnist_classifier.py --data-dir data --epochs 14
```

Then evaluate a run:

```bash
python eval.py \
  --run-dir outputs/runs/distill/hclt-dense-h64-vanilla-noem-step20000-seed0 \
  --times 0,0.25,0.5,0.75,1 \
  --num-generated 10000
```

Reported metrics include discrete test bpd, FID-MNIST, KID-MNIST,
precision/recall, IS-MNIST, class coverage, throughput, memory, and parameter
count. Sampling is evaluated with direct PC sampling, coupled path sampling,
Euler, RK4, and adaptive Dopri5 trajectories.

### Experiment launchers

| Command | Purpose |
|---|---|
| `bash scripts/run_standard_vanilla_noem.sh` | Equal-budget four-architecture benchmark |
| `FLOW2PC_GPU=1 bash scripts/evaluate_vanilla_noem_checkpoints.sh` | 5k/10k/15k/20k Pareto evaluation |
| `bash scripts/run_dense_hclt_time_tuning_noem.sh` | Polynomial/Fourier and learning-rate sweep |
| `bash scripts/run_beta_timestep_ablation.sh` | Uniform versus endpoint-biased time sampling |
| `FLOW2PC_GPU=0 bash scripts/run_endpoint_em_then_distill.sh` | Endpoint EM followed by distillation |
| `bash scripts/run_published_endpoint_hclt.sh` | Published categorical HCLT endpoint check |

Endpoint EM is controlled by `--endpoint-pretraining {none,em}`. Its defaults
are 20 full-batch epochs, pseudocount `0.1`, and minimum standard deviation
`0.01`.

The categorical endpoint check uses HCLT `M=16`, batch size 1024, 100
mini-batch EM epochs, and 20 full-batch EM epochs. It targets the 1.26 bpd
result reported by
[Liu et al. (ICLR 2022)](https://starai.cs.ucla.edu/papers/LiuICLR22.pdf).

## Synthetic analytic-oracle benchmark

`synthetic_experiment.py` uses a fixed 2D six-component Gaussian-mixture CFM
path with exact density, velocity, and query values. It isolates PC
approximation error.

This is an oracle ablation, not the proposal's full pretrained-flow experiment.
The full experiment must first train and freeze a neural flow on the synthetic
data, then distill that checkpoint. The analytic model should be used only for
ground-truth evaluation in that experiment.

### Fixed setup

| Setting | Value |
|---|---:|
| Student | time-dependent Gaussian-mixture PC |
| Capacities | `K={2,4,8}` |
| Steps | 5,000 |
| Batch size | 2,048 |
| Time order | 4 |
| Optimizer | AdamW, `lr=1e-3`, weight decay `0.01` |
| Gradient clipping | `1.0` |

The benchmark reports velocity MSE, endpoint KL, density L1, sliced
Wasserstein distance, query errors, and the adjoint-weighted residual.

### Reproduction sequence

Run these in order. Completed matching runs are reused.

```bash
export FLOW2PC_PYTHON=/mnt/disk1/aiotlab/envs/uffm/bin/python
export FLOW2PC_GPU=1

bash scripts/run_synthetic_distill_only.sh
bash scripts/run_synthetic_coupled.sh
bash scripts/run_synthetic_density_timestep_tuning.sh
bash scripts/run_synthetic_density_timestep_refinement.sh
bash scripts/run_synthetic_density_timestep_confirmation.sh
bash scripts/run_synthetic_timestep_significance.sh
bash scripts/run_synthetic_density_weight_trend.sh
```

The final two stages use:

- eight matched seeds and an exact paired two-sided sign-flip test to choose
  uniform or `Beta(1,0.2)` timestep sampling;
- three matched seeds and `lambda_den={0,1,3,10,30,100}` to select the density
  weight.

Synthetic outputs contain `config.json`, `training_history.json`,
`results.csv`, `results.json`, `students.pt`, and diagnostic figures.

## Logging and outputs

- W&B project: `flow2pc`
- Use `--wandb-group` for experiment families and `--wandb-run-name` for each
  configuration.
- Use `--wandb-mode offline` or `--wandb-mode disabled` when required.
- Local runs are written under `outputs/runs/`; benchmark summaries are under
  `outputs/benchmark/`.
- `outputs/` and credentials are not committed.

## Development

```bash
make lint PY=/mnt/disk1/aiotlab/envs/uffm/bin/python
make test PY=/mnt/disk1/aiotlab/envs/uffm/bin/python
```

## Layout

```text
train.py                              teacher training and PC distillation
eval.py                               MNIST likelihood and sample evaluation
synthetic_experiment.py               analytic-oracle synthetic benchmark
train_endpoint_hclt.py                categorical endpoint reproduction
train_mnist_classifier.py             fixed MNIST evaluator
repc/flow.py                          flow teacher
repc/pc.py                            Gaussian-mixture PC
repc/hclt_pc.py                       dense and Monarch HCLT PCs
repc/region_pc.py                     Poon--Domingos spatial PC
repc/synthetic.py                     analytic branching CFM path
scripts/run_synthetic_*.sh            synthetic experiment pipeline
summarize_synthetic_*.py              synthetic aggregate reports
```
