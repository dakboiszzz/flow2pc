# Recompile-Probabilistic

Recompiling a pretrained flow-matching model into a **chain of Probabilistic
Circuits**: tractable exact likelihood (UQ) + one-step generation, trained
with a single velocity-matching loss (see `slides/report300526.pptx`).

## Method

A time-indexed PC with Gaussian leaves (time-polynomial parameters, no
constant term ⇒ `p_0 = N(0, I)` exactly) and time-independent sum weights
induces a velocity field via three rules — leaf `μ′ + L′L⁻¹(z − μ)`,
product = concatenation, sum = responsibility-weighted average. By
Proposition 1 it satisfies the continuity equation by construction, so the
only objective is `E ||u_root(Z_t) − u_θ(Z_t)||²` against the frozen flow.

## Repo layout (Hydra + Lightning)

```
train.py / eval.py       entry points (Hydra + Lightning)
Makefile                 make test | lint | flow-mnist | distill-ddpm | ...
pyproject.toml           installable package + pinned deps
configs/                 hydra config groups
  task/     flow | distill
  model/    flow_unet flow_mlp flow_unet_cifar flow_unet_tin |
            pc_shallow pc_deep pc_wide pc_patch pc_patch_cifar pc_patch_tin pc_general
  data/     mnist | cifar10 | tinyimagenet | miniboone
  trainer/  logger/      lightning trainer, wandb + csv loggers
repc/                    core library
  pc.py                  shallow mixture PC
  deep_pc.py             deep PD region-graph PC (structure via sum weights)
  wide_pc.py             per-scope-size channel widths
  patch_pc.py            multivariate patch leaves (full time-dep Cholesky)
  clt_structure.py       data-learned Chow-Liu region graph (HCLT-style)
  general_pc.py          random balanced partition trees (non-image data)
  flow.py                UNet (2/3-stage) and MLPFlow velocity nets
  pretrained.py          off-the-shelf diffusion teachers (no retraining)
  metrics.py             FID, Hutchinson LL, ODE integrators, samplers
  lightning/             LightningModules + DataModules
tests/                   verification battery (pytest, all circuit families)
scripts/                 baselines (from-scratch Monarch-HCLT EM training)
third_party/             vendored MonarchCircuits (gitignored;
                         github.com/wangben88/MonarchCircuits)
outputs/
  runs/<task>/<ts>/      one dir per run: ckpts, csv, wandb, eval/
  figures/ logs/ eval/
```

## Setup

```bash
conda activate uffm            # or: pip install -r requirements.txt
cp .env.example .env           # put your WANDB_API_KEY here (gitignored)
```

## Usage

```bash
# 1. pretrain a flow
python train.py task=flow data=mnist model=flow_unet
python train.py task=flow data=miniboone model=flow_mlp trainer.max_steps=20000

# 2. recompile it into a PC (single loss, no ad-hoc terms)
python train.py task=distill data=mnist model=pc_deep \
    task.flow_ckpt=outputs/runs/flow/<ts>/checkpoints/last.ckpt
python train.py task=distill data=miniboone model=pc_general \
    task.flow_ckpt=<...> task.flow_model._target_=repc.flow.MLPFlow

# warm-start / longer grind (loss ↓ ⇒ fidelity ↑, see report Figure 7)
python train.py task=distill ... task.warm_start=<prev-ckpt> task.lr=8e-4

# 3. evaluate a distill run
python eval.py --run-dir outputs/runs/distill/<ts>
```

All runs log to **W&B** (project `recompile-probabilistic`) and to local CSV
in the run directory simultaneously.

## Results (MNIST, summary)

- Exact log-likelihood up to 2217 nats, 2500× faster than the flow's
  Hutchinson estimate; perfect MNIST/FashionMNIST OOD separation.
- Reconfiguration fidelity is a monotone function of the single loss
  (FID 259 → 169 as loss 0.127 → 0.095, diagonal leaves).
- Hybrid sampling (one-step PC teleport to t₀, teacher on [t₀,1]):
  FID 21.5 at half the teacher's cost; FID 15.4 (< teacher's 20.4) at 70%.
- Full report: see the project artifact (private link in lab notes).
