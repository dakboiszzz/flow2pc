# flow2pc

Recompiling a pretrained flow-matching model into a **probabilistic circuit
(PC)**: tractable exact inference (likelihood, moments, uncertainty
decomposition) from a model you only need to *distill*, not retrain.

Two parts:

| dir | what | scale |
|---|---|---|
| [`toy/`](toy/) | **Replica-Flow2PC** — compile a two-replica (pair) flow teacher into a synchronized pair PC for exact aleatoric/epistemic uncertainty decomposition; observability theory + 5 benchmarks | analytic teachers, d ≤ 32 |
| [`cifar/`](cifar/) | **Recompile-Probabilistic** — recompile image flow/diffusion teachers into deep PCs (Monarch/HCLT-style region graphs) with a single velocity-matching loss | MNIST / CIFAR-10 / TinyImageNet |

## Method in one paragraph

A time-indexed PC with Gaussian leaves (polynomial or time-MLP paths, boundary
`p_0 = N(0, I)` structural) and time-independent sum weights induces a velocity
field by three rules — leaf `u = μ′ + (σ′/σ)(z − μ)`, product = concatenation,
sum = responsibility-weighted average — and satisfies the continuity equation
by construction. Matching this velocity to a frozen teacher's field
(`E‖U − V‖²`) *reconfigures* the flow into the PC, with a Grönwall/Wasserstein
error certificate. For uncertainty decomposition the student is a **pair** PC
`Q²(z₁,z₂) = Σ_h π_h q_h(z₁) q_h(z₂)` (only the hypothesis channel shared):
`Σ_epi = Cov_h(μ_h)`, `Σ_ale = E_h[diag σ_h²]` are then exact reads off the
parameters — 0-NFE, no sampling.

Final training objective (both parts): `L = L_vel + λ·L_density` (λ ≈ 0.3,
pathwise). The velocity term preserves the teacher's dynamics; the density term
supplies the state information the velocity loss provably under-observes — see
[`toy/docs/asymmetry.md`](toy/docs/asymmetry.md) for the observability
analysis and the 5-seed ablation.

## Key toy result (5 seeds, matched compute)

| loss | err_epi | err_ale | velocity defect |
|---|---|---|---|
| velocity only | 0.093 | 1.49 ± 2.15 (drifts) | **0.009** |
| density only | 0.021 | 0.008 | 0.105 |
| **velocity + density (λ=0.3)** | **0.0013** | **0.0010** | 0.011 |

The compiled PC returns exact moments at 0 NFE (~0.12 ms, 480-edge circuit)
vs. Monte-Carlo flow integration (K NFE/sample) or MeanFlow (1 NFE/sample).

## Quickstart

```bash
conda env create -f cifar/requirements.txt   # or use an env with torch>=2.1

# toy: verification battery + the decisive ablation
cd toy
pytest tests/ -q                      # 25 tests
python run_ablation_seeds.py VD0.3    # velocity+density, 5 seeds
python run_gramian.py                 # observability Gramian verification

# cifar: pretrain a flow, then recompile it into a PC
cd ../cifar
python train.py task=flow data=cifar10 model=flow_unet_cifar
python train.py task=distill data=cifar10 model=pc_patch_cifar \
    task.flow_ckpt=outputs/runs/flow/<ts>/checkpoints/last.ckpt
```

Each subdirectory has its own README with full usage.

## License

MIT
