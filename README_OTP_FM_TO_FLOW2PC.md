**Distilling OTP-FM into Flow2PC on Beijing and Gulf of Mexico**

This is a code-based integration proposal and teacher selection, prepared after
reading the current neighboring `../flow2pc` checkout. No teacher/student
training, model inference, dependency installation, or modification of that
repository was performed. The adapters and training entry points proposed here
are **not yet implemented**. Existing Flow2PC commands cannot load OTP-FM as-is.

**Updated recommendation:** use OTP-FM's own **conditional interpolant** to
generate intermediate samples, and query the frozen EMA instantaneous velocity
teacher on those samples. For our W2 checkpoints, compute the small direct
linear solve once and cache each coupled tuple's correction coefficients.
There is **no ODE rollout per distillation update**, no learned generator to
train, and no GAN. The previous ODE-first proposal was a conservative reference
construction, but it was unnecessarily expensive as the default for this
workflow. ODE evaluation is an optional, separate reference diagnostic.

The [teacher manifest](reproduction/distillation_teachers.json) records exact
paths, SHA-256 hashes, scores, data/time contracts, alternatives and reviewed
source hashes. It identifies the conditional-interpolant route and explicitly
marks its reference evaluation and student training as not yet performed.

**What your current Flow2PC code does.** The relevant implementations are
[ALI teacher/generator wrappers](../flow2pc/repc/flow.py),
[the generic loss](../flow2pc/train.py),
[multi-marginal path sampling](../flow2pc/repc/single_cell.py),
[single-cell training](../flow2pc/scripts/train_single_cell.py), and
[the HCLT student](../flow2pc/repc/hclt_pc.py).

Your ALI path uses two frozen interfaces:

```text
samples = generator(x0, x1, t)
target  = teacher(t, samples)
loss    = mean((pc.velocity(samples, t) - target)**2)
```

`ALICFMGeneratorWrapper` constructs
`(1-t)*x0 + t*x1 + t*(1-t)*f_phi(x0,x1,t)`. Its endpoints are the supplied
`x0` and `x1`. `ALICFMTeacherWrapper` reconstructs a different velocity MLP.
The single-cell path samples OT-coupled endpoints through an observed middle
population, then evaluates the generator at continuous global time. The generic
path independently samples endpoint pools. These are distinct coupling choices.

The single-cell trainer uses velocity MSE only; observed NLL is a diagnostic.
The ST launcher instead enables an observed-marginal NLL term by default.
Therefore, there is not one universal ALI distillation objective across all
launchers. In either case, the PC update is not itself GAN training: it consumes
the already trained ALI networks. There is no discriminator to port into the
OTP-FM student loss.

In the files reviewed, the student is a **single time-conditioned probabilistic
circuit** with a spatial/feature HCLT, not a separate conditional PC transition
for every time interval. `sample_path` shares latent choices and Gaussian noise
across times to produce a coupled path. If this is what you mean by a chain of
PC marginals, the existing representation can be reused. A literal temporal
Markov chain is a further extension, described below.

**How OTP-FM differs.** The learned model is a consistency/MeanFlow model
`v_theta(x, t1, t2)`. For these checkpoints, `v_func(x,t1,t2)` calls the network
with inputs `(x, t1, dt=t2-t1)`. Its diagonal is an instantaneous field:

```text
T(t, x) = model.v_func_ema(x, t, t)     # dt = 0
```

The off-diagonal value predicts a mean velocity over a finite interval. It is
not interchangeable with the instantaneous field. In particular,
`model.sample(..., n_steps, ema=True)` updates
`x_next = x + (t_next-t)*v_ema(x,t,t_next)` per marginal segment. Our saved
reproduction scores were measured using this finite-step consistency sampler.

OTP-FM constructs conditional training paths from multi-marginal tuples using
potential-driven corrections and a fixed-point/direct solver; it does not train
an adversarial interpolant. The preferred W2 presets use precomputed OT coupling
and the linear-gradient/direct-solver construction. The experiment label `W2`
still uses `potential="w2inf"` in the released code; do not replace the potential
class based on its label. These semantics are supported by the
[model implementation](src/otpfm/otpfm.py),
[solver](src/otpfm/solvers.py), and
[paper's consistency-model and algorithm sections](https://arxiv.org/html/2606.05327v1#S2.SS3).

| Required role | ALI-CFM route in Flow2PC | Recommended OTP-FM route |
| --- | --- | --- |
| Velocity oracle | Separate pretrained velocity MLP | Frozen EMA `v_func_ema(x,t,t)` |
| Continuous path samples | Learned endpoint-conditioned interpolant | Analytic conditional OTP interpolant with cached corrections |
| Starting distribution | Whatever the ALI bundle specifies | First component of the normalized coupled data tuple |
| Conditioning inputs | Paired endpoints | Endpoints and all observed intermediate samples in the tuple |
| Extra generator/discriminator training | Already embodied in the ALI bundle | None for the initial port |
| PC supervision | Velocity matching, plus launcher-specific extras | Velocity matching after handling the source density |

EMA and live weights are two versions of the same learned architecture, not the
velocity/interpolator pair. Use EMA for teacher velocity targets. For these W2
presets the conditional interpolant is determined by data coupling, times,
potentials and the chosen correction scale; it does not evaluate either neural
network to sample a point after its coefficients have been cached.

**Choose these teachers first.** For distillation I prioritize the lowest
recorded average holdout W2 of an individual completed checkpoint, rather than
proximity to the paper's five-seed mean. Both selected models are W2, seed 44,
from `main-reproduction`:

| Dataset | Model / selected epoch | Recorded holdout W2 | Recorded training W2 | Recorded finite-step sampler |
| --- | --- | --- | --- | --- |
| Beijing | W2, seed 44, epoch **120** | **16.345044** | 19.979231 | 20 steps per marginal |
| GoM | W2, seed 44, epoch **1876** | **0.104112** | 0.075894 | 25 steps per marginal |

Checkpoint paths relative to this OTP-FM checkout:

```text
outputs/reproduction/main-reproduction/beijing/W2/seed44/models/best.pt
outputs/reproduction/main-reproduction/gom/W2/seed44/models/best.pt
```

Use the adjacent `config.json`, `best_metrics.json`, `provenance.json`, and
`result.json` for each. The checkpoint's `epochs` field is the configured total,
not its selected epoch; `best_metrics.json` identifies the selected epoch.
Do not substitute the final `models/model.pt` checkpoint.

Individual teacher runs:
[Beijing](https://wandb.ai/pqhhust-hanoi-university-of-science-and-technology/flow2pc/runs/kq4zwy8l)
and [GoM](https://wandb.ai/pqhhust-hanoi-university-of-science-and-technology/flow2pc/runs/phofe5eh).
Static archive inspection confirmed that both checkpoints contain EMA weights,
and the current model/factory source hashes match their saved training snapshots.

These are **best recorded finite-step candidates, not yet reranked under the
new interpolant-based distillation protocol**. Evaluation protocols differ
across some Beijing attempts. Selection
also uses benchmark holdouts already used for teacher checkpoint selection; it
does not create an untouched test set. Before fixing a teacher for student
comparisons, evaluate its frozen targets and interpolant under a common
protocol. Never reuse 16.345044 or 0.104112 as if they were scores of the
conditional interpolant itself or of a diagonal-ODE evaluation.

Useful alternatives for that common-protocol comparison are:

| Dataset | Alternative | Recorded holdout / training W2 | Why retain it |
| --- | --- | --- | --- |
| Beijing | `paper_times/W2/seed43`, epoch 380 | 16.947443 / 17.051560 | Best seed in the selected paper-aligned W2 cohort; exact times and 50-step evaluation |
| GoM | Original `W2/seed46`, epoch 2000 | 0.106211 / 0.062810 | Slightly higher holdout error, lower observed-time error |

The manifest contains their exact paths. For five-teacher robustness studies,
retain all five W2 seeds instead of mixing seed-specific variants. The earlier
[reproduction README](README_REPRODUCTION.md) deliberately selected complete
cohorts to match paper means; this document selects individual candidate
teachers for student fidelity. Those are different selection questions.

**Reconstruct and freeze the complete OTP-FM model.** An OTP-FM checkpoint is
not a Flow2PC `model_type="flow"` bundle, an ALI velocity MLP, or an ALI
generator. It stores `model_state_dict`, including live and EMA networks.
Potentials, exact marginal times, normalization and architecture settings must
come from the saved config/data contract; the weights alone are insufficient.

Illustrative adapter code, not installed or executed:

```python
import json
from pathlib import Path
import torch
from torch import nn
from experiments.train import create_model, create_potentials

def load_otpfm(run_dir, dim, device):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    # Use only our trusted local checkpoint; it also contains loss-history objects.
    saved = torch.load(run_dir / "models/best.pt",
                       map_location="cpu", weights_only=False)
    model = create_model(config, dim, create_potentials(config), "cpu")
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.to(device).eval()
    model.requires_grad_(False)
    if model.flownet_ema is None:
        raise ValueError("Selected teacher is expected to have EMA weights")
    return model, config

class OTPVelocity(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, time, values):
        if values.ndim != 2 or values.shape[1] != self.model.d:
            raise ValueError("Expected [batch, data_dim] normalized values")
        times = torch.as_tensor(time, device=values.device,
                                dtype=values.dtype).reshape(-1)
        if times.numel() == 1:
            times = times.expand(len(values))
        if times.numel() != len(values):
            raise ValueError("Expected a scalar time or one time per sample")
        return self.model.v_func_ema(values, times, times)
```

`create_model` does not download data or compute data OT alignments. Do not use
the ALI state-dict-to-MLP heuristic to reconstruct OTP-FM: it would lose the
residual architecture, positional time embeddings, normalization layers and EMA
selection. Some OTP-FM time/potential tensors are ordinary attributes rather
than registered buffers; the existing solver explicitly moves them where
needed. Avoid assuming `.to(device)` migrates arbitrary config attributes when
implementing the analytic sampler below.

The new Flow2PC integration should add the OTP-FM root and its `src/` to the
import path explicitly or use a dedicated importable adapter. Do not route
OTP-FM through `--teacher-arch alicfm`. Preserve the existing `uffm` environment;
no installation or environment change is part of this proposal.

**Use the existing conditional interpolant directly.** This is the construction
inside `forward_with_loss`, not a separately saved neural `generator.pt`.
For a normalized, coupled observed tuple
`z=(x0,x_t1,...,x_tK,x1)`, let

```text
H_k(t) = model.x_time_dep(potential_k, t)
J_k(t) = model.v_time_dep(potential_k, t) = dH_k(t)/dt
F_k    = strength_k * (Z_k - x_tk)

Gamma_alpha(z,t) = x0 + (x1-x0)*t + alpha * sum_k F_k*H_k(t)
U_alpha(z,t)     = x1-x0          + alpha * sum_k F_k*J_k(t)
```

The conditional interpolation and closed-form W2 fixed point are described in
[paper equations 17–20](https://arxiv.org/html/2606.05327v1#S4.SS1).
Here `Z` is the full-strength direct solution used by the released code:

```text
A_ij   = H_j(t_i) * strength_j
Z_base = x0 + (x1-x0) * [t_1,...,t_K]
Z      = (I-A)^(-1) * (Z_base - A*x_intermediate)
```

For our chosen teachers, **K=7 for Beijing and K=3 for GoM**. Factor the fixed
`I-A` matrix once, solve/cache `F` for the observed OT-coupled tuples once, and
evaluate `Gamma` at arbitrary continuous times. No trajectory integration,
discriminator, generator training or teacher-network evaluation is needed to
produce those input samples. Intermediate marginals are soft constraints;
`Gamma(z,t_k)` need not exactly equal the supplied intermediate sample.
At endpoints, `Gamma(z,0)=x0` and `Gamma(z,1)=x1` for the supplied tuple.

This conditioning is important: preserve the original OT chains through **all
observed marginals**, rather than draw only independent endpoints or insert a
held-out population. The original loaders construct these chains with row-argmax
OT maps. In Beijing, marginal sizes differ, so their induced tuple marginals
can be reweighted by duplicated indices. Do not replace that coupling silently
or claim that exact tuple endpoints imply every empirical marginal is recovered
with its original weights. Cache the chain indices and input hashes.

The original code applies `alpha` outside the correction after its full-strength
solve; do not instead change `A` to `alpha*A` when reproducing that implementation.
For a checkpoint-matched initial experiment, use the last logged training
correction strength at the selected epoch:

| Teacher | Fixed checkpoint-matched alpha | Source |
| --- | --- | --- |
| Beijing W2 seed 44, epoch 120 | **0.916572766752024** | `train/otp_alpha` at epoch 120 in saved `history.jsonl` |
| GoM W2 seed 44, epoch 1876 | **0.9997395918739371** | `train/otp_alpha` at epoch 1876 in saved `history.jsonl` |

This is a disclosed sampling-policy choice, not proof of perfect teacher/path
agreement: the teacher trained across earlier curriculum values too. A separate
`alpha=1` experiment represents the full-strength conditional path. Do not
change the alpha schedule during PC training without labeling it as a different
sampling objective.

Illustrative cache construction and sampling for the selected `W2InfPotential`
plus OT-coupling presets; this code is documentation, not a validated adapter:

```python
from otpfm.potentials import W2InfPotential

@torch.no_grad()
def build_interpolant_cache(model, tuples, alpha):
    # tuples: [N,K+2,D], already normalized and coupled using observed data only.
    potentials = list(model.potentials.values())
    if not all(isinstance(p, W2InfPotential) for p in potentials):
        raise ValueError("This fast linear cache is for the selected W2 presets")
    K, N, D = len(potentials), len(tuples), tuples.shape[-1]
    if K == 0 or tuples.shape[1] != K + 2:
        raise ValueError("Tuple ordering must match every intermediate potential")
    knots = tuples.new_tensor(list(model.potentials.keys()))
    strengths = tuples.new_tensor([p.strength for p in potentials])
    # H[i,j] = correction from potential j evaluated at knot i.
    H = torch.stack([model.x_time_dep(p, knots[:, None]).reshape(K)
                     for p in potentials], dim=1)
    A = H * strengths[None, :]
    lu, pivots = torch.linalg.lu_factor(torch.eye(K, device=A.device,
                                               dtype=A.dtype) - A)
    x0, base = tuples[:, 0], tuples[:, -1] - tuples[:, 0]
    middle = tuples[:, 1:-1].transpose(0, 1)  # [K,N,D]
    bases = x0[None] + knots[:, None, None] * base[None]
    rhs = bases - torch.einsum("ij,jnd->ind", A, middle)
    Z = torch.linalg.lu_solve(lu, pivots, rhs.reshape(K, -1)).reshape(K, N, D)
    forces = strengths[:, None, None] * (Z - middle)
    return {"x0": x0, "base": base, "forces": forces, "alpha": float(alpha)}

@torch.no_grad()
def sample_interpolant(model, cache, rows, times):
    # rows selects cached tuples; times has one global time per selected row.
    potentials = list(model.potentials.values())
    t = times.reshape(-1, 1).to(cache["x0"])
    H = torch.stack([model.x_time_dep(p, t) for p in potentials])
    J = torch.stack([model.v_time_dep(p, t) for p in potentials])
    F = cache["forces"][:, rows]  # [K,B,D]; strength already included.
    base = cache["base"][rows]
    samples = cache["x0"][rows] + t * base + cache["alpha"] * (F * H).sum(0)
    conditional_velocity = base + cache["alpha"] * (F * J).sum(0)
    return samples, conditional_velocity
```

Keep cache rows, query times, potentials and tensors on compatible devices.
Validate tuple/time ordering, finite values, the relative linear-system residual
and both endpoint identities before using the cache. Setup still has a one-time
data-coupling cost. The factorization and force cache are rebuilt when config,
times, strengths, widths, normalization or tuple membership change; query times
alone do not require rebuilding the cache.

There is a source-code performance trap in `solve_X_tks`: auto-selection can
leave `self.solver_type=None` while the actual solver is `DirectSolver`, causing
unnecessary learned-position initialization and fallback inverse calculations.
The explicit linear cache above bypasses that path without changing the teacher
or training equations. Do not switch a pretrained model to a new potential just
to avoid it. The fast cache is not a generic replacement for nonlinear MMD/KL
fixed-point solves.

Then preserve your existing teacher-velocity distillation objective:

```python
with torch.no_grad():
    rows = draw_training_tuple_rows(batch_size)
    times = torch.rand(batch_size, device=device)
    samples, conditional_velocity = sample_interpolant(model, cache, rows, times)
    target = teacher(times, samples)  # frozen diagonal EMA, not conditional_velocity
student.train()
loss = (student.velocity(samples, times) - target).square().mean()
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

Thus each update needs direct interpolation, one batched teacher velocity query,
and the student update. Do not wrap `student.velocity` in `no_grad`; its circuit
flow calculations need parameter gradients.

`U_alpha(z,t)` is a **conditional** velocity. The marginal target is its
conditional expectation given `Gamma(z,t)=x`. Therefore it is generally
incorrect to require `T(t,Gamma)` to equal every sampled `U_alpha` pointwise:
conditional variability creates a nonzero error even for an ideal marginal
teacher. Log a conditional-FM diagnostic, but interpret it with that variability
and the pretrained teacher's approximation error. Direct regression to
`conditional_velocity` would instead train the PC by conditional flow matching;
it changes the experiment from frozen-teacher velocity distillation and must be
labeled separately.

Do not leave `generator=None` in the generic Flow2PC `distillation_loss`: that
falls back to a Gaussian straight line. Add an explicit tuple-interpolant sampler
used identically by training and validation. An endpoint-only generator API is
insufficient unless it also carries the correct intermediate tuple conditioning.

**ODE and learned-interpolator alternatives are optional.** A diagonal EMA ODE
still provides a useful learned-flow reference and can be evaluated once outside
student updates. It is not required to supply every training batch. If a
separately learned generator is an architectural requirement, it can regress to
cached conditional trajectories without a GAN; matching the teacher ODE rather
than conditional paths would be a different reference choice. Do not treat
`model.x_t` or `x0+t*v_ema(x0,0,t)` as automatically exact versions of either
reference. Few-step consistency prediction, analytic conditional interpolation,
and the diagonal ODE define distinct finite-model sampling procedures.

There is a concrete diagnostic issue to avoid in the existing
[ALI compatibility script](../flow2pc/scripts/diagnose_alicfm_compatibility.py):
`autograd.grad(samples.sum(), times)` sums coordinate derivatives. In GoM it
returns one scalar per sample, not the required 2D derivative, and broadcasting
it against a 2D velocity is incorrect. For an independent-sample differentiable
generator, use a componentwise Jacobian or
`torch.func.jvp(lambda t: G(x0,t), (times,), (ones_like(times),))[1]`, which returns
the full `[B,D]` directional derivative. See the
[PyTorch JVP documentation](https://docs.pytorch.org/docs/2.6/generated/torch.func.jvp.html).

**Keep the teacher's data and time coordinates.** Use the same raw input bytes,
cast/reshape operations and `StandardScaler` fit as the OTP-FM loader. The
checkpoint does not contain the fitted scaler. Reconstruct it once from the
verified input manifest and all original marginals, then persist mean/scale,
feature order, dtype, source hash and normalized populations in the new teacher
bundle. The existing normalization used held-out marginals; retaining that
convention must be disclosed rather than described as a training-only fit.

| Contract | Beijing | GoM |
| --- | --- | --- |
| Raw input | `/data/lthung/otp-fm/beijing/dingling_pm25_13marginals.npy` | `/data/lthung/otp-fm/gom/GoMvortex_data.npy` |
| Dimension | 1 | 2 |
| Full observation times | `i/12`, `i=0..12` | `i/8`, `i=0..8` |
| Observed indices | `[0,1,3,4,6,7,9,10,12]` | `[0,2,4,6,8]` |
| Holdout indices | `[2,5,8,11]` | `[1,3,5,7]` |
| Initial source population | First monthly PM2.5 cloud, 744 observations | First particle cloud, 111 observations |
| Source law | Empirical normalized first cloud | Empirical normalized first cloud |

Do not apply ALI min/max scaling or assume a standard-normal source. Do not
compress the nine observed Beijing times into an equally spaced nine-point
grid. Keep original month indices and a list of four holdouts, rather than the
single integer `heldout` used by the current single-cell contract.

If `x_phys = mean + scale*x_norm`, then
`v_phys = scale*v_norm` in normalized global time. If time is also converted to
a physical span `S`, velocities acquire an additional division by `S`.
Prefer training entirely in the teacher's normalized coordinates and only
inverse-transform for metrics/plots. A Gaussian PC's physical log density is
`log p_phys(x) = log p_norm(x_norm) - sum(log(scale))`.

Use only observed populations for student updates, topology fitting and source
density fitting. Partition the original coupled tuples into fixed training and
validation sets by their source-row indices. Intermediate/target observations
can be reused across tuples, so this is a source-row split, not a claim that all
coordinates in validation are unseen. Recomputing OT separately on disjoint
per-population splits is possible but changes the coupling and must be reported.
Continuous interpolant samples at a held-out **time** are valid distillation
inputs; real held-out **observations** are excluded from updates and selection.
The teacher previously trained on the complete observed clouds, so this split
measures student fidelity rather than generalization to cells unseen by the
teacher.

**Handle the PC source density explicitly.** Velocity matching identifies a
transported density only with its initial condition. Beijing/GoM do not have
`p_0=N(0,I)`. The existing free-source Fourier setting and
`initialize_source` are useful, but initialization alone does not guarantee
that subsequent velocity-only updates preserve or correctly fit that source.

For a clean velocity-only distillation experiment, first fit the student source
density on source-training observations, select that boundary fit by source
validation NLL, and preserve it during path distillation. Fit the continuous
Gaussian source density by MLE; the categorical MNIST endpoint EM routine is
not the corresponding source-fitting procedure. One explicit
parameterization is:

```text
mu_j(t)       = frozen_mu_j0       + A_j * [phi(t) - phi(0)]
log_sigma_j(t)= frozen_log_sigma_j0+ B_j * [phi(t) - phi(0)]
```

Keep the fitted time-independent circuit weights fixed as well if claiming
the entire initial density is fixed. This requires a small emission/boundary
extension: simply turning `fourier_zero_center=True` on in the existing class
would force zero means/unit scales again, because it has no separate learned
source offsets. This source fitting is a proposed addition to the current
single-cell velocity-only recipe, not something we already ran.

An alternative is jointly learning the source with an explicit source-NLL term
and reporting its coefficient. That changes the objective and must be labeled
as an ablation; do not silently add the ST launcher's all-observed NLL loss.
Keep observed marginal NLL diagnostic-only for the initial velocity-only study.
Also monitor scale-clamp saturation: the current HCLT clamps log-scales without
masking their reported time derivatives; saturated leaves need consistent
derivatives to preserve the claimed density/velocity relationship.

For the first experiment, reuse a dense Gaussian-leaf HCLT with Fourier time
conditioning: parents `[-1]` for Beijing and `[-1,0]` for GoM. In 1D the circuit
reduces to a time-dependent Gaussian mixture; in 2D the tree connects the two
coordinates. The spatial PD image structure is unnecessary here. A proposed
starting configuration is 64 latent states, 16 time features, batch size 64,
Adam at 1e-3, gradient clipping 1, 20,000 velocity steps, validation every 500
steps, and student seeds 0/1/2. These are starting choices, not tuned results.
Report actual parameter counts, runtimes and memory rather than importing the
MNIST capacity defaults. Increase capacity only after source-fit and
teacher-fidelity checks identify a limitation.

**If you require a literal temporal PC chain.** For intervals `[a_k,b_k]`, a
local student receiving `s=(t-a_k)/(b_k-a_k)` must match

```text
u_local(x,s) = (b_k-a_k) * T(a_k+(b_k-a_k)*s, x)
```

The interval factor is necessary. Use actual global observed times; in Beijing
the interval lengths are unequal. Sample intervals in proportion to duration
if the intended objective is uniform in global time.

Independent local PCs do not automatically share boundary marginals or define
transition distributions. To claim
`q(x_0,...,x_K)=q_0(x_0)*product_k q(x_{k+1}|x_k)`, you need normalized
conditional kernels or an explicitly specified deterministic transport; shared
boundary distributions are additionally needed to match any claimed segment
marginals. This is not provided by a list of `pc.log_prob(x,t)` calls.
`ProbabilisticCircuit.path_log_prob` currently stacks marginal log densities;
their sum is not a learned temporal joint log likelihood. Direct latent-coupled
PC paths and ODE trajectories can share marginals without being the same
pathwise coupling. Start with the existing global time-conditioned circuit;
add explicit temporal kernels only if that joint-distribution requirement is
part of the method you intend to evaluate.

**Implementation changes still required in Flow2PC.**

| Area | Required change |
| --- | --- |
| Teacher loading | Explicit OTP-FM adapter/factory branch; strict config reconstruction, EMA selection and checkpoint hashes |
| Dataset contract | Beijing/GoM ragged clouds, scaler metadata, exact global times and multiple holdouts |
| Path sampling | Cached conditional OTP tuple sampler shared by training and validation; preserve coupling and fixed alpha |
| Generic CLI | Add OTP-FM/Beijing/GoM dispatch, or preferably a dedicated trainer/evaluator; current choices do not accept them |
| Validation | Replace the generic Gaussian-CFM fallback; it currently ignores an ALI generator in its validation path too |
| Student boundary | Free empirical source density with a specified fitting/anchoring policy |
| Metrics | W2 in physical coordinates, separate legacy and common-protocol scores, and four held-out marginals |
| Tracking | Reuse the existing single-cell reference hashes and online W&B handling, with OTP-specific config/metrics |

The reusable pieces are `pc.velocity`, marginal likelihood, direct sampling,
gradient/update logic, checkpointing and tracking. The current
`repc.single_cell_eval.integrate` can evaluate the new `teacher(t,x)` interface,
but its data/holdout contract and metric aggregation need adaptation. Its
`emd` uses Euclidean transport cost and computes **W1**, not OTP-FM's **W2**.
For W2, use squared Euclidean cost, normalized empirical masses, and take the
square root of the optimal cost.

Possible new files would be `repc/otpfm.py`, `repc/otpfm_data.py`, and dedicated
teacher/student evaluation and distillation scripts. These names are proposals;
they have not been created in the neighboring repository. The illustrative
code here is not a ready-to-run end-to-end launcher.

**Evaluate the interpolant and frozen target before training students.** Reuse
the selected checkpoint, EMA, coupled tuples, scaler and physical coordinates.
Check the cached linear-system residual and endpoint identities, then measure
the interpolant's marginal W2 and the frozen teacher's conditional-FM diagnostic
on fixed validation tuples/times. Preserve the existing finite-step teacher
scores separately. The analytic path, finite-step learned predictions and
diagonal ODE are distinct references; the previously recorded scores do not
establish the analytic path's accuracy. An ODE reference is optional and can be
evaluated once outside student training. It is not a prerequisite for sampling
distillation batches.

Keep the legacy reproduction score, including Beijing truncation and GoM's
nearby-time minimum, as a separately labeled reproduction reference. For a
common distillation comparison, evaluate interpolant and student at the **exact
observation times** with fixed tuple/sample seeds. Use
proper unequal-size empirical W2 rather than dropping reference observations
to whichever sample count is smaller. Begin with the available source sizes
(744 Beijing, 111 GoM) and fixed source subsets where a validation split is
required. Record every actual sample count, coupling, alpha and solver setting.

Report these references separately:

1. Original finite-step EMA teacher predictions, retaining their evaluation protocol.
2. Direct conditional-interpolant samples from the cached observed tuples.
3. Direct student PC samples, testing its learned density including source fit.

If needed, add teacher and student diagonal-ODE samples from the same empirical
source as a separate vector-field diagnostic. No such rollouts are needed for
student updates. Matching the frozen teacher on interpolant samples does not
guarantee exact recovery of interpolant marginals: that also depends on how
well the learned teacher represents their marginal velocity and on the PC
source density. Report these approximation errors rather than identifying the
conditional interpolant with the learned teacher flow.

Report W2 per timepoint, holdout and observed-time means, student/teacher
differences, validation absolute/relative velocity MSE, source NLL and source
W2, marginal NLL with coordinate conventions, sampling throughput, NFE and
model sizes. Choose the student checkpoint by validation velocity MSE, with a
previously fixed source-density policy. Holdout W2 is a diagnostic, not a
student hyperparameter-selection target. This does not undo the teacher's
earlier benchmark-based selection; report that provenance honestly.

For future runs use W&B project `flow2pc` and distinct groups such as
`otpfm-beijing-w2-teacher44` and `otpfm-gom-w2-teacher44`. First log a separate
teacher/interpolant reference containing checkpoint/config/data/source hashes,
coupling/cache hashes, exact times, path type `conditional_otpfm_cached`, fixed
alpha, and measured reference scores. Record ODE tolerances only if that optional
diagnostic is run. Student
runs should record that reference hash, PC boundary policy, objective and selected
checkpoint, per-timepoint metrics, plots and artifacts. No such new W&B run or
interpolant evaluation or ODE reference was created while writing this document.

The reviewed Flow2PC HEAD was
`a691b0c815c3f4b30b9049473b12d4bee9b9f2ab`, with existing uncommitted changes
including `repc/flow.py`, `repc/data.py` and README edits. The manifest captures
the reviewed source contents rather than treating HEAD as a clean snapshot.
Those edits were left untouched. Existing checkpoints, datasets and reproduction
documents were retained. Only this README and its local teacher manifest were
added for this task.
