# MMFM on shared MFM datasets

The 20 profiles in `data/mfm/catalog.json` have MMFM pretrain, HCLT/Gaussian
distill and teacher/student evaluation configs. These are new unconditional
adaptations using native MMFM models and interpolants, with minibatch OT chains.
They are not claims of reproducing the upstream conditional experiments.

Install Flow2PC's `mmfm` extra alongside dependencies needed by the dataset.
The source checkout defaults to `../pretrain/MMFM`; source file hashes are stored
in teacher checkpoints. No upstream source, data or checkpoints are vendored.

Configs read the existing prepared MFM snapshots under
`data/mfm/prepared/PROFILE/seed42`. Prepare them using the existing MFM data
workflow first. The same arrays, splits, normalization and held-out frames are
used by both methods. No data preparation or training is run by a dry-run.

```bash
bash scripts/run_mmfm.sh pretrain cite50 --dry-run
bash scripts/run_mmfm.sh pretrain cite50
bash scripts/run_mmfm.sh distill cite50 hclt
bash scripts/run_mmfm.sh distill cite50 gaussian
bash scripts/run_mmfm.sh evaluate cite50 teacher
bash scripts/run_mmfm.sh evaluate cite50 hclt
```

Replace `cite50` with any catalog profile. Outputs go to `outputs/mmfm/PROFILE/`.
For resume, set `training.resume_checkpoint` for teacher training or
`distillation.resume_checkpoint` for a student, and increase the epoch/step budget.

Native MMFM pretraining learns the conditional-flow target of its stochastic
interpolant. Distillation instead matches the frozen teacher's predicted velocity
using the shared velocity MSE. No NLL is optimized. Validation draws only from
observed validation marginals; temporal holdouts are evaluation references.

Evaluation writes metrics, trajectories and ODE line figures. High-dimensional
figures display the first two coordinates, not PCA recomputed for plotting.
These figures are in teacher input coordinates; they do not reproduce the native
LiDAR terrain background. Numerical behavior has not been verified by running
training, evaluation or tests in this implementation session.
