# Completed ALI-CFM native-versus-migrated comparison

Inspected 2026-10-30; full comparison completed at 17:54 UTC. For the six
single-cell recipes and ST, seed 42 / held-out index 1, the migrated training
reproduced the native training exactly through the final CFM update. This is
numerical evidence from completed runs, superseding the earlier partial-trace
and static-only assessments for these specific configurations.

## Evidence

| Dataset | Pretrain updates | GAN updates | CFM updates | All logged training scalars | Saved G / D / CFM tensors |
| --- | ---: | ---: | ---: | --- | --- |
| CITE5 | 2,000 | 70,000 | 2,001 | Exactly equal at every update | Byte-identical |
| CITE50 | 2,000 | 70,000 | 2,001 | Exactly equal at every update | Byte-identical |
| CITE100 | 2,000 | 70,000 | 2,001 | Exactly equal at every update | Byte-identical |
| Multi5 | 2,000 | 70,000 | 2,001 | Exactly equal at every update | Byte-identical |
| Multi50 | 2,000 | 70,000 | 2,001 | Exactly equal at every update | Byte-identical |
| Multi100 | 2,000 | 70,000 | 2,001 | Exactly equal at every update | Byte-identical |
| ST | 1,000 | 70,000 | 10,000 | Exactly equal at every update | Byte-identical |

There were **2,975,006 scalar comparisons and zero differences**. The comparison
included pretrain MSE, GAN discriminator loss, total generator loss, adversarial
generator loss, regularization term, real/fake probabilities, and CFM velocity
MSE. Every expected update was present; neither side had unmatched entries after
stage/step alignment. No smoothing, rounding or tolerance was used for equality.

All three model state dictionaries matched for all seven datasets: 21 model
states containing 140 tensors. Per-model SHA256 fingerprints include ordered
parameter names, shapes, dtypes and contiguous tensor bytes. Native checkpoint
`ot_cfm_model` maps to migrated `cfm_model`; interpolant and discriminator keys
match directly. The serialized checkpoint files differ because Flow2PC also
stores optimizer/RNG/manifest metadata; the trained model tensors are identical.
Each migrated checkpoint's file hash also matched its saved manifest, and the
selected native source hashes matched the saved source provenance. Seeds were
42 on both sides.

Native ST is the working historical implementation at `501e9db`, as recorded by
the control manifest. Single-cell uses the supplied current working checkout.
ST's native W&B definition injects a stale `t=1_seed=42_cfm_step=0` alongside
the actual `t=1_seed=42_cfm_cfm_step` written by the CFM loop. Comparison uses
the latter. For example, native/migrated ST CFM losses at updates 0, 1 and 9,999
are respectively 0.009881116449832916, 0.12566649913787842 and
0.005138293839991093 on both sides.

## The two native failures

Native CITE5 and CITE100 finished pretraining, GAN training, CFM training and
held-out EMD calculation, then saved their checkpoints. Both subsequently failed
at `wandb.log({'interpolant_results': wandb.Table(...)})`, when W&B tried to
authenticate while creating a table artifact:

```text
AuthenticationError: Failed to verify credentials with https://api.wandb.ai:
Failed to execute API request: the service process is busy and did not respond in time.
```

The error indicates a W&B service/API timeout during the final upload, not proof
of an invalid API key. No training failure, OOM or missing model weights was found
in these two controls. Final results CSVs are absent because the native driver
writes them after uploading the tables; the final scalar EMDs remain in local
W&B summaries/history. Native retraining is unnecessary for the migration check.

These failures differ from the earlier migrated CITE5/Multi5 empty-split plotting
failure, which was fixed separately. Do not conflate those two batches.

## Saved held-out EMD: different evaluation protocols

| Dataset | Native evaluator EMD | Flow2PC shared evaluator EMD | Matching final CFM loss |
| --- | ---: | ---: | ---: |
| CITE5 | 0.9295303822 | 0.9650360072 | 0.0394868627 |
| CITE50 | 41.0524368286 | 41.8294296390 | 0.6513556242 |
| CITE100 | 48.5560607910 | 49.2685220349 | 1.6808505058 |
| Multi5 | 1.0232651234 | 1.0432501927 | 0.0303399973 |
| Multi50 | 45.3113174438 | 46.7420292909 | 0.6817308664 |
| Multi100 | 52.4802398682 | 53.6775234299 | 1.9722263813 |
| ST | 103.1444702148 | 120.9873878476 | 0.0051382938 |

These EMD columns are not a numerical parity comparison. Native single-cell
evaluation uses complete populations, while the shared evaluation caps source
and metric populations at 2,048. Native ST evaluates full populations with
TorchDyn Dopri5; the saved shared ST protocol uses RK4 and a 512-point metric cap.
The integration and cost-computation implementations also differ. The exact
weights prove the trained networks agree; the EMD difference arises downstream
in the evaluation path. This inspection did not isolate the contribution of
each evaluation setting or establish matched-protocol evaluation parity.

## Artifacts and scope

- Native logs: `/data/lthung/flow2pc/alicfm-native-controls/_launcher/logs/<dataset>.log`.
- Native checkpoints: below each `<dataset>/seed42-t1/**/files/checkpoints/t=1_seed=42_ali_cfm.pth`.
- Migrated runs: `/data/lthung/flow2pc/framework/native-teachers/<dataset>/teacher/`.
- Detailed comparison report: `/tmp/alicfm_final_parity.json`.
- Read-only inspection script used: `/tmp/inspect_alicfm_native_parity.py`.

Only existing logs, configs, manifests and CPU checkpoint tensors were read.
No model was instantiated, no forward pass/ODE rollout/EMD was recomputed, and
no training, evaluation, unit tests or W&B network calls were started. Existing
jobs and outputs were left unchanged.

The demonstrated scope is these seven seed-42 first-fold recipes. It does not
establish other seeds/folds, cell-tracking parity, the new GOM/Beijing adaptation,
or identical evaluation results across differing protocols.
