# Codex instructions — Flow2PC

This repository is a research project that **reconfigures pretrained flow-matching teachers into time-dependent probabilistic circuits**. Read `docs/ARCHITECTURE_SPEC.md` for the authoritative proposed pipeline and `README_PROPOSAL.md` for the mentor-facing overview. These docs are a **proposal**: inspect the live code and clearly distinguish implemented from planned behavior.

## Architecture boundaries

- Shared `flow2pc.data`: loaders/metadata for synthetic and real multi-marginal datasets, without bundling full datasets in Git.
- `flow2pc.pretrain.models`: independently trained original methods **MFM, MMFM, OTP-FM, ALI-CFM** behind common registry and checkpoint/velocity adapter contracts. Their training loops/losses remain method-specific.
- `flow2pc.distill`: loads **frozen teacher checkpoints**, makes valid time/state queries, and trains PC students in `flow2pc.pc`.
- Teacher code must not import `distill` or `pc`. Datasets must not import teacher code.
- The long-term CLI is `python main.py --config <config.yaml>` with `stage: pretrain|distill`.

## Scientific non-negotiables

- Keep physical observation time, model-normalized time, and preprocessing metadata explicit and consistent.
- Do not invent trajectory pairings in unpaired marginal data.
- Do not assume the current MNIST noise-to-data path is valid for every multi-marginal teacher.
- Do not assume a standard Gaussian initial PC distribution fits real earliest-time marginals. Flag incompatibility and obtain a design decision.
- Do not replace published algorithms with a generic TorchCFM loss or claim placeholder models are actual reproductions.
- Respect optional method-specific conditioning; a velocity adapter must fail clearly if required context is missing.

## Existing repo compatibility

- The original PC baseline is `3eb65ef67e8643dd844c9a9e819899316ffff542`. MNIST trainers, evaluation, classifier, benchmark commands and their wrappers have since been retired at the user's explicit request.
- Inspect `src/flow2pc/pc/`, `src/flow2pc/pretrain/models/alicfm/`, the shared utilities, and relevant tests before editing. Keep the HCLT model and all directly or indirectly imported helpers intact.
- Preserve real-data distillation model/checkpoint/teacher interfaces. Before archiving a file, inspect its callers and transitive imports. Do not remove mixed-purpose files just because some of their branches are unused.
- The user requested scripts only: do not run training, distillation, evaluation or tests without a new explicit request. Read-only inspection and file edits remain allowed.
- Reuse and test existing PC implementations rather than silently rewriting their mathematics.
- Never automatically download large datasets or commit raw data, processed arrays, checkpoints, logs, or secrets.

## Development workflow

1. Read current code and the spec; produce a short, scoped implementation plan.
2. Work on **one approved phase at a time**. Phase 1 is the installable skeleton/CLI/registries/toy dataset/checkpoint contract, **not all four research paper integrations**.
3. Make minimal reviewable changes, add tests for contracts and regressions, and run available lint/tests.
4. State precisely what was implemented, which tests were actually run, limitations, and remaining integration work.
5. Do not commit/push, install system-wide dependencies, vendor unlicensed code, or change scientific semantics without explicit request.

## Shared ownership

- Contributor A: ALI-CFM and OTP-FM method folders.
- Contributor B: MFM and MMFM method folders.
- Shared registries/data/CLI/checkpoint utilities: coordinated changes and review.

When instructions conflict or science semantics are unclear, **report the conflict and propose options instead of guessing**.
