# ALI-CFM source attribution

The `_source/` modules were migrated from the user-supplied `../ali-cfm`
checkout and its ST-era Git history of **Multi-Marginal Flow Matching with Adversarially Learnt
Interpolants**, by Oskar Kviman, Kirill Tamogashev, Nicola Branchini,
Víctor Elvira, Jens Lagergren, and Esmeralda S. Whitammer.

Repository: https://github.com/mmacosha/adversarially-learned-interpolants

The supplied upstream `pyproject.toml` declares `license = { text = "MIT" }`
and lists Oskar Kviman, Kirill Tamogashev and Nicola Branchini as authors.
No standalone LICENSE file was present in that checkout. This notice records
the supplied declaration; it does not assign authorship of that code to the
Flow2PC contributors. Source revisions, exact file hashes and migration edits
are recorded in `SOURCE_PROVENANCE.json`.

Network layers, state-dict keys, regularizers, OT sampling, staged objectives,
and optimizer updates are retained from the recorded local source revisions,
including its previously applied numerical repairs. This is not a claim that
all experiments reproduce the paper. The new trainer selects and validates a
supported configuration and handles logging/checkpoints outside the algorithm.

The ST profile restores the matching piecewise method, derivative and training
loops from `501e9db6a3bcb1b96eb893882b62a56f78fe1823`, before the later
single-cell refactor made its configured call incompatible. This selection is
isolated from the current single-cell profile; it does not replace ST's
regularizer with a different branch. See `docs/ALICFM_DATASETS.md` for the
preprocessing and reproduction limits.
