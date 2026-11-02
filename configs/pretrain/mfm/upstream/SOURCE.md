# MFM configuration snapshot

The 22 YAML files under this directory were copied byte-for-byte from
`/data/pqhung_group/pdtung/pretrain/metric-flow-matching/configs` on 2026-10-31.
The source checkout was at commit `e44e03ec6207d2b111c1d51a03dcc53a98f87b95`
with local, uncommitted changes to its configurations. This directory therefore
records the live checkout, not a clean Git revision. The accompanying `LICENSE`
is the source repository's MIT license.

Seventeen YAML files set `mfm: true`; the other five set `mfm: false` and are
CFM reference configurations. The Flow2PC `stage: pretrain` MFM launcher only
accepts the former. Runnable Flow2PC wrappers are in `../runs/`, with
`../arch.yaml` covering `arch/ot-mfm.yaml`.
