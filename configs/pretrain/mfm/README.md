# MFM pretraining

`upstream/` contains all 22 YAML files from the MFM source checkout, unchanged.
Five are CFM references (`mfm: false`). `arch.yaml` and the 16 YAML files in
`runs/` are Flow2PC wrappers for the 17 MFM configurations. Each wrapper calls
the authentic two-phase trainer in `pretrain/metric-flow-matching`, using the
copied native configuration and objective. Change `model.params.config_path`,
`data.name`, `data.root`, and `output.dir` together for another MFM dataset.
The native configuration must declare `mfm: true`, a matching data type, and
`seeds: [<top-level seed>]`.

For example:

```bash
python main.py --config configs/pretrain/mfm/runs/single_cell/5dims/ot-mfm_cite.yaml --validate-only
```

MFM's original CLI expects data under `<working_dir>/data`. This integration
creates `<output.dir>/data` as a link to `data.root` when training starts.
GeoPath and flow checkpoints remain in the original Lightning format under
`<output.dir>/checkpoints/` and can be used by the existing MFM adapter.
