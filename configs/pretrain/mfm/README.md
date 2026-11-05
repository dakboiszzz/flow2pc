# MFM pretraining

`upstream/` contains all 22 YAML files from the MFM source checkout, unchanged.
Five are CFM references (`mfm: false`). The 17 YAML files in `native/` are
Flow2PC wrappers for the original MFM configurations. Each wrapper calls
the authentic two-phase trainer in `pretrain/metric-flow-matching`, using the
copied native configuration and objective. Change `model.params.config_path`,
`data.name`, `data.root`, and `output.dir` together for another MFM dataset.
The native configuration must declare `mfm: true`, a matching data type, and
`seeds: [<top-level seed>]`.

For example:

```bash
python main.py --config configs/pretrain/mfm/native/mfm_cite5_ot.yaml --validate-only
```

MFM's original CLI expects data under `<working_dir>/data`. This integration
creates `<output.dir>/data` as a link to `data.root` when training starts.
GeoPath and flow checkpoints remain in the original Lightning format under
`<output.dir>/checkpoints/` and can be used by the existing MFM adapter.

The `mfm_<dataset>.yaml` files directly in this folder use the shared prepared
snapshot pipeline. They export a bound `checkpoints/{flow,geopath}.pt` pair.
Their corresponding students are `configs/distill/mfm/mfm_<dataset>_<student>.yaml`.
Native students remain under `configs/distill/mfm/native/`. These are two
existing data/checkpoint paths for the same method, not additional model types.
