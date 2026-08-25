# Flow2PC

Flow2PC distills a pretrained flow-matching model into a time-indexed
probabilistic circuit (PC). The PC provides direct samples and exact
log-likelihoods at every time along the learned path.

## Method

For a data sample `x`, Gaussian noise `z0`, and time `t`:

```text
z_t = (1 - t) z0 + t x
L = E ||u_pc(z_t, t) - u_flow(z_t, t)||²
```

The PC uses Gaussian leaves whose means and log-scales are polynomial in time.
Their parameterization gives `p_0 = N(0, I)`. Time-independent sum weights
combine the leaf velocities into a valid PC velocity field.

Two structures share the same API:

| Student | Structure |
|---|---|
| `gaussian` | shallow mixture of Gaussian product nodes |
| `region` | hierarchical image region graph |

```python
pc.sample(num_samples, time)
pc.log_prob(values, time)
pc.sample_path(num_samples, times)
pc.path_log_prob(path_values, times)
```

`sample_path` keeps the circuit choices and Gaussian noise fixed across time,
producing a coupled trajectory through the PC marginals.

## Usage

Install with Python 3.11 or newer:

```bash
python -m pip install -e ".[dev]"
```

Train a flow teacher:

```bash
python train.py flow
```

Distill a region-graph PC:

```bash
python train.py distill \
  --teacher-checkpoint outputs/runs/flow/<run>/checkpoints/last.pt \
  --student region
```

Evaluate samples and exact likelihoods along the PC path:

```bash
python eval.py \
  --run-dir outputs/runs/distill/<run> \
  --times 0,0.25,0.5,0.75,1
```

Each run contains `config.json`, `metrics.csv`, and a self-describing checkpoint.
Evaluation writes path metrics and one sample grid per requested time to
`<run>/eval/`.

## Layout

```text
train.py           flow training and PC distillation
eval.py            path sampling and exact likelihood evaluation
repc/flow.py       flow teacher
repc/pc.py         shared path API and Gaussian PC
repc/region_pc.py  hierarchical region-graph PC
repc/data.py       MNIST loader
```
