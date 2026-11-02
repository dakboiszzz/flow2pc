# Local launch recipes and the recovered multi-GPU queue

`local_runs/` is ignored by Git. Put machine-specific launch recipes there; shared
launching code stays in `scripts/` and `src/flow2pc/utils/`. The local recipe
prepared on this machine is `local_runs/check_all_alicfm.sh`. Ignored recipes
will not be included in a push; the equivalent tracked command is shown below.

The scheduler reuses `choose_device`, `next_task` and `stop_workers` from the
previous successful `run_single_cell_all_sweeps.py` implementation. The tracking
scheduler also reused those helpers. Their function bodies remain equivalent;
`SCHEDULER_SOURCE.json` records the original source. The controller adopts the
later `nvidia-smi` memory-monitor approach instead of `torch.cuda.mem_get_info()`.
It imports no Torch and allocates no CUDA context.

## Preview and run

```bash
# Inspect the plan, GPU memory and completed-run reuse. No jobs or outputs.
bash local_runs/check_all_alicfm.sh --dry-run

# Run all ten dataset checks across the visible GPUs.
bash local_runs/check_all_alicfm.sh

# Select physical nvidia-smi GPU indices and keep two workers per GPU.
GPUS=0,1,2,3 WORKERS_PER_GPU=2 bash local_runs/check_all_alicfm.sh

# Select only some datasets, or repeat evaluation without retraining.
bash local_runs/check_all_alicfm.sh --datasets st knot cell_tracking
bash local_runs/check_all_alicfm.sh --rerun-evaluation
```

The ten tags are `cite5 cite50 cite100 multi5 multi50 multi100 eb st cell_tracking
knot`. These are ten configured checks, not a hyperparameter grid. Teacher
training uses the original full budgets when no teacher checkpoint is present.
Students use their configured full budgets and resume supported checkpoints.

The equivalent command available in Git on another machine is:

```bash
python scripts/run_alicfm_checks.py --gpus auto --workers-per-gpu 2
```

`FLOW2PC_PYTHON` selects the interpreter used by the local recipe. Activate the
existing environment with the project dependencies before launching.

## Concurrency and reuse

Each dataset queues its teacher training/reference evaluation first. Its student
becomes eligible as soon as that teacher is ready; it does not wait for all
other teachers. Completed matching teacher/student runs with evaluation metrics
and figure files are skipped. Finished workers return their slots to the shared
pool, and tasks rotate between datasets to avoid starving a dataset.

Defaults allow two workers per GPU. The old 30-second warm-up reservation is
retained, with stage-specific memory accounting. By default a teacher needs
4 GiB free to launch and a student needs 8 GiB. These are launch thresholds,
not guarantees of peak usage or reservations against unrelated processes.
The scheduler keeps useful lighter tasks eligible while students wait for space.
Allocation failures with a usable checkpoint can retry once with a 60-second
GPU cooldown. It does not silently reduce model size, batch size, or budgets.

You can adjust `TEACHER_MIN_FREE_GB`, `STUDENT_MIN_FREE_GB`, `CPU_THREADS_PER_RUN`,
`WORKERS_PER_GPU`, and `GPUS` in the local recipe/environment. CLI flags include
`--max-parallel`, `--workers-per-gpu`, `--teacher-min-free-gb`,
`--student-min-free-gb`, `--min-free-gb` (both stages), and `--oom-retries`.

GPUs are physical `nvidia-smi` indices or GPU UUIDs. An inherited
`CUDA_VISIBLE_DEVICES` allocation is respected. Each child receives its selected
GPU UUID and uses `cuda:0` internally. The launcher requires at least two GPUs;
it does not silently fall back to a single GPU or CPU. W&B initialization uses
the previous shared startup-lock convention; training runs concurrently once
initialization finishes. Every worker explicitly logs online to project `flow2pc`.

Only resumable allocation failures are retried. Teacher training currently saves
its complete artifact at the end, so an earlier failed teacher training attempt
cannot resume from a partial directory. Such directories are preserved and
reported, while other datasets continue. Student runs now save a step-zero
checkpoint before their first update, as well as normal best/last checkpoints.

## Files and control

Defaults keep the existing check layout:

```text
/data/lthung/flow2pc/framework/migration-check/
  cite5/{teacher,student}/...
  cite50/{teacher,student}/...
  ...
  _launcher/
    status.json
    summary.csv
    logs/cite50-teacher.log
    logs/cite50-student.log
    ...
```

Per-run `eval/media.json` lists every figure path. W&B uses the same persistent
teacher/student run IDs as the sequential check script. The CSV includes
teacher EMD references beside student EMD; each dataset retains its own units
and protocol, so the table is not a cross-dataset ranking.

Set `FLOW2PC_DATA_ROOT` for the input data and `FLOW2PC_CHECKS_ROOT` for the common
output root. The plural variable is a base containing one subfolder per dataset;
the sequential script's `FLOW2PC_CHECK_ROOT` specifies a single dataset's folder.
Configuration changes to scientific settings require a fresh output root.

One shared launcher owns an output root at a time. Dataset locks are shared with
`check_alicfm_pipeline.sh`, preventing duplicate checkpoint/W&B writers.
Ctrl-C or SIGTERM stops only this launcher's worker process groups. Existing
checkpoints and logs remain available for resuming.

The launcher has received static checks only. No training, evaluation or tests
were run while preparing it.
