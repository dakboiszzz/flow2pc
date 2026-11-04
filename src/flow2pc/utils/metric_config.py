"""Translate the unified schema to existing backend settings without changing defaults.

Relative paths retain the historical repository-root convention.
"""
import copy
from pathlib import Path

import yaml

from flow2pc.utils.experiment import ExperimentConfig
from flow2pc.utils.paths import REPO_ROOT
MODELS = {name: {"load": True, "pretrain": True} for name in ("mfm", "mmfm")}

DATASETS = ('arch', 'sphere', 'cite', 'multi', 'eb', 'lidar', 'afhq', 'mfm_snapshots')


def mapping(value, label, allowed=None):
    if allowed is not None and (not isinstance(value, dict) or set(value) - set(allowed)):
        raise ValueError(f"Invalid {label}; allowed keys: {sorted(allowed)}")
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be a mapping')
    return value


def load_config(path, overrides=()):
    config = mapping(yaml.safe_load(Path(path).read_text()), 'config')
    for override in overrides:
        key, sep, value = override.partition('=')
        if not sep or not key or any(not part for part in key.split('.')):
            raise ValueError('--set requires KEY=VALUE')
        target = config
        parts = key.split('.')
        for part in parts[:-1]:
            target = mapping(target.setdefault(part, {}), part)
        target[parts[-1]] = yaml.safe_load(value)
    return config


def prepare(raw):
    c = copy.deepcopy(raw)
    allowed = {'stage', 'seed', 'model', 'teacher', 'data', 'student',
               'training', 'distillation', 'evaluation', 'output', 'wandb'}
    if set(c) - allowed:
        raise ValueError(f'Unknown config keys: {sorted(set(c) - allowed)}')
    stage = c.get('stage')
    if stage not in ('pretrain', 'distill', 'evaluate'):
        raise ValueError('stage must be pretrain, distill, or evaluate')
    if stage == 'pretrain':
        if c.get('model', {}).get('name') == 'mmfm':
            return stage, prepare_mmfm(c)
        from flow2pc.pretrain.models.mfm.trainer import prepare as prepare_mfm

        return stage, prepare_mfm(c)
    output = mapping(c.get('output', {}), 'output')
    if set(output) - {'dir'}:
        raise ValueError('output only supports dir')
    if stage == 'evaluate':
        e = mapping(c.get('evaluation'), 'evaluation')
        if set(e) - {'backend', 'checkpoint', 'device', 'params', 'role'}:
            raise ValueError('Unknown evaluation keys')
        if e.get('backend') not in ('mfm', 'mmfm') or not e.get('checkpoint'):
            raise ValueError('Standalone evaluation requires backend: mfm/mmfm and checkpoint')
        if e.get('role', 'student') not in ('teacher', 'student'):
            raise ValueError('evaluation.role must be teacher or student')
        if e['backend'] == 'mfm' and e.get('role', 'student') != 'student':
            raise ValueError('MFM stage: evaluate expects a student checkpoint')
        return stage, {**e, "output_dir": output.get("dir")}
    data = mapping(c.get('data'), 'data')
    if set(data) - {'name', 'root', 'params'}:
        raise ValueError('data supports name, root, params')
    if data.get('name') not in DATASETS:
        raise ValueError(f'Unsupported data: {data.get("name")}; only MFM datasets are enabled; DGP-i remains deferred')
    teacher = mapping(c.get('teacher'), 'teacher')
    if set(teacher) - {'name', 'checkpoint', 'generator_checkpoint', 'params'}:
        raise ValueError('Unknown teacher keys')
    name = teacher.get('name')
    if name not in MODELS or not MODELS[name]['load']:
        raise ValueError(f'Teacher is unavailable: {name}')
    if not teacher.get('checkpoint'):
        raise ValueError('teacher.checkpoint is required')
    student = mapping(c.get('student', {}), 'student')
    if set(student) - {'name', 'params'}:
        raise ValueError('student supports name and params')
    settings = dict(mapping(c.get('distillation', {}), 'distillation'))
    if 'seed' in c:
        if 'seed' in settings:
            raise ValueError('Set seed once at the top level')
        settings['seed'] = c['seed']
    if 'dir' in output:
        if 'output_dir' in settings:
            raise ValueError('Set output.dir instead of distillation.output_dir')
        settings['output_dir'] = output['dir']
    t = {'name': 'metric_flow' if name == 'mfm' else name,
         'checkpoint_path': teacher['checkpoint']}
    if 'params' in teacher:
        t['options'] = dict(mapping(teacher['params'], 'teacher.params'))
    if teacher.get('generator_checkpoint'):
        t['generator_path'] = teacher['generator_checkpoint']
    d = {'name': data['name'], **mapping(data.get('params', {}), 'data.params')}
    if name == 'mmfm':
        d = {'name': data['name'], 'options': mapping(data.get('params', {}), 'data.params')}
    if 'root' in data:
        d['data_dir'] = data['root']
    st = dict(mapping(student.get('params', {}), 'student.params'))
    if 'name' in student:
        st['pc_structure'] = student['name']
    exp = ExperimentConfig(teacher=t, dataset=d, student=st, training=settings,
                           evaluation=c.get('evaluation', {}), wandb=c.get('wandb', {}))
    if name in ('mfm', 'mmfm'):
        from flow2pc.utils.mfm_config import validate_config
        validate_config(exp)
    return stage, exp


def _resolved_path(value):
    value = Path(value).expanduser()
    return str(value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve())


def prepare_mmfm(raw):
    """Resolve native MMFM settings without importing its model or loading data."""
    import math

    from flow2pc.data.metric_profiles import vector_profile
    from flow2pc.data.mfm_catalog import profile as snapshot_profile
    from flow2pc.utils.paths import MFM_PRETRAIN_ROOT, MMFM_ROOT

    c = copy.deepcopy(raw)
    mapping(c, "MMFM config", {"stage", "model", "data", "training", "seed", "output", "evaluation", "wandb"})
    if c.get("stage") != "pretrain" or c.get("model", {}).get("name") != "mmfm":
        raise ValueError("MMFM pretraining requires stage: pretrain and model.name: mmfm")
    mapping(c["model"], "model", {"name", "params"})
    params = mapping(c["model"].get("params", {}), "model.params", {
        "source_root", "x_latent_dim", "time_embed_dim", "num_out_layers", "activation",
        "conditional_model", "interpolation", "flow_variance", "coupling",
    })
    source_root = _resolved_path(params.get("source_root", MMFM_ROOT))
    seed = c.get("seed", 42)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    data = mapping(c.get("data", {}), "data", {"name", "root", "params"})
    name = data.get("name")
    options = mapping(data.get("params", {}), "data.params", {
        "source_root", "whiten", "heldout_index", "train_fraction", "profile",
    })
    if name == "mfm_snapshots":
        spec = snapshot_profile(options.get("profile"))
        if set(options) - {"profile"}:
            raise ValueError("Prepared snapshots accept only data.params.profile")
        profile = {"repo_path": str(MFM_PRETRAIN_ROOT), "args": {"dim": spec["dim"], "whiten": False}, "profile": options["profile"]}
    else:
        profile = vector_profile(name, seed, **options)
    profile["repo_path"] = _resolved_path(profile["repo_path"])
    if type(profile["args"]["whiten"]) is not bool:
        raise ValueError("data.params.whiten must be boolean")
    model_config = {
        "data_dim": profile["args"]["dim"], "conditional_model": False,
        "x_latent_dim": params.get("x_latent_dim", 64),
        "time_embed_dim": params.get("time_embed_dim", 64),
        "num_out_layers": params.get("num_out_layers", 3),
        "activation": params.get("activation", "LeakyReLU"),
        "normalization": None,
    }
    if params.get("conditional_model", False) is not False:
        raise ValueError("This integration supports unconditional MMFM only")
    for key in ("x_latent_dim", "time_embed_dim", "num_out_layers"):
        if type(model_config[key]) is not int or model_config[key] < 2:
            raise ValueError(f"model.params.{key} must be an integer >= 2")
    if model_config["activation"] not in ("SELU", "ReLU", "LeakyReLU"):
        raise ValueError("Unsupported native MMFM activation")
    matcher = {
        "coupling": params.get("coupling", "minibatch_ot"),
        "interpolation": params.get("interpolation", "cubic"),
        "flow_variance": params.get("flow_variance", "adaptive2-4-0.01"),
    }
    if matcher["coupling"] != "minibatch_ot" or matcher["interpolation"] not in ("cubic", "linear", "lagrange"):
        raise ValueError("Use minibatch_ot and a native cubic/linear/lagrange interpolant")
    sigma = matcher["flow_variance"]
    if isinstance(sigma, (int, float)) and not isinstance(sigma, bool):
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("flow_variance must be finite and nonnegative")
    elif isinstance(sigma, str):
        pieces = sigma.split("-")
        if len(pieces) != 3 or pieces[0] not in ("adaptive1", "adaptive2", "adaptive3", "adaptive4"):
            raise ValueError("Invalid native MMFM adaptive variance")
        maximum, minimum = float(pieces[1]), float(pieces[2])
        if not (math.isfinite(maximum) and math.isfinite(minimum) and maximum > 0 and minimum >= 0):
            raise ValueError("Invalid adaptive variance coefficients")
    else:
        raise ValueError("Invalid flow_variance")
    defaults = {
        "epochs": 300, "batch_size": 128, "lr": 0.01, "gradient_clip": 1.0,
        "device": "auto", "steps_per_epoch": None, "validation_batches": 4,
        "resume_checkpoint": None,
    }
    training = {**defaults, **mapping(c.get("training", {}), "training", defaults)}
    for key in ("epochs", "batch_size", "validation_batches"):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    count = training["steps_per_epoch"]
    if count is not None and (type(count) is not int or count < 1):
        raise ValueError("steps_per_epoch must be null or a positive integer")
    for key in ("lr", "gradient_clip"):
        if not math.isfinite(float(training[key])) or float(training[key]) <= 0:
            raise ValueError(f"training.{key} must be positive and finite")
    if training["resume_checkpoint"]:
        training["resume_checkpoint"] = _resolved_path(training["resume_checkpoint"])
    output = mapping(c.get("output", {}), "output", {"dir"})
    if not output.get("dir"):
        raise ValueError("output.dir is required")
    evaluation = {"enabled": True, "samples": 128, "projections": 32, "ode_substeps": 16, "seed": 1234,
                  **mapping(c.get("evaluation", {}), "evaluation", {"enabled", "samples", "projections", "ode_substeps", "seed"})}
    for key in ("samples", "projections", "ode_substeps"):
        if type(evaluation[key]) is not int or evaluation[key] < 1:
            raise ValueError(f"evaluation.{key} must be a positive integer")
    wandb = {"mode": "disabled", **mapping(c.get("wandb", {}), "wandb", {"mode", "project", "group", "run_name"})}
    if wandb["mode"] not in ("disabled", "online", "offline"):
        raise ValueError("Invalid wandb.mode")
    return {
        "method": "mmfm", "source_root": source_root, "model_config": model_config,
        "matcher": matcher, "seed": seed, "dataset_profile": profile,
        "dataset_config": {
            "name": name, "data_dir": _resolved_path(data.get("root", MFM_PRETRAIN_ROOT / "data")),
            "seed": seed, "train_fraction": options.get("train_fraction", 0.9),
            "batch_size": training["batch_size"],
        },
        "training": training, "output_dir": _resolved_path(output["dir"]),
        "evaluation": evaluation, "wandb": wandb,
    }
