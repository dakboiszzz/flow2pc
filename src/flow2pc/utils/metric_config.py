"""Translate the unified schema to existing backend settings without changing defaults.

Relative paths retain the historical repository-root convention.
"""
import copy
from pathlib import Path

import yaml

from flow2pc.utils.experiment import ExperimentConfig
MODELS = {"mfm": {"load": True, "pretrain": True}}

DATASETS = ('arch', 'sphere', 'cite', 'multi', 'eb', 'lidar', 'afhq')


def mapping(value, label):
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
        from flow2pc.pretrain.models.mfm.trainer import prepare
        return stage, prepare(c)
    output = mapping(c.get('output', {}), 'output')
    if set(output) - {'dir'}:
        raise ValueError('output only supports dir')
    if stage == 'evaluate':
        e = mapping(c.get('evaluation'), 'evaluation')
        if set(e) - {'backend', 'checkpoint', 'device', 'params'}:
            raise ValueError('Unknown evaluation keys')
        if e.get('backend') != 'mfm' or not e.get('checkpoint'):
            raise ValueError('Standalone evaluation currently requires backend: mfm and checkpoint')
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
    if 'root' in data:
        d['data_dir'] = data['root']
    st = dict(mapping(student.get('params', {}), 'student.params'))
    if 'name' in student:
        st['pc_structure'] = student['name']
    exp = ExperimentConfig(teacher=t, dataset=d, student=st, training=settings,
                           evaluation=c.get('evaluation', {}), wandb=c.get('wandb', {}))
    if name == 'mfm':
        from flow2pc.utils.mfm_config import validate_config
        validate_config(exp)
    return stage, exp
