"""Evaluation of unconditional MMFM teacher and student checkpoints."""
from pathlib import Path
import numpy as np
import torch
from flow2pc.utils.checkpoint import atomic_json


def evaluate_teacher_checkpoint(checkpoint, *, output_dir=None, device='cpu', settings=None):
    from flow2pc.pretrain.models.mmfm.adapter import load_teacher, load_teacher_data
    from flow2pc.evaluation.metric_flow import evaluation_seed, integrate, sliced_wasserstein
    options = {'samples': 128, 'projections': 32, 'ode_substeps': 16, 'seed': 1234, **(settings or {})}
    device = torch.device(device)
    teacher = load_teacher({'checkpoint_path': str(checkpoint)}, device)
    data = load_teacher_data(teacher)
    destination = Path(output_dir or Path(checkpoint).resolve().parents[1] / 'reevaluation')
    destination.mkdir(parents=True, exist_ok=True)
    with evaluation_seed(options['seed'], device), torch.no_grad():
        source = data.validation_marginals[0]
        initial = source[torch.randint(len(source), (options['samples'],))].to(device)
        trajectory = integrate(teacher.velocity, initial, data.times.to(device), options['ode_substeps'])
        scores = {}
        for i in range(len(data.times)):
            pool = data.test_marginals[i] if hasattr(data, 'test_marginals') and i in data.heldout_indices else data.validation_marginals[i]
            reference = pool[torch.randint(len(pool), (options['samples'],))].to(device)
            scores[str(i)] = {'teacher_ode_swd': sliced_wasserstein(trajectory[i], reference, options['projections']), 'heldout': i in data.heldout_indices}
        np.savez_compressed(destination / 'trajectories.npz', teacher_ode=trajectory.cpu().numpy(), times=data.times.numpy())
    report = {'method': 'mmfm', 'by_time': scores, 'times': data.times.tolist()}
    save_figures(destination)
    atomic_json(destination / 'evaluation.json', report)
    return report


def evaluate_checkpoint(checkpoint, *, role='student', output_dir=None, device='cpu', settings=None):
    if role == 'teacher':
        return evaluate_teacher_checkpoint(checkpoint, output_dir=output_dir, device=device, settings=settings)
    if role != 'student':
        raise ValueError('Unknown evaluation role')
    from flow2pc.distill.student import build_student_from_config
    from flow2pc.distill.runner import _identity, load_resources
    from flow2pc.evaluation.metric_flow import evaluate
    from flow2pc.utils.experiment import ExperimentConfig
    from flow2pc.utils.mfm_config import checkpoint_training_config, validate_config
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload.get('model_type') != 'mmfm_pc_distill':
        raise ValueError('Expected an MMFM-distilled student checkpoint')
    config = ExperimentConfig(**checkpoint_training_config(payload))
    validate_config(config)
    target = torch.device(device)
    teacher, data = load_resources(config, target)
    if _identity(teacher, data) != payload['identity']:
        raise ValueError('Teacher or dataset differs from checkpoint')
    student = build_student_from_config(payload['model_config']).to(target)
    student.load_state_dict(payload['state_dict'], strict=True)
    destination = Path(output_dir or Path(checkpoint).resolve().parents[1] / 'reevaluation')
    destination.mkdir(parents=True, exist_ok=True)
    report = evaluate(student, teacher, data, {**config.evaluation, **(settings or {})}, target, destination)
    save_figures(destination)
    atomic_json(destination / 'evaluation.json', report)
    return report


def save_figures(directory):
    """Render saved ODE paths; high-dimensional datasets show coordinates 1 and 2."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(directory)
    with np.load(directory / 'trajectories.npz') as arrays:
        times = arrays['times']
        for name in ('teacher_ode', 'student_ode'):
            if name not in arrays:
                continue
            paths = arrays[name]
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111, projection='3d' if paths.shape[-1] == 3 else None)
            for i in range(min(64, paths.shape[1])):
                if paths.shape[-1] == 1:
                    ax.plot(times, paths[:, i, 0], alpha=0.3)
                elif paths.shape[-1] == 3:
                    ax.plot(*paths[:, i, :3].T, alpha=0.3)
                else:
                    ax.plot(*paths[:, i, :2].T, alpha=0.3)
            ax.set_title(name + ' (teacher input coordinates)')
            fig.savefig(directory / f'{name}.png', dpi=160, bbox_inches='tight')
            plt.close(fig)
