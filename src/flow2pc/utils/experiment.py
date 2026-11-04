"""Configuration value object for the original MFM distillation runner."""

from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    teacher: dict
    dataset: dict
    student: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)
    wandb: dict = field(default_factory=dict)
    task: str = "distill"


def run_experiment(config):
    from flow2pc.distill.runner import run

    if config.teacher.get("name") not in ("metric_flow", "mmfm"):
        raise ValueError("This runner supports MFM and MMFM")
    return run(config)
