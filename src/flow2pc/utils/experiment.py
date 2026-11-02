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
    from flow2pc.distill.mfm import run

    if config.teacher.get("name") != "metric_flow":
        raise ValueError("This runner only supports MFM")
    return run(config)
