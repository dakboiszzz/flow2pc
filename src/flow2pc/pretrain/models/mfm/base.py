"""Common teacher result; raw model/state dicts retain their original layout."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoadedTeacher:
    model: Any
    metadata: dict
    flow_matching_config: dict = field(default_factory=dict)
    generator: Any = None
    conditional: bool = False
    matcher_factory: Any = None
    path_sampler: Any = None

    def velocity(self, values, time, condition=None):
        if self.conditional:
            if condition is None:
                raise ValueError("This teacher requires condition")
            return self.model(time, values, condition)
        if condition is not None:
            raise ValueError("This teacher does not accept condition")
        return self.model(time, values)
