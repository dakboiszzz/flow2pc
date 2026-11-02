"""Lazy registries: listing methods never loads models, data or GPUs."""

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class Entry:
    target: str | None
    description: str

    @property
    def status(self):
        return "implemented" if self.target is not None else "unavailable"


class Registry:
    def __init__(self, kind):
        self.kind = kind
        self._entries = {}

    def register(self, name, target, description):
        if name in self._entries:
            raise ValueError(f"Duplicate {self.kind}: {name}")
        if target is not None and ":" not in target:
            raise ValueError("Targets must have the form module:attribute")
        self._entries[name] = Entry(target, description)

    def describe(self):
        return {
            name: {"status": entry.status, "description": entry.description}
            for name, entry in sorted(self._entries.items())
        }

    def require(self, name):
        if not isinstance(name, str) or name not in self._entries:
            raise ValueError(
                f"Unknown {self.kind} {name!r}; known names: "
                + ", ".join(sorted(self._entries))
            )
        entry = self._entries[name]
        if entry.target is None:
            raise NotImplementedError(f"{name}: {entry.description}")
        return entry

    def get(self, name):
        module, attribute = self.require(name).target.split(":", 1)
        return getattr(import_module(module), attribute)
