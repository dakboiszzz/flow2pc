"""Small explicit registry with lazy imports and duplicate protection."""

from importlib import import_module


class Registry:
    def __init__(self, kind):
        self.kind = kind
        self._entries = {}

    def register(self, name, factory):
        if name in self._entries:
            raise ValueError(f"Duplicate {self.kind}: {name}")
        self._entries[name] = factory

    def names(self):
        return tuple(self._entries)

    def get(self, name):
        if name not in self._entries:
            raise ValueError(
                f"Unknown {self.kind}: {name}; available: {', '.join(self.names())}"
            )
        factory = self._entries[name]
        if isinstance(factory, str):
            module, attr = factory.split(":")
            factory = getattr(import_module(module), attr)
        return factory
