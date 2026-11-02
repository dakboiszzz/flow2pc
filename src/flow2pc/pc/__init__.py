"""Student model API; repc retains aliases for the existing model imports."""

from importlib import import_module

_EXPORTS = {
    "ProbabilisticCircuit": "flow2pc.pc.base",
    "TimeDependentHCLTPC": "flow2pc.pc.hclt_pc",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(_EXPORTS[name]), name)
