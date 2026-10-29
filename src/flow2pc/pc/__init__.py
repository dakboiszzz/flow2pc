"""Original literature-aligned PCs, organized under one implementation owner.

The repc namespace provides compatibility aliases to these exact class objects.
"""

from importlib import import_module

_EXPORTS = {
    "ProbabilisticCircuit": "flow2pc.pc.base",
    "TimeDependentGaussianPC": "flow2pc.pc.gaussian_pc",
    "TimeDependentHCLTPC": "flow2pc.pc.hclt_pc",
    "RegionGraphPC": "flow2pc.pc.region_pc",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(_EXPORTS[name]), name)
