"""Compatibility entry point; implementation lives in flow2pc.legacy."""
import importlib
import sys

_implementation = importlib.import_module('flow2pc.legacy.summarize_dense_tuning')

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
