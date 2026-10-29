"""Compatibility entry point; implementation lives in flow2pc.legacy."""
import importlib
import sys

_implementation = importlib.import_module('flow2pc.legacy.train_mnist_classifier')

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
