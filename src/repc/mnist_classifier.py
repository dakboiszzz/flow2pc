"""Compatibility alias for the original repc import path."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module('flow2pc.legacy.mnist_classifier')
