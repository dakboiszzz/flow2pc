"""Real-data ALI-CFM compatibility imports; MNIST code is retired."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("flow2pc.pretrain.models.alicfm.checkpoint_adapter")
