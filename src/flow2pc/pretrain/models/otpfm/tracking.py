"""Adapt the original reproduction callback to the shared local/W&B logger."""

import json

from ._source.reproduction import RunLogger


class Summary:
    def __init__(self, logger):
        self.logger = logger

    def update(self, values):
        if self.logger.run is not None:
            with self.logger.remote("OTP-FM summary"):
                self.logger.run.summary.update(values)

    def __setitem__(self, name, value):
        self.update({name: value})


class RunSink:
    def __init__(self, logger):
        self.logger, self.summary = logger, Summary(logger)

    def define_metric(self, *args, **kwargs):
        if self.logger.run is not None:
            with self.logger.remote("OTP-FM metric definition"):
                self.logger.run.define_metric(*args, **kwargs)

    def log(self, values):
        try:
            json.dumps(values, allow_nan=False)
        except TypeError:
            if self.logger.run is not None:
                with self.logger.remote("OTP-FM media"):
                    self.logger.run.log(values)
        else:
            self.logger.log(values)


class NativeObserver:
    def __init__(self, logger, folder, dataset, native_config):
        self.original = RunLogger(
            RunSink(logger), folder, dataset, native_config["holdout_times"],
            selection_rule=native_config.get("checkpoint_selection", "mean_w2"),
        )
        self.best_state = None
        self.best_step = None
        self.best_alpha = None

    @property
    def best(self):
        return self.original.best

    def __call__(self, trainer, epoch):
        previous = self.original.best
        self.original(trainer, epoch)
        if self.original.best is not previous:
            # Copying saved model values consumes no random draws and does not
            # change any native forward/loss/backward/optimizer/EMA operation.
            self.best_state = {name: value.detach().cpu().clone() for name, value in trainer.model.state_dict().items()}
            self.best_step = trainer.global_step
            history = trainer.losses.get("otp_alpha", [])
            self.best_alpha = float(history[-1][1]) if history else float(trainer.curriculum(0))
