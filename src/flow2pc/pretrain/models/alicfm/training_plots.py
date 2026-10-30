"""Native training callbacks; random draws remain part of the training stream."""

from pathlib import Path

from flow2pc.pretrain.models.alicfm._source import training_utils


def native_plot_function(dataset, evaluation):
    if dataset.metadata.name != "st":
        return training_utils.sc_plot_fn
    from flow2pc.pretrain.models.alicfm._source.st_plot import Plotter

    directory = Path(evaluation.get("image_dir") or (
        dataset.root / "spatial_transcriptomics/aligned/ref_U5_warped_images"
    ))
    return Plotter(
        str(directory), list(dataset.metadata.model_times), coordinate_scaling=1
    ).plot_fn


def bind_logger(plot_function, logger):
    import wandb

    class Sink:
        Image = wandb.Image

        def log(self, values):
            if logger.run is not None:
                logger.run.log(values)

    def callback(*args):
        values = list(args)
        values[-2] = Sink()
        return plot_function(*values)

    return callback
