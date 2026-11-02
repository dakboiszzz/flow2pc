"""Native ST training Plotter from ALI-CFM 501e9db; selected methods unchanged."""

import os
import numpy as np
import torch as T
import matplotlib.pyplot as plt
from PIL import Image

def denormalize(x, min_max):
    if min_max is None:
        return x
    min_, max_ = (m.to(x.device) for m in min_max)
    return x * (max_ - min_) + min_



class Plotter(object):
    def __init__(self, path_to_images, time_stamps, slide_names=None, coordinate_scaling=100, ds=None):
        self.rgb_images = []
        self.slide_names = [u for u in ["U2", "U3", "U4", "U5"]] if slide_names is None else slide_names
        for u in self.slide_names:
            img_name = f"{u}_aligned_to_U5.tif"
            path_to_image = os.path.join(path_to_images, img_name)
            # load image using PIL and convert to numpy array (PIL natively reads in RGB)
            self.rgb_images.append(np.asarray(Image.open(path_to_image)))
        self.time_stamps = time_stamps
        self.n_slides = len(self.time_stamps)
        self.coordinate_scaling = coordinate_scaling
        self.ds = ds


    def overlay_spot_coordinates(self, s_t, slide, ax):
        # x_list is a list of coordinate matrices, e.g. containing four matrices of interpolants
        # observed is a boolean array indicating if a slide was used during training
        ax.imshow(self.rgb_images[slide])
        # s_t = s_t * self.ds.scale.cpu() + self.ds.shift.cpu() if self.ds is not None else s_t
        x = s_t[:, 1] * self.coordinate_scaling
        y = s_t[:, 0] * self.coordinate_scaling
        ax.scatter(x, y, alpha=0.2, color='red')
        #ax.set_title(f""
        #             f"{self.slide_names[slide]}" +
        #             " (Observed;" * observed +
        #             " (Unobserved;" * (1 - observed) +
        #             f" t = {t})"
        #             )
        ax.axis("off")


    def plot_fn(self, interpolant, epoch, seed, t_max, data, ot_sampler, device, metric_prefix, train_timesteps, wandb, min_max):
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        x0 = data[0].to(device)
        x1 = data[-1].to(device)
        M = min(x0.shape[0], x1.shape[0])
        N = min(512, M)
        idx = np.random.choice(np.arange(M), N, replace=False)
        x0, x1 = ot_sampler.sample_plan(x0[idx], x1[idx])

        for slide, s_t in enumerate(data):
            s_t = denormalize(s_t, min_max)
            self.overlay_spot_coordinates(s_t.to('cpu'), slide, axes[0, slide])

            t = self.time_stamps[slide]
            t = T.ones(N, dtype=T.float32, device=x0.device) * t
            t = t.unsqueeze(-1)
            if isinstance(interpolant, list):
                xt_fake = interpolant[slide]
            else:
                with T.no_grad():
                    xt_fake = interpolant(x0, x1, t)
            xt_fake = denormalize(xt_fake, min_max)
            self.overlay_spot_coordinates(xt_fake.to('cpu'), slide, axes[1, slide])
        plt.tight_layout()
        if wandb is not None:
            wandb.log({f"{metric_prefix}/interpolants": wandb.Image(fig), f"{metric_prefix}_step": epoch})
            plt.close(fig)
        else:
            plt.show()


