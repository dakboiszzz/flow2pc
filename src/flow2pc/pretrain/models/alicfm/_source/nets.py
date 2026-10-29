import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from numpy.matlib import randn
from torch.func import functional_call, jvp

from .land_metric import compute_time_dependent_metric


class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim

        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w), torch.nn.SELU(),
            torch.nn.Linear(w, w), torch.nn.SELU(),
            torch.nn.Linear(w, w), torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(torch.nn.Module):
    def __init__(self, in_dim, w=64, apply_sigmoid: bool = True):
        super().__init__()
        self.apply_sigmoid = apply_sigmoid
        self.net = torch.nn.Sequential(
            nn.Linear(in_dim, w), nn.ELU(),
            nn.Linear(w, w), nn.ELU(),
            nn.Linear(w, 1), nn.Sigmoid() if apply_sigmoid else nn.Identity()
        )

    def forward(self, x):
        return self.net(x)


def get_marginals(xts):
    xs, ts = [], []

    for t_, x_ in xts.items():
        ts.append(t_ * torch.ones(x_.size(0), 1, device=x_.device))
        xs.append(x_)

    xs = torch.cat(xs, dim=0)
    ts = torch.cat(ts, dim=0)

    # print(xs.shape, ts.shape)
    # assert 0

    return xs, ts


class TrainableInterpolant(nn.Module):
    def __init__(self, dim, h_dim, t_smooth=0.01, time_varying=False, regulariser='linear'):
        super().__init__()
        self.time_varying = time_varying
        self.t_smooth = t_smooth
        self.interpolant_net = nn.Sequential(
            nn.Linear(2 * dim + 1, h_dim), nn.ELU(),
            nn.Linear(h_dim, h_dim), nn.ELU(),
            nn.Linear(h_dim, dim)
        )
        self.regularizer = regulariser

    def forward(self, x0, x1, t, training=True):
        t = t[..., None] if t.ndim == 1 else t

        xt = self.linear_interpolant(x0, x1, t)
        if training:
            t_input = t + torch.randn_like(t) * self.t_smooth if self.t_smooth > 0 else t
        else:
            t_input = t
        input_ = torch.cat([x0, x1, t_input], dim=-1)

        correction = self.interpolant_net(input_)
        return xt + t * (1 - t) * correction

    @staticmethod
    def linear_interpolant(x0, x1, t):
        t = t[..., None] if t.ndim == 1 else t
        return t * x1 + (1 - t) * x0

    def get_reg_term(self, x0, x1, t, xt_fake, xt):
        if self.regularizer == 'length':
            return self.compute_length_reg_term(x0, x1)
        elif self.regularizer == 'piecewise':
            return self.compute_piecewise_reg_term(x0, x1, xt, t)
        elif self.regularizer == 'regression':
            return self.compute_regression_term(xt_fake, xt)
        elif self.regularizer == 'linear':
            return self.compute_linear_reg_term(x0, x1, t, xt_fake)
        else:
            raise ValueError(f"Unknown regularizer type: {self.regularizer}")

    def compute_length_reg_term(
        self, xts, metric='l2', land_gamma=0.2, land_t_gamma=0.2,
        num_t_steps=3, h=0.001, reference_data=None, land_epsilon=1e-6,
    ):
        """Equations 15-16: integrated squared curvature, using finite differences."""
        x0, x1 = xts[0], xts[1]
        samples, sample_times = reference_data if reference_data is not None else get_marginals(xts)
        # Float64 retains the paper's finite-difference estimator while avoiding
        # false curvature from float32 cancellation at the released h=0.001.
        parameters = {name: value.double() for name, value in self.named_parameters()}
        x0_double, x1_double = x0.double(), x1.double()
        samples, sample_times = samples.double(), sample_times.double()

        def path_at(time):
            return functional_call(
                self, parameters, (x0_double, x1_double, time),
                {"training": False},
            )

        losses = []
        for _ in range(num_t_steps):
            t = torch.rand(x0.shape[0], 1, device=x0.device).double()
            center = path_at(t)
            curvature = (path_at(t + h) + path_at(t - h) - 2 * center) / h**2
            if metric == 'l2':
                penalty = curvature.square().sum(-1)
            elif metric in {'land', 'land_norm'}:
                diagonal = compute_time_dependent_metric(
                    center, t, samples, sample_times, land_gamma, land_t_gamma,
                    normalize_t=(metric == 'land_norm'), rho=land_epsilon,
                )
                penalty = (curvature.square() * diagonal).sum(-1)
            else:
                raise ValueError(f"Unknown metric {metric}")
            losses.append(penalty.mean())
        return torch.stack(losses).mean()

    def compute_piecewise_reg_term(self, x0, x1, xt, t_exact, metric='l2'):
        t = t_exact + torch.randn_like(t_exact)
        t_left = t[t < t_exact]
        t_right = t[t >= t_exact]

        x0_ = x0[torch.randperm(x0.shape[0])[:t_left.shape[0]]]
        x1_ = x1[torch.randperm(x1.shape[0])[:t_right.shape[0]]]

        xt_linear = torch.zeros_like(xt)
        xt_linear[(t < t_exact).squeeze()] = (
            (t_exact[t < t_exact] - t_left).unsqueeze(-1) * x0_ + \
            t_left.unsqueeze(-1) * xt[(t < t_exact).squeeze()]
        ) / t_exact[t < t_exact].unsqueeze(-1)

        xt_linear[(t >= t_exact).squeeze()] = (
            (1 - t_right).unsqueeze(-1) * xt[(t >= t_exact).squeeze()] + \
            t_right.unsqueeze(-1) * x1_
        ) / (1 - t_exact[t >= t_exact].unsqueeze(-1))

        t = t[..., None] if t.ndim == 1 else t
        xt = self.linear_interpolant(x0, x1, t)
        input_ = torch.cat([x0, x1, t], dim=-1)
        correction = self.interpolant_net(input_)
        xt_interpolant = xt + t * (1 - t) * correction
        diff = xt_linear - xt_interpolant

        if metric == 'l2':
            ret_term = (diff).pow(2).mean()

        elif metric == 'land':
            xs, ts = get_marginals(xts)
            G = compute_time_dependent_metric(
                xt_linear, t, xs, ts,
                gamma=land_gamma,
                t_gamma=land_t_gamma,
                normalize_t=False,
            )
            ret_term = ((diff)**2 * G).mean()

        elif metric == 'land_norm':
            xs, ts = get_marginals(xts)
            G = compute_time_dependent_metric(
                xt_linear, t, xs, ts,
                gamma=land_gamma,
                t_gamma=land_t_gamma,
                normalize_t=True,
            )
            ret_term = ((diff)**2 * G).mean()

        return (xt_linear - xt_interpolant).pow(2).mean()

    def compute_regression_term(self, x_hat, xt):
        return (x_hat - xt).pow(2).mean()

    def compute_linear_reg_term(self, x0, x1, t, xt):
        correction = xt - self.linear_interpolant(x0, x1, t)
        batch_size = correction.shape[0]
        if correction.ndim == 1:
            reg_term = correction.pow(2)
        else:
            raise ValueError(f"Unknown metric {metric}")

        return ret_term

    def compute_linear_reg_term(
            self, xt_pred, xts, t,
            metric='l2', land_gamma=0.2, land_t_gamma=0.2,
        ):
        x0, x1 = xts.get(0), xts.get(1)
        correction = xt_pred - self.linear_interpolant(x0, x1, t)

        if metric == 'l2':
            reg_term = correction.reshape(correction.size(0), -1).pow(2).sum(-1)

        elif metric == 'land':
            xs, ts = get_marginals(xts)
            G = compute_time_dependent_metric(
                xt_pred, t, xs, ts,
                gamma=land_gamma,
                t_gamma=land_t_gamma,
                normalize_t=False,
            )

        elif metric == 'land_norm':
            xs, ts = get_marginals(xts)

            G = compute_time_dependent_metric(
                xt_pred, t, xs, ts,
                gamma=land_gamma,
                t_gamma=land_t_gamma,
                normalize_t=True,
            )

            assert correction.size(1) == G.size(1)
            reg_term = torch.sqrt(((correction**2) * G).sum(dim=-1))

        else:
            raise ValueError(f"Unknown metric {metric}")

        return reg_term.mean()

    def dI_dt(self, x0, x1, t):
        t = t[..., None] if t.ndim == 1 else t
        # Each trajectory has one time coordinate. Forward-mode computes the
        # same noiseless derivative without a Jacobian sized by output dimension.
        _, velocity = jvp(
            lambda time: self(x0, x1, time, training=False),
            (t,), (torch.ones_like(t),),
        )
        return velocity


class TrainableInterpolantST(TrainableInterpolant):
    def __init__(self, dim, h_dim, t_smooth=0.01,
                 time_varying=False, regulariser='linear'):
        super().__init__(
            dim=dim, h_dim=h_dim, t_smooth=t_smooth,
            time_varying=time_varying, regulariser=regulariser
        )
        # self.interpolant_net = RotationGenerator(dim=dim)
        self.interpolant_net = nn.Sequential(
            nn.Linear(2 * dim + 1, h_dim), nn.ELU(),
            nn.Linear(h_dim, h_dim), nn.ELU(),
            nn.Linear(h_dim, h_dim), nn.ELU(),
            nn.Linear(h_dim, dim), nn.Sigmoid()
        )
