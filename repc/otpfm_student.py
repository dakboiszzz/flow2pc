"""Shared ALI velocity student, with an optional anchored-source experiment."""

import copy

import torch
from torch import nn

from repc.hclt_pc import TimeDependentHCLTPC
from repc.single_cell import build_single_cell_student, initialize_source


def build_student(dim, num_latents=64, time_embedding_dim=16, seed=0, recipe="ali"):
    if recipe == "anchored":
        return AnchoredHCLT(dim, num_latents, time_embedding_dim, seed)
    if recipe != "ali" or dim not in (1, 2):
        raise ValueError("expected ali/anchored recipe and dimension 1 or 2")
    return build_single_cell_student({
        "name": "hclt", "parents": [-1] if dim == 1 else [-1, 0],
        "num_latents": num_latents, "time_embedding_dim": time_embedding_dim,
        "time_feature_seed": seed, "fourier_zero_center": False})


def initialize_student(student, training, validation, recipe, seed, source_steps=1000,
                       source_lr=.001, log=None):
    if recipe == "anchored":
        return fit_source(student, training, validation, source_steps, source_lr, log)
    if recipe != "ali":
        raise ValueError("unknown student recipe")
    initialize_source(student, training, torch.Generator().manual_seed(seed + 20000))
    return {"fit_steps": 0, "policy": "ALI_source_initialization_only_all_student_parameters_trainable"}


def build_optimizer(student, recipe, lr, steps):
    if recipe == "ali":
        optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=.01)
        return optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    if recipe != "anchored":
        raise ValueError("unknown student recipe")
    return torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=lr), None


class AnchoredHCLT(TimeDependentHCLTPC):
    def __init__(self, dim, num_latents=64, time_embedding_dim=16, seed=0):
        if dim not in (1, 2):
            raise ValueError("OTP real-data students currently support dimensions 1 and 2")
        super().__init__(parents=[-1] if dim == 1 else [-1, 0], num_latents=num_latents,
                         time_conditioning="fourier", time_embedding_dim=time_embedding_dim,
                         time_feature_seed=seed, fourier_zero_center=True)
        self.register_buffer("source_mean", torch.zeros(dim, num_latents))
        self.register_buffer("source_log_scale", torch.zeros(dim, num_latents))
        self.mean_coefficients.data.zero_()
        self.log_scale_coefficients.data.zero_()
        self.root_logits.requires_grad_(False)
        self.transition_logits.requires_grad_(False)

    def leaf_parameters(self, time):
        basis, derivative = self._time_basis(time)
        equation = "dhp,p->dh" if time.ndim == 0 else "dhp,bp->bdh"
        mean = self.source_mean + torch.einsum(equation, self.mean_coefficients, basis)
        mean_dt = torch.einsum(equation, self.mean_coefficients, derivative)
        raw_scale = self.source_log_scale + torch.einsum(equation, self.log_scale_coefficients, basis)
        scale_dt = torch.einsum(equation, self.log_scale_coefficients, derivative)
        active = (raw_scale > -20) & (raw_scale < 20)
        return mean, mean_dt, raw_scale.clamp(-20, 20), scale_dt * active

    @torch.no_grad()
    def set_source(self, state):
        self.source_mean.copy_(state["mean"])
        self.source_log_scale.copy_(state["log_scale"])
        self.root_logits.copy_(state["root_logits"])
        self.transition_logits.copy_(state["transition_logits"])


class SourceHCLT(TimeDependentHCLTPC):
    """Time-independent source fit with an explicit minimum std of .01 normalized units."""
    def __init__(self, dim, latents, values):
        super().__init__([-1] if dim == 1 else [-1, 0], num_latents=latents)
        self.source_mu = nn.Parameter(values[torch.randint(len(values), (latents,), device=values.device)].T.clone())
        sigma = values.std(0).clamp_min(.02) * .3
        self.source_logs = nn.Parameter(sigma.log()[:, None].expand(dim, latents).clone())
        self.mean_coefficients.requires_grad_(False)
        self.log_scale_coefficients.requires_grad_(False)

    def leaf_parameters(self, time):
        mu, logs = self.source_mu, self.source_logs.clamp(min=-4.605170186, max=20)
        if time.ndim:
            mu, logs = mu[None], logs[None]
        return mu, torch.zeros_like(mu), logs, torch.zeros_like(logs)


def fit_source(student, training, validation, steps=1000, lr=.001, log=None):
    """Select only by source-validation NLL; no held-out-time observations."""
    source = SourceHCLT(student.dim, student.num_latents, training).to(training.device)
    optimizer = torch.optim.Adam([p for p in source.parameters() if p.requires_grad], lr=lr)
    zero = training.new_tensor(0.)
    best, best_state, best_step = float("inf"), None, 0
    for step in range(steps + 1):
        if step:
            loss = -source.log_prob(training, zero).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite source MLE objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(source.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        if step % 25 == 0 or step == steps:
            with torch.no_grad():
                score = float(-source.log_prob(validation, zero).mean())
            if score < best:
                best, best_step = score, step
                best_state = copy.deepcopy(source.state_dict())
            if log:
                log({"source/step": step, "source/validation_nll_normalized": score})
    source.load_state_dict(best_state)
    with torch.no_grad():
        mean, _, logs, _ = source.leaf_parameters(zero)
        student.set_source({"mean": mean, "log_scale": logs, "root_logits": source.root_logits,
                            "transition_logits": source.transition_logits})
    return {"fit_steps": steps, "best_step": best_step, "validation_nll_normalized": best,
            "minimum_std_normalized": .01,
            "policy": "source_MLE_selected_by_validation_NLL_then_frozen_emissions_and_circuit_weights"}
