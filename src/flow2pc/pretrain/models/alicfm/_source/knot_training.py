"""Knot sampling and update loops from the supplied ALI-CFM train_teacher.py.

Source revision: a35963ef226e1bbdc6344a944c50e0f42ed67335.
The original nested sampler is lifted to module scope. Run settings and a logging
sink are injected; model construction and artifact handling belong to the caller.
Full NumPy curves, sampling order, casts and optimizer update order are retained.
"""

import numpy as np
import torch

from .knot import get_dx_dt


def get_knot_data(size=120, std=0.01):
    t = np.linspace(0, 3, size)
    t_x = 3 * (t / t.max()) - 1.5
    t1, t2, t3 = t_x[:size // 3], t_x[size // 3:-size // 3], t_x[-size // 3:]
    x1 = 3 * (t1 + 0.5)
    x2 = np.cos(2 * np.pi * (t2 - 0.75))
    x3 = 3 * (t3 - 0.5)
    x = np.concatenate([x1, x2, x3]).reshape(-1, 1) + np.random.randn(size, 1) * std

    t_y = 3 * (t / t.max()) - 1.5
    t1, t2, t3 = t_y[:size // 3], t_y[size // 3:-size // 3], t_y[-size // 3:]
    y1 = - np.tanh(5 * (t1 + 1)) / 2 + 0.5
    y2 = np.sin(2 * np.pi * (t2 - 0.75)) + 1
    y3 = np.tanh(5 * (t3 - 1)) / 2 + 0.5
    y = np.concatenate([y1, y2, y3]).reshape(-1, 1) + np.random.randn(size, 1) * std

    return np.concatenate([x, y], axis=1), t / 3.0


def train(
    velocity_model,
    generator_model,
    discriminator,
    opt_G,
    opt_D,
    opt_V,
    *,
    device,
    batch_size=128,
    n_epochs_interpolant=50000,
    n_epochs_flow=30000,
    size=120,
    std=0.01,
    correct_coeff=1.0,
    logger=None,
):
    for epoch in range(n_epochs_interpolant):
        xt_list = []
        for _ in range(batch_size):
            xt, t_arr = get_knot_data(size=size, std=std)
            xt_list.append(xt)
        xt_all = torch.tensor(np.stack(xt_list)).float().to(device)
        t_all = torch.tensor(t_arr).float().to(device)

        idx = np.random.randint(1, len(t_arr) - 1)
        t_val = t_all[idx].view(1, 1).expand(batch_size, 1)
        x0 = xt_all[:, 0, :]
        x1 = xt_all[:, -1, :]
        xt_real = xt_all[:, idx, :]

        # Train Discriminator
        opt_D.zero_grad()
        with torch.no_grad():
            xt_fake = generator_model(x0, x1, t_val)

        real_inputs = torch.cat([xt_real, t_val], dim=-1)
        fake_inputs = torch.cat([xt_fake.detach(), t_val], dim=-1)

        real_logits = discriminator(real_inputs)
        fake_logits = discriminator(fake_inputs)

        d_loss = torch.nn.functional.softplus(-real_logits).mean() + torch.nn.functional.softplus(fake_logits).mean()
        d_loss.backward()
        opt_D.step()

        # Train Generator
        opt_G.zero_grad()
        xt_fake = generator_model(x0, x1, t_val)
        fake_inputs = torch.cat([xt_fake, t_val], dim=-1)
        fake_logits = discriminator(fake_inputs)

        g_loss_gan = torch.nn.functional.softplus(-fake_logits).mean()

        # Native coefficient is 1.0; an explicit config may override it.
        linear_interpolant = (1 - t_val) * x0 + t_val * x1
        reg_loss = torch.nn.functional.mse_loss(xt_fake, linear_interpolant)

        g_loss = g_loss_gan + correct_coeff * reg_loss
        g_loss.backward()
        opt_G.step()

        if epoch % 10000 == 0 or epoch == n_epochs_interpolant - 1:
            print(f"Epoch {epoch:6d} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | Reg: {reg_loss.item():.4f}")

        if logger is not None:
            logger.log(
                {
                    "ali/gan/step": epoch,
                    "ali/gan/d_loss": d_loss.item(),
                    "ali/gan/g_loss": g_loss.item(),
                    "ali/gan/regularizer": reg_loss.item(),
                }
            )

    for epoch in range(n_epochs_flow):
        xt_list = []
        for _ in range(batch_size):
            xt, t_arr = get_knot_data(size=size, std=std)
            xt_list.append(xt)
        xt_all = torch.tensor(np.stack(xt_list)).float().to(device)

        t_val = torch.rand(batch_size, 1, device=device)
        x0 = xt_all[:, 0, :].to(device)
        x1 = xt_all[:, -1, :].to(device)

        opt_V.zero_grad()

        dx_dt = get_dx_dt(generator_model, x0, x1, t_val)

        with torch.no_grad():
            xt_fake = generator_model(x0, x1, t_val)

        vt = velocity_model(xt_fake, t_val)
        v_loss = torch.nn.functional.mse_loss(vt, dx_dt.detach())
        v_loss.backward()
        opt_V.step()

        if epoch % 10000 == 0 or epoch == n_epochs_flow - 1:
            print(f"Epoch {epoch:6d} | V Loss: {v_loss.item():.6f}")

        if logger is not None:
            logger.log({"ali/cfm/step": epoch, "ali/cfm/loss": v_loss.item()})
