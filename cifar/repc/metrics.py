"""Shared evaluation utilities: FID, flow likelihood, ODE integrators,
feature classifiers, and fast circuit samplers.

Single home for everything that was previously duplicated across scripts.
FID caveat: features come from small dataset-specific CNNs trained here —
values are internally comparable per dataset, NOT Inception-scale.
"""

import math
import os

import numpy as np
import torch
from torch import nn

LOG2PI = math.log(2.0 * math.pi)


# ------------------------------------------------------------- integrators
@torch.no_grad()
def euler(flow, z, t0=0.0, t1=1.0, steps=100):
    """dz/dt = flow(z, t), Euler, from t0 to t1."""
    dt = (t1 - t0) / steps
    n = z.shape[0]
    for i in range(steps):
        t = torch.full((n,), t0 + i * dt, device=z.device)
        z = z + dt * flow(z, t)
    return z


@torch.no_grad()
def heun(flow, z, t0=0.0, t1=1.0, steps=50):
    """2nd-order Heun; ~Euler quality at half the steps (2 NFE per step)."""
    dt = (t1 - t0) / steps
    n = z.shape[0]
    for i in range(steps):
        t = torch.full((n,), t0 + i * dt, device=z.device)
        k1 = flow(z, t)
        t2 = torch.full((n,), min(t0 + (i + 1) * dt, t1), device=z.device)
        k2 = flow(z + dt * k1, t2)
        z = z + dt * 0.5 * (k1 + k2)
    return z


# ---------------------------------------------------------- flow likelihood
def flow_log_prob(flow, x, steps=100, probes=2, seed=0):
    """log p_1(x) via instantaneous change of variables, integrating x back
    to t=0 with Hutchinson divergence estimation (Rademacher probes)."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    z = x.clone()
    div_acc = torch.zeros(x.shape[0], device=x.device)
    dt = 1.0 / steps
    for i in range(steps, 0, -1):
        t = torch.full((x.shape[0],), (i - 0.5) * dt, device=x.device)
        with torch.enable_grad():
            z_in = z.detach().requires_grad_(True)
            u = flow(z_in, t)
            div = torch.zeros(x.shape[0], device=x.device)
            for _ in range(probes):
                v = torch.randint(
                    0, 2, u.shape, device=x.device, generator=g, dtype=u.dtype
                )
                v = 2 * v - 1
                (Jv,) = torch.autograd.grad(u, z_in, grad_outputs=v, retain_graph=True)
                div = div + (v * Jv).flatten(1).sum(1)
            div = div / probes
        z = z - dt * u.detach()
        div_acc = div_acc + dt * div.detach()
    logp0 = -0.5 * (z.detach().flatten(1).pow(2).sum(1) + z[0].numel() * LOG2PI)
    return logp0 - div_acc


# -------------------------------------------------------------------- FID
class FeatureCNN(nn.Module):
    """Small classifier used as the FID feature extractor."""

    def __init__(self, in_ch, n_classes, widths=(64, 128, 256), feat_dim=256):
        super().__init__()
        layers = []
        c = in_ch
        for w in widths[:-1]:
            layers += [nn.Conv2d(c, w, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            c = w
        layers += [
            nn.Conv2d(c, widths[-1], 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
            nn.Linear(widths[-1] * 4, feat_dim),
            nn.ReLU(),
        ]
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        return self.head(self.features(x))


def get_feature_net(
    dataset_name, train_loader, device, cache_dir="outputs/eval", epochs=3
):
    """Train (or load cached) FID feature net for a dataset."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"featnet_{dataset_name}.pt")
    x0, y0 = next(iter(train_loader))
    n_classes = int(y0.max().item()) + 1 if y0.dim() == 1 else 10
    net = FeatureCNN(x0.shape[1], max(n_classes, 10)).to(device)
    if os.path.exists(path):
        net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        return net.eval()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = nn.functional.cross_entropy(net(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    torch.save(net.state_dict(), path)
    return net.eval()


@torch.no_grad()
def fid(feature_net, real, fake, batch=500):
    """Frechet distance in the feature net's penultimate space."""

    def stats(imgs):
        fs = []
        for i in range(0, len(imgs), batch):
            fs.append(feature_net.features(imgs[i : i + batch]).cpu().double())
        f = torch.cat(fs)
        return f.mean(0), torch.cov(f.T)

    mu1, c1 = stats(real)
    mu2, c2 = stats(fake)
    diff = (mu1 - mu2).pow(2).sum()
    eig = torch.linalg.eigvals(c1 @ c2).real.clamp_min(0)
    return (diff + c1.trace() + c2.trace() - 2 * eig.sqrt().sum()).item()


# ------------------------------------------------------------ fast samplers
@torch.no_grad()
def sample_region_pc(pc, n, t, seed=0):
    """Numpy top-down ancestral sampler for DeepPC-style circuits
    (flat-buffer addressing: groups with childA/childB buffer positions)."""
    rng = np.random.default_rng(seed)
    mu, _, logsig, _ = pc.leaf_params(t)
    mu = mu.detach().cpu().numpy()
    sig = torch.exp(logsig).detach().cpu().numpy()
    w = [torch.softmax(sl, dim=-1).detach().cpu().numpy() for sl in pc.sum_logits]
    root_w = torch.softmax(pc.root_logits, 0).detach().cpu().numpy()
    pos2group = {}
    childA, childB = [], []
    base = pc.n_leaves
    for gi, g in enumerate(pc.groups):
        rows = g.get("bufpos")
        if rows is None:
            rows = range(base, base + len(g["pos"]))
            base += len(g["pos"])
        else:
            rows = rows.tolist()
        for j, p in enumerate(rows):
            pos2group[p] = (gi, j)
        childA.append(g["childA"].cpu().numpy())
        childB.append(g["childB"].cpu().numpy())
    S = pc.S
    out = torch.empty(n, pc.D)
    for si in range(n):
        ch = rng.choice(len(root_w), p=root_w)
        stack = [(pc.root_pos, ch)]
        while stack:
            p, c = stack.pop()
            if p < pc.n_leaves:
                out[si, p] = float(mu[p, c] + sig[p, c] * rng.standard_normal())
                continue
            gi, j = pos2group[p]
            k = rng.choice(w[gi].shape[-1], p=w[gi][j, c])
            cand, rem = divmod(k, S * S)
            i, jj = divmod(rem, S)
            stack.append((int(childA[gi][j, cand]), i))
            stack.append((int(childB[gi][j, cand]), jj))
    return out
