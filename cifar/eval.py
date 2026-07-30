"""Evaluate a finished distill run directory.

Usage:
  python eval.py --run-dir outputs/runs/distill/<timestamp> [--n-samples 5000]

Reads the run's composed config (.hydra/config.yaml) and last checkpoint,
then reports exact test log-likelihood, sample grids (images) or moment
errors (tabular), and writes everything into <run_dir>/eval/.
"""

import argparse
import os

import hydra
import torch
from omegaconf import OmegaConf
from train import build_net, load_env, strip_prefix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-samples", type=int, default=5000)
    args = ap.parse_args()
    load_env()
    run_dir = args.run_dir
    n_samples = args.n_samples
    device = "cuda" if torch.cuda.is_available() else "cpu"

    OmegaConf.register_new_resolver(
        "hydra",
        lambda p: {"runtime.cwd": os.getcwd(), "runtime.output_dir": run_dir}[p],
        replace=True,
    )
    rc = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    assert rc.task.name == "distill", "eval.py evaluates distill runs"

    datamodule = hydra.utils.instantiate(rc.data)
    datamodule.prepare_data()
    datamodule.setup()

    pc = build_net(rc.model, datamodule.dim).to(device).eval()
    ckpt_path = os.path.join(run_dir, "checkpoints", "last.ckpt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    pc.load_state_dict(strip_prefix(ckpt["state_dict"], "pc."))

    out = os.path.join(run_dir, "eval")
    os.makedirs(out, exist_ok=True)
    t1 = torch.tensor(1.0, device=device)

    # exact log-likelihood on the test split
    lls = []
    with torch.no_grad():
        for batch in datamodule.val_dataloader():
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            lls.append(pc.log_prob(x.to(device).flatten(1), t1).cpu())
    lls = torch.cat(lls)
    print(
        f"exact test LL: {lls.mean():.2f} +- {lls.std():.2f} nats "
        f"({lls.mean() / datamodule.dim:.4f} nats/dim)"
    )

    # samples
    with torch.no_grad():
        try:
            xs = pc.sample(n_samples, t1)
        except TypeError:  # families with generator= signature
            xs = pc.sample(n_samples, t1, generator=None)
    xs = xs.to(device) if torch.is_tensor(xs) else xs

    if datamodule.shape is not None:  # images: save a grid
        from torchvision.utils import save_image

        imgs = xs[:64].reshape(-1, *datamodule.shape)
        save_image(
            imgs.clamp(-1, 1) * 0.5 + 0.5, os.path.join(out, "samples.png"), nrow=8
        )
        print(f"sample grid -> {out}/samples.png")
    else:  # tabular: moment errors against test data
        Xte = torch.cat(
            [
                (b[0] if isinstance(b, (list, tuple)) else b)
                for b in datamodule.val_dataloader()
            ]
        ).to(device)
        m_err = (xs.mean(0) - Xte.mean(0)).abs().mean()
        c_err = (torch.cov(xs.T) - torch.cov(Xte.T)).abs().mean()
        print(f"one-step samples: mean abs err {m_err:.4f}, cov abs err {c_err:.4f}")

    torch.save({"ll": lls}, os.path.join(out, "metrics.pt"))
    print(f"metrics -> {out}/metrics.pt")


if __name__ == "__main__":
    main()
