"""Train a Monarch-HCLT probabilistic circuit from scratch on CIFAR-10.

Adapted from wangben88/MonarchCircuits (ICML 2025, arXiv:2506.12383), which
trains on ImageNet32/64. Single-GPU (their loop minus DDP), raw RGB
Categorical(256) leaves so samples live in the same space as our
recompiled students. Maximum-likelihood via pyjuice minibatch EM.

Usage:
    python scripts/train_monarch_cifar10.py --hidden-size 256 --device 1
"""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import torch
import torchvision

REPO = Path(__file__).resolve().parent.parent
MONARCH = Path(
    os.environ.get("MONARCH_DIR", str(REPO / "third_party" / "MonarchCircuits"))
)
sys.path.insert(0, str(MONARCH))
sys.setrecursionlimit(15000)

import pyjuice as juice  # noqa: E402
from monarch_juice.layers.monarchlayer import (  # noqa: E402
    create_dense_layer,
    create_monarch_layers,
)
from monarch_juice.structures.HCLTMonarch import HCLTGeneral  # noqa: E402


def rgb_to_ycocgr(x):
    """Lossless YCoCg-R (paper's transform): integer bijection on RGB.
    x is (N, 3, H, W) long in [0, 256). Y in [0,256), Co/Cg in [0,512)."""
    R, G, B = x[:, 0], x[:, 1], x[:, 2]
    Co = R - B
    tmp = B + Co.div(2, rounding_mode="floor")
    Cg = G - tmp
    Y = tmp + Cg.div(2, rounding_mode="floor")
    return torch.stack((Y, Co + 256, Cg + 256), dim=1)


def ycocgr_to_rgb(x):
    """Exact inverse of rgb_to_ycocgr; x is (N, 3, H, W) long."""
    Y, Co, Cg = x[:, 0], x[:, 1] - 256, x[:, 2] - 256
    tmp = Y - Cg.div(2, rounding_mode="floor")
    G = Cg + tmp
    B = tmp - Co.div(2, rounding_mode="floor")
    R = B + Co
    return torch.stack((R, G, B), dim=1).clamp(0, 255)


def load_cifar10(root: str, ycocgr: bool = False, hflip: bool = False):
    """CIFAR-10 as (N, 3072) ints in CHW variable order (matches HCLT's
    v // (H*W) channel convention). Optional lossless YCoCg-R transform
    (paper pipeline) and horizontal-flip augmentation of the train set."""
    tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
    te = torchvision.datasets.CIFAR10(root, train=False, download=True)

    def prep(d, flip):
        x = torch.from_numpy(d.data).permute(0, 3, 1, 2).long()  # (N,3,32,32)
        if flip:
            x = torch.cat([x, x.flip(-1)])
        if ycocgr:
            x = rgb_to_ycocgr(x)
        return x.reshape(len(x), -1)

    return prep(tr, hflip), prep(te, False)


@torch.no_grad()
def eval_ll(pc, data, batch_size, device):
    tot, n = 0.0, 0
    for s in range(0, len(data), batch_size):
        x = data[s : s + batch_size].to(device).long()
        tot += pc(x).sum().item()
        n += len(x)
    return tot / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer-type", choices=["monarch", "dense"], default="monarch")
    p.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="must be a square number for monarch",
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--minibatch-schedule",
        type=str,
        default="20000,20",
        help="pairs of (samples-per-EM-update, epochs)",
    )
    p.add_argument("--clt-samples", type=int, default=20000)
    p.add_argument(
        "--ycocgr",
        action="store_true",
        help="lossless YCoCg-R transform + Categorical(512) + per-channel tied leaves (paper pipeline)",
    )
    p.add_argument(
        "--hflip",
        action="store_true",
        help="horizontal-flip augmentation of the train set",
    )
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--data-root", type=str, default=str(REPO / "data"))
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default=None)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.device}")
    out = (
        Path(args.out_dir)
        if args.out_dir
        else (
            REPO
            / "outputs"
            / "runs"
            / "monarch_scratch"
            / f"{args.layer_type}_hs{args.hidden_size}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        )
    )
    out.mkdir(parents=True, exist_ok=True)

    run = None
    if os.environ.get("WANDB_API_KEY"):
        import wandb

        run = wandb.init(
            project="recompile-probabilistic",
            name=args.wandb_name
            or f"monarch-scratch-cifar10-{args.layer_type}-hs{args.hidden_size}",
            config=vars(args),
            dir=str(out),
        )

    train, test = load_cifar10(args.data_root, ycocgr=args.ycocgr, hflip=args.hflip)
    D = train.shape[1]
    print(
        f"> CIFAR-10 loaded: train {tuple(train.shape)} test {tuple(test.shape)}"
        f" (ycocgr={args.ycocgr}, hflip={args.hflip})"
    )

    print("> Building HCLT (Chow-Liu tree from data) ...")
    clt_x = train[torch.randperm(len(train))[: args.clt_samples]].float().to(device)
    layer_fn = (
        create_monarch_layers if args.layer_type == "monarch" else create_dense_layer
    )
    block = (
        int(args.hidden_size**0.5) if args.layer_type == "monarch" else args.hidden_size
    )
    assert block * block == args.hidden_size or args.layer_type == "dense", (
        "hidden_size must be square for monarch"
    )
    ns = HCLTGeneral(
        clt_x,
        num_latents=args.hidden_size,
        layer_fn=layer_fn,
        num_bins=64,
        sigma=0.02,
        chunk_size=32,
        block_size=block,
        homogeneous_inputs=True,
        homog_by_channel=args.ycocgr,
        input_node_params={"num_cats": 512 if args.ycocgr else 256},
    )
    del clt_x

    pc = juice.compile(ns)
    pc.print_statistics()
    pc.to(device)
    n_params = sum(p_.numel() for p_ in pc.parameters())
    print(f"> compiled; {n_params / 1e6:.2f}M params")
    if run:
        run.summary["n_params"] = n_params

    schedule = [int(x) for x in args.minibatch_schedule.split(",")]
    sizes, epochs = schedule[::2], schedule[1::2]
    tot_epochs = sum(epochs)
    era, epochs_in_era = 0, 0
    nats2bpd = 1.0 / (D * math.log(2.0))

    pc.init_param_flows(flows_memory=0.0)
    v0 = eval_ll(pc, test, args.batch_size, device)
    print(f"> initial val LL {v0:.2f} ({-v0 * nats2bpd:.4f} bpd)")

    csv_f = open(out / "metrics.csv", "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["epoch", "train_ll", "val_ll", "val_bpd", "step_size", "secs"])

    step_count, best_val = 0, -float("inf")
    for epoch in range(tot_epochs):
        t0 = time.time()
        niters = (
            len(train) // args.batch_size
            if sizes[era] == 0
            else max(1, sizes[era] // args.batch_size)
        )
        perm = torch.randperm(len(train))
        ll_sum, nb, update_count, step_size = 0.0, 0, 0, 0.0
        for s in range(0, len(train) - args.batch_size + 1, args.batch_size):
            x = train[perm[s : s + args.batch_size]].to(device).long()
            lls = pc(x, propagation_alg="LL")
            pc.backward(
                x,
                flows_memory=1.0,
                allow_modify_flows=False,
                propagation_alg="LL",
                logspace_flows=True,
            )
            ll_sum += lls.mean().item()
            nb += 1
            step_count += 1
            if step_count >= niters:
                step_count = 0
                update_count += 1
                frac = (epoch + update_count * sizes[era] / len(train)) / tot_epochs
                step_size = 0.5 * (1 + math.cos(math.pi * frac))
                pc.mini_batch_em(step_size=step_size, pseudocount=1e-8)
                pc.init_param_flows(flows_memory=0.0)

        epochs_in_era += 1
        if epochs_in_era > epochs[era]:
            epochs_in_era, era = 0, min(era + 1, len(sizes) - 1)

        train_ll = ll_sum / max(nb, 1)
        val_ll = eval_ll(pc, test, args.batch_size, device)
        secs = time.time() - t0
        print(
            f"[Epoch {epoch + 1}/{tot_epochs}] train LL {train_ll:.2f} | "
            f"val LL {val_ll:.2f} ({-val_ll * nats2bpd:.4f} bpd) | "
            f"step {step_size:.3f} | {secs:.0f}s",
            flush=True,
        )
        writer.writerow(
            [epoch + 1, train_ll, val_ll, -val_ll * nats2bpd, step_size, secs]
        )
        csv_f.flush()
        if run:
            run.log(
                {
                    "epoch": epoch + 1,
                    "train_ll": train_ll,
                    "val_ll": val_ll,
                    "val_bpd": -val_ll * nats2bpd,
                    "step_size": step_size,
                }
            )
        if val_ll > best_val:
            best_val = val_ll
            juice.save(str(out / "best.jpc"), pc)

    juice.save(str(out / "last.jpc"), pc)
    print(
        f"> done. best val LL {best_val:.2f} ({-best_val * nats2bpd:.4f} bpd); saved to {out}"
    )

    print("> sampling 64 images ...")
    try:
        from pyjuice.queries import sample

        samples = sample(pc, num_samples=64, conditional=False)
        torch.save(samples.cpu(), out / "samples.pt")
        imgs = samples.reshape(-1, 3, 32, 32).float().cpu() / 255.0
        torchvision.utils.save_image(imgs, out / "samples.png", nrow=8)
        print(f"> samples saved to {out / 'samples.png'}")
    except Exception as e:  # sampling API varies across pyjuice versions
        print(f"> sampling failed ({e}); model saved, sample offline")

    if run:
        run.summary["best_val_ll"] = best_val
        run.summary["best_val_bpd"] = -best_val * nats2bpd
        run.finish()
    csv_f.close()


if __name__ == "__main__":
    main()
