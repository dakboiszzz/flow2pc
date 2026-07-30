"""Ancestral sampler for pyjuice node graphs, written against the UNCOMPILED
CircuitNodes structure so it is correct for any edge layout — including
MonarchCircuits' sparse/permuted product layers, which pyjuice's compiled
sampler (2.2.0 and 2.6.1) mishandles (wrong sample marginals despite correct
likelihoods; see project notes).

Usage:
    python scripts/monarch_ns_sampler.py <run_dir> <n_samples> [--device 0]
Writes <run_dir>/samples5k_ns.pt with shape (n, num_vars) long.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.setrecursionlimit(15000)


def build_tables(ns):
    """Flatten the graph once into numpy lookup tables keyed by id(node)."""
    tables = {}
    for node in ns:
        t = {}
        if node.is_input():
            t["kind"] = "input"
            t["var"] = int(next(iter(node.scope)))
            nc = node.dist.num_cats
            src = (
                node.get_source_ns()
                if node._params is None and node.is_tied()
                else node
            )
            p = src._params.detach().cpu().numpy().reshape(-1, nc)
            t["cum"] = np.cumsum(p, axis=1)
        elif node.is_sum():
            t["kind"] = "sum"
            eb = node.edge_ids.cpu().numpy()  # (2, E)
            chbs = node.chs[0].block_size
            params = node._params.detach().cpu().numpy()  # (E, bs, chbs)
            # per parent block: list of (edge_idx, child_block); weights per offset
            per_block = {}
            for e in range(eb.shape[1]):
                per_block.setdefault(int(eb[0, e]), []).append((e, int(eb[1, e])))
            cums, childs = {}, {}
            for b, lst in per_block.items():
                w = np.concatenate(
                    [params[e, :, :] for e, _ in lst], axis=1
                )  # (bs, k*chbs)
                cums[b] = np.cumsum(w / w.sum(axis=1, keepdims=True), axis=1)
                childs[b] = np.array([cb for _, cb in lst])
            t["cums"], t["childs"], t["chbs"] = cums, childs, chbs
        else:
            t["kind"] = "prod"
            t["sparse"] = (
                node.edge_ids.shape[0] == node.num_node_blocks * node.block_size
            )
            t["edge_ids"] = node.edge_ids.cpu().numpy()
            t["ch_bs"] = [c.block_size for c in node.chs]
        tables[id(node)] = t
    return tables


def sample_one(ns, tables, rng, num_vars):
    out = np.zeros(num_vars, dtype=np.int64)
    stack = [(ns, 0, 0)]
    while stack:
        node, b, i = stack.pop()
        t = tables[id(node)]
        if t["kind"] == "input":
            cum = t["cum"][b * node.block_size + i]
            out[t["var"]] = int(np.searchsorted(cum, rng.random() * cum[-1]))
        elif t["kind"] == "sum":
            cum = t["cums"][b][i]
            k = int(np.searchsorted(cum, rng.random() * cum[-1]))
            e, j = divmod(k, t["chbs"])
            stack.append((node.chs[0], int(t["childs"][b][e]), j))
        else:
            flat = b * node.block_size + i
            for c, ch in enumerate(node.chs):
                if t["sparse"]:
                    cf = int(t["edge_ids"][flat, c])
                    stack.append((ch, cf // t["ch_bs"][c], cf % t["ch_bs"][c]))
                else:
                    stack.append((ch, int(t["edge_ids"][b, c]), i))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("n_samples", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-name", default="samples5k_ns.pt")
    args = ap.parse_args()

    sys.path.insert(
        0,
        str(Path(__file__).resolve().parent.parent / "third_party" / "MonarchCircuits"),
    )
    import pyjuice as juice

    ns = juice.load(f"{args.run_dir}/best.jpc")
    num_vars = max(max(n.scope) for n in ns if n.is_input()) + 1
    print(f"> building tables ({num_vars} vars) ...", flush=True)
    tables = build_tables(ns)
    rng = np.random.default_rng(args.seed)
    out_path = Path(args.run_dir) / args.out_name

    samples, t0 = [], time.time()
    for k in range(args.n_samples):
        samples.append(sample_one(ns, tables, rng, num_vars))
        if (k + 1) % 250 == 0:
            el = time.time() - t0
            print(
                f"{k + 1}/{args.n_samples} ({el / 60:.1f} min, {el / (k + 1):.2f}s/sample)",
                flush=True,
            )
            torch.save(torch.from_numpy(np.stack(samples)), out_path)
    torch.save(torch.from_numpy(np.stack(samples)), out_path)
    print(f"DONE {len(samples)} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
