"""Benchmark 4 verification: the aleatoric/epistemic split makes active
identification more efficient. Information-gain acquisition (which needs the
aleatoric level to discount noise) reduces reducible uncertainty faster and
wastes fewer queries in the high-noise region than a total-uncertainty signal
(what a one-replica model gives)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.active_learning import HeteroGP, run_active  # noqa: E402


def test_infogain_beats_total_uncertainty():
    gp = HeteroGP(lengthscale=0.4, sig_lo=0.05, sig_hi=0.6, region=(0.5, 1.5))
    iv, tv = torch.zeros(40), torch.zeros(40)
    ira, tra = 0.0, 0.0
    for fi in range(30):
        i, ia = run_active(gp, "infogain", 40, seed=fi, f_seed=1000 + fi)
        t, ta = run_active(gp, "total", 40, seed=fi, f_seed=1000 + fi)
        iv += torch.tensor(i)
        tv += torch.tensor(t)
        ira += ia
        tra += ta
    iv /= 30
    tv /= 30
    ira /= 30
    tra /= 30
    # info-gain reduces reducible uncertainty much faster at 20 queries
    assert iv[19] < 0.5 * tv[19], f"infogain var {iv[19]:.3f} not << total {tv[19]:.3f}"
    # and spends fewer queries in the irreducible-noise region
    assert ira < tra, f"infogain R_a frac {ira:.2f} not < total {tra:.2f}"
