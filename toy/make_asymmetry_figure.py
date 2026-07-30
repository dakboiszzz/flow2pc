"""Figure for the asymmetry write-up: under a pure velocity loss the epistemic
error converges monotonically while the aleatoric trace drifts *away* from its
target (1.0), climbing past it; a mechanism-derived tail-broadening term holds it.

Trajectories are the logged runs (sota_push_m96.log, broaden_300k.log)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

# step, err_epi, err_ale, ale_tr  (from outputs/logs)
PLAIN = [
    (0, 0.9998, 3.0116, 4.012),
    (25000, 0.2299, 0.3636, 0.784),
    (50000, 0.1581, 0.4410, 1.029),
    (75000, 0.1203, 0.5456, 1.156),
    (100000, 0.1078, 0.7547, 1.283),
    (125000, 0.0999, 0.9297, 1.380),
]
BROAD = [
    (0, 0.9998, 2.9915, 3.992),
    (50000, 0.1292, 0.3343, 0.666),
    (100000, 0.0941, 0.2705, 0.729),
    (150000, 0.0858, 0.2523, 0.748),
    (200000, 0.0850, 0.2441, 0.756),
    (250000, 0.0850, 0.2405, 0.760),
    (299999, 0.0850, 0.2394, 0.761),
]

INK = "#1a1a1a"
EPI = "#2166ac"  # blue = mean/bulk (well-constrained)
ALE = "#b2182b"  # red = scale/tails (under-constrained)
GRID = "#d9d9d9"

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.1))

# ---- (a) the asymmetry under a pure velocity loss --------------------------
sp = [r[0] / 1000 for r in PLAIN[1:]]
ax[0].plot(
    sp,
    [r[1] for r in PLAIN[1:]],
    "-o",
    color=EPI,
    lw=2,
    ms=4,
    label="epistemic err (mean structure)",
)
ax[0].plot(
    sp,
    [r[2] for r in PLAIN[1:]],
    "-o",
    color=ALE,
    lw=2,
    ms=4,
    label="aleatoric err (scale structure)",
)
ax[0].axhline(0, color=GRID, lw=1)
ax[0].set_title(
    "(a)  pure velocity loss:\nmean converges, scale diverges", fontsize=11, color=INK
)
ax[0].set_xlabel("training steps (k)")
ax[0].set_ylabel("relative error")
ax[0].legend(frameon=False, fontsize=8.5, loc="upper left")
ax[0].annotate(
    "veldef → 0\n(loss is being minimized)",
    xy=(125, 0.10),
    xytext=(60, -0.02),
    fontsize=8,
    color=INK,
    arrowprops=dict(arrowstyle="->", color=INK, lw=0.8),
)

# ---- (b) the aleatoric trace drifts through its target ---------------------
ax[1].axhline(1.0, color=INK, lw=1.2, ls="--", label="ground-truth ale. trace = 1.0")
ax[1].plot(
    [r[0] / 1000 for r in PLAIN[1:]],
    [r[3] for r in PLAIN[1:]],
    "-o",
    color=ALE,
    lw=2,
    ms=4,
    label="pure velocity (drifts up, past 1.0)",
)
ax[1].plot(
    [r[0] / 1000 for r in BROAD[1:]],
    [r[3] for r in BROAD[1:]],
    "-s",
    color="#238b45",
    lw=2,
    ms=4,
    label="+ tail broadening (mechanism fix, stable)",
)
ax[1].set_title(
    "(b)  aleatoric trace vs its target:\nvelocity loss cannot pin the scale",
    fontsize=11,
    color=INK,
)
ax[1].set_xlabel("training steps (k)")
ax[1].set_ylabel("aleatoric trace  (target 1.0)")
ax[1].legend(frameon=False, fontsize=8.5, loc="upper right")

for a in ax:
    a.spines[["top", "right"]].set_visible(False)
    a.grid(True, color=GRID, lw=0.6, alpha=0.6)
    a.tick_params(labelsize=9)

fig.tight_layout()
out = Path(__file__).resolve().parent / "outputs" / "figures"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "asymmetry.png", dpi=160, bbox_inches="tight")
print(f"saved {out / 'asymmetry.png'}")
