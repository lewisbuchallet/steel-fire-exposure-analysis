"""
Makes the two FDS figures for the report and puts all four report figures in ./figures/

Needs, in the same folder:
    compartment3_hrr.csv        (FDS heat release record, Run 2)
    compartment1_devc.csv       (thermocouples, Run 1)
    compartment3_devc.csv       (thermocouples, Run 2)
Run:  python make_figs.py
Also copies steel_comparison.png and parametric_compare.png into figures/ if they exist
(they are written by analyse_all.py and parametric_curve.py).
"""
import csv
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# Colours: categorical slots 1 and 2 of the reference palette, fixed per entity in every panel
C_FIRE = "#2a78d6"   # above fire
C_FAR = "#eb6834"    # far ceiling
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
})


def read_fds_csv(path):
    """FDS csv: row 1 units, row 2 names. Returns {name: values} including Time; duplicate times dropped."""
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    names = [n.strip().strip('"') for n in rows[1]]
    data = np.array([[float(x) for x in r] for r in rows[2:]])
    _, idx = np.unique(data[:, 0], return_index=True)
    return {n: data[idx, i] for i, n in enumerate(names)}


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# ---------------- Figure 1: HRR of Run 2 ----------------
if os.path.exists("compartment3_hrr.csv"):
    h = read_fds_csv("compartment3_hrr.csv")
    t = h["Time"]
    q = h["HRR"] / 1000.0                                  # MW

    def prescribed(ts):
        out = np.where(ts <= 300, 5.0 * (ts / 300.0) ** 2, 5.0)
        out = np.where(ts > 1500, np.clip(5.0 * (2100 - ts) / 600.0, 0, None), out)
        return out

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.plot(t / 60, q, color=C_FIRE, lw=0.9, label="FDS heat release rate")
    tp = np.linspace(0, t[-1], 500)
    ax.plot(tp / 60, prescribed(tp), color=INK, lw=1.4, ls="--", label="Prescribed")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Heat release rate (MW)")
    ax.set_xlim(0, t[-1] / 60)
    ax.set_ylim(0, 6.5)
    style(ax)
    ax.legend(frameon=False, loc="lower center", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fds_hrr.png"), dpi=200)
    plt.close(fig)

    # Integrate FDS heat release rate to obtain total energy (MJ)
    FDS_energy_MJ = np.trapezoid(h["HRR"], t) / 1000.0

    print("wrote figures/fds_hrr.png")
    print(f"Run 2 total energy released: {FDS_energy_MJ:.2f} MJ")

else:
    print("[skip] compartment3_hrr.csv not found")

# ---------------- Figure 2: thermocouples, both runs ----------------
runs = [("Run 1 (0.96 MW)", "compartment1_devc.csv"), ("Run 2 (5 MW)", "compartment3_devc.csv")]
have = [(n, f) for n, f in runs if os.path.exists(f)]
if have:
    fig, axes = plt.subplots(1, len(have), figsize=(7.5, 3.3), sharey=True, squeeze=False)
    for ax, (name, f) in zip(axes[0], have):
        d = read_fds_csv(f)
        tm = d["Time"] / 60
        ax.plot(tm, d["TC_ceiling_fire"], color=C_FIRE, lw=1.0, label="Ceiling above fire")
        ax.plot(tm, d["TC_ceiling_far"], color=C_FAR, lw=1.0, label="Ceiling away from fire")
        ax.set_title(name, color=INK, fontsize=9, loc="left")
        ax.set_xlabel("Time (min)")
        ax.set_xlim(0, tm[-1])
        ax.set_ylim(0, 1000)
        style(ax)
    axes[0][0].set_ylabel("Thermocouple temperature (°C)")
    axes[0][0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fds_gas_temperature.png"), dpi=200)
    plt.close(fig)
    print("wrote figures/fds_gas_temperature.png")
else:
    print("[skip] no devc csv found")

# ---------------- copy the other two figures ----------------
for name in ("steel_comparison.png", "parametric_compare.png"):
    if os.path.exists(name):
        shutil.copy(name, os.path.join(OUT, name))
        print("copied", name, "to figures/")
    else:
        print("[missing]", name, "- run analyse_all.py / parametric_curve.py first")

