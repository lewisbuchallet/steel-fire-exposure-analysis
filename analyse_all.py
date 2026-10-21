# Source - https://stackoverflow.com/a/57409500
# Posted by Maeaex1
# Retrieved 2026-10-14, License - CC BY-SA 4.0

import csv
import os
import time
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import parametric_curve as pc

start_time = time.time()

AM_V_LIST = (100.0, 200.0, 300.0)
T_END = 3 * 3600.0

# k_y,theta from EN 1993-1-2 Table 3.1, FROM MEMORY. CHECK against your copy.
KY_T = [20, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
KY_V = [1.0, 1.0, 1.0, 1.0, 1.0, 0.78, 0.47, 0.23, 0.11, 0.06, 0.04, 0.02, 0.0]


def read_devc(path):
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    names = [n.strip().strip('"') for n in rows[1]]
    data = np.array([[float(x) for x in r] for r in rows[2:]])
    t, idx = np.unique(data[:, 0], return_index=True)
    return t, {n: data[idx, i] for i, n in enumerate(names) if i > 0}


def steel_am(t, tg, alpha, am_v):
    s = np.zeros_like(t)
    s[0] = 20.0
    for i in range(1, len(t)):
        g, m = tg[i - 1], s[i - 1]
        q = alpha * (g - m) + 0.7 * 5.67e-8 * (
            (g + 273.15) ** 4 - (m + 273.15) ** 4
        )
        s[i] = m + am_v / (pc.ca(m) * 7850.0) * q * (t[i] - t[i - 1])
    return s


def t_equivalent(peak, t, s_iso):
    k = np.argmax(s_iso >= peak)
    return None if s_iso[k] < peak else t[k] / 60.0


def ky(theta):
    return float(np.interp(theta, KY_T, KY_V))


t = np.arange(0, T_END + 1, 1.0)
exposures = {
    "ISO 834": (pc.iso834(t), pc.ALPHA_ISO),
    "Parametric (Annex A)": (pc.parametric(t), pc.ALPHA_PARAM),
}

# Updated run definitions to change legend labels
runs = (
    ("Run 1 (0.96MW)", "compartment1_devc.csv"),
    ("Run 2 (5MW)", "compartment3_devc.csv"),
)

for label, fname in runs:
    if not os.path.exists(fname):
        print(f"[skip] {fname} not found")
        continue
    tf, cols = read_devc(fname)
    for dev in ("TC_ceiling_fire", "TC_ceiling_far"):
        g = np.interp(t, tf, cols[dev], right=cols[dev][-1])
        g[t > tf[-1]] = np.nan
        exposures[f"FDS {label} {dev.replace('TC_ceiling_', '')}"] = (
            g,
            pc.ALPHA_PARAM,
        )
    print(
        f"{fname}: {tf[0]:.0f}-{tf[-1]:.0f} s, {len(tf)} samples, devices:"
        f" {', '.join(cols)}"
    )

results = {}
lines = []
for am in AM_V_LIST:
    lines.append(f"\nAm/V = {am:.0f} 1/m")
    lines.append(
        f"{'exposure':36s} {'peak gas':>9s} {'peak steel':>11s}"
        f" {'at (min)':>9s} {'t_e (min)':>10s} {'k_y':>6s}"
    )
    s_iso = steel_am(t, exposures["ISO 834"][0], exposures["ISO 834"][1], am)
    for name, (g, alpha) in exposures.items():
        ok = ~np.isnan(g)
        tt, gg = t[ok], g[ok]
        s = steel_am(tt, gg, alpha, am)
        pk, ipk = s.max(), s.argmax()
        te = t_equivalent(pk, t, s_iso) if name != "ISO 834" else None
        results[(am, name)] = (tt, gg, s)
        lines.append(
            f"{name:36s} {gg.max():9.0f} {pk:11.0f} {tt[ipk]/60:9.1f} "
            f"{('-' if te is None else f'{te:.1f}'):>10s} {ky(pk):6.2f}"
        )
text = "\n".join(lines)
print(text)
print(
    "\nNOTE: ISO 834 'peak' is the value at the end of the 3 h window, not a"
    " real peak."
)
open("results_table.txt", "w").write(text + "\n")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for name in exposures:
    tt, gg, s = results[(200.0, name)]
    sty = "k--" if name == "ISO 834" else "-"
    ax[0].plot(tt / 60, gg, sty, label=name)
    ax[1].plot(tt / 60, s, sty, label=name)
ax[0].set_title("Gas temperature")
ax[1].set_title("Unprotected steel, Am/V = 200 1/m")
for a in ax:
    a.set_xlim(0, 60)
    a.set_xlabel("time (min)")
    a.set_ylabel("temperature (C)")
    a.grid(alpha=0.3)
ax[0].legend(fontsize=7)
plt.tight_layout()
plt.savefig("steel_comparison.png", dpi=150)
print("wrote steel_comparison.png and results_table.txt")

print("Process finished --- %s seconds ---" % (time.time() - start_time))