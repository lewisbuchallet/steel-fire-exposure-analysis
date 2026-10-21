"""
EN 1991-1-2 Annex A parametric fire curve for the 4 x 4 x 2.4 m compartment,
plus unprotected-steel temperature (EN 1993-1-2 4.2.5.1) under ISO 834, the parametric
curve, and (if compartment1_devc.csv is next to this script) your FDS thermocouples.

CHECK EVERY INPUT in the block below against your own copy of the code.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- inputs ----------------
LX, LY, H = 4.0, 4.0, 2.4        # internal room dimensions, m
AV, HEQ = 1.0 * 2.0, 2.0         # opening area (m2) and weighted opening height (m)
RHO, C, LAM = 2300.0, 880.0, 1.6  # boundary density kg/m3, specific heat J/kgK, conductivity W/mK
QFD = 511.0                      # design fire load density per floor area, MJ/m2 (office, 80% fractile; CHECK Table E.4)
T_LIM_MIN = 20.0                 # 25 slow / 20 medium / 15 fast growth (office: medium)
AM_V = 200.0                     # steel section factor, 1/m (CHANGE to your section)
ALPHA_PARAM, ALPHA_ISO = 35.0, 25.0
# ----------------------------------------

AF = LX * LY
AT = 2 * AF + 2 * (LX + LY) * H          # all enclosure surfaces incl. opening
QTD = QFD * AF / AT                       # MJ/m2 of total enclosure area
O = AV * np.sqrt(HEQ) / AT
b = np.sqrt(RHO * C * LAM)
T_LIM = T_LIM_MIN / 60.0                  # h
G_REF = (0.04 / 1160.0) ** 2

print(f"Af={AF:.1f} m2  At={AT:.1f} m2  O={O:.4f} m^0.5 (valid 0.02-0.20: {0.02<=O<=0.20})")
print(f"b={b:.0f} J/m2s^0.5K (valid 100-2200: {100<=b<=2200})  qt,d={QTD:.1f} MJ/m2 (valid 50-1000: {50<=QTD<=1000})")

gamma = (O / b) ** 2 / G_REF
tmax_vent = 0.2e-3 * QTD / O              # h
fuel_controlled = tmax_vent < T_LIM

def heat(ts):
    return 20 + 1325 * (1 - 0.324 * np.exp(-0.2 * ts) - 0.204 * np.exp(-1.7 * ts) - 0.472 * np.exp(-19 * ts))

if not fuel_controlled:
    tmax = tmax_vent
    g_heat = gamma
    tstar_max = tmax * gamma
    x = 1.0
    cool_tstar_max = tstar_max
else:
    o_lim = 0.1e-3 * QTD / T_LIM
    g_heat = (o_lim / b) ** 2 / G_REF
    if O > 0.04 and QTD < 75 and b < 1160:
        g_heat *= 1 + ((O - 0.04) / 0.04) * ((QTD - 75) / 75) * ((1160 - b) / 1160)
    tmax = T_LIM
    tstar_max = tmax * g_heat
    cool_tstar_max = tmax_vent * gamma
    x = T_LIM * gamma / cool_tstar_max

theta_max = heat(tstar_max)
print(f"Gamma={gamma:.3f}  t_max(vent)={tmax_vent*60:.1f} min  {'FUEL' if fuel_controlled else 'VENTILATION'}-controlled  "
      f"t_max={tmax*60:.1f} min  theta_max={theta_max:.0f} C")

def parametric(t_s):
    th = t_s / 3600.0
    out = np.zeros_like(th)
    heating = th <= tmax
    out[heating] = heat(th[heating] * g_heat)
    ts = th[~heating] * gamma
    if cool_tstar_max <= 0.5:
        out[~heating] = theta_max - 625 * (ts - cool_tstar_max * x)
    elif cool_tstar_max < 2:
        out[~heating] = theta_max - 250 * (3 - cool_tstar_max) * (ts - cool_tstar_max * x)
    else:
        out[~heating] = theta_max - 250 * (ts - cool_tstar_max * x)
    return np.maximum(out, 20.0)

def iso834(t_s):
    return 20 + 345 * np.log10(8 * t_s / 60.0 + 1)

def ca(m):
    if m < 600: return 425 + 0.773*m - 1.69e-3*m**2 + 2.22e-6*m**3
    if m < 735: return 666 + 13002/(738 - m)
    if m < 900: return 545 + 17820/(m - 731)
    return 650.0

def steel(t, tg, alpha):
    s = np.zeros_like(t); s[0] = 20.0
    for i in range(1, len(t)):
        g, m = tg[i-1], s[i-1]
        q = alpha*(g-m) + 0.7*5.67e-8*((g+273.15)**4 - (m+273.15)**4)
        s[i] = m + AM_V/(ca(m)*7850.0)*q*(t[i]-t[i-1])
    return s

t = np.arange(0, 3 * 3600 + 1, 1.0)
g_iso, g_par = iso834(t), parametric(t)
s_iso, s_par = steel(t, g_iso, ALPHA_ISO), steel(t, g_par, ALPHA_PARAM)

for mins in (5, 10, 20, 30, 60, 90, 120):
    i = mins * 60
    print(f"{mins:4d} min  gas ISO {g_iso[i]:5.0f} param {g_par[i]:5.0f} | steel ISO {s_iso[i]:5.0f} param {s_par[i]:5.0f}")
print(f"peak steel: ISO(3h) {s_iso.max():.0f} C, parametric {s_par.max():.0f} C at {t[s_par.argmax()]/60:.0f} min")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(t/60, g_iso, "k--", label="ISO 834")
ax[0].plot(t/60, g_par, label="EN 1991-1-2 Annex A")
ax[1].plot(t/60, s_iso, "k--", label="steel, ISO 834")
ax[1].plot(t/60, s_par, label="steel, parametric")
ax[0].set_title("Gas temperature"); ax[1].set_title(f"Unprotected steel, Am/V={AM_V:.0f} 1/m")
for a in ax:
    a.set_xlabel("time (min)"); a.set_ylabel("temperature (C)"); a.grid(alpha=0.3); a.legend()
plt.tight_layout(); plt.savefig("parametric_compare.png", dpi=150)