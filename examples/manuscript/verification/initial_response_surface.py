#!/usr/bin/env python3

from pathlib import Path
from manuscript_paths import RESULT_ROOT, SAMPLE_ROOT
import csv
import math

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

PARAM = Path(
    str(SAMPLE_ROOT / 'parameters/Stehekin_test_params_20160327.nc')
)

DOMAIN = Path(
    str(SAMPLE_ROOT / 'parameters/domain.stehekin.20151028.nc')
)

OUT = Path(
    str(RESULT_ROOT / 'A_initial_response_surface')
)

OUT.mkdir(parents=True, exist_ok=True)

GAPS = np.array([
    1.0,
    2.0,
    5.0,
    10.0,
    25.0,
    50.0,
    75.0,
    100.0,
    150.0,
    250.0,
    400.0,
    500.0,
    750.0,
])

KA_SCALES = [
    ("ka1",     1.0),
    ("ka0p1",   0.1),
    ("ka0p01",  0.01),
    ("ka0p001", 0.001),
]

NIU_F = 1.25               # m^-1
RUNOFF_STEPS_PER_DAY = 24
DRAIN_FRACTION = 0.01

layer = -1


# ============================================================
# Read VIC parameters
# ============================================================

with nc.Dataset(PARAM) as ds, nc.Dataset(DOMAIN) as dom:

    depth = np.ma.asarray(ds["depth"][layer])
    kbot = np.ma.asarray(ds["Ksat"][layer])
    bubble = np.ma.asarray(ds["bubble"][layer])
    expt = np.ma.asarray(ds["expt"][layer])
    init = np.ma.asarray(ds["init_moist"][layer])
    resid = np.ma.asarray(ds["resid_moist"][layer])
    bulk = np.ma.asarray(ds["bulk_density"][layer])
    soilrho = np.ma.asarray(ds["soil_density"][layer])

    lat = np.asarray(ds["lat"][:])
    lon = np.asarray(ds["lon"][:])

    mask = np.asarray(dom["mask"][:]) > 0


theta_sat = 1.0 - bulk / soilrho

theta_liq = init / (depth * 1000.0)

se = theta_liq / theta_sat
se = np.ma.clip(se, 1.0e-8, 1.0)

b = 0.5 * (expt - 3.0)

psi_e = -0.01 * bubble

psi = psi_e * se ** (-b)

equilibrium_gap = -psi


valid = (
    mask
    & ~np.ma.getmaskarray(depth)
    & ~np.ma.getmaskarray(kbot)
    & ~np.ma.getmaskarray(init)
    & ~np.ma.getmaskarray(psi)
)


# ============================================================
# Initial downward flux limiter
#
# Frozen soil is OFF for this Stehekin experiment, so initial
# ice storage is zero.
# ============================================================

residual_mm = resid * depth * 1000.0

available_mm = np.maximum(
    0.0,
    np.asarray(init - residual_mm)
)

fraction_limit_mm = (
    DRAIN_FRACTION * np.asarray(init)
)

max_down_step_mm = np.minimum(
    available_mm,
    fraction_limit_mm
)


# ============================================================
# Helpers
# ============================================================

def compressed(a):
    return np.asarray(np.ma.asarray(a))[valid]


def evaluate(gap, ka):
    """
    Current Scheidegger/VIC geometry:

        q = Ka * (1 + psi/gap)

    positive = recharge
    negative = upward GW flux
    """

    q_raw = ka * (1.0 + psi / gap)

    q_step = q_raw / RUNOFF_STEPS_PER_DAY

    q_applied_step = np.array(q_step, dtype=float)

    positive = q_applied_step > 0.0

    q_applied_step[positive] = np.minimum(
        q_applied_step[positive],
        max_down_step_mm[positive]
    )

    q_applied_day_equiv = (
        q_applied_step *
        RUNOFF_STEPS_PER_DAY
    )

    limited = (
        positive &
        (q_step > max_down_step_mm)
    )

    return q_raw, q_applied_day_equiv, limited


def summarize(mode, gap, ka_scale, ka, qraw, qapp, limited):

    x = compressed(qraw)
    y = compressed(qapp)
    kval = compressed(ka)
    lim = np.asarray(limited)[valid]

    eps = 1.0e-10

    return {
        "mode": mode,
        "gap_m": float(gap),
        "ka_scale": ka_scale,

        "Ka_min_mm_day": float(kval.min()),
        "Ka_mean_mm_day": float(kval.mean()),
        "Ka_max_mm_day": float(kval.max()),

        "qraw_min_mm_day": float(x.min()),
        "qraw_mean_mm_day": float(x.mean()),
        "qraw_median_mm_day": float(np.median(x)),
        "qraw_max_mm_day": float(x.max()),

        "qapplied_min_mm_day": float(y.min()),
        "qapplied_mean_mm_day": float(y.mean()),
        "qapplied_median_mm_day": float(np.median(y)),
        "qapplied_max_mm_day": float(y.max()),

        "upward_fraction": float(np.mean(x < -eps)),
        "downward_fraction": float(np.mean(x > eps)),
        "nearzero_fraction": float(np.mean(np.abs(x) <= eps)),

        "downward_limited_fraction": float(np.mean(lim)),
    }


# ============================================================
# Equilibrium-state table
# ============================================================

eq_rows = []

for j in range(valid.shape[0]):
    for i in range(valid.shape[1]):

        if not valid[j, i]:
            continue

        eq_rows.append({
            "lat": float(lat[j]),
            "lon": float(lon[i]),
            "depth_bottom_layer_m": float(depth[j, i]),
            "Kbot_mm_day": float(kbot[j, i]),
            "initial_moist_mm": float(init[j, i]),
            "initial_relative_saturation": float(se[j, i]),
            "bubble_cm": float(bubble[j, i]),
            "expt": float(expt[j, i]),
            "Campbell_b": float(b[j, i]),
            "psi_bottom_m": float(psi[j, i]),
            "equilibrium_gap_m": float(equilibrium_gap[j, i]),
        })


with (OUT / "equilibrium_cells.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=eq_rows[0].keys())
    w.writeheader()
    w.writerows(eq_rows)


# ============================================================
# Full response-surface tables
# ============================================================

summary_rows = []
cell_rows = []


for gap in GAPS:

    # --------------------------------------------------------
    # Constant Ka/Kbot scale experiments
    # --------------------------------------------------------

    for mode, scale in KA_SCALES:

        ka = kbot * scale

        qraw, qapp, limited = evaluate(
            gap,
            ka
        )

        summary_rows.append(
            summarize(
                mode,
                gap,
                scale,
                ka,
                qraw,
                qapp,
                limited
            )
        )

        for j in range(valid.shape[0]):
            for i in range(valid.shape[1]):

                if not valid[j, i]:
                    continue

                cell_rows.append({
                    "mode": mode,
                    "gap_m": float(gap),
                    "lat": float(lat[j]),
                    "lon": float(lon[i]),
                    "Ka_mm_day": float(ka[j, i]),
                    "psi_bottom_m": float(psi[j, i]),
                    "equilibrium_gap_m":
                        float(equilibrium_gap[j, i]),
                    "qraw_mm_day":
                        float(qraw[j, i]),
                    "qapplied_initial_mm_day_equiv":
                        float(qapp[j, i]),
                    "downward_limited":
                        int(limited[j, i]),
                })


    # --------------------------------------------------------
    # Niu exponential depth-decay experiment
    # --------------------------------------------------------

    niu_scale = (
        1.0 - math.exp(-NIU_F * gap)
    ) / (NIU_F * gap)

    ka = kbot * niu_scale

    qraw, qapp, limited = evaluate(
        gap,
        ka
    )

    summary_rows.append(
        summarize(
            "niu_f1p25",
            gap,
            niu_scale,
            ka,
            qraw,
            qapp,
            limited
        )
    )

    for j in range(valid.shape[0]):
        for i in range(valid.shape[1]):

            if not valid[j, i]:
                continue

            cell_rows.append({
                "mode": "niu_f1p25",
                "gap_m": float(gap),
                "lat": float(lat[j]),
                "lon": float(lon[i]),
                "Ka_mm_day": float(ka[j, i]),
                "psi_bottom_m": float(psi[j, i]),
                "equilibrium_gap_m":
                    float(equilibrium_gap[j, i]),
                "qraw_mm_day":
                    float(qraw[j, i]),
                "qapplied_initial_mm_day_equiv":
                    float(qapp[j, i]),
                "downward_limited":
                    int(limited[j, i]),
            })


with (OUT / "summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=summary_rows[0].keys()
    )
    w.writeheader()
    w.writerows(summary_rows)


with (OUT / "cell_response.csv").open("w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=cell_rows[0].keys()
    )
    w.writeheader()
    w.writerows(cell_rows)


# ============================================================
# Figure A1
#
# Raw q versus water-table gap, Ka = Kbot
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

for j in range(valid.shape[0]):
    for i in range(valid.shape[1]):

        if not valid[j, i]:
            continue

        q = (
            float(kbot[j, i]) *
            (
                1.0 +
                float(psi[j, i]) / GAPS
            )
        )

        ax.plot(GAPS, q, marker="o", linewidth=1)

ax.axhline(0.0, linewidth=1)
ax.set_xscale("log")
ax.set_yscale("symlog", linthresh=1.0)

ax.set_xlabel(
    "Groundwater-table gap below VIC soil base (m)"
)
ax.set_ylabel(
    "Initial soil-groundwater exchange (mm day$^{-1}$)"
)

ax.set_title(
    "Initial groundwater exchange response: "
    "$K_a = K_{sat,3}$"
)

fig.tight_layout()

fig.savefig(
    OUT / "fig_A1_q_vs_gap_Ka_Ksat.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# Figure A2
#
# Raw q versus gap using Niu depth-averaged conductivity
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

for j in range(valid.shape[0]):
    for i in range(valid.shape[1]):

        if not valid[j, i]:
            continue

        values = []

        for gap in GAPS:

            scale = (
                1.0 - math.exp(-NIU_F * gap)
            ) / (NIU_F * gap)

            ka = (
                float(kbot[j, i]) *
                scale
            )

            values.append(
                ka *
                (
                    1.0 +
                    float(psi[j, i]) / gap
                )
            )

        ax.plot(
            GAPS,
            values,
            marker="o",
            linewidth=1
        )

ax.axhline(0.0, linewidth=1)
ax.set_xscale("log")
ax.set_yscale("symlog", linthresh=1.0)

ax.set_xlabel(
    "Groundwater-table gap below VIC soil base (m)"
)
ax.set_ylabel(
    "Initial soil-groundwater exchange (mm day$^{-1}$)"
)

ax.set_title(
    "Initial groundwater exchange with "
    "Niu depth-decaying conductivity"
)

fig.tight_layout()

fig.savefig(
    OUT / "fig_A2_q_vs_gap_Niu_Ka.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# Figure A3
#
# Equilibrium groundwater gap versus initial saturation
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    compressed(se),
    compressed(equilibrium_gap),
    s=45
)

ax.set_xlabel(
    "Initial bottom-layer relative saturation (-)"
)
ax.set_ylabel(
    "Zero-flux groundwater gap (m)"
)

ax.set_title(
    "Hydraulic-equilibrium groundwater depth "
    "implied by initial VIC soil moisture"
)

fig.tight_layout()

fig.savefig(
    OUT / "fig_A3_equilibrium_gap.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# Figure A4
#
# Median initial applied flux for Ka scale experiments
# ============================================================

matrix = []

for mode, scale in KA_SCALES:

    row = []

    for gap in GAPS:

        ka = kbot * scale

        _, qapp, _ = evaluate(
            gap,
            ka
        )

        row.append(
            float(
                np.median(
                    compressed(qapp)
                )
            )
        )

    matrix.append(row)


matrix = np.asarray(matrix)

fig, ax = plt.subplots(figsize=(10, 5))

im = ax.imshow(
    matrix,
    aspect="auto"
)

ax.set_xticks(
    np.arange(len(GAPS))
)

ax.set_xticklabels(
    [f"{x:g}" for x in GAPS],
    rotation=45,
    ha="right"
)

ax.set_yticks(
    np.arange(len(KA_SCALES))
)

ax.set_yticklabels(
    [name for name, _ in KA_SCALES]
)

ax.set_xlabel(
    "Groundwater-table gap below soil base (m)"
)

ax.set_ylabel(
    "$K_a/K_{sat,3}$ experiment"
)

ax.set_title(
    "Median initial applied groundwater exchange "
    "(mm day$^{-1}$ equivalent)"
)

fig.colorbar(
    im,
    ax=ax,
    label="Exchange (mm day$^{-1}$)"
)

fig.tight_layout()

fig.savefig(
    OUT / "fig_A4_initial_flux_matrix.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# Markdown experiment record
# ============================================================

eq = compressed(equilibrium_gap)

with (OUT / "EXPERIMENT_A.md").open("w") as f:

    f.write("# Experiment A — Initial hydraulic response surface\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "Quantify the initial VIC soil-groundwater exchange as a "
        "function of prescribed groundwater-table depth and effective "
        "aquifer/interface hydraulic conductivity before performing "
        "additional dynamic simulations.\n\n"
    )

    f.write("## Baseline equation\n\n")
    f.write(
        "`q = Ka * (1 + psi_bottom / gap)`\n\n"
    )

    f.write(
        "Positive q denotes VIC-to-groundwater recharge; "
        "negative q denotes groundwater-to-VIC exchange.\n\n"
    )

    f.write("## Initial equilibrium gaps\n\n")
    f.write(
        f"- minimum: {eq.min():.3f} m\n"
        f"- mean: {eq.mean():.3f} m\n"
        f"- median: {np.median(eq):.3f} m\n"
        f"- maximum: {eq.max():.3f} m\n\n"
    )

    f.write("## Conductivity experiments\n\n")
    f.write(
        "- Ka = Ksat3\n"
        "- Ka = 0.1 Ksat3\n"
        "- Ka = 0.01 Ksat3\n"
        "- Ka = 0.001 Ksat3\n"
        "- Niu exponential depth-decay, f = 1.25 m^-1\n\n"
    )

    f.write("## Files\n\n")
    f.write("- `summary.csv`\n")
    f.write("- `cell_response.csv`\n")
    f.write("- `equilibrium_cells.csv`\n")
    f.write("- `fig_A1_q_vs_gap_Ka_Ksat.png`\n")
    f.write("- `fig_A2_q_vs_gap_Niu_Ka.png`\n")
    f.write("- `fig_A3_equilibrium_gap.png`\n")
    f.write("- `fig_A4_initial_flux_matrix.png`\n")


# ============================================================
# Console summary
# ============================================================

print()
print("===== EQUILIBRIUM GAP =====")
print(f"min    = {eq.min():.3f} m")
print(f"mean   = {eq.mean():.3f} m")
print(f"median = {np.median(eq):.3f} m")
print(f"max    = {eq.max():.3f} m")

print()
print("===== SELECTED RESPONSE-SURFACE RESULTS =====")

print(
    f"{'mode':12s} "
    f"{'gap':>7s} "
    f"{'qraw_min':>13s} "
    f"{'qraw_med':>13s} "
    f"{'qraw_max':>13s} "
    f"{'qapp_med':>13s} "
    f"{'up_frac':>8s} "
    f"{'limited':>8s}"
)

for r in summary_rows:

    if r["gap_m"] not in (
        1.0,
        10.0,
        50.0,
        100.0,
        250.0,
        500.0
    ):
        continue

    print(
        f"{r['mode']:12s} "
        f"{r['gap_m']:7.1f} "
        f"{r['qraw_min_mm_day']:13.3f} "
        f"{r['qraw_median_mm_day']:13.3f} "
        f"{r['qraw_max_mm_day']:13.3f} "
        f"{r['qapplied_median_mm_day']:13.3f} "
        f"{r['upward_fraction']:8.3f} "
        f"{r['downward_limited_fraction']:8.3f}"
    )

print()
print("Results saved in:")
print(OUT)
