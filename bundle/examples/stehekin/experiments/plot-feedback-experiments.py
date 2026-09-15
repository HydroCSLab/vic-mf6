#!/usr/bin/env python3
"""Audit the archived process runs and draw figures from their outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
import numpy as np
import pandas as pd


EXAMPLE_DIR = Path(__file__).resolve().parents[1]
BUNDLE_DIR = EXAMPLE_DIR.parents[1]
COLORS = {
    "baseline": "#376996",
    "stock": "#666666",
    "pumped": "#b35a40",
    "replay": "#84699c",
    "tight-baseline": "#4b8b7a",
    "tight-pumped": "#4b8b7a",
    "pumped-3mm": "#c68b30",
}
LABELS = {
    "baseline": "two-way, no withdrawal",
    "stock": "VIC, ARNO boundary",
    "pumped": "two-way, withdrawal",
    "replay": "prescribed-exchange withdrawal",
    "tight-baseline": "low-K aquitard, no withdrawal",
    "tight-pumped": "low-K aquitard, withdrawal",
    "pumped-3mm": "threefold withdrawal",
}


def load_runs(campaign: Path) -> dict:
    runs = {}
    for name in (*COLORS, "baseline-6h", "pumped-6h", "high-snow"):
        directory = campaign / name
        provenance = json.loads((directory / "provenance.json").read_text())
        if provenance.get("status") != "completed":
            raise ValueError(f"run is not completed: {directory}")
        runs[name] = {
            "table": pd.read_csv(directory / "timeseries.csv"),
            "fields": dict(np.load(directory / "fields.npz")),
            "settings": provenance,
            "directory": directory,
        }
        if name != "stock":
            runs[name]["budgets"] = pd.read_csv(directory / "budgets.csv")
    return runs


def save(fig, figures: Path, name: str):
    fig.savefig(figures / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(figures / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def style_time(ax, ylabel, title, shade=True):
    ax.set(xlabel="simulation day", ylabel=ylabel, title=title)
    if shade:
        ax.axvspan(20, 45, color="#e8ded6", alpha=0.4, zorder=-10)
    ax.grid(alpha=0.2)


def audit(runs, mapping, analysis_dir):
    """Test matched controls and physical accounting independently of plotting."""
    checks = []

    def check(name, value, limit, comparison="<="):
        passed = (value <= limit) if comparison == "<=" else (value > limit)
        checks.append(
            dict(
                check=name,
                value=float(value),
                criterion=f"{comparison} {limit:g}",
                passed=bool(passed),
            )
        )

    metrics = []
    for name, run in runs.items():
        t = run["table"]
        f = run["fields"]
        check(f"{name}: VIC water balance mm", t.max_vic_water_error_mm.max(), 1e-7)
        check(f"{name}: mapping m3", t.mapping_error_m3.abs().max(), 1e-5)
        row = dict(
            case=name,
            interval_days=run["settings"]["interval"],
            evap_mm=t.OUT_EVAP.sum(),
            runoff_mm=t.OUT_RUNOFF.sum(),
            baseflow_mm=t.OUT_BASEFLOW.sum(),
            exchange_mm=t.OUT_GW_EXCHANGE.sum(),
            final_soil_mm=t.filter(like="OUT_SOIL_MOIST_").iloc[-1].sum(),
        )
        if name != "stock":
            b = run["budgets"]
            h = f["heads"].reshape(-1, 3, 48)
            check(f"{name}: cell budget m3", b.max_cell_residual_m3.max(), 1e-3)
            check(f"{name}: API rate m3/day", t.api_error_m3_day.max(), 1e-5)
            check(
                f"{name}: upper saturated thickness m", (h[:, 0] + 20).min(), 0.0, ">"
            )
            check(
                f"{name}: aquitard confined head margin m",
                (h[:, 1] + 20).min(),
                0.0,
                ">",
            )
            check(
                f"{name}: deep confined head margin m", (h[:, 2] + 25).min(), 0.0, ">"
            )
            row.update(
                withdrawal_m3=-b.withdrawal_m3.sum(),
                storage_change_m3=b.storage_change_m3.sum(),
                maximum_cell_residual_m3=b.max_cell_residual_m3.max(),
            )
        metrics.append(row)
    # before withdrawal starts, each pair has exactly the same equations and
    # state. this control detects inadvertent differences in forcing or restart.
    for pumped, baseline in (
        ("pumped", "baseline"),
        ("tight-pumped", "tight-baseline"),
        ("pumped-3mm", "baseline"),
        ("pumped-6h", "baseline-6h"),
    ):
        p, b = runs[pumped], runs[baseline]
        before = p["table"].time_days.to_numpy() <= 20
        for var in ("heads", "OUT_SOIL_MOIST", "OUT_GW_EXCHANGE", "OUT_EVAP"):
            check(
                f"{pumped}: pre-withdrawal equality {var}",
                np.max(np.abs(p["fields"][var][before] - b["fields"][var][before])),
                1e-9,
            )
    check(
        "replay: prescribed interface identity m3",
        np.max(
            np.abs(
                runs["replay"]["fields"]["interface_m3"]
                - runs["baseline"]["fields"]["interface_m3"]
            )
        ),
        1e-8,
    )
    # the change in soil storage must balance the change in exchange and other
    # VIC outputs. identical initial states cancel from this paired budget.
    contrasts = []
    areas = mapping["areas"]
    for pumped, baseline in (
        ("pumped", "baseline"),
        ("tight-pumped", "tight-baseline"),
        ("pumped-3mm", "baseline"),
        ("pumped-6h", "baseline-6h"),
    ):
        p, b = runs[pumped], runs[baseline]
        pt, bt = p["table"], b["table"]
        delta_exchange = pt.OUT_GW_EXCHANGE.sum() - bt.OUT_GW_EXCHANGE.sum()
        delta_evap = pt.OUT_EVAP.sum() - bt.OUT_EVAP.sum()
        delta_runoff = pt.OUT_RUNOFF.sum() - bt.OUT_RUNOFF.sum()
        delta_baseflow = pt.OUT_BASEFLOW.sum() - bt.OUT_BASEFLOW.sum()
        state_columns = [
            "OUT_SOIL_MOIST_1",
            "OUT_SOIL_MOIST_2",
            "OUT_SOIL_MOIST_3",
            "OUT_SWE",
            "OUT_SNOW_CANOPY",
            "OUT_WDEW",
        ]
        delta_storage = (pt[state_columns].iloc[-1] - bt[state_columns].iloc[-1]).sum()
        error = (
            delta_exchange + delta_evap + delta_runoff + delta_baseflow + delta_storage
        )
        check(f"{pumped}: paired land water closure mm", abs(error), 1e-7)
        contrasts.append(
            dict(
                case=pumped,
                control=baseline,
                upward_support_loss_mm=delta_exchange,
                evap_difference_mm=delta_evap,
                final_land_storage_difference_mm=delta_storage,
                paired_water_residual_mm=error,
                support_loss_m3=delta_exchange * 0.001 * areas.sum(),
                maximum_upper_drawdown_m=np.max(
                    b["fields"]["heads"][:, :48] - p["fields"]["heads"][:, :48]
                ),
                maximum_deep_drawdown_m=np.max(
                    b["fields"]["heads"][:, 96:] - p["fields"]["heads"][:, 96:]
                ),
            )
        )
    checks = pd.DataFrame(checks)
    checks.to_csv(analysis_dir / "checks.csv", index=False)
    pd.DataFrame(metrics).to_csv(analysis_dir / "run-summary.csv", index=False)
    contrasts = pd.DataFrame(contrasts)
    contrasts.to_csv(analysis_dir / "paired-effects.csv", index=False)
    if not checks.passed.all():
        raise ValueError(checks.loc[~checks.passed].to_string(index=False))
    print(f"passed {len(checks)} recorded control and budget checks")
    print(contrasts.to_string(index=False))
    return contrasts


def grids(runs, mapping, geometry, figures):
    collection = json.loads(
        (runs["baseline"]["directory"] / "grids.geojson").read_text()
    )
    polys = {k: [] for k in ("vic", "mf6")}
    origin = np.array([geometry["west"], geometry["south"]])
    for feature in collection["features"]:
        coords = (np.asarray(feature["geometry"]["coordinates"][0]) - origin) / 1000
        polys[feature["properties"]["grid"]].append(coords)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9), layout="constrained")

    def draw(ax, kind, values=None, color="none", edge=".5", alpha=1.0, cmap="viridis"):
        patches = [Polygon(p) for p in polys[kind]]
        options = {} if values is not None else {"facecolor": color}
        p = PatchCollection(
            patches, edgecolor=edge, linewidth=0.7, alpha=alpha, cmap=cmap, **options
        )
        if values is not None:
            p.set_array(np.asarray(values))
        ax.add_collection(p)
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set(
            xlabel="easting from domain edge (km)",
            ylabel="northing from domain edge (km)",
        )
        return p

    ax = axes[0, 0]
    draw(ax, "mf6", edge=".85")
    c = draw(ax, "vic", mapping["areas"] / 1e6, edge="white", cmap="Blues")
    for i, p in enumerate(polys["vic"]):
        ax.text(*p[:-1].mean(axis=0), str(i), ha="center", va="center", fontsize=8)
    ax.set_title("(a) 16 active VIC cells; original footprint")
    fig.colorbar(c, ax=ax, shrink=0.75, label="authoritative VIC area (km²)")
    ax = axes[0, 1]
    draw(ax, "mf6", color="#e9e4d9", edge=".35")
    draw(ax, "vic", edge="#376996", alpha=0.9)
    for node in geometry["pumped_nodes"]:
        ax.plot(
            *polys["mf6"][node][:-1].mean(axis=0), marker="v", color="#b35a40", ms=7
        )
    ax.set_title("(b) actual nonmatching overlay; 48 cells per layer")
    ax = axes[1, 0]
    im = ax.imshow(
        mapping["weights"],
        aspect="auto",
        cmap="Blues",
        vmin=0,
        vmax=1,
        interpolation="none",
    )
    ax.set(
        title="(c) exact intersection fractions",
        xlabel="upper MF6 cell index",
        ylabel="VIC cell index",
    )
    fig.colorbar(im, ax=ax, shrink=0.75, label=r"$w_{ij}=a_{ij}/\sum_j a_{ij}$")
    ax = axes[1, 1]
    width = (geometry["east"] - geometry["west"]) / 1000
    for top, bottom, color, label in (
        (0, -20, "#dbe9f1", "convertible aquifer"),
        (-20, -25, "#c8bd9a", "aquitard"),
        (-25, -75, "#cadad1", "deep confined aquifer"),
    ):
        ax.add_patch(
            Rectangle((0, bottom), width, top - bottom, facecolor=color, edgecolor=".5")
        )
        ax.text(width * 0.02, (top + bottom) / 2, label, va="center", fontsize=9)
    x = (np.arange(8) + 0.5) * geometry["dx"] / 1000
    for name, style in (("baseline", "-"), ("pumped", "--")):
        h = runs[name]["fields"]["heads"][44].reshape(3, 6, 8)
        ax.plot(x, h[0, 2], style, color=COLORS[name], label=f"upper head, {name}")
    for col in (3, 4):
        ax.plot([x[col], x[col]], [3, -55], color=COLORS["pumped"], lw=1.6)
        ax.plot([x[col], x[col]], [-50, -58], color=COLORS["pumped"], lw=5)
    ax.annotate(
        "VIC soil-base interface",
        (width * 0.75, 0),
        (width * 0.5, 10),
        fontsize=8,
        arrowprops=dict(arrowstyle="->"),
    )
    ax.set(
        xlim=(0, width),
        ylim=(-78, 15),
        xlabel="distance along row 2 (km)",
        ylabel="elevation in local datum (m)",
        title="(d) layered section and day-45 heads",
    )
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.04), fontsize=7, frameon=False)
    save(fig, figures, "fig-coupled-grid-overlay")
    return polys


def snowmelt(runs, figures):
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10), layout="constrained")
    t = runs["baseline"]["table"]
    days = t.time_days
    ax = axes[0, 0]
    ax.bar(days, t.OUT_SNOWF, color="#9ebbd0", label="snowfall")
    ax.plot(days, t.OUT_SNOW_MELT, color="#376996", label="snowmelt")
    style_time(
        ax,
        "water depth (mm/day)",
        "(a) prescribed accumulation and simulated melt",
        False,
    )
    ax.legend(frameon=False)
    ax = axes[0, 1]
    ax.plot(days, t.OUT_SWE, color="#376996", label="snow water equivalent")
    style_time(ax, "snow storage (mm)", "(b) snow release precedes the dry-down", False)
    for name in ("stock", "baseline"):
        r = runs[name]["table"]
        for ax, column, title, ylabel, cumulative in (
            (
                axes[1, 0],
                "OUT_SOIL_MOIST_3",
                "(c) bottom-layer soil water",
                "storage (mm)",
                False,
            ),
            (
                axes[1, 1],
                "OUT_EVAP",
                "(d) cumulative evapotranspiration",
                "water depth (mm)",
                True,
            ),
            (
                axes[2, 0],
                "OUT_RUNOFF",
                "(e) cumulative surface runoff",
                "water depth (mm)",
                True,
            ),
        ):
            ax.plot(
                days,
                r[column].cumsum() if cumulative else r[column],
                label=LABELS[name],
                color=COLORS[name],
            )
            style_time(ax, ylabel, title, False)
    axes[1, 0].legend(frameon=False)
    ax = axes[2, 1]
    ax.plot(
        days, -t.OUT_GW_EXCHANGE, color=COLORS["baseline"], label="groundwater support"
    )
    ax.plot(
        days,
        runs["stock"]["table"].OUT_BASEFLOW,
        color=COLORS["stock"],
        label="ARNO drainage",
    )
    style_time(
        ax,
        "lower-boundary magnitude (mm/day)",
        "(f) distinct lower-boundary water pathways",
        False,
    )
    ax.legend(frameon=False)
    save(fig, figures, "fig-snowmelt-drydown-response")


def withdrawal(runs, mapping, figures):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), layout="constrained")
    baseline = runs["baseline"]
    t = baseline["table"].time_days.to_numpy()
    for name in ("pumped", "pumped-3mm", "replay"):
        f = runs[name]["fields"]
        b = baseline["fields"]
        style = "--" if name == "replay" else "-"
        dh = (f["heads"] - b["heads"]).reshape(-1, 3, 48)
        axes[0, 0].plot(
            t, -dh[:, 0].min(axis=1), style, color=COLORS[name], label=LABELS[name]
        )
        dq = np.average(
            f["OUT_GW_EXCHANGE"] - b["OUT_GW_EXCHANGE"],
            axis=1,
            weights=mapping["areas"],
        )
        axes[0, 1].plot(t, dq, style, color=COLORS[name])
        ds = np.average(
            (f["OUT_SOIL_MOIST"] - b["OUT_SOIL_MOIST"]).sum(axis=1),
            axis=1,
            weights=mapping["areas"],
        )
        axes[1, 0].plot(t, ds, style, color=COLORS[name])
        axes[1, 1].plot(t, dq.cumsum(), style, color=COLORS[name])
    for ax, title, ylabel in (
        (axes[0, 0], "(a) maximum upper-aquifer drawdown", "drawdown (m)"),
        (
            axes[0, 1],
            "(b) reduced upward groundwater support",
            "exchange change (mm/day)",
        ),
        (
            axes[1, 0],
            "(c) change in soil-water availability",
            "soil-storage change (mm)",
        ),
        (axes[1, 1], "(d) accumulated reduction in support", "water depth (mm)"),
    ):
        style_time(ax, ylabel, title)
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, figures, "fig-withdrawal-feedback")


def aquitard(runs, mapping, figures):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), layout="constrained")
    for pumped, control, color, label in (
        ("pumped", "baseline", "#376996", "aquitard K = 0.02 m/day"),
        ("tight-pumped", "tight-baseline", "#4b8b7a", "aquitard K = 0.002 m/day"),
    ):
        f, b = runs[pumped]["fields"], runs[control]["fields"]
        t = runs[pumped]["table"].time_days
        dh = (b["heads"] - f["heads"]).reshape(-1, 3, 48)
        axes[0, 0].plot(t, dh[:, 2].max(axis=1), color=color, label=label)
        axes[0, 1].plot(t, dh[:, 0].max(axis=1), color=color)
        dv = (f["vertical_down_m3"] - b["vertical_down_m3"]).sum(axis=1)
        axes[1, 0].plot(t, dv / 1e6, color=color)
        dq = np.average(
            f["OUT_GW_EXCHANGE"] - b["OUT_GW_EXCHANGE"],
            axis=1,
            weights=mapping["areas"],
        )
        axes[1, 1].plot(t, dq, color=color)
    for ax, title, ylabel in (
        (axes[0, 0], "(a) deep aquifer response", "maximum drawdown (m)"),
        (axes[0, 1], "(b) upper aquifer response", "maximum drawdown (m)"),
        (
            axes[1, 0],
            "(c) additional leakage out of the upper aquifer",
            r"downward volume ($10^6$ m³/day)",
        ),
        (axes[1, 1], "(d) response transmitted to VIC", "support reduction (mm/day)"),
    ):
        style_time(ax, ylabel, title)
    axes[0, 0].legend(frameon=False)
    save(fig, figures, "fig-aquitard-transmission")


def spatial(runs, polys, geometry, figures):
    f, b = runs["pumped"]["fields"], runs["baseline"]["fields"]
    final = 44
    values = [
        (
            (f["heads"] - b["heads"])[final, :48],
            "mf6",
            "(a) upper head change, day 45",
            "m",
        ),
        (
            (f["heads"] - b["heads"])[final, 96:],
            "mf6",
            "(b) deep head change, day 45",
            "m",
        ),
        (
            (f["OUT_SOIL_MOIST"] - b["OUT_SOIL_MOIST"])[final].sum(axis=0),
            "vic",
            "(c) soil-storage change, day 45",
            "mm",
        ),
        (
            (f["OUT_GW_EXCHANGE"] - b["OUT_GW_EXCHANGE"])[:45].sum(axis=0),
            "vic",
            "(d) cumulative support reduction to day 45",
            "mm",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9), layout="constrained")
    for ax, (value, kind, title, units) in zip(axes.flat, values, strict=True):
        patches = PatchCollection(
            [Polygon(p) for p in polys[kind]],
            cmap="RdBu_r",
            edgecolor=".5",
            linewidth=0.4,
        )
        patches.set_array(value)
        limit = max(np.max(np.abs(value)), 1e-12)
        patches.set_clim(-limit, limit)
        ax.add_collection(patches)
        if kind == "mf6":
            ax.add_collection(
                PatchCollection(
                    [Polygon(p) for p in polys["vic"]],
                    facecolor="none",
                    edgecolor=".25",
                    linewidth=0.7,
                )
            )
        for node in geometry["pumped_nodes"]:
            ax.plot(*polys["mf6"][node][:-1].mean(axis=0), "v", color="black", ms=5)
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set(title=title, xlabel="easting (km)", ylabel="northing (km)")
        fig.colorbar(patches, ax=ax, shrink=0.8, label=units)
    save(fig, figures, "fig-spatial-feedback-response")


def closure_and_interval(runs, contrasts, figures):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), layout="constrained")
    for name in ("baseline", "pumped", "tight-pumped", "replay"):
        b = runs[name]["budgets"]
        axes[0, 0].plot(
            b.time_days,
            b.storage_change_m3.cumsum() / 1e6,
            color=COLORS[name],
            label=LABELS[name],
        )
        axes[0, 0].plot(
            b.time_days,
            (b.interface_m3 + b.withdrawal_m3).cumsum() / 1e6,
            ":",
            color="black",
            lw=0.7,
        )
        axes[0, 1].semilogy(
            b.time_days, b.max_cell_residual_m3.clip(lower=1e-14), color=COLORS[name]
        )
    axes[0, 1].axhline(1e-3, color=".4", linestyle="--", label="acceptance tolerance")
    axes[0, 0].legend(frameon=False, fontsize=7)
    style_time(
        axes[0, 0],
        r"cumulative volume ($10^6$ m³)",
        "(a) storage (color) and external fluxes (dotted)",
    )
    style_time(
        axes[0, 1],
        "maximum cell residual (m³)",
        "(b) closure including withdrawal and vertical flow",
    )
    for p, b, color, label in (
        ("pumped", "baseline", "#376996", "one-day coupling"),
        ("pumped-6h", "baseline-6h", "#b35a40", "six-hour coupling"),
    ):
        t = runs[p]["table"].time_days
        difference = runs[p]["table"].OUT_GW_EXCHANGE - runs[b]["table"].OUT_GW_EXCHANGE
        axes[1, 0].plot(t, difference.cumsum(), color=color, label=label)
        axes[1, 1].plot(t, runs[b]["table"].OUT_SOIL_MOIST_3, color=color, label=label)
    style_time(
        axes[1, 0],
        "cumulative support reduction (mm)",
        "(c) refinement of the paired withdrawal effect",
    )
    style_time(
        axes[1, 1],
        "bottom-layer water (mm)",
        "(d) refinement of the unperturbed soil trajectory",
    )
    axes[1, 0].legend(frameon=False)
    save(fig, figures, "fig-feedback-budget-refinement")


def snow_sensitivity(runs, mapping, figures):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), layout="constrained")
    for name, color, label in (
        ("baseline", "#376996", "80 mm snowfall input"),
        ("high-snow", "#4b8b7a", "240 mm snowfall input"),
    ):
        r = runs[name]
        t = r["table"].time_days
        axes[0, 0].plot(t, r["table"].OUT_SWE, color=color, label=label)
        axes[0, 1].plot(t, r["table"].OUT_GW_EXCHANGE, color=color)
        q = r["fields"]["OUT_GW_EXCHANGE"]
        positive = np.maximum(q, 0) @ mapping["areas"] * 0.001
        axes[1, 0].plot(t, positive.cumsum() / 1e6, color=color)
        heads = r["fields"]["heads"][:, :48]
        average = (
            (heads @ mapping["weights"].T) @ mapping["areas"] / mapping["areas"].sum()
        )
        axes[1, 1].plot(t, average, color=color)
    axes[0, 1].axhline(0, color=".4", linestyle=":")
    axes[0, 1].set_ylim(-3.5, 0.35)
    axes[0, 1].set_xlim(10, 60)
    for ax, title, ylabel in (
        (axes[0, 0], "(a) snow storage and melt duration", "SWE (mm)"),
        (
            axes[0, 1],
            "(b) exchange during melt and dry-down",
            "signed exchange (mm/day)",
        ),
        (
            axes[1, 0],
            "(c) accumulated downward transfer",
            r"gross downward volume ($10^6$ m³)",
        ),
        (
            axes[1, 1],
            "(d) groundwater state returned to VIC",
            "footprint-weighted upper head (m)",
        ),
    ):
        style_time(ax, ylabel, title, False)
    axes[0, 0].legend(frameon=False)
    save(fig, figures, "fig-snowmelt-exchange-reversal")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=BUNDLE_DIR / "runs/vicmf6-feedback/campaign-01",
    )
    parser.add_argument(
        "--analysis-dir", type=Path, default=BUNDLE_DIR / "analysis/vicmf6-feedback"
    )
    parser.add_argument("--figure-dir", type=Path, default=EXAMPLE_DIR / "figures")
    args = parser.parse_args()
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    runs = load_runs(args.campaign_dir)
    mapping = dict(np.load(args.campaign_dir / "baseline/mapping.npz"))
    geometry = json.loads((args.campaign_dir / "baseline/geometry.json").read_text())
    contrasts = audit(runs, mapping, args.analysis_dir)
    polys = grids(runs, mapping, geometry, args.figure_dir)
    snowmelt(runs, args.figure_dir)
    withdrawal(runs, mapping, args.figure_dir)
    aquitard(runs, mapping, args.figure_dir)
    spatial(runs, polys, geometry, args.figure_dir)
    closure_and_interval(runs, contrasts, args.figure_dir)
    snow_sensitivity(runs, mapping, args.figure_dir)
    print(f"wrote seven figures to {args.figure_dir}")


if __name__ == "__main__":
    main()
