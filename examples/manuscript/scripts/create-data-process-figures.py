#!/usr/bin/env python3
"""Export only the process fields needed by the manuscript figures and audits.

The CSV files retain full precision, cell ordering, polygon vertices, and units.
Restart files, unused fields, binary budgets, and raw metadata stay in the run.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CASES = ["baseline", "stock", "pumped", "replay", "tight-baseline", "tight-pumped",
         "pumped-3mm", "baseline-6h", "pumped-6h", "high-snow"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    settings, timeseries, budgets, vic_fields, gw_fields = [], [], [], [], []

    def save(name, data):
        data.to_csv(a.output_dir / f"data-process-{name}.csv", index=False)

    for case in CASES:
        root = a.campaign_dir / case
        metadata = json.loads((root / "provenance.json").read_text())
        if metadata["status"] != "completed":
            raise ValueError(f"Incomplete case: {case}")
        settings.append(dict(name=case, **{key: metadata[key] for key in
            ["case", "days", "interval", "aquitard_k", "pumping_mm_day", "snowfall_mm_day", "scale", "length"]}))
        table = pd.read_csv(root / "timeseries.csv")
        timeseries.append(table.assign(case=case))
        fields = np.load(root / "fields.npz")
        count, cells = fields["OUT_GW_EXCHANGE"].shape
        values = dict(case=case, step=np.repeat(np.arange(count), cells), cell=np.tile(np.arange(cells), count))
        for key in ["OUT_GW_EXCHANGE", "OUT_EVAP"]:
            values[key] = fields[key].ravel()
        for layer in range(3):
            values[f"OUT_SOIL_MOIST_{layer + 1}"] = fields["OUT_SOIL_MOIST"][:, layer, :].ravel()
        vic_fields.append(pd.DataFrame(values))
        if case != "stock":
            budgets.append(pd.read_csv(root / "budgets.csv").assign(case=case))
            heads = fields["heads"].reshape(count, 3, 48)
            values = dict(case=case, step=np.repeat(np.arange(count), 48), cell=np.tile(np.arange(48), count))
            for layer in range(3):
                values[f"head_layer_{layer + 1}_m"] = heads[:, layer, :].ravel()
            for key in ["interface_m3", "vertical_down_m3"]:
                values[key] = fields[key].ravel()
            gw_fields.append(pd.DataFrame(values))
    for name, values in [("cases", pd.DataFrame(settings)), ("timeseries", pd.concat(timeseries)),
                          ("budgets", pd.concat(budgets)), ("vic-fields", pd.concat(vic_fields)),
                          ("groundwater-fields", pd.concat(gw_fields))]:
        save(name, values)
    root = a.campaign_dir / "baseline"
    mapping = np.load(root / "mapping.npz")
    rows = []
    for i, j in zip(*np.where(mapping["weights"] > 0)):
        rows.append(dict(vic_cell=i, groundwater_cell=j, fraction=mapping["weights"][i, j],
                         overlap_area_m2=mapping["transfer_areas"][i, j]))
    save("overlaps", pd.DataFrame(rows))
    save("vic-cells", pd.DataFrame(dict(cell=np.arange(16), area_m2=mapping["areas"],
         row=mapping["rows"], col=mapping["cols"], lat=mapping["lat"], lon=mapping["lon"])))
    rows = []
    for key, value in json.loads((root / "geometry.json").read_text()).items():
        if isinstance(value, list):
            rows.extend(dict(parameter=key, index=i, value=v) for i, v in enumerate(value))
        else:
            rows.append(dict(parameter=key, index="", value=value))
    save("geometry", pd.DataFrame(rows))
    rows = []
    counters = {"vic": 0, "mf6": 0}
    for feature in json.loads((root / "grids.geojson").read_text())["features"]:
        grid = feature["properties"]["grid"]
        for vertex, (x, y) in enumerate(feature["geometry"]["coordinates"][0]):
            rows.append(dict(grid=grid, cell=counters[grid], vertex=vertex, x_m=x, y_m=y))
        counters[grid] += 1
    save("vertices", pd.DataFrame(rows))
    print("Exported process plotting inputs as CSV")


if __name__ == "__main__":
    main()
