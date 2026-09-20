#!/usr/bin/env python3
"""Run the packaged groundwater grid without VIC exchange and compare heads."""
import argparse
from pathlib import Path
import subprocess

import flopy
import numpy as np
import pandas as pd

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--acceptance-dir", type=Path, required=True)
p.add_argument("--output-dir", type=Path, required=True)
p.add_argument("--install-dir", type=Path, default=Path("/opt/vicmf6"))
a = p.parse_args()
a.output_dir.mkdir(parents=True, exist_ok=False)
work = a.output_dir / "mf6"
subprocess.run(["python", str(a.install_dir / "src/vic-mf6/examples/stehekin/create_mf6.py"),
                "--workspace", str(work)], check=True)
with (a.output_dir / "mf6.log").open("w") as stream:
    subprocess.run([str(a.install_dir / "bin/mf6")], cwd=work, stdout=stream, stderr=subprocess.STDOUT, check=True)
head_file = flopy.utils.HeadFile(work / "stehekin.hds", precision="double")
times = head_file.get_times()
control = head_file.get_alldata().reshape(len(times), -1)
tables = a.acceptance_dir / "postprocessing/tables"
coupled = pd.read_csv(tables / "mf6_heads.csv")
cells = pd.read_csv(tables / "mf6_cells.csv").sort_values("node")
volumes = pd.read_csv(tables / "vic_exchange_daily.csv")
sim = flopy.mf6.MFSimulation.load(sim_ws=work, verbosity_level=0)
model = sim.get_model()
coefficients = np.asarray(model.sto.ss.array).ravel() * (cells.top_m - cells.bottom_m).to_numpy() * cells.area_m2.to_numpy()
stats, balance = [], []
cumulative = 0.
for i, day in enumerate(times):
    values = coupled.loc[coupled.time_days == day].sort_values("node").head_m.to_numpy()
    assert len(values) == len(control[i])
    difference = values - control[i]
    stats.append(dict(time_days=day, coupled_min_m=values.min(), coupled_mean_m=values.mean(), coupled_max_m=values.max(),
        standalone_min_m=control[i].min(), standalone_mean_m=control[i].mean(), standalone_max_m=control[i].max(),
        difference_max_abs_m=abs(difference).max(), difference_rmse_m=np.sqrt(np.mean(difference**2))))
    cumulative += volumes.iloc[i].net_m3
    storage = float(coefficients @ difference)
    residual = storage - cumulative
    assert abs(residual) < 1e-5, (day, residual)
    balance.append(dict(step=i+1, cumulative_interface_m3=cumulative, storage_difference_m3=storage, residual_m3=residual))
pd.DataFrame(stats).to_csv(a.output_dir / "data-mf6-head-statistics.csv", index=False)
pd.DataFrame(balance).to_csv(a.output_dir / "data-storage-vs-interface.csv", index=False)
pd.DataFrame(dict(node=cells.node, coupled_head_m=values, standalone_head_m=control[-1], difference_m=difference)).to_csv(
    a.output_dir / "data-mf6-final-head-by-node.csv", index=False)
