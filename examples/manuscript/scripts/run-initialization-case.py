#!/usr/bin/env python3
"""Run the installed process example with a specified groundwater/land start.

Only generated inputs and the initial restart choice change. Component sources
and binaries remain untouched. Outputs, including native model files, stay in
the caller's external run directory. A prepared state is a controlled alternative
initial condition, not a claim of equilibrium or a continuous seasonal hindcast.
"""
import argparse
import csv
import hashlib
import importlib.util
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial-head', type=float, default=-2.)
    p.add_argument('--prepared-from', type=Path)
    p.add_argument('--install-dir', type=Path, default=Path('/opt/vicmf6'))
    a, remaining = p.parse_known_args()
    import numpy as np
    import flopy
    source = a.install_dir/'examples/stehekin/experiments/run-feedback-experiment.py'
    spec = importlib.util.spec_from_file_location('original_process', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    original_groundwater, original_window = module.build_groundwater, module.vic_window
    restart = None
    if a.prepared_from:
        restart = sorted((a.prepared_from/'vic').glob('window-*/state*.nc'))[-1]
        initial = np.load(a.prepared_from/'fields.npz')['heads'][-1].reshape(3, 6, 8)
    else:
        initial = a.initial_head

    def groundwater(args, geometry, run_dir):
        path = original_groundwater(args, geometry, run_dir)
        sim = flopy.mf6.MFSimulation.load(sim_ws=path, verbosity_level=0)
        sim.get_model().ic.strt.set_data(initial)
        sim.write_simulation(silent=True)
        with (run_dir/'initialization.csv').open('w') as f:
            w = csv.writer(f); w.writerow(['parameter','value'])
            w.writerows([['uniform_initial_head_m', a.initial_head if restart is None else 'not uniform'],
                         ['prepared_from', str(a.prepared_from or '')],
                         ['vic_restart', str(restart or '')],
                         ['original_runner_sha256', hashlib.sha256(source.read_bytes()).hexdigest()],
                         ['wrapper_sha256', hashlib.sha256(Path(__file__).read_bytes()).hexdigest()]])
        return path

    def window(args, geometry, prefix, run_dir, step, state, heads):
        if step == 0 and restart is not None:
            state = restart
        return original_window(args, geometry, prefix, run_dir, step, state, heads)

    module.build_groundwater, module.vic_window = groundwater, window
    sys.argv = [str(source), *remaining]
    module.main()


if __name__ == '__main__':
    main()
