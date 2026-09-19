#!/usr/bin/env python3
"""Run a bounded 15-case sensitivity campaign in the installed VIC–MF6 image.

Raw products go to --output-dir, outside the paper source tree. No component
source is modified. Perturbations bracket effective conductance and initial
soil water; they are local sensitivity tests, not calibration or spin-up.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--install-dir', type=Path, default=Path('/opt/vicmf6'))
    p.add_argument('--workers', type=int, default=3)
    a = p.parse_args()
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out/'cases').exists():
        p.error('output already contains cases; use a fresh directory')
    import netCDF4 as nc
    import numpy as np
    example = a.install_dir/'examples/stehekin'
    runner = example/'experiments/run-feedback-experiment.py'
    (out/'cases').mkdir(); (out/'logs').mkdir()
    shutil.copy2(__file__, out/Path(__file__).name)
    shutil.copy2(runner, out/'run-feedback-experiment.py')
    with (out/'source-hashes.csv').open('w') as f:
        w=csv.writer(f); w.writerow(['file','sha256'])
        for file in [Path(__file__),runner,a.install_dir/'bin/vic_image.exe',a.install_dir/'lib/libmf6.so']:
            w.writerow([str(file),hashlib.sha256(file.read_bytes()).hexdigest()])
    groups=[('nominal',10.,1.),('conductance-half',20.,1.),('conductance-double',5.,1.),('soil-drier',10.,.9),('soil-wetter',10.,1.1)]
    jobs=[]
    for group,length,factor in groups:
        sample=out/'samples'/group
        (sample/'parameters').mkdir(parents=True); (sample/'forcings').mkdir()
        names=[('domain_stehekin_20151028.nc','parameters/domain.stehekin.20151028.nc'),('stehekin_parameters_20160327.nc','parameters/Stehekin_test_params_20160327.nc'),('stehekin_forcings_10_days_1949.nc','forcings/Stehekin_image_test.forcings_10days.1949.nc')]
        for source,target in names:shutil.copy2(example/'input'/source,sample/target)
        with nc.Dataset(sample/'parameters/Stehekin_test_params_20160327.nc','r+') as ds:
            initial=ds['init_moist'][:]; changed=initial*factor
            capacity=ds['depth'][:]*1000*(1-ds['bulk_density'][:]/ds['soil_density'][:])
            assert np.ma.all(changed>=0) and np.ma.all(changed<=capacity), 'initial moisture outside storage capacity'
            ds['init_moist'][:]=changed
        lines=(example/'stehekin.global.txt').read_text().splitlines()
        for i,line in enumerate(lines):
            if line.startswith('DOMAIN '):lines[i]='DOMAIN '+str(sample/'parameters/domain.stehekin.20151028.nc')
            if line.startswith('PARAMETERS '):lines[i]='PARAMETERS '+str(sample/'parameters/Stehekin_test_params_20160327.nc')
        (sample/'parameters/Stehekin_image_test.global.txt').write_text('\n'.join(lines)+'\n')
        for case in ['baseline','high-snow','pumped']:
            name=group+'--'+case
            cmd=[sys.executable,str(runner),'--run-dir',str(out/'cases'/name),'--sample-dir',str(sample),'--vic-exe',str(a.install_dir/'bin/vic_image.exe'),'--mf6-library',str(a.install_dir/'lib/libmf6.so'),'--coupler-dir',str(a.install_dir/'src/vic-mf6'),'--days','60','--pumping-mm-day','1','--length',str(length),'--case','pumped' if case=='pumped' else 'baseline','--snowfall-mm-day','24' if case=='high-snow' else '8']
            jobs.append((name,group,case,length,factor,cmd))
    with (out/'design.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['name','group','case','resistance_length_m','initial_soil_factor'])
        w.writerows(j[:5] for j in jobs)
    def run(job):
        name,*_=job;started=time.monotonic()
        with (out/'logs'/f'{name}.log').open('w') as log:
            result=subprocess.run(job[-1],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        return name,result.returncode,round(time.monotonic()-started,2)
    failed=[]
    with (out/'execution.csv').open('w') as f, ThreadPoolExecutor(max_workers=a.workers) as pool:
        w=csv.writer(f);w.writerow(['name','exit_code','elapsed_seconds']);f.flush()
        for future in as_completed([pool.submit(run,j) for j in jobs]):
            row=future.result();w.writerow(row);f.flush();print(row,flush=True)
            if row[1]:failed.append(row[0])
    if failed:raise SystemExit('Failed cases: '+', '.join(failed))

if __name__=='__main__':main()
