#!/usr/bin/env python3
"""Run initialization sensitivity and paired controller-path reproduction.

Use the pinned installed runtime. This driver creates fresh directories and
records all failures; it never overwrites previous campaigns. Raw output is
external to the manuscript. The preparation test resets the experiment calendar
to the same January date while retaining both models' final physical state.
It therefore isolates alternative starting states, not seasonal chronology.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--install-dir',type=Path,default=Path('/opt/vicmf6'))
    p.add_argument('--workers',type=int,default=3)
    a=p.parse_args(); out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import yaml
    example=a.install_dir/'examples/stehekin'
    sample=out/'sample'; (sample/'parameters').mkdir(parents=True); (sample/'forcings').mkdir()
    for source,target in [('domain_stehekin_20151028.nc','parameters/domain.stehekin.20151028.nc'),('stehekin_parameters_20160327.nc','parameters/Stehekin_test_params_20160327.nc'),('stehekin_forcings_10_days_1949.nc','forcings/Stehekin_image_test.forcings_10days.1949.nc')]:
        shutil.copy2(example/'input'/source,sample/target)
    lines=(example/'stehekin.global.txt').read_text().splitlines()
    for i,line in enumerate(lines):
        if line.startswith('DOMAIN '): lines[i]='DOMAIN '+str(sample/'parameters/domain.stehekin.20151028.nc')
        if line.startswith('PARAMETERS '): lines[i]='PARAMETERS '+str(sample/'parameters/Stehekin_test_params_20160327.nc')
    (sample/'parameters/Stehekin_image_test.global.txt').write_text('\n'.join(lines)+'\n')
    (out/'logs').mkdir(); (out/'cases').mkdir()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMPI_ALLOW_RUN_AS_ROOT='1',OMPI_ALLOW_RUN_AS_ROOT_CONFIRM='1')
    wrapper=Path(__file__).with_name('run-initialization-case.py')
    shutil.copy2(__file__,out/Path(__file__).name); shutil.copy2(wrapper,out/wrapper.name)
    common=['--sample-dir',str(sample),'--vic-exe',str(a.install_dir/'bin/vic_image.exe'),'--mf6-library',str(a.install_dir/'lib/libmf6.so'),'--coupler-dir',str(a.install_dir/'src/vic-mf6'),'--days','60','--pumping-mm-day','1']
    rows=[]
    def run(name,cmd):
        start=time.monotonic()
        with (out/'logs'/f'{name}.log').open('w') as f:
            r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        result=[name,r.returncode,time.monotonic()-start]; print(result,flush=True)
        return result
    def batch(groups):
        jobs=[]
        for group,extra in groups:
            for case in ['baseline','high-snow','pumped']:
                name=group+'--'+case
                cmd=[sys.executable,str(wrapper),'--install-dir',str(a.install_dir),*extra,*common,'--run-dir',str(out/'cases'/name),'--case','pumped' if case=='pumped' else 'baseline','--snowfall-mm-day','24' if case=='high-snow' else '8']
                jobs.append((name,cmd))
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for future in as_completed([pool.submit(run,*job) for job in jobs]): rows.append(future.result())
        with (out/'execution.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['case','exit_code','seconds']);w.writerows(rows)
        if any(row[1] for row in rows): raise RuntimeError('A process case failed; inspect logs and execution.csv')
    batch([('nominal',[]),('head-shallow',['--initial-head','-1']),('head-deep',['--initial-head','-5'])])
    batch([('prepared',['--prepared-from',str(out/'cases/nominal--baseline')])])

    # Recreate the same 60-day decks through the public MPI controller.
    runner=example/'experiments/run-feedback-experiment.py'
    spec=importlib.util.spec_from_file_location('process',runner)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for case in ['baseline','pumped']:
        d=out/f'controller--{case}';d.mkdir()
        args=argparse.Namespace(sample_dir=sample,days=60,interval=1.,case=case,aquitard_k=.02,pumping_mm_day=1.)
        geometry=module.build_geometry(sample/'parameters',d)
        prefix=module.build_forcing(sample,d,60,8.)
        workspace=module.build_groundwater(args,geometry,d)
        template=(out/'cases'/f'nominal--{case}'/'vic/window-0000/global.txt').read_text().splitlines()
        excluded={'NRECS','STATENAME','STATEYEAR','STATEMONTH','STATEDAY','STATESEC','STATE_FORMAT','INIT_STATE'}
        template=[line for line in template if not line.split() or line.split()[0] not in excluded]
        template=[f'FORCING1 {prefix}' if line.startswith('FORCING1 ') else line for line in template]
        template += ['NRECS 1440']
        global_file=d/'global.txt';global_file.write_text('\n'.join(template)+'\n')
        exchange=d/'exchange.csv'
        with exchange.open('w') as f:
            w=csv.writer(f); w.writerow(['vic_id','vic_row','vic_col','mf6_model','mf6_node','overlap_area_m2','vic_area_m2','vic_lat','vic_lon','vic_interface_elevation_m'])
            for i,j in zip(*np.where(geometry['transfer_areas']>0)):
                w.writerow([i,int(geometry['rows'][i]),int(geometry['cols'][i]),'FLOW',int(j+1),geometry['transfer_areas'][i,j],geometry['areas'][i],geometry['lat'][i],geometry['lon'][i],0.])
        config={'run':{'directory':str(d/'run')},'mf6':{'namefile':str(workspace/'mfsim.nam'),'library':str(a.install_dir/'lib/libmf6.so'),'max_solve_iterations':200},'vic':{'global_file':str(global_file),'executable':str(a.install_dir/'bin/vic_image.exe'),'mpi_processes':1,'omp_threads':1,'preload_library':str(a.install_dir/'lib/libvic_parent_disconnect.so')},'coupling':{'exchange_table':str(exchange),'interval_days':1.,'scheme':'explicit','exchange_length_m':10.,'exchange_conductivity_scale':.001,'head_transform':'pressure_head_from_interface_elevation','conservation_absolute_tolerance_m3':1e-5,'conservation_relative_tolerance':1e-11,'api_absolute_tolerance_m3_per_day':1e-5},'diagnostics':{'verbosity':'info','write_rank_logs':True}}
        cfg=d/'config.yml';cfg.write_text(yaml.safe_dump(config))
        rows.append(run('controller--'+case,['mpirun','--oversubscribe','-np','2',str(a.install_dir/'src/vic-mf6/vicmf6'),'run','-c',str(cfg)]))
        with (out/'execution.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['case','exit_code','seconds']);w.writerows(rows)
        if rows[-1][1]: raise RuntimeError('Controller case failed; inspect its log')


if __name__=='__main__':main()
