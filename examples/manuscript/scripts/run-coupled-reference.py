#!/usr/bin/env python3
"""Live MF6 prescribed-flow and coupled linear-storage reference tests.

The upper store is an exactly integrated linear reservoir, NOT VIC. The closed
form solves both stores simultaneously; the numerical path freezes groundwater
head during the upper-store step and advances the production MF6 API adapter.
This isolates sign, storage, and sequential coupling accuracy, not soil physics.
"""
import argparse
import csv
from datetime import datetime
import logging
import hashlib
import shutil
from pathlib import Path
import sys
import math


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--install-dir',type=Path,default=Path('/opt/vicmf6'))
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__, a.output_dir/Path(__file__).name)
    with (a.output_dir/'source-hashes.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['file','sha256'])
        for file in [Path(__file__),a.install_dir/'lib/libmf6.so',a.install_dir/'src/vic-mf6/src/vicmf6/mf6.py']:
            w.writerow([str(file),hashlib.sha256(file.read_bytes()).hexdigest()])
    import flopy
    import numpy as np
    from mpi4py import MPI
    sys.path.insert(0,str(a.install_dir/'src/vic-mf6/src'))
    from vicmf6.config import Mf6Config
    from vicmf6.mf6 import Mf6Runtime
    rows=[];traces=[]
    cs,cg,conductance=.3,.2,.02
    for kind in ['prescribed','coupled']:
        for sign in [-1,1]:
            previous=None
            for dt in ([1.] if kind=='prescribed' else [1.,.5,.25,.125]):
                duration=1. if kind=='prescribed' else 10.
                count=round(duration/dt);name=f'{kind}-{sign:+d}-{dt:g}'
                workspace=a.output_dir/name
                sim=flopy.mf6.MFSimulation(sim_name=name,sim_ws=workspace)
                flopy.mf6.ModflowTdis(sim,time_units='DAYS',nper=count,perioddata=[(dt,1,1.)]*count)
                flopy.mf6.ModflowIms(sim,outer_dvclose=1e-12,inner_dvclose=1e-12,rcloserecord=1e-12,outer_maximum=100,inner_maximum=100)
                gwf=flopy.mf6.ModflowGwf(sim,modelname='FLOW',save_flows=True)
                flopy.mf6.ModflowGwfdis(gwf,nlay=1,nrow=1,ncol=1,delr=1.,delc=1.,top=-200.,botm=-400.)
                flopy.mf6.ModflowGwfic(gwf,strt=-100.)
                flopy.mf6.ModflowGwfnpf(gwf,icelltype=0,k=1.)
                flopy.mf6.ModflowGwfsto(gwf,iconvert=0,ss=.001,sy=0.,transient={0:True},save_flows=True)
                flopy.mf6.ModflowGwfapi(gwf,pname='EXCHANGE',maxbound=1,save_flows=True)
                flopy.mf6.ModflowGwfoc(gwf,head_filerecord='flow.hds',budget_filerecord='flow.cbc',saverecord=[('HEAD','ALL'),('BUDGET','ALL')])
                sim.write_simulation(silent=True)
                config=Mf6Config(workspace=workspace,library=a.install_dir/'lib/libmf6.so',simulation_namefile='mfsim.nam',start_time=datetime(2000,1,1),api_packages={'FLOW':'EXCHANGE'},solution_ids={'FLOW':1},total_time_days=duration,time_step_boundaries_days=tuple(np.arange(1,count+1)*dt),max_solve_iterations=100)
                runtime=Mf6Runtime(config,model_name='FLOW',coupled_nodes=[1],logger=logging.getLogger('reference'))
                hg0=-100.;hs0=hg0+sign*4.;hs=hs0;hg=hg0;api=mass=error=0.
                runtime.initialize(MPI.COMM_SELF.py2f())
                try:
                    for step in range(count):
                        t=(step+1)*dt
                        if kind=='prescribed':
                            volume=sign*.2*dt;expected_g=hg0+sign*.2*t/cg;expected_s=hs0
                        else:
                            new_s=hg+(hs-hg)*math.exp(-conductance*dt/cs)
                            volume=cs*(hs-new_s);hs=new_s
                            equilibrium=(cs*hs0+cg*hg0)/(cs+cg)
                            difference=(hs0-hg0)*math.exp(-conductance*(1/cs+1/cg)*t)
                            expected_s=equilibrium+cg/(cs+cg)*difference
                            expected_g=equilibrium-cs/(cs+cg)*difference
                        result=runtime.advance_to(t,np.array([volume]),api_tolerance_m3_per_day=1e-10)
                        hg=float(result.head_m[0]);api=max(api,result.maximum_api_error_m3_per_day)
                        residual=abs(cs*(hs-hs0)+cg*(hg-hg0)) if kind=='coupled' else abs(cg*(hg-hg0)-sign*.2*t)
                        mass=max(mass,residual);error=max(error,abs(hg-expected_g),abs(hs-expected_s))
                        traces.append([name,t,hs,expected_s,hg,expected_g,volume,residual])
                finally:runtime.finalize()
                ratio=previous/error if previous is not None else ''
                passed=mass<1e-10 and api<1e-10 and (error<1e-10 if kind=='prescribed' else (previous is None or error<previous))
                rows.append([name,kind,sign,dt,error,mass,api,ratio,passed]);previous=error
    with (a.output_dir/'reference-summary.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['name','kind','sign','interval_days','max_head_error_m','max_mass_residual_m3','max_api_error_m3_day','error_reduction_ratio','passed']);w.writerows(rows)
    with (a.output_dir/'reference-traces.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['name','time_days','upper_head_m','exact_upper_head_m','groundwater_head_m','exact_groundwater_head_m','exchange_m3','mass_residual_m3']);w.writerows(traces)
    for row in rows:print(row,flush=True)
    if not all(row[-1] for row in rows):raise SystemExit('reference checks failed')

if __name__=='__main__':main()
