#!/usr/bin/env python3
"""Test the actual VIC C exchange kernel against an independent nonlinear ODE.

Compile the unchanged installed calc_groundwater_exchange.c with its real VIC
headers. Couple that function to a live MF6 cell; independently integrate the
Campbell soil-store and groundwater equations with DOP853. This evaluates the
nonlinear interface kernel under its own closure, not complete VIC column physics
or the physical adequacy of a fixed vadose resistance.
"""
import argparse
import csv
import ctypes
from datetime import datetime
import hashlib
import logging
from pathlib import Path
import shutil
import subprocess
import sys


def write_csv(path, rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--install-dir',type=Path,default=Path('/opt/vicmf6'))
    p.add_argument('--reference-dir',type=Path,required=True)
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    import flopy
    import netCDF4 as nc
    import numpy as np
    import pandas as pd
    from mpi4py import MPI
    sys.path.insert(0,str(a.install_dir/'src/vic-mf6/src'))
    from vicmf6.config import Mf6Config
    from vicmf6.mf6 import Mf6Runtime
    source=a.install_dir/'src/vic/vic/vic_run/src/calc_groundwater_exchange.c'
    include=source.parents[1]/'include'
    wrapper=out/'kernel-wrapper.c'
    wrapper.write_text('''#include <vic_run.h>
double evaluate(double storage, double ice, double maximum, double residual,
                double depth, double bubble, double exponent, double head,
                double length, double conductivity, int steps) {
    soil_con_struct soil = {0};
    soil.depth[2]=depth; soil.max_moist[2]=maximum;
    soil.resid_moist[2]=residual; soil.bubble[2]=bubble; soil.expt[2]=exponent;
    return calc_groundwater_exchange_head(&soil,2,storage,ice,0.0,head,
                                          length,conductivity,1.0,steps);
}
''')
    subprocess.run(['gcc','-O2','-shared','-fPIC','-I'+str(include),str(source),str(wrapper),'-lm','-o',str(out/'kernel.so')],check=True)
    lib=ctypes.CDLL(str(out/'kernel.so'));kernel=lib.evaluate
    kernel.argtypes=[ctypes.c_double]*10+[ctypes.c_int];kernel.restype=ctypes.c_double
    params=a.install_dir/'examples/stehekin/input/stehekin_parameters_20160327.nc'
    domain=a.install_dir/'examples/stehekin/input/domain_stehekin_20151028.nc'
    with nc.Dataset(domain) as d: cells=np.argwhere(d['mask'][:]==1)
    with nc.Dataset(params) as d:
        # The first active cell is selected before seeing outcomes.
        r,c=map(int,cells[0]);depth=float(d['depth'][2,r,c]);bubble=float(d['bubble'][2,r,c]);expt=float(d['expt'][2,r,c])
        capacity=1000*depth*(1-float(d['bulk_density'][2,r,c])/float(d['soil_density'][2,r,c]))
        conductivity=float(d['Ksat'][2,r,c])*.001
    residual=0.;length=10.;cg=150.;b=(expt-3)/2
    def pressure(storage): return -.01*bubble*(np.clip(storage/capacity,1e-8,1.))**(-b)
    def c_step(storage,head,steps,ice=0.,res=0.,k=conductivity):
        return kernel(storage,ice,capacity,res,depth,bubble,expt,head,length,k,steps)
    metadata=[dict(parameter=k,value=v) for k,v in dict(row=r,col=c,depth_m=depth,capacity_mm=capacity,bubble_cm=bubble,expt=expt,conductivity_mm_day=conductivity,length_m=length,groundwater_storage_mm_per_m=cg).items()]
    write_csv(out/'parameters.csv',metadata)
    sources=[source,params,a.install_dir/'bin/vic_image.exe',a.install_dir/'lib/libmf6.so',Path(__file__)]
    write_csv(out/'source-hashes.csv',[dict(file=str(s),sha256=hashlib.sha256(s.read_bytes()).hexdigest()) for s in sources])
    shutil.copy2(__file__,out/Path(__file__).name);shutil.copy2(source,out/source.name)

    # Point checks independently evaluate sign, equilibrium, availability and ice.
    point=[]
    for fraction in [.2,.5,.8]:
        storage=fraction*capacity
        for offset in [-2.,0.,2.]:
            head=pressure(storage)+offset
            expected=-conductivity*offset/length/24
            actual=c_step(storage,head,24)
            point.append(dict(case=f'saturation-{fraction}-offset-{offset}',expected_mm=expected,actual_mm=actual,error_mm=abs(actual-expected)))
    for name,storage,ice,res in [('liquid-limit',2.,0.,0.),('residual-limit',20.,0.,19.),('ice-limit',2.,20.,19.)]:
        head=pressure(storage)-100.
        actual=c_step(storage,head,1,ice=ice,res=res,k=1e8)
        expected=min(storage,max(0.,storage+ice-res))
        point.append(dict(case=name,expected_mm=expected,actual_mm=actual,error_mm=abs(actual-expected)))
    write_csv(out/'kernel-checks.csv',point)
    assert max(x['error_mm'] for x in point)<1e-10

    rows=[];traces=[]
    duration=10.
    # Refinement changes the exchange-kernel and MF6 step together; it is not
    # a claim that every VIC production configuration supports these settings.
    for direction in ['upward','downward']:
        s0=.6*capacity;h0=float(pressure(s0))+ (2. if direction=='upward' else -2.)
        reference=pd.read_csv(a.reference_dir/'reference.csv')
        reference=reference[reference.direction==direction].reset_index(drop=True)
        assert abs(reference.storage_mm.iloc[0]-s0)<1e-10 and abs(reference.head_m.iloc[0]-h0)<1e-10
        reference_delta=float(reference.tolerance_difference.max())
        previous=None
        for steps in [1,2,4,8,24]:
            dt=1./steps;count=int(duration*steps);name=f'{direction}-{steps}'
            workspace=out/name
            sim=flopy.mf6.MFSimulation(sim_name=name,sim_ws=workspace)
            flopy.mf6.ModflowTdis(sim,time_units='DAYS',nper=count,perioddata=[(dt,1,1.)]*count)
            flopy.mf6.ModflowIms(sim,outer_dvclose=1e-12,inner_dvclose=1e-12,rcloserecord=1e-12,outer_maximum=100,inner_maximum=100)
            gwf=flopy.mf6.ModflowGwf(sim,modelname='FLOW',save_flows=True)
            flopy.mf6.ModflowGwfdis(gwf,nlay=1,nrow=1,ncol=1,delr=1.,delc=1.,top=-200.,botm=-300.)
            flopy.mf6.ModflowGwfic(gwf,strt=h0)
            flopy.mf6.ModflowGwfnpf(gwf,icelltype=0,k=1.)
            flopy.mf6.ModflowGwfsto(gwf,iconvert=0,ss=cg/1000/100,sy=0.,transient={0:True})
            flopy.mf6.ModflowGwfapi(gwf,pname='EXCHANGE',maxbound=1,save_flows=True)
            sim.write_simulation(silent=True)
            # FloPy's default decimal output rounds 1/24 day. The adapter
            # intentionally rejects that mismatch; preserve the exact schedule.
            (workspace/sim.tdis.filename).write_text(
                'BEGIN OPTIONS\n TIME_UNITS DAYS\nEND OPTIONS\n'
                f'BEGIN DIMENSIONS\n NPER {count}\nEND DIMENSIONS\n'
                'BEGIN PERIODDATA\n'+f' {dt:.17g} 1 1.0\n'*count+'END PERIODDATA\n')
            (workspace/gwf.ic.filename).write_text(
                f'BEGIN GRIDDATA\n STRT\n CONSTANT {h0:.17g}\nEND GRIDDATA\n')
            config=Mf6Config(workspace=workspace,library=a.install_dir/'lib/libmf6.so',simulation_namefile='mfsim.nam',start_time=datetime(2000,1,1),api_packages={'FLOW':'EXCHANGE'},solution_ids={'FLOW':1},total_time_days=duration,time_step_boundaries_days=tuple(np.arange(1,count+1)*dt),max_solve_iterations=100)
            runtime=Mf6Runtime(config,model_name='FLOW',coupled_nodes=[1],logger=logging.getLogger('nonlinear-reference'))
            runtime.initialize(MPI.COMM_SELF.py2f());s=s0;h=h0;es=eh=mass=api=0.
            try:
                for i in range(count):
                    q=c_step(s,h,steps);s-=q
                    result=runtime.advance_to((i+1)*dt,np.array([q*.001]),api_tolerance_m3_per_day=1e-10)
                    h=float(result.head_m[0]);ref=reference.iloc[round((i+1)*dt*24)]
                    assert abs(ref.time_days-(i+1)*dt)<1e-12
                    exact_s,exact_h=ref.storage_mm,ref.head_m
                    es=max(es,abs(s-exact_s));eh=max(eh,abs(h-exact_h));mass=max(mass,abs(s-s0+cg*(h-h0)))
                    api=max(api,result.maximum_api_error_m3_per_day)
                    assert 0<s<capacity
                    traces.append(dict(case=name,direction=direction,time_days=(i+1)*dt,storage_mm=s,reference_storage_mm=exact_s,head_m=h,reference_head_m=exact_h,exchange_mm=q))
            finally: runtime.finalize()
            passed=mass<1e-10 and api<1e-10 and (previous is None or es<previous)
            rows.append(dict(case=name,direction=direction,interval_days=dt,max_storage_error_mm=es,max_head_error_m=eh,max_balance_residual_mm=mass,max_api_error_m3_day=api,reference_tolerance_difference=reference_delta,error_ratio=previous/es if previous else '',passed=passed))
            previous=es
    write_csv(out/'nonlinear-summary.csv',rows);write_csv(out/'nonlinear-traces.csv',traces)
    for row in rows: print(row,flush=True)
    assert all(row['passed'] for row in rows)


if __name__=='__main__':main()
