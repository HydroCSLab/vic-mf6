#!/usr/bin/env python3
"""Independently integrate the stated nonlinear closure with SciPy DOP853.

No model C function is used to calculate this reference solution.
"""
import argparse
import hashlib
from pathlib import Path
import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--parameter-file',type=Path,required=True)
p.add_argument('--domain-file',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
with nc.Dataset(a.domain_file) as d:r,c=map(int,np.argwhere(d['mask'][:]==1)[0])
with nc.Dataset(a.parameter_file) as d:
    depth=float(d['depth'][2,r,c]);bubble=float(d['bubble'][2,r,c]);expt=float(d['expt'][2,r,c])
    capacity=1000*depth*(1-float(d['bulk_density'][2,r,c])/float(d['soil_density'][2,r,c]))
    conductivity=float(d['Ksat'][2,r,c])*.001
b=(expt-3)/2;cg=150.;length=10.
def pressure(storage):return -.01*bubble*np.clip(storage/capacity,1e-8,1.)**(-b)
rows=[]
for direction,offset in [('upward',2.),('downward',-2.)]:
    initial=[.6*capacity,pressure(.6*capacity)+offset]
    def rhs(t,y):
        q=conductivity/length*(pressure(y[0])-y[1])
        return [-q,q/cg]
    times=np.arange(241)/24
    one=solve_ivp(rhs,[0,10],initial,method='DOP853',rtol=1e-11,atol=1e-12,t_eval=times)
    two=solve_ivp(rhs,[0,10],initial,method='DOP853',rtol=1e-13,atol=1e-14,t_eval=times)
    assert one.success and two.success
    for i,t in enumerate(times):rows.append(dict(direction=direction,time_days=t,storage_mm=two.y[0,i],head_m=two.y[1,i],tolerance_difference=max(abs(one.y[:,i]-two.y[:,i]))))
pd.DataFrame(rows).to_csv(a.output_dir/'reference.csv',index=False)
pd.DataFrame([dict(file=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in [a.parameter_file,a.domain_file,Path(__file__)]]).to_csv(a.output_dir/'source-hashes.csv',index=False)
print(pd.DataFrame(rows).groupby('direction').tolerance_difference.max())
