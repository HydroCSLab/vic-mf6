#!/usr/bin/env python3
"""Separate volume preservation from unresolved within-cell flux variation.

Exercise the production mapper on a prescribed heterogeneous-head example.
The comparison is with the same uncapped linear law evaluated by overlap;
it is not a Richards reference or a replacement coupled simulation.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import netCDF4 as nc

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--coupler-dir',type=Path,required=True)
p.add_argument('--campaign-dir',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(a.coupler_dir/'src'))
from vicmf6.exchange import ExchangeTable

rows=[]
for name,heads in [('uniform',[-3.,-3.]),('mixed-direction',[-3.,-1.]),('unequal-downward',[-4.,-3.])]:
    for refined in [False,True]:
        table=ExchangeTable.from_records([dict(vic_id=str(j if refined else 0),vic_row=0,vic_col=j if refined else 0,mf6_model='FLOW',mf6_node=j+1,overlap_area_m2=500.,vic_area_m2=500. if refined else 1000.) for j in range(2)])
        contribution=table.head_contribution_for_model('FLOW',np.array(heads))
        mean=table.finish_head_mapping(contribution.head_area_sum_m3,contribution.area_sum_m2)
        depth=.1*(-2.-mean)  # mm/day; same conductance at both intersections
        actual=table.map_vic_depth_to_model('FLOW',depth).volume_by_node_m3
        expected=.1*(-2.-np.array(heads))*.001*500.
        assert abs(actual.sum()-expected.sum())<1e-12
        if refined:assert np.max(abs(actual-expected))<1e-12
        for node in range(2):rows.append(dict(case=name,vic_cells=2 if refined else 1,groundwater_node=node+1,groundwater_head_m=heads[node],mapped_volume_m3_day=actual[node],overlap_reference_m3_day=expected[node],error_m3_day=actual[node]-expected[node]))
pd.DataFrame(rows).to_csv(a.output_dir/'data-spatial-mapping-diagnostic.csv',index=False)

# Diagnose the head-heterogeneity term in the actual process geometry. Adding
# c_i*(mean_head_i-head_j) to each mapped depth preserves its cell integral.
# This is an uncapped-law diagnostic, not a rerun with a different soil closure.
root=a.campaign_dir;mapping=np.load(root/'cases/nominal--baseline/mapping.npz')
weights,areas=mapping['weights'],mapping['transfer_areas'];rr,cc=mapping['rows'],mapping['cols']
with nc.Dataset(root/'sample/parameters/Stehekin_test_params_20160327.nc') as d:conductance=np.asarray(d['Ksat'][2])[rr,cc]*.001/10
result=[]
for case in ['baseline','high-snow','pumped']:
    fields=np.load(root/'cases'/f'nominal--{case}'/'fields.npz')
    previous=np.vstack([np.full(48,-2.),fields['heads'][:-1,:48]])
    for i,head in enumerate(previous):
        mean=weights@head
        correction=conductance[:,None]*(mean[:,None]-head[None,:])*areas*.001
        mapped=fields['OUT_GW_EXCHANGE'][i,:,None]*areas*.001
        local=mapped+correction
        assert np.max(abs(correction.sum(axis=1)))<1e-6
        result.append(dict(case=case,time_days=i+1,maximum_subcell_head_deviation_m=np.max(np.abs(mean[:,None]-head[None,:])[areas>0]),gross_redistribution_correction_m3=np.abs(correction).sum()/2,gross_mapped_volume_m3=np.abs(mapped).sum(),sign_different_overlaps=int(np.sum((mapped*local<0)&(areas>0)))))
pd.DataFrame(result).to_csv(a.output_dir/'data-spatial-process-diagnostic.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
print(pd.DataFrame(result).groupby('case').agg({'maximum_subcell_head_deviation_m':'max','gross_redistribution_correction_m3':'sum','gross_mapped_volume_m3':'sum','sign_different_overlaps':'max'}))
