#!/usr/bin/env python3
"""Audit initialization cases, controller equivalence and layer-level response."""
import argparse
from pathlib import Path
import flopy
import netCDF4 as nc
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    root=a.campaign_dir;execution=pd.read_csv(root/'execution.csv')
    assert len(execution)==14 and (execution.exit_code==0).all()
    summary=[];checks=[];curves=[];equivalence=[];layer_rows=[]
    def check(name,value,tolerance):
        checks.append(dict(check=name,value=float(value),limit=tolerance,passed=bool(value<=tolerance)))
    def store(t): return t[['OUT_SOIL_MOIST_1','OUT_SOIL_MOIST_2','OUT_SOIL_MOIST_3','OUT_SWE','OUT_WDEW','OUT_SNOW_CANOPY']].sum(axis=1)
    for group in ['nominal','head-shallow','head-deep','prepared']:
        tables={};fields={}
        for case in ['baseline','high-snow','pumped']:
            d=root/'cases'/f'{group}--{case}';t=pd.read_csv(d/'timeseries.csv');b=pd.read_csv(d/'budgets.csv');f=np.load(d/'fields.npz')
            assert len(t)==len(b)==60
            for title,val,tol in [('groundwater cell balance',b.max_cell_residual_m3.abs().max(),1e-3),('VIC balance',t.max_vic_water_error_mm.abs().max(),1e-7),('API',t.api_error_m3_day.abs().max(),1e-5),('mapping',t.mapping_error_m3.abs().max(),1e-5),('saturation',max(0.,-20-f['heads'][:,:48].min()),0.),('confinement',max(0.,-20-f['heads'][:,48:96].min(),-25-f['heads'][:,96:].min()),0.)]:check(group+'/'+case+'/'+title,val,tol)
            tables[case]=t;fields[case]=f
            for _,row in t.iterrows():curves.append(dict(group=group,case=case,time_days=row.time_days,exchange_mm_day=row.OUT_GW_EXCHANGE,soil_storage_mm=sum(row[f'OUT_SOIL_MOIST_{i}'] for i in [1,2,3])))
        base,pump,snow=[tables[k] for k in ['baseline','pumped','high-snow']]
        loss=(pump.OUT_GW_EXCHANGE-base.OUT_GW_EXCHANGE).sum();ds=store(pump).iloc[-1]-store(base).iloc[-1]
        de=(pump.OUT_EVAP-base.OUT_EVAP).sum();dr=(pump.OUT_RUNOFF-base.OUT_RUNOFF).sum()
        check(group+'/paired balance',abs(ds+loss+de+dr+(pump.OUT_BASEFLOW-base.OUT_BASEFLOW).sum()),1e-7)
        for key in ['heads','OUT_SOIL_MOIST','OUT_GW_EXCHANGE']:check(group+'/before withdrawal/'+key,np.max(abs(fields['pumped'][key][:20]-fields['baseline'][key][:20])),1e-10)
        summary.append(dict(group=group,baseline_upward_net_mm=-base.OUT_GW_EXCHANGE.sum(),first10_upward_mm=-base.OUT_GW_EXCHANGE.iloc[:10].sum(),snow_downward_days=int((snow.OUT_GW_EXCHANGE>0).sum()),supply_loss_mm=loss,storage_change_mm=ds,ET_change_mm=de,runoff_change_mm=dr))
        layer_rows.append(dict(group=group,layer1_change_mm=pump.OUT_SOIL_MOIST_1.iloc[-1]-base.OUT_SOIL_MOIST_1.iloc[-1],layer2_change_mm=pump.OUT_SOIL_MOIST_2.iloc[-1]-base.OUT_SOIL_MOIST_2.iloc[-1],layer3_change_mm=pump.OUT_SOIL_MOIST_3.iloc[-1]-base.OUT_SOIL_MOIST_3.iloc[-1],root_storage_change_mm=pump.OUT_ROOTMOIST.iloc[-1]-base.OUT_ROOTMOIST.iloc[-1],transpiration_change_mm=(pump.OUT_TRANSP_VEG-base.OUT_TRANSP_VEG).sum(),bare_evaporation_change_mm=(pump.OUT_EVAP_BARE-base.OUT_EVAP_BARE).sum()))

    mapping=np.load(root/'cases/nominal--baseline/mapping.npz');rr,cc=mapping['rows'],mapping['cols']
    for case in ['baseline','pumped']:
        d=root/f'controller--{case}';reference=np.load(root/'cases'/f'nominal--{case}'/'fields.npz')
        heads=flopy.utils.HeadFile(d/'mf6/flow.hds',precision='double').get_alldata().reshape(60,-1)
        records=[('heads',float(np.max(abs(heads-reference['heads']))),'m')]
        files=sorted((d/'run/vic/outputs').glob('window-*/hydro*.nc'));assert len(files)==60
        keys=[k for k in reference.files if k.startswith('OUT_')];values={k:[] for k in keys}
        for file in files:
            with nc.Dataset(file) as ds:
                for k in keys: values[k].append(np.asarray(ds[k][0])[...,rr,cc])
        for k in keys: records.append((k,float(np.max(abs(np.array(values[k])-reference[k]))),'native output unit'))
        for name,error,unit in records:
            equivalence.append(dict(case=case,field=name,max_absolute_difference=error,unit=unit))
            check('controller/'+case+'/'+name,error,1e-8)
        diagnostic=pd.read_csv(d/'run/diagnostics/coupling_windows.csv')
        check('controller/'+case+'/API',diagnostic.maximum_api_rate_error_m3_per_day.max(),1e-5)
        check('controller/'+case+'/water balance',diagnostic.maximum_vic_water_error_mm.max(),1e-7)

    # Daily restart snapshots diagnose whether the perturbed bottom store
    # approaches the critical moisture used in VIC's transpiration calculation.
    with nc.Dataset(root/'sample/parameters/Stehekin_test_params_20160327.nc') as d:
        capacity=d['depth'][:]*1000*(1-d['bulk_density'][:]/d['soil_density'][:])
        critical=d['Wcr_FRACT'][2]*capacity[2]
        active=(d['Cv'][:][:,None,:,:]>0)&(d['AreaFract'][:][None,:,:,:]>0)
    margins=[]
    for case in ['baseline','pumped']:
        for day,file in enumerate(sorted((root/'cases'/f'nominal--{case}'/'vic').glob('window-*/state*.nc')),start=1):
            if day<20:continue
            with nc.Dataset(file) as d:
                moisture=d['STATE_SOIL_MOISTURE'][:,:,2,:,:]
                # Frozen-soil physics is disabled in these process runs.
                ice=d['STATE_SOIL_ICE'][:,:,2,:,:,:]
                assert np.ma.max(abs(ice))==0
                margin=np.ma.masked_where(~active,moisture-critical)
                margins.append(dict(case=case,time_days=day,minimum_bottom_water_above_critical_mm=float(margin.min())))
    for name,data in [('initialization-summary',summary),('initialization-curves',curves),('review-checks',checks),('controller-equivalence',equivalence),('layer-response',layer_rows),('soil-stress-margin',margins)]:pd.DataFrame(data).to_csv(a.output_dir/f'data-{name}.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False));print(pd.DataFrame(equivalence).groupby('case').max_absolute_difference.max())
    print(pd.DataFrame(margins).groupby('case').minimum_bottom_water_above_critical_mm.min())
    print(sum(c['passed'] for c in checks),'/',len(checks),'checks passed')
    assert all(c['passed'] for c in checks)


if __name__=='__main__':main()
