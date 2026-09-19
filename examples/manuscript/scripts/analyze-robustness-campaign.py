#!/usr/bin/env python3
"""Audit the local sensitivity runs and export compact paper tables as CSV."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    design=pd.read_csv(a.campaign_dir/'design.csv');execution=pd.read_csv(a.campaign_dir/'execution.csv')
    assert len(execution)==len(design)==15 and (execution.exit_code==0).all(), 'campaign incomplete or failed'
    rows=[];checks=[];curves=[]
    def check(name,value,limit):
        checks.append(dict(check=name,value=float(value),limit=limit,passed=bool(value<=limit)))
    def storage(t):return t[['OUT_SOIL_MOIST_1','OUT_SOIL_MOIST_2','OUT_SOIL_MOIST_3','OUT_SWE','OUT_WDEW','OUT_SNOW_CANOPY']].sum(axis=1)
    for group,part in design.groupby('group',sort=False):
        tables={};fields={}
        for case in ['baseline','high-snow','pumped']:
            path=a.campaign_dir/'cases'/f'{group}--{case}'
            t=pd.read_csv(path/'timeseries.csv');b=pd.read_csv(path/'budgets.csv');f=np.load(path/'fields.npz')
            assert len(t)==60 and len(b)==60
            check(f'{group}/{case}/groundwater budget',b.max_cell_residual_m3.abs().max(),1e-3)
            check(f'{group}/{case}/VIC balance',t.max_vic_water_error_mm.abs().max(),1e-7)
            check(f'{group}/{case}/API application',t.api_error_m3_day.abs().max(),1e-5)
            check(f'{group}/{case}/mapping',t.mapping_error_m3.abs().max(),1e-5)
            h=f['heads'].reshape(60,3,48)
            check(f'{group}/{case}/upper saturation violation',max(0.,float(-20-h[:,0].min())),0.)
            check(f'{group}/{case}/confined-layer violation',max(0.,float(-20-h[:,1].min()),float(-25-h[:,2].min())),0.)
            tables[case]=t;fields[case]=f
            for _,r in t.iterrows():curves.append(dict(group=group,case=case,time_days=r.time_days,exchange_mm_day=r.OUT_GW_EXCHANGE,soil_storage_mm=r.OUT_SOIL_MOIST_1+r.OUT_SOIL_MOIST_2+r.OUT_SOIL_MOIST_3))
        base=tables['baseline'];snow=tables['high-snow'];pump=tables['pumped']
        delta=pump.OUT_GW_EXCHANGE-base.OUT_GW_EXCHANGE
        loss=delta.sum();ds=storage(pump).iloc[-1]-storage(base).iloc[-1]
        de=(pump.OUT_EVAP-base.OUT_EVAP).sum();dr=(pump.OUT_RUNOFF-base.OUT_RUNOFF).sum();db=(pump.OUT_BASEFLOW-base.OUT_BASEFLOW).sum()
        check(f'{group}/paired water balance',abs(ds+de+dr+db+loss),1e-7)
        check(f'{group}/pre-withdrawal head agreement',np.max(np.abs(fields['pumped']['heads'][:20]-fields['baseline']['heads'][:20])),1e-10)
        check(f'{group}/pre-withdrawal soil agreement',np.max(np.abs(fields['pumped']['OUT_SOIL_MOIST'][:20]-fields['baseline']['OUT_SOIL_MOIST'][:20])),1e-10)
        positive=snow.loc[snow.OUT_GW_EXCHANGE>0,'time_days']
        draw=fields['baseline']['heads']-fields['pumped']['heads']
        rows.append(dict(group=group,resistance_length_m=part.resistance_length_m.iloc[0],initial_soil_factor=part.initial_soil_factor.iloc[0],baseline_upward_net_mm=-base.OUT_GW_EXCHANGE.sum(),baseline_first10_upward_mm=-base.OUT_GW_EXCHANGE.iloc[:10].sum(),snow_downward_days=len(positive),snow_first_downward_day=positive.min() if len(positive) else np.nan,snow_last_downward_day=positive.max() if len(positive) else np.nan,cumulative_positive_domain_mean_mm=snow.OUT_GW_EXCHANGE.clip(lower=0).sum(),snow_peak_downward_mm_day=snow.OUT_GW_EXCHANGE.max(),pumping_supply_loss_mm=loss,storage_difference_mm=ds,ET_difference_mm=de,runoff_difference_mm=dr,max_upper_drawdown_m=draw[:,:48].max(),max_deep_drawdown_m=draw[:,96:].max()))
    summary=pd.DataFrame(rows);summary.to_csv(a.output_dir/'data-robustness-summary.csv',index=False)
    pd.DataFrame(curves).to_csv(a.output_dir/'data-robustness-curves.csv',index=False)
    pd.DataFrame(checks).to_csv(a.output_dir/'data-robustness-checks.csv',index=False)
    print(summary.to_string(index=False));print(f'{sum(c["passed"] for c in checks)}/{len(checks)} checks passed')
    if not all(c['passed'] for c in checks):raise SystemExit('Some checks failed; retain failed evidence')

if __name__=='__main__':main()
