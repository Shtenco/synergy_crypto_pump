#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

SPECS = {
    'up10_15m': (3, 0.10),
    'up20_1h': (12, 0.20),
    'up50_6h': (72, 0.50),
    'up100_24h': (288, 1.00),
}


def metrics(df: pd.DataFrame, ret_col: str, notional: float=50.0):
    if df.empty:
        return {'trades':0,'net_pnl_usd':0.0,'pnl_pct_initial_equity':0.0}
    r = pd.to_numeric(df[ret_col], errors='coerce').dropna().to_numpy(float)
    pnl = r * notional
    gp = float(pnl[pnl>0].sum()) if np.any(pnl>0) else 0.0
    gl = float(-pnl[pnl<0].sum()) if np.any(pnl<0) else 0.0
    order = df.loc[pd.to_numeric(df[ret_col], errors='coerce').notna()].copy()
    order['_pnl'] = pd.to_numeric(order[ret_col], errors='coerce') * notional
    order = order.sort_values(['exit_timestamp','entry_timestamp','symbol'], kind='stable')
    eq = order['_pnl'].cumsum().to_numpy(float)
    peaks = np.maximum.accumulate(np.r_[0.0, eq])
    dd = np.r_[0.0, eq] - peaks
    max_dd = float(dd.min())
    mean = float(np.mean(r)); sd = float(np.std(r, ddof=1)) if len(r)>1 else 0.0
    return {
        'trades': int(len(r)),
        'net_pnl_usd': float(pnl.sum()),
        'pnl_pct_initial_equity': float(pnl.sum()/5000.0),
        'expectancy_usd': float(pnl.mean()),
        'win_rate': float(np.mean(r>0)),
        'profit_factor': (float(gp/gl) if gl>0 else (float('inf') if gp>0 else None)),
        'max_realized_drawdown_usd': max_dd,
        'mean_return': mean,
        'median_return': float(np.median(r)),
        'sharpe_per_trade': (float(mean/sd*np.sqrt(len(r))) if sd>0 else None),
        'tp_hit_rate': float((order['exit_reason']=='take_profit').mean()),
        'avg_holding_minutes': float(pd.to_numeric(order['minutes_held'], errors='coerce').mean()),
    }


def remove_top_1pct(df, ret_col):
    if df.empty: return df
    n=max(1,int(math.ceil(len(df)*0.01)))
    return df.sort_values(ret_col, ascending=False).iloc[n:].copy()


def portfolio_filter(df: pd.DataFrame, max_concurrent=10):
    if df.empty: return df
    x=df.sort_values(['entry_timestamp','probability'], ascending=[True,False], kind='stable')
    active=[]; out=[]
    for idx,row in x.iterrows():
        et=row['entry_timestamp']
        active=[a for a in active if a[0] > et]
        active_symbols={a[1] for a in active}
        if row['symbol'] in active_symbols or len(active)>=max_concurrent:
            continue
        out.append(idx)
        active.append((row['exit_timestamp'], row['symbol']))
    return x.loc[out].copy()


def build_ledger(d, selected, spec_name, entry_mode, fee_bps=10.0, slippage_bps=25.0):
    H, target = SPECS[spec_name]
    by_symbol={s:g.reset_index(drop=True) for s,g in d.groupby('symbol',sort=False)}
    index_maps={s:{t:i for i,t in enumerate(g['timestamp'])} for s,g in by_symbol.items()}
    rows=[]
    leg=(fee_bps+slippage_bps)/10000.0
    leg2=2*leg
    for sr in selected.itertuples(index=False):
        g=by_symbol.get(sr.symbol); im=index_maps.get(sr.symbol)
        if g is None or sr.timestamp not in im: continue
        i=im[sr.timestamp]
        if i+H>=len(g): continue
        if entry_mode=='signal_close':
            entry_i=i; entry_ts=g.loc[i,'timestamp']; entry=float(g.loc[i,'close']); first=i+1
        else:
            entry_i=i+1; entry_ts=g.loc[entry_i,'timestamp']; entry=float(g.loc[entry_i,'open']); first=entry_i
        if not np.isfinite(entry) or entry<=0: continue
        last=i+H
        tp=entry*(1+target)
        exit_i=last; exit_price=float(g.loc[last,'close']); reason='horizon_close'
        for j in range(first,last+1):
            hi=float(g.loc[j,'high'])
            if np.isfinite(hi) and hi>=tp:
                exit_i=j; exit_price=tp; reason='take_profit'; break
        if not np.isfinite(exit_price) or exit_price<=0: continue
        gross=exit_price/entry-1.0
        net=(exit_price*(1-leg))/(entry*(1+leg))-1.0
        net2=(exit_price*(1-leg2))/(entry*(1+leg2))-1.0
        rows.append({
            'signal_timestamp':sr.timestamp,'entry_timestamp':entry_ts,'exit_timestamp':g.loc[exit_i,'timestamp'],
            'symbol':sr.symbol,'probability':float(sr.probability),'rank':int(sr.rank),'entry_mode':entry_mode,
            'exit_reason':reason,'bars_held':int(exit_i-entry_i+1 if entry_mode=='next_bar_open' else exit_i-entry_i),
            'minutes_held':float((g.loc[exit_i,'timestamp']-entry_ts).total_seconds()/60.0),
            'entry_price':entry,'exit_price':exit_price,'gross_return':gross,'net_return_35bps_leg':net,'net_return_70bps_leg':net2,
        })
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('labeled'); ap.add_argument('models_dir'); ap.add_argument('output_dir')
    ap.add_argument('--top-k',type=int,default=10); ap.add_argument('--notional-usd',type=float,default=50.0)
    args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    d=pd.read_parquet(args.labeled)
    d['timestamp']=pd.to_datetime(d['timestamp'],utc=True)
    d=d.sort_values(['symbol','timestamp'],kind='stable').reset_index(drop=True)
    times=np.array(sorted(d['timestamp'].unique()))
    split=int(len(times)*0.75); test_times=set(times[split:])
    report={'protocol':'HISTORICAL_V2_TOP10_NO_STOP','rows':int(len(d)),'unique_timestamps':int(len(times)),
            'test_timestamp_min':str(times[split]),'test_timestamp_max':str(times[-1]),'top_k':args.top_k,
            'notional_usd':args.notional_usd,'costs':{'fee_bps_per_leg':10.0,'slippage_bps_per_leg':25.0},'targets':{}}
    mdir=Path(args.models_dir)
    for name,(H,thr) in SPECS.items():
        mp=next(mdir.rglob(f'catboost_baseline_{name}.cbm'))
        model=CatBoostClassifier(); model.load_model(str(mp))
        features=list(model.feature_names_)
        target_col=f'target__{name}'
        mask=d['timestamp'].isin(test_times) & d[target_col].notna()
        test=d.loc[mask, ['timestamp','symbol',target_col,*features]].copy()
        p=model.predict_proba(test[features].astype(np.float32))[:,1]
        scored=test[['timestamp','symbol']].copy(); scored['probability']=p
        scored['rank']=scored.groupby('timestamp',sort=False)['probability'].rank(method='first',ascending=False).astype(int)
        selected=scored[scored['rank']<=args.top_k].sort_values(['timestamp','rank'])
        tgt=d.loc[mask,target_col].astype(float).to_numpy()
        top_target=d.loc[selected.index,target_col].astype(float)
        target_report={'base_rate':float(tgt.mean()),'precision_at_top10':float(top_target.mean()),'selected_signals':int(len(selected)),
                       'horizon_bars':H,'threshold':thr,'entry_modes':{}}
        for mode in ['signal_close','next_bar_open']:
            ledger=build_ledger(d,selected,name,mode)
            ledger.to_parquet(out/f'{name}_{mode}_all_top10.parquet',index=False)
            port=portfolio_filter(ledger,max_concurrent=10)
            port.to_parquet(out/f'{name}_{mode}_portfolio10.parquet',index=False)
            target_report['entry_modes'][mode]={
                'all_top10':{
                    'gross':metrics(ledger,'gross_return',args.notional_usd),
                    'net_35bps_leg':metrics(ledger,'net_return_35bps_leg',args.notional_usd),
                    'costs_x2_70bps_leg':metrics(ledger,'net_return_70bps_leg',args.notional_usd),
                    'top1pct_removed_net':metrics(remove_top_1pct(ledger,'net_return_35bps_leg'),'net_return_35bps_leg',args.notional_usd),
                },
                'portfolio_max10_one_per_symbol':{
                    'gross':metrics(port,'gross_return',args.notional_usd),
                    'net_35bps_leg':metrics(port,'net_return_35bps_leg',args.notional_usd),
                    'costs_x2_70bps_leg':metrics(port,'net_return_70bps_leg',args.notional_usd),
                    'top1pct_removed_net':metrics(remove_top_1pct(port,'net_return_35bps_leg'),'net_return_35bps_leg',args.notional_usd),
                }
            }
        report['targets'][name]=target_report
    (out/'historical_no_stop_report.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    print(json.dumps(report,indent=2,default=str))

if __name__=='__main__': main()
