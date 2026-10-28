import pickle, json, numpy as np, pandas as pd
from paths import DATA_DIR, SPEC_DIR
B=str(DATA_DIR)
out=pickle.load(open(f'{B}/results.pkl','rb')); cfg=json.load(open(SPEC_DIR / 'windows.json'))
spec=pd.read_csv(SPEC_DIR / 'Features.csv').set_index('Feature ID'); st=pd.read_csv(f'{B}/build_status.csv')
W=['A','B','C']; LBL={'A':'A (2024-01-02+)','B':'B (2024-11-06+, post-election)','C':'C (2026-03-31+, post-selloff low)'}
# ---- final selected
fin=[]
for w in W:
    log,counts,final,reserve,res,cal,info=out[w]
    for f in final+reserve:
        r=log.loc[f]
        fin.append(dict(window=w,role='final' if f in final else 'reserve',id=f,name=r['name'],category=r.category,priority=r.priority,
            reason_kept=r.reason,economic_logic=r.econ_logic,econ_flag=r.econ_flag,stability_flags=r.stability_flags,
            wf_auc=r.wf_auc,ic_spearman=r.ic,ic_t=r.ic_t,sign_consistency=r.sign_consistency,ic_hiVIX=r.ic_hiVIX,ic_loVIX=r.ic_loVIX,
            enet_stab_freq=r.stab_freq,lgbm_perm_dlogloss=r.perm_logloss,mean_abs_shap=r.mean_abs_shap,combined_score=r.combined_score,missing_frac=r.missing_frac))
fin=pd.DataFrame(fin)
sets={w:set(out[w][2]) for w in W}
allf=sorted(set().union(*sets.values()))
ov=pd.DataFrame([dict(id=f,name=fin[fin.id==f].name.iloc[0],category=fin[fin.id==f].category.iloc[0],
     in_A='Y' if f in sets['A'] else ('reserve' if f in out['A'][3] else ''),in_B='Y' if f in sets['B'] else ('reserve' if f in out['B'][3] else ''),in_C='Y' if f in sets['C'] else ('reserve' if f in out['C'][3] else ''),
     n_windows_final=sum(f in sets[w] for w in W)) for f in allf]).sort_values(['n_windows_final','in_B','in_A'],ascending=False)
# ---- fates
fa=st.rename(columns={'id':'id','Feature Name':'name','status':'build_status','note':'build_note'})
for w in W:
    log=out[w][0]
    t=log[['fate','stage','reason']].copy(); t.columns=[f'{w}_fate',f'{w}_stage',f'{w}_reason']
    fa=fa.merge(t,left_on='id',right_index=True,how='left')
for w in W:
    fa[f'{w}_fate']=fa[f'{w}_fate'].fillna('not built'); fa[f'{w}_stage']=fa[f'{w}_stage'].fillna('build'); fa[f'{w}_reason']=fa[f'{w}_reason'].fillna(fa.build_note)
# ---- models
res=pd.concat([out[w][4] for w in W]); res['window']=res.window.map(LBL)
br=res[res.model=='base'][['window','split','acc','logloss','brier']].rename(columns={'acc':'base_acc','logloss':'base_ll','brier':'base_brier'})
res=res.merge(br,on=['window','split']); res['acc_minus_base']=res.acc-res.base_acc; res['logloss_minus_base']=res.logloss-res.base_ll; res['brier_minus_base']=res.brier-res.base_brier
res['majority_class_acc']=np.maximum(res.up_rate,1-res.up_rate)
res=res.drop(columns=['base_ll','base_brier'])
cal=pd.concat([out[w][5] for w in W]); cal['window']=cal.window.map(LBL)
# ---- window comparison
wc=[]
for w in W:
    log,counts,final,reserve,r,_,info=out[w]
    d=dict(window=LBL[w],**{k:v for k,v in info.items() if k!='window'},**{'funnel_'+k:v for k,v in counts.items()})
    for _,row in r.iterrows(): d[f"{row.model}|{row.split}|auc"]=row.auc; d[f"{row.model}|{row.split}|acc"]=row.acc
    wc.append(d)
wc=pd.DataFrame(wc)
inter=sets['A']&sets['B']&sets['C']; ab=sets['A']&sets['B']
jac=pd.DataFrame([[len(sets[a]&sets[b])/len(sets[a]|sets[b]) for b in W] for a in W],index=W,columns=W)
# ---- coverage
cov=pd.DataFrame(json.load(open(f'{B}/raw/download_log.json')))
cov=cov[['source']+[c for c in cov.columns if c!='source']]
extra=pd.DataFrame([
 dict(source='Nasdaq Trader',series='consolidated share/dollar volume',start='',end='',n='',note='NOT OBTAINED: no verified free historical daily archive; F128,F129,F131 excluded'),
 dict(source='Optional calendars (src 26-29)',series='peer/hyperscaler earnings, NVIDIA events, index rebalances',note='NOT OBTAINED: proposed sources without point-in-time announcement archive; F250-F255 excluded'),
 dict(source='BEA schedule',series='PCE release dates',note='Past dates proxied by ALFRED PCEPI release vintages; future dates not parsed from bea.gov -> days_to_next_pce NaN for the last ~17 holdout rows (imputed)'),
 dict(source='FRED ICE BofA OAS',series='BAMLH0A0HYM2/BAMLC0A0CM',note='FRED keeps only ~3y (from 2023-10-24); hy_oas_z252 invalid before ~Nov 2024'),
 dict(source='FRED SP500',series='SP500',note='FRED keeps only 10y (from 2016-10-24); ample for this study'),
 dict(source='Cboe put/call',series='daily market statistics JSON',note='cdn.cboe.com daily JSON 2023-08-01..2026-10-22, 811/811 days; lagged 1 session because historical posting time vs 18:00 ET is unverified (R09)'),
 dict(source='Cboe legacy P/C CSV archive',series='totalpc.csv etc.',note='ends 2019 - not used'),
 dict(source='Crypto (Coinbase)',series='BTC-USD, ETH-USD hourly',note='exact 18:00 ET and 16:00 ET prices from hourly candle closes (DST handled); no approximation needed'),
 dict(source='SEC EDGAR',series='NVDA submissions',note='acceptanceDateTime (UTC) converted to ET; earnings 8-Ks are accepted ~20:20-21:30 ET, i.e. after the 18:00 cutoff, so earnings_8k_today=1 on the reaction day t+1 per rule R12'),
 dict(source='FOMC',series='federalreserve.gov calendar 2021-2027',note='56 scheduled decisions; Aug-2025 notation vote excluded; rate change from FRED DFEDTARU'),
 dict(source='BLS/ALFRED release dates',series='CPI, Employment Situation, PPI (PPIFIS), JOLTS',note='past = ALFRED vintages that add a new month; future = BLS schedule pages. Caveat: 2025 shutdown-delayed releases were not known in advance'),
])
cov=pd.concat([cov,extra],ignore_index=True)
with pd.ExcelWriter(f'{B}/NVDA_Feature_Selection_Results.xlsx',engine='xlsxwriter') as xw:
    pd.DataFrame({'Item':['Study','Target','Windows (dev)','Locked holdout','Purged/embargo','Window C start evidence','Warm-up','Caveat'],
                  'Value':['NVDA next-day direction feature selection (selection study only; no production model)','y=1 if NVDA close[t+1] > close[t]; features as of 18:00 ET on t',
                           'A 2024-01-02..2026-07-21; B 2024-11-06..2026-07-21; C 2026-03-31..2026-07-21','2026-07-23..2026-10-21 (64 sessions; label for 10-21 uses 10-22 close); evaluated once',
                           '2026-07-22 dropped between dev and holdout; walk-forward folds use 1-day gap',cfg['c_start_evidence'],'2022-01 onward used only to compute lookbacks',
                           'Dev walk-forward scores for REDUCED sets are optimistically biased: selection used the same validation folds. Holdout is the only clean estimate.']}).to_excel(xw,sheet_name='README',index=False)
    fin[fin.window=='A'].to_excel(xw,sheet_name='Final_Selected_A',index=False); fin[fin.window=='B'].to_excel(xw,sheet_name='Final_Selected_B',index=False); fin[fin.window=='C'].to_excel(xw,sheet_name='Final_Selected_C',index=False)
    ov.to_excel(xw,sheet_name='Window_Overlap',index=False)
    wc.T.reset_index().to_excel(xw,sheet_name='Window_Comparison',index=False,header=False)
    jac.to_excel(xw,sheet_name='Window_Comparison',startcol=6,startrow=1)
    res.to_excel(xw,sheet_name='Model_Comparison',index=False); cal.to_excel(xw,sheet_name='Calibration',index=False)
    fa.to_excel(xw,sheet_name='Feature_Fates',index=False); cov.to_excel(xw,sheet_name='Data_Coverage',index=False)
    for w in W: out[w][0].reset_index().rename(columns={'index':'id'}).to_excel(xw,sheet_name=f'Funnel_Stats_{w}',index=False)
pickle.dump(dict(fin=fin,ov=ov,res=res,wc=wc,jac=jac,inter=inter,ab=ab,cov=cov,fa=fa,cal=cal),open(f'{B}/summary.pkl','wb'))
print(len(inter),[out['A'][0].at[f,'name'] for f in inter]); print(jac); print(ov.head(15).to_string())
