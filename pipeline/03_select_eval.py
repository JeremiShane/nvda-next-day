"""Selection funnel per window (A,B,C) on dev only + evaluation on dev walk-forward and locked holdout."""
import json, warnings, numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import lightgbm as lgb, shap
from paths import DATA_DIR, SPEC_DIR
warnings.filterwarnings('ignore')
B=str(DATA_DIR); rng=np.random.default_rng(7)
X=pd.read_parquet(f'{B}/features.parquet'); st=pd.read_csv(f'{B}/build_status.csv'); cfg=json.load(open(SPEC_DIR / 'windows.json'))
spec=pd.read_csv(SPEC_DIR / 'Features.csv').set_index('Feature ID')
built=[c for c in st[st.status=='built'].id]+['F237_cos']
HO_START,HO_END,DEV_END=cfg['holdout_start'],cfg['holdout_end'],cfg['dev_end']
PRI={'High':0,'Medium':1,'Low':2}
def pri(f): return PRI.get(spec.Priority.get(f[:4],'Low'),2)
def name(f): return spec['Feature Name'].get(f[:4],f)+('_cos' if f.endswith('_cos') else '')
def cat(f): return spec.Category.get(f[:4],'')
X=X[X.y.notna()]

def folds(n,min_train,val,gap=1):
    out=[]; s=min_train
    while s+gap<n:
        e=min(s+gap+val,n); out.append((np.arange(0,s),np.arange(s+gap,e))); s=e-gap
        if n-s-gap<max(5,val//2): break
    return out
def prep(tr,te,feats):
    a=tr[feats].copy(); b=te[feats].copy()
    lo=a.quantile(.01); hi=a.quantile(.99); a=a.clip(lo,hi,axis=1); b=b.clip(lo,hi,axis=1)   # winsorize on train fold
    med=a.median(); a=a.fillna(med).fillna(0); b=b.fillna(med).fillna(0)
    sc=StandardScaler().fit(a); return sc.transform(a),sc.transform(b)
LOGIT_C=0.05
def fit_pred(model,tr,te,feats):
    if model=='base': return np.full(len(te),tr.y.mean())
    if model=='logit':
        a,b=prep(tr,te,feats); m=LogisticRegression(C=LOGIT_C,penalty='l2',max_iter=2000).fit(a,tr.y); return m.predict_proba(b)[:,1]
    if model=='lgbm':
        m=lgb.LGBMClassifier(n_estimators=150,learning_rate=0.02,num_leaves=4,max_depth=2,min_child_samples=max(10,len(tr)//15),
                             subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=5.0,verbose=-1,random_state=1,n_jobs=1)
        m.fit(tr[feats],tr.y); return m.predict_proba(te[feats])[:,1]
def metrics(y,p,base_rate_acc=None):
    y=np.asarray(y); p=np.clip(np.asarray(p),1e-4,1-1e-4)
    auc=roc_auc_score(y,p) if len(set(y))>1 and np.std(p)>0 else 0.5
    return dict(n=len(y),auc=auc,logloss=log_loss(y,p,labels=[0,1]),brier=brier_score_loss(y,p),acc=accuracy_score(y,p>0.5),up_rate=y.mean())
def boot_ci(y,p,stat,B_=1000,block=5):
    y=np.asarray(y); p=np.asarray(p); n=len(y); vals=[]
    nb=int(np.ceil(n/block))
    for _ in range(B_):
        st_=rng.integers(0,n-block+1,nb); ix=np.concatenate([np.arange(s,s+block) for s in st_])[:n]
        yy,pp=y[ix],p[ix]
        if stat=='auc':
            if len(set(yy))<2: continue
            vals.append(roc_auc_score(yy,pp) if np.std(pp)>0 else 0.5)
        else: vals.append(accuracy_score(yy,pp>0.5))
    return np.percentile(vals,[2.5,97.5])
def calib(y,p,bins=5):
    d=pd.DataFrame({'y':y,'p':p}); d['b']=pd.qcut(d.p.rank(method='first'),bins,labels=False)
    g=d.groupby('b').agg(p_mean=('p','mean'),y_rate=('y','mean'),n=('y','size')); return g
def mde(n,p0=0.5):  # min detectable accuracy edge (2-sided 5%, 80% power), and AUC (Hanley-McNeil approx)
    acc=(1.96+0.84)*np.sqrt(p0*(1-p0)/n)
    se_auc=np.sqrt((0.25+ (n/2-1)*(1/12) *2/ (n/2)) / (n/2) ) if n>4 else np.nan
    q1=0.5/(2-0.5); q2=2*0.25/(1+0.5); se=np.sqrt((0.25+(n/2-1)*(q1-0.25)+(n/2-1)*(q2-0.25))/((n/2)**2))
    return acc,(1.96+0.84)*se

def econ(f):
    c=cat(f); n=name(f)
    weak=['days_to_year_end','days_to_quarter_end','days_since_last_holiday','days_to_next_holiday','month_of_year','days_since_10q_10k','days_since_8k_any','days_since_jolts','days_to_next_jolts','days_since_ppi','days_to_next_ppi','days_since_pce','days_to_next_pce','day_of_week','days_to_month_end','days_to_quad_witching','unrate','cpi_yoy','core_cpi_yoy','pce_yoy','core_pce_yoy','ust_10y','real_yield_10y','breakeven_10y','hy_oas','ig_oas','slope_2s10s','slope_3m10y','days_since_form4_cluster','form4_count_5d','nfp_change_k','unrate_chg_1m','core_cpi_mom','last_fomc_rate_change_bp','days_since_fomc','days_since_cpi','days_since_jobs']
    if n.replace('_cos','') in weak: return ('weak','Slow-moving level / calendar counter: in a short sample it mostly proxies for time or regime, not next-day news; statistically-only if it survives')
    R={'NVDA Price & Returns':'Short-horizon momentum/reversal and overnight-vs-intraday pressure in NVDA itself can carry into the next session (order-flow persistence, liquidity-driven reversal).',
       'NVDA Direction & Acceleration':'Change in NVDA momentum/volatility signals shifting positioning that can spill into the next day.',
       'NVDA Volatility & Range':'Volatility state sets the size of next-day moves and drives volatility-targeting/option-hedging flows; high vol tends to coincide with mean reversion.',
       'NVDA Volume':'Abnormal volume marks information arrival or forced flows whose price impact can persist or reverse next day.',
       'NVDA Technical Indicators':'Overbought/oversold and trend-distance measures capture short-term mean reversion and trend-following flows widely watched by traders.',
       'NVDA Price Key Points (Time Since)':'Time since highs/lows/crosses proxies for trend maturity and breakout/pullback dynamics.',
       'Broad Market':'NVDA is a high-beta index heavyweight; market-wide risk appetite and index momentum/reversal transmit directly.',
       'Relative Strength':'Idiosyncratic out/under-performance vs market/sector tends to partially reverse or persist (lead-lag, ETF arbitrage).',
       'Beta & Correlation':'Co-movement regime: when NVDA trades as a macro/beta asset, index signals matter more.',
       'Semiconductor Sector':'Sector ETF flows and semi-cycle news move NVDA; sector moves can lead the stock.',
       'Semiconductor Peers':'Peer moves (AMD, TSM, AVGO, MU) carry AI/semiconductor demand news and supply-chain read-throughs.',
       'Mega-cap Tech':'Hyperscalers are NVDA\'s largest customers; their moves reflect AI capex sentiment.',
       'Bonds & Credit ETFs':'Rates and credit risk appetite affect discount rates for long-duration growth stocks and risk-on/off flows.',
       'Commodities & Dollar ETFs':'Dollar strength and commodity/haven moves reflect global risk appetite and liquidity.',
       'S&P 500 Index (FRED)':'Lagged index return: market reversal/momentum.',
       'Nasdaq Market Activity':'Nasdaq index move: tech risk appetite.',
       'Volatility (VIX)':'Implied volatility/risk premium: spikes in fear tend to precede rebounds (vol risk premium), and vol regime scales moves.',
       'Volatility Term Structure & Cross-Asset':'Inverted short-dated vol (VIX9D/VIX) signals near-term stress; vol-of-vol signals hedging demand.',
       'Options Sentiment (Put/Call)':'Put/call extremes are classic contrarian sentiment signals for next-day reversals.',
       'Treasury Yields & Curve':'Rate shocks reprice long-duration tech; curve moves reflect growth/policy expectations.',
       'Real Yields & Breakevens':'Real yields are the discount rate for long-duration growth equities like NVDA.',
       'Credit Spreads (OAS)':'Widening spreads signal risk-off and tighter financial conditions.',
       'U.S. Dollar (FRED)':'Dollar moves affect multinational earnings and global liquidity.',
       'Crypto':'Crypto trades after the equity close; BTC moves (esp. 16:00-18:00 ET) reflect overnight speculative risk appetite and spill into high-beta tech.',
       'Macro: Inflation':'Inflation drives Fed path and discount rates.','Macro: Employment':'Labor data drives Fed path and growth expectations.','Macro: PCE':'Fed-preferred inflation gauge.',
       'Event: Economic Releases':'Scheduled macro releases the next morning raise next-day variance and can shift risk premia (pre-announcement drift).',
       'Event: Fed / FOMC':'FOMC decision days/pre-FOMC drift are documented equity effects.',
       'Event: NVDA Filings & Earnings':'Earnings and filings are the largest NVDA-specific information events; post-earnings drift and pre-earnings positioning.',
       'Calendar & Seasonality':'Turn-of-month flows, pre-holiday and options-expiration hedging flows are documented calendar effects.'}
    return ('ok',R.get(c,'Plausible but indirect link.'))

def univariate(D,feats,fl):
    rows=[]
    vixmed=D.F132.median()
    for f in feats:
        aucs=[];
        for tr,va in fl:
            a=D.iloc[tr]; b=D.iloc[va]; m=a[[f,'ret_next']].dropna()
            if len(m)<20 or b[f].notna().sum()<5 or m[f].nunique()<2: continue
            sgn=np.sign(spearmanr(m[f],m.ret_next)[0]) or 1
            bb=b[[f,'y']].dropna()
            if bb.y.nunique()<2 or bb[f].nunique()<2: aucs.append(0.5); continue
            aucs.append(roc_auc_score(bb.y,sgn*bb[f]))
        m=D[[f,'ret_next','F132']].dropna()
        ic=spearmanr(m[f],m.ret_next)[0] if m[f].nunique()>1 else 0
        # sub-period sign consistency: calendar quarters (A,B) or months (C)
        per=m.index.to_period('Q' if len(D)>200 else 'M')
        subs=[spearmanr(g[f],g.ret_next)[0] for _,g in m.groupby(per) if len(g)>=15 and g[f].nunique()>1]
        subs=[s for s in subs if np.isfinite(s)]
        cons=np.mean(np.sign(subs)==np.sign(ic)) if subs and ic!=0 else np.nan
        hi=m[m.F132>vixmed]; lo=m[m.F132<=vixmed]
        ich=spearmanr(hi[f],hi.ret_next)[0] if hi[f].nunique()>1 else np.nan; icl=spearmanr(lo[f],lo.ret_next)[0] if lo[f].nunique()>1 else np.nan
        rows.append(dict(id=f,wf_auc=np.mean(aucs) if aucs else np.nan,wf_auc_folds=len(aucs),ic=ic,ic_t=ic*np.sqrt(max(len(m)-2,1)),sign_consistency=cons,n_sub=len(subs),ic_hiVIX=ich,ic_loVIX=icl,
                         vix_flip=bool(np.isfinite(ich) and np.isfinite(icl) and np.sign(ich)!=np.sign(icl) and min(abs(ich),abs(icl))>0.05)))
    return pd.DataFrame(rows).set_index('id')

def multivariate(D,feats,fl,nboot=30):
    freq=pd.Series(0.0,index=feats); cnt=0; perm=pd.Series(0.0,index=feats); sh=pd.Series(0.0,index=feats); nf=0
    for tr,va in fl:
        a=D.iloc[tr]; b=D.iloc[va]
        A,Bv=prep(a,b,feats); ya=a.y.values
        for k in range(nboot):
            ix=rng.choice(len(a),len(a)//2,replace=False)
            m=LogisticRegression(penalty='elasticnet',solver='saga',l1_ratio=0.7,C=0.1,max_iter=3000).fit(A[ix],ya[ix])
            freq+=(np.abs(m.coef_[0])>1e-6); cnt+=1
        m=lgb.LGBMClassifier(n_estimators=150,learning_rate=0.02,num_leaves=4,max_depth=2,min_child_samples=max(10,len(a)//15),subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=5.0,verbose=-1,random_state=1,n_jobs=1).fit(a[feats],ya)
        if b.y.nunique()<2: continue
        p0=m.predict_proba(b[feats])[:,1]; base=log_loss(b.y,p0,labels=[0,1])
        for f in feats:
            d=[]
            for r in range(3):
                bp=b[feats].copy(); bp[f]=rng.permutation(bp[f].values); d.append(log_loss(b.y,m.predict_proba(bp)[:,1],labels=[0,1])-base)
            perm[f]+=np.mean(d)
        sv=shap.TreeExplainer(m).shap_values(b[feats]); sv=sv[1] if isinstance(sv,list) else sv
        sh+=pd.Series(np.abs(sv).mean(0),index=feats); nf+=1
    return pd.DataFrame({'stab_freq':freq/cnt,'perm_logloss':perm/max(nf,1),'mean_abs_shap':sh/max(nf,1)})

def run_window(W):
    s=cfg['windows'][W]['start']; D=X.loc[s:DEV_END].copy(); H=X.loc[HO_START:HO_END].copy(); n=len(D)
    mt,vl={'A':(250,65),'B':(200,50),'C':(40,12)}[W]
    fl=folds(n,mt,vl)
    log=pd.DataFrame(index=built); log['name']=[name(f) for f in built]; log['category']=[cat(f) for f in built]; log['priority']=[spec.Priority.get(f[:4]) for f in built]
    log['fate']='kept'; log['stage']=''; log['reason']=''
    def cut(fs,stage,reason):
        for f in fs:
            if log.at[f,'fate']=='kept': log.loc[f,['fate','stage','reason']]=['cut',stage,reason if isinstance(reason,str) else reason[f]]
    alive=lambda: [f for f in built if log.at[f,'fate']=='kept']
    counts={'built':len(built)}
    # a. data quality
    miss=D[built].isna().mean(); log['missing_frac']=miss
    cut([f for f in built if miss[f]>0.10],'a. data quality',{f:f'missing {miss[f]:.0%} of dev rows (short source history)' for f in built})
    for f in alive():
        v=D[f].dropna(); nu=v.nunique(); top=v.value_counts(normalize=True).iloc[0] if len(v) else 1
        if nu<=1: cut([f],'a. data quality','constant in this window')
        elif nu==2 and (1-top)*len(v)<10: cut([f],'a. data quality',f'binary flag with only {int(round((1-top)*len(v)))} events in dev (cannot estimate)')
        elif nu>2 and top>0.95: cut([f],'a. data quality',f'near-zero variance (top value {top:.0%})')
        elif cat(f).startswith('Macro') and nu<12: cut([f],'a. data quality',f'monthly macro with only {nu} distinct values in window: a step function / time proxy')
    if 'F239' in alive(): cut(['F239'],'a. data quality','identical to F238 by construction (t month-end <=> t+1 month start)')
    counts['a_quality']=len(alive())
    # b. redundancy: |spearman|>0.85 clusters, then VIF<10
    fs=alive(); rho=D[fs].rank().corr().abs().fillna(0).to_numpy(copy=True); np.fill_diagonal(rho,1)
    Z=linkage(squareform(1-rho,checks=False),'average'); cl=fcluster(Z,0.15,'distance'); log.loc[fs,'cluster']=cl
    for c in set(cl):
        mem=[f for f,k in zip(fs,cl) if k==c]
        if len(mem)<2: continue
        keep=sorted(mem,key=lambda f:(pri(f),miss[f],len(str(spec['Formula / Definition'].get(f[:4],''))),f))[0]
        cut([f for f in mem if f!=keep],'b. redundancy (|rho|>0.85)',{f:f'cluster {c}: |Spearman| with {keep} ({name(keep)}) > 0.85; kept higher-priority/simpler member' for f in mem})
    fs=alive(); M=D[fs].fillna(D[fs].median()); M=(M-M.mean())/M.std()
    while True:
        vif=pd.Series([variance_inflation_factor(M.values,i) for i in range(M.shape[1])],index=M.columns)
        if vif.max()<10: break
        cand=vif[vif>=10].index; worst=max(cand,key=lambda f:(round(vif[f],0)>=vif.max()*0.9, pri(f), vif[f]))
        cut([worst],'b. redundancy (VIF)',f'VIF {vif[worst]:.1f} > 10 (linear combination of other survivors)'); M=M.drop(columns=worst)
    log.loc[M.columns,'vif']=vif
    counts['b_redundancy']=len(alive())
    # c. univariate signal & stability
    U=univariate(D,alive(),fl); log=log.join(U,how='left')
    for f in alive():
        u=U.loc[f]; flags=[]
        if u.vix_flip: flags.append('IC flips sign between high/low VIX')
        if np.isfinite(u.sign_consistency) and u.sign_consistency<0.5: flags.append(f'IC sign consistent in only {u.sign_consistency:.0%} of sub-periods')
        log.at[f,'stability_flags']='; '.join(flags)
        if (not np.isfinite(u.wf_auc) or u.wf_auc<=0.5) and abs(u.ic_t)<1.0:
            cut([f],'c. univariate',f'no walk-forward signal (WF AUC {u.wf_auc:.3f}, IC t {u.ic_t:.2f})')
        elif u.wf_auc<0.5 and flags:
            cut([f],'c. univariate',f'WF AUC {u.wf_auc:.3f} < 0.5 and unstable ({"; ".join(flags)})')
    counts['c_univariate']=len(alive())
    # d. multivariate inside training folds; importance on validation folds
    fs=alive(); Mv=multivariate(D,fs,fl,nboot=30 if W!='C' else 50); log=log.join(Mv,how='left')
    Mv['score']=(Mv.stab_freq.rank(pct=True)+Mv.perm_logloss.rank(pct=True)+Mv.mean_abs_shap.rank(pct=True)+U.loc[fs,'wf_auc'].rank(pct=True))/4
    log.loc[fs,'combined_score']=Mv.score
    for f in fs:
        r=Mv.loc[f]
        if r.stab_freq<0.2 and r.perm_logloss<=0 and U.at[f,'wf_auc']<0.52:
            cut([f],'d. multivariate',f'rarely selected by elastic-net (freq {r.stab_freq:.2f}), permutation importance <= 0 and weak WF AUC')
    counts['d_multivariate']=len(alive())
    # e. economic logic
    for f in built:
        e,txt=econ(f); log.at[f,'econ_logic']=txt; log.at[f,'econ_flag']=e
    fs=alive(); weak=[f for f in fs if log.at[f,'econ_flag']=='weak']
    sc=log.loc[fs,'combined_score']
    for f in weak:
        if sc[f]<sc.quantile(0.75): cut([f],'e. economic logic','weak economic rationale (time/regime proxy) and not a top-quartile statistical survivor')
        else: log.at[f,'reason']='STATISTICAL-ONLY survivor (weak economic logic) - treat with suspicion'
    counts['e_econ']=len(alive())
    # final ~15-30: rank by combined score, prefer stable (no flags), cap 25
    fs=alive(); strong=[f for f in fs if log.at[f,'econ_flag']!='weak']; weakf=[f for f in fs if f not in strong]
    rk=log.loc[strong].assign(pen=lambda d: d.stability_flags.fillna('').str.len().gt(0).astype(int)*0.1+(d.econ_flag=='weak')*0.15)
    rk['final_score']=rk.combined_score-rk.pen; rk=rk.sort_values('final_score',ascending=False)
    target=25 if W!='C' else 15
    final=list(rk.index[:target]); wk=log.loc[weakf].sort_values('combined_score',ascending=False)
    reserve=list(rk.index[target:target+10-min(3,len(wk))])+list(wk.index[:3])   # up to 3 statistical-only survivors go to reserve, never final
    for f in reserve: log.loc[f,['fate','stage','reason']]=['reserve','final ranking','statistical-only survivor (weak economic logic): reserve only' if f in weakf else 'survived funnel but ranked below final cutoff']
    for f in [f for f in fs if f not in final+reserve]: log.loc[f,['fate','stage','reason']]=['cut','final ranking','survived funnel but ranked outside final set + reserve']
    for f in final:
        log.at[f,'stage']='final'; 
        if not log.at[f,'reason']: log.at[f,'reason']=f"combined rank score {log.at[f,'combined_score']:.2f} (stab freq {log.at[f,'stab_freq']:.2f}, WF AUC {log.at[f,'wf_auc']:.3f}, perm dLL {log.at[f,'perm_logloss']*1e3:.2f}e-3)"
    counts['final']=len(final); counts['reserve']=len(reserve)
    # ---- evaluation ----
    res=[]; preds={}
    allf=built
    for mname,feats in [('base',[]),('logit_full',allf),('logit_reduced',final),('lgbm_full',allf),('lgbm_reduced',final)]:
        mdl=mname.split('_')[0]
        P=[];Y=[]
        for tr,va in fl:
            P+=list(fit_pred(mdl,D.iloc[tr],D.iloc[va],feats)); Y+=list(D.iloc[va].y)
        m=metrics(Y,P); ci=boot_ci(Y,P,'auc'); ca=boot_ci(Y,P,'acc')
        res.append(dict(window=W,model=mname,split='dev walk-forward',**m,auc_lo=ci[0],auc_hi=ci[1],acc_lo=ca[0],acc_hi=ca[1],n_feats=len(feats)))
        ph=fit_pred(mdl,D,H,feats); m=metrics(H.y,ph); ci=boot_ci(H.y.values,ph,'auc'); ca=boot_ci(H.y.values,ph,'acc')
        res.append(dict(window=W,model=mname,split='holdout',**m,auc_lo=ci[0],auc_hi=ci[1],acc_lo=ca[0],acc_hi=ca[1],n_feats=len(feats)))
        preds[(mname,'dev')]=(Y,P); preds[(mname,'holdout')]=(list(H.y),list(ph))
    cal=[]
    for (mn,sp),(yy,pp) in preds.items():
        if mn=='base': continue
        g=calib(yy,pp); g['window']=W; g['model']=mn; g['split']=sp; cal.append(g.reset_index())
    a_dev,auc_dev=mde(sum(len(v) for _,v in fl)); a_ho,auc_ho=mde(len(H))
    info=dict(window=W,start=s,dev_end=DEV_END,dev_rows=n,folds=len(fl),val_rows=sum(len(v) for _,v in fl),holdout_rows=len(H),dev_up_rate=D.y.mean(),
              mde_acc_dev=a_dev,mde_auc_dev=auc_dev,mde_acc_holdout=a_ho,mde_auc_holdout=auc_ho,fold_spec=f'expanding; min train {mt}, val blocks {vl}, gap 1')
    log['window']=W
    return log,counts,final,reserve,pd.DataFrame(res),pd.concat(cal),info

out={}
for W in ['A','B','C']:
    print('== window',W,flush=True); out[W]=run_window(W); print(out[W][1]); print([name(f) for f in out[W][2]]); print(out[W][4][['model','split','auc','auc_lo','auc_hi','acc','logloss','brier','up_rate']].round(3).to_string())
import pickle; pickle.dump(out,open(f'{B}/results.pkl','wb'))
