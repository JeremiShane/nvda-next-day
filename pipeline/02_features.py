"""Build F001-F255 point-in-time features as of 18:00 ET on trading day t. Output features.parquet + build_status.csv"""
import os, re, json, glob, warnings
import numpy as np, pandas as pd, pandas_market_calendars as mcal
from paths import DATA_DIR, RAW_DIR, SPEC_DIR
warnings.filterwarnings('ignore')
B=str(DATA_DIR); RAW=str(RAW_DIR)
LAST_BAR='2026-10-22'
spec=pd.read_csv(SPEC_DIR / 'Features.csv')
cal=mcal.get_calendar('XNYS')
sched=cal.schedule('2021-06-01','2027-12-31')
TD=pd.DatetimeIndex(sched.index.normalize())           # all trading days incl. future (known schedule)
early=set(TD[(sched.market_close.dt.tz_convert('America/New_York').dt.hour<16).values])
idx=TD[(TD>='2022-01-03')&(TD<=LAST_BAR)]
pos=pd.Series(np.arange(len(TD)),index=TD)
F={}; status={}
def put(fid,s,note=''):
    F[fid]=pd.Series(s,index=idx).astype(float) if not isinstance(s,pd.Series) else s.reindex(idx).astype(float)
    status[fid]=('built',note)
def excl(fid,why): status[fid]=('excluded',why)

# ---------- Yahoo ----------
y=pd.read_parquet(f'{RAW}/yahoo.parquet'); y.index=pd.to_datetime(y.index).normalize()
def px(t,f='Close'): return y[t][f].reindex(idx)
lr=lambda s,k=1: np.log(s/s.shift(k))
def z60(r):  # R18: today / stdev of trailing 60 obs ending t-1, min 40 obs
    return r/ r.shift(1).rolling(60,min_periods=40).std()
def zwin(s,w): return (s-s.rolling(w,min_periods=int(w*.8)).mean())/s.rolling(w,min_periods=int(w*.8)).std()
def rvol(r,w): return r.rolling(w,min_periods=w).std()*np.sqrt(252)
def wilder(s,n): return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
def rsi(c,n):
    d=c.diff(); g=wilder(d.clip(lower=0),n); l=wilder((-d).clip(lower=0),n); return 100-100/(1+g/l)
def days_since_true(flag,cap):
    out=[];last=None
    for i,v in enumerate(flag.values):
        if v==True: last=i
        out.append(np.nan if last is None else min(i-last,cap))
    return pd.Series(out,index=flag.index,dtype=float)
def days_since_argmax(s,w,low=False):
    f=(lambda a: (len(a)-1-np.argmin(a))) if low else (lambda a:(len(a)-1-np.argmax(a)))
    return s.rolling(w,min_periods=w).apply(f,raw=True)
C,O,H,L,V=[px('NVDA',k) for k in ['Close','Open','High','Low','Volume']]
r1=lr(C)
put('F001',r1); put('F002',lr(C,2)); put('F003',lr(C,5)); put('F004',lr(C,10)); put('F005',lr(C,20))
put('F006',np.log(O/C.shift())); put('F007',np.log(C/O))
put('F008',np.where(H==L,0.5,(C-L)/(H-L)))
sd60=r1.shift(1).rolling(60,min_periods=40).std()
put('F009',r1/sd60); put('F010',lr(C,5)/(sd60*np.sqrt(5)))
put('F011',np.sign(r1)); put('F012',r1.abs())
st=[];c=0
for v in np.sign(r1).fillna(0).values:
    c = (c+1 if c>0 else 1) if v>0 else ((c-1 if c<0 else -1) if v<0 else 0); c=max(min(c,10),-10); st.append(c)
put('F013',pd.Series(st,index=idx))
put('F014',(r1>0).rolling(5).sum()/5); put('F015',(r1>0).rolling(10).sum()/10)
put('F016',r1-r1.shift(1).rolling(5).mean()); put('F017',lr(C,5)-lr(C,5).shift(5))
rv5=rvol(r1,5); rv20=rvol(r1,20)
put('F018',rv5-rv5.shift(5))
lv=np.log(V); vz20=(lv-lv.rolling(20).mean())/lv.rolling(20).std()
put('F019',vz20-vz20.shift()); put('F020',np.log(H/L))
tr=pd.concat([H-L,(H-C.shift()).abs(),(L-C.shift()).abs()],axis=1).max(axis=1)
put('F021',wilder(tr,14)/C); put('F022',rv5); put('F023',rv20); put('F024',rv5/rv20)
put('F025',np.sqrt((1/(4*np.log(2)))*(np.log(H/L)**2).rolling(20).mean()*252))
put('F026',vz20); put('F027',zwin(np.log(C*V),60)); put('F028',np.sign(r1)*vz20)
rsi14=rsi(C,14); put('F029',rsi14); put('F030',rsi(C,2))
macd=C.ewm(span=12,adjust=False).mean()-C.ewm(span=26,adjust=False).mean(); hist=macd-macd.ewm(span=9,adjust=False).mean()
put('F031',macd/C); put('F032',hist/C)
sma=lambda s,n: s.rolling(n,min_periods=n).mean()
d20=C/sma(C,20)-1; d50=C/sma(C,50)-1
put('F033',d20); put('F034',d50); put('F035',C/sma(C,200)-1)
sdc=C.rolling(20).std(); put('F036',(C-(sma(C,20)-2*sdc))/(4*sdc))
put('F037',C/H.rolling(20).max()-1); put('F038',C/H.rolling(252,min_periods=252).max()-1)
put('F039',rsi14.diff()); put('F040',rsi14-rsi14.shift(5))
hn=hist/C; put('F041',hn.diff())
put('F042',np.where((hist.shift()<=0)&(hist>0),1,np.where((hist.shift()>=0)&(hist<0),-1,0)))
put('F043',d20-d20.shift(5)); put('F044',d50-d50.shift(5))
put('F045',days_since_argmax(H,252)); put('F046',days_since_argmax(H,20)); put('F047',days_since_argmax(L,20,low=True))
big=r1.abs()>=np.log(1.05); put('F048',days_since_true(big,120))
lbs=pd.Series(np.where(big,np.sign(r1),np.nan),index=idx).ffill(); put('F049',lbs.where(F['F048'].notna()&(F['F048']<120),0))
def cross(c,n,cap):
    dd=np.sign(c-sma(c,n)); cr=(dd!=dd.shift())&dd.notna()&dd.shift().notna()
    ds=days_since_true(cr,cap); dirn=pd.Series(np.where(cr,dd,np.nan),index=c.index).ffill(); return ds,dirn
a,b=cross(C,50,120); put('F050',a); put('F051',b); a,b=cross(C,200,252); put('F052',a); put('F053',b)
R={t:lr(px(t)) for t in ['SPY','QQQ','DIA','IWM','SMH','SOXX','AMD','AVGO','TSM','INTC','MU','ARM','MSFT','GOOGL','AMZN','META','AAPL','TLT','IEF','HYG','LQD','GLD','USO','UUP','^IXIC']}
R5={t:lr(px(t),5) for t in R}
put('F054',R['SPY']); put('F055',R5['SPY']); put('F056',R['QQQ']); put('F057',R5['QQQ']); put('F058',R['DIA']); put('F059',R['IWM'])
put('F060',px('SPY')/sma(px('SPY'),200)-1); put('F061',px('QQQ')/sma(px('QQQ'),50)-1)
spyrv=rvol(R['SPY'],20); put('F062',spyrv); put('F063',z60(R['SPY'])); put('F064',z60(R['QQQ']))
put('F065',r1-R['SPY']); put('F066',lr(C,5)-R5['QQQ']); put('F067',lr(C,20)-lr(px('QQQ'),20))
put('F068',r1-R['SMH']); put('F069',lr(C,5)-R5['SMH']); put('F070',lr(C,20)-lr(px('SOXX'),20))
peers=['AMD','AVGO','TSM','INTC','MU','ARM']
P1=pd.concat([R[p] for p in peers],axis=1); P5=pd.concat([R5[p] for p in peers],axis=1)
put('F071',lr(C,5)-P5.mean(axis=1))
beta=lambda a,b,w: a.rolling(w,min_periods=w).cov(b)/b.rolling(w,min_periods=w).var()
put('F072',r1-beta(r1,R['QQQ'],60).shift(1)*R['QQQ'])
put('F073',((np.sign(r1)==np.sign(R['SMH']))&(r1!=0)&(R['SMH']!=0)).astype(float))
put('F074',((np.sign(r1)==np.sign(R['QQQ']))&(r1!=0)&(R['QQQ']!=0)).astype(float))
put('F075',beta(r1,R['QQQ'],60)); put('F076',beta(r1,R['SMH'],60)); put('F077',r1.rolling(20).corr(R['SPY']))
put('F078',R['SMH']); put('F079',R5['SMH']); put('F080',R['SOXX']); put('F081',R5['SMH']*0+lr(px('SMH'),20)-lr(px('QQQ'),20))
put('F082',z60(R['SMH'])); put('F083',np.sign(R['SMH'])); put('F084',days_since_argmax(px('SMH','High'),20))
put('F085',(P1>0).sum(axis=1)/P1.notna().sum(axis=1)); pm=P1.mean(axis=1); put('F086',pm); put('F087',P5.mean(axis=1))
put('F088',P1.std(axis=1))
ab=pd.concat([(px(p)>sma(px(p),50)).where(sma(px(p),50).notna()) for p in peers],axis=1); put('F089',ab.mean(axis=1))
for fid,t in [('F090','AMD'),('F091','AVGO'),('F092','TSM'),('F093','MU'),('F096','INTC'),('F097','ARM')]: put(fid,R[t])
put('F094',z60(pm)); put('F095',(P1>0).rolling(5).sum().sum(axis=1)/P1.notna().rolling(5).sum().sum(axis=1))
for fid,t in [('F098','AMD'),('F099','AVGO'),('F100','TSM'),('F101','MU')]: put(fid,R5[t])
for fid,t in [('F102','AMD'),('F103','AVGO'),('F104','TSM'),('F105','MU'),('F106','INTC'),('F107','ARM')]: put(fid,z60(R[t]))
M=pd.concat([R[t] for t in ['MSFT','GOOGL','AMZN','META','AAPL']],axis=1)
put('F108',(M>0).sum(axis=1)/5); mm=M.mean(axis=1); put('F109',mm); put('F110',M.iloc[:,:4].mean(axis=1)); put('F111',z60(mm))
put('F112',R['TLT']); put('F113',R5['TLT']); put('F114',R['HYG'])
put('F115',lr(px('HYG')/px('LQD'),5)); put('F116',lr(px('HYG')/px('IEF'),5)); put('F117',zwin(np.log(px('HYG')/px('IEF')),60))
put('F118',z60(R['TLT'])); put('F119',z60(R['HYG']))
put('F120',R['GLD']); put('F121',R['USO']); put('F122',R['UUP']); put('F123',R5['UUP']); put('F124',z60(R['UUP'])); put('F125',z60(R['GLD'])); put('F126',z60(R['USO']))
put('F130',R['^IXIC'],'Nasdaq Composite close from Yahoo ^IXIC (not Nasdaq Trader)')
for f in ['F128','F129','F131']: excl(f,'Nasdaq Trader consolidated share/dollar volume: no free, verifiable historical daily archive obtained; not faked')

# ---------- FRED daily (lag 1 business day, R03/R23) ----------
def fred(s):
    x=pd.read_csv(f'{RAW}/fred_{s}.csv'); x.columns=['d','v']; x['v']=pd.to_numeric(x.v,errors='coerce'); x=x.dropna()
    x['d']=pd.to_datetime(x.d); return x.set_index('d').v
def lag1(s):   # value for obs date d usable from next trading day; align to calendar w/ ffill, then lag 1 session
    a=s.reindex(TD.union(s.index)).ffill().reindex(TD)
    return a.shift(1).reindex(idx)
sp=lag1(fred('SP500')); put('F127',np.log(sp/sp.shift(1)),'FRED SP500 lag 1 day (FRED only keeps ~10y)')
dfii=lag1(fred('DFII10')); d10=lag1(fred('DGS10'))
put('F171',dfii); put('F172',100*(dfii-dfii.shift(5))); be=d10-dfii; put('F173',be)
put('F174',100*dfii.diff()); put('F175',z60(100*dfii.diff())); put('F176',100*be.diff()); put('F177',100*(be-be.shift(5)))
hy=lag1(fred('BAMLH0A0HYM2')); ig=lag1(fred('BAMLC0A0CM'))
put('F178',hy,'FRED ICE BofA history only from 2023-10-24 (license limit)'); put('F179',100*(hy-hy.shift(5)))
put('F180',(hy-hy.rolling(252,min_periods=252).mean())/hy.rolling(252,min_periods=252).std(),'needs 252 obs; FRED OAS history starts 2023-10-24 so valid only from ~late 2024')
put('F181',ig); put('F182',100*(ig-ig.shift(5))); put('F183',100*hy.diff()); put('F184',100*ig.diff()); put('F185',z60(100*hy.diff())); put('F186',z60(100*ig.diff()))
usd=fred('DTWEXBGS')
avail=usd.index+pd.to_timedelta((7-usd.index.dayofweek)%7+0,unit='D'); avail=pd.DatetimeIndex([a if a>d else a+pd.Timedelta(days=7) for a,d in zip(avail,usd.index)])
# obs dated d (Mon-Fri) assumed released on the following Monday H.10 (4:15pm ET) -> usable that Monday 18:00
ua=pd.Series(usd.values,index=avail); ua=ua.groupby(level=0).last()
ul=np.log(usd); ch=(ul-ul.shift(5)); cha=pd.Series(ch.values,index=avail).groupby(level=0).last()
put('F187',cha.reindex(TD.union(cha.index)).ffill().reindex(idx),'DTWEXBGS: obs usable from next Monday (weekly H.10) - conservative')

# ---------- Treasury (default lag 1, R10) ----------
t=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{RAW}/treasury_*.csv'))]); t['Date']=pd.to_datetime(t.Date); t=t.set_index('Date').sort_index()
y10=lag1(t['10 Yr']); y2=lag1(t['2 Yr']); y3m=lag1(t['3 Mo'])
put('F160',y10); put('F161',100*y10.diff()); put('F162',100*y2.diff()); put('F163',100*(y10-y10.shift(5)))
s210=y10-y2; s3m=y10-y3m; put('F164',s210); put('F165',s3m); put('F166',s210-s210.shift(5))
put('F167',z60(100*y10.diff())); put('F168',z60(100*y2.diff())); put('F169',100*s210.diff()); put('F170',100*s3m.diff())

# ---------- Cboe vol ----------
def cboe(s):
    x=pd.read_csv(f'{RAW}/cboe_{s}.csv'); x.columns=[c.upper() for c in x.columns]; x['DATE']=pd.to_datetime(x.DATE); x=x.set_index('DATE')
    col='CLOSE' if 'CLOSE' in x else x.columns[-1]; return x[col].astype(float).reindex(idx)
vix=cboe('VIX'); v9=cboe('VIX9D'); vv=cboe('VVIX'); ovx=cboe('OVX'); gvz=cboe('GVZ')
put('F132',vix); put('F133',vix.diff()); put('F134',np.log(vix/vix.shift(5))); vz=zwin(vix,60); put('F135',vz)
put('F136',vix-100*spyrv); put('F137',z60(vix.diff())); put('F138',vz-vz.shift(5)); put('F139',days_since_true(vix>30,252),'NaN until first VIX>30 observed in data (Aug 2024 / Apr 2025 spikes)')
p90=vix.shift(1).rolling(252,min_periods=200).quantile(.9); put('F140',days_since_true((vix>p90)&p90.notna(),252))
rt=v9/vix; put('F141',rt); put('F142',rt.diff()); put('F143',vv); put('F144',np.log(ovx/ovx.shift(5))); put('F145',np.log(gvz/gvz.shift(5)))
put('F146',rt-rt.shift(5)); put('F147',vv.diff()); put('F148',np.log(vv/vv.shift(5)))

# ---------- Cboe put/call (lag 1 session: posting time vs 18:00 ET not verified, R09) ----------
rows={}
for f in glob.glob(f'{RAW}/cboe_pc/*.json'):
    d=json.load(open(f)); r={x['name']:float(x['value']) for x in d['ratios']}; rows[pd.Timestamp(os.path.basename(f)[:10])]=r
pc=pd.DataFrame(rows).T.sort_index()
def pcs(name): return pc[name].reindex(TD).shift(1).reindex(idx)
tot=pcs('TOTAL PUT/CALL RATIO'); eq=pcs('EQUITY PUT/CALL RATIO')
nt='Cboe daily market statistics JSON (cdn.cboe.com), lagged 1 session (posting time not verified per R09)'
put('F149',tot,nt); put('F150',tot.rolling(5).mean(),nt); put('F151',zwin(tot,60),nt); put('F152',eq,nt); put('F153',eq.rolling(5).mean(),nt); put('F154',zwin(eq,60),nt)
put('F155',pcs('INDEX PUT/CALL RATIO'),nt); put('F156',pcs('EXCHANGE TRADED PRODUCTS PUT/CALL RATIO'),nt); put('F157',pcs('CBOE VOLATILITY INDEX (VIX) PUT/CALL RATIO'),nt)
put('F158',tot.diff(),nt); put('F159',eq.diff(),nt)

# ---------- Crypto: exact 18:00 ET and 16:00 ET from hourly candles ----------
def crypto(prod):
    c=pd.read_csv(f'{RAW}/coinbase_{prod}_1h.csv'); c['end']=pd.to_datetime(c.time,unit='s',utc=True)+pd.Timedelta(hours=1)
    c['et']=c.end.dt.tz_convert('America/New_York'); c=c.set_index('et').close
    def at(h):
        s=c[c.index.hour==h]; s.index=pd.DatetimeIndex(s.index.date); return s[~s.index.duplicated()]
    return at(18),at(16)
b18,b16=crypto('BTC-USD'); e18,_=crypto('ETH-USD')
bt=b18.reindex(idx); put('F188',np.log(bt/bt.shift(1)),'close of the hourly candle ending 18:00 ET'); et_=e18.reindex(idx); put('F189',np.log(et_/et_.shift(1)))
bcal=np.log(b18/b18.shift(1)).asfreq('D'); put('F190',(bcal.rolling(20,min_periods=18).std()*np.sqrt(365)).reindex(idx))
put('F191',np.log(b18/b16).reindex(idx)); put('F192',z60(F['F188']))

# ---------- Monthly macro via ALFRED vintages (point-in-time, released 8:30am ET) ----------
def vint(s):
    out={}
    for f in sorted(glob.glob(f'{RAW}/alfred_{s}/*.csv')):
        v=pd.Timestamp(os.path.basename(f)[:10]); x=pd.read_csv(f); x.columns=['d','v']; x['v']=pd.to_numeric(x.v,errors='coerce'); x=x.dropna()
        if len(x): out[v]=pd.Series(x.v.values,index=pd.to_datetime(x.d))
    return out
def pit(s,fn):
    vs=vint(s); vals=pd.Series({v:fn(ser) for v,ser in vs.items()}).sort_index()
    rel=[]; prev=None
    for v in sorted(vs):   # a release = vintage that adds a new observation month (filters revision-only / non-release vintages)
        last=vs[v].index.max()
        if prev is None or last>prev: rel.append(v)
        prev=last if prev is None else max(prev,last)
    return vals.reindex(TD.union(vals.index)).ffill().reindex(idx), rel[1:]
yoy=lambda s: np.log(s.iloc[-1]/s.get(s.index[-1]-pd.DateOffset(months=12),np.nan))
mom=lambda s: np.log(s.iloc[-1]/s.iloc[-2])
cpi,cpid=pit('CPIAUCSL',yoy); put('F193',cpi,'ALFRED vintage as of release date')
cc,_=pit('CPILFESL',yoy); put('F194',cc); cm,_=pit('CPILFESL',mom); put('F195',cm)
nfp,nfpd=pit('PAYEMS',lambda s: s.iloc[-1]-s.iloc[-2]); put('F197',nfp,'first-release vintage')
ur,_=pit('UNRATE',lambda s:s.iloc[-1]); put('F198',ur); uc,_=pit('UNRATE',lambda s:s.iloc[-1]-s.iloc[-2]); put('F199',uc)
pce,pced=pit('PCEPI',yoy); put('F200',pce); cp,_=pit('PCEPILFE',yoy); put('F201',cp)
_,ppid=pit('PPIFIS',lambda s:s.iloc[-1]); _,joltd=pit('JTSJOL',lambda s:s.iloc[-1])

# release dates: ALFRED vintage dates (past) + BLS/BEA published schedules (future, after last vintage)
MON={m:i for i,m in enumerate(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],1)}
def sched_dates(name):
    p=f'{RAW}/{name}.html'
    if not os.path.exists(p): return []
    h=open(p,errors='ignore').read(); out=[]
    for m,d,yv in re.findall(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2}),\s+(20\d\d)',h):
        try: out.append(pd.Timestamp(int(yv),MON[m.lower()],int(d)))
        except: pass
    return out
def evset(vd,schedname):
    s=set(pd.Timestamp(v) for v in vd)
    lastv=max(s); fut=[d for d in sched_dates(schedname) if d>lastv and d in set(TD)]
    return sorted(s|set(fut)), len(fut)
CPI_D,nf1=evset(cpid,'bls_sched_cpi'); JOB_D,nf2=evset(nfpd,'bls_sched_empsit'); PPI_D,nf3=evset(ppid,'bls_sched_ppi'); JOL_D,nf4=evset(joltd,'bls_sched_jolts'); PCE_D,nf5=evset(pced,'bea_schedule')
print('future sched dates added',nf1,nf2,nf3,nf4,nf5)
def ev_series(dates):
    d=pd.DatetimeIndex([x for x in dates if x in pos.index]); return d
def today(dates): s=set(ev_series(dates)); return pd.Series([float(x in s) for x in idx],index=idx)
def nextday(dates):
    s=set(ev_series(dates)); return pd.Series([float(TD[pos[x]+1] in s) for x in idx],index=idx)
def days_to(dates,cap):
    p=np.sort(pos[ev_series(dates)].values); out=[]
    for x in idx:
        i=pos[x]; j=np.searchsorted(p,i,side='right'); out.append(min(p[j]-i,cap) if j<len(p) else np.nan)
    return pd.Series(out,index=idx,dtype=float)
def days_since(dates,cap):
    p=np.sort(pos[ev_series(dates)].values); out=[]
    for x in idx:
        i=pos[x]; j=np.searchsorted(p,i,side='right')-1; out.append(min(i-p[j],cap) if j>=0 else np.nan)
    return pd.Series(out,index=idx,dtype=float)
nb='release dates = ALFRED vintage dates (actual release days); future from BLS schedule page. Caveat: 2025 shutdown-delayed releases were not known in advance'
put('F196',today(CPI_D),nb); put('F202',days_to(CPI_D,30),nb); put('F203',nextday(CPI_D),nb); put('F204',nextday(JOB_D),nb)
put('F205',nextday(PPI_D),nb+' (PPIFIS vintages)'); put('F206',nextday(JOL_D),nb+' (JTSJOL vintages)'); put('F207',days_since(CPI_D,30))
put('F208',days_to(JOB_D,30)); put('F209',days_since(JOB_D,30)); put('F210',days_to(PPI_D,30)); put('F211',days_since(PPI_D,30)); put('F212',days_to(JOL_D,40)); put('F213',days_since(JOL_D,30))
put('F214',days_to(PCE_D,30),'PCE dates from ALFRED PCEPI vintage dates (proxy for BEA schedule; source proposed/not in inventory)'); put('F215',days_since(PCE_D,30),'as F214')

# ---------- FOMC ----------
h=open(f'{RAW}/fomc_calendar.html',errors='ignore').read()
fomc=[]
for blk in re.split(r'<h4[^>]*>',h)[1:]:
    m=re.search(r'(20\d\d) FOMC Meetings',blk[:200]);
    if not m: continue
    yr=int(m.group(1))
    for mon,days in re.findall(r'fomc-meeting__month[^>]*>\s*<strong>([A-Za-z/]+)</strong>.*?fomc-meeting__date[^>]*>([^<]+)<',blk,flags=re.S):
        if 'notation' in days.lower() or 'unscheduled' in days.lower(): continue
        dd=re.findall(r'\d+',days); mo=mon.split('/')[-1][:3].lower()
        if not dd or mo not in MON: continue
        try: fomc.append(pd.Timestamp(yr,MON[mo],int(dd[-1])))
        except: pass
fomc=sorted(set(fomc)); print('FOMC decisions parsed',len(fomc),fomc[:2],fomc[-2:])
put('F217',days_to(fomc,40)); put('F218',nextday(fomc)); put('F219',today(fomc)); put('F221',days_since(fomc,40))
tu=fred('DFEDTARU'); chg={}
for d in fomc:
    if d> tu.index.max(): continue
    a=tu[:d-pd.Timedelta(days=1)].iloc[-1]; b=tu[:d+pd.Timedelta(days=3)].iloc[-1]; chg[d]=100*(b-a)
cs=pd.Series(chg); put('F220',cs.reindex(TD.union(cs.index)).ffill().reindex(idx),'DFEDTARU change around scheduled decision days (known 2:00 pm ET)')
mins=[TD[TD>=d+pd.Timedelta(days=21)][0] for d in fomc]
put('F222',days_to(mins,40),'DERIVED: decision + 21 calendar days (next trading day if not a session)')
bo=pd.Series(0.0,index=idx)
for d in fomc:
    start=d-pd.Timedelta(days=1)  # meetings are 2-day (1st day = d-1) for all scheduled meetings 2022-2026
    sat=start-pd.Timedelta(days=(start.dayofweek-5)%7 or 7); sat2=sat-pd.Timedelta(days=7)
    bo[(idx>=sat2)&(idx<=d+pd.Timedelta(days=1))]=1.0
put('F223',bo,'blackout: 2nd Saturday before meeting start through day after meeting')

# ---------- EDGAR ----------
e=json.load(open(f'{RAW}/edgar_nvda.json'))['filings']['recent']; ed=pd.DataFrame(e)
ed['acc']=pd.to_datetime(ed.acceptanceDateTime,utc=True).dt.tz_convert('America/New_York')
# window (18:00 t-1, 18:00 t] -> assign each filing to first trading day whose 18:00 >= acceptance
def assign(ts):
    d=ts.tz_localize(None); day=d.normalize(); 
    if d.hour>=18: day=day+pd.Timedelta(days=1)
    j=TD.searchsorted(day); return TD[j] if j<len(TD) else pd.NaT
ed['t']=[assign(x) for x in ed.acc]
k8=ed[ed.form.isin(['8-K','8-K/A'])]; ern=k8[k8['items'].str.contains('2.02',na=False)]
ERN=sorted(set(ern.t.dropna())); print('earnings 8-Ks',len(ERN),[str(x.date()) for x in ERN[-6:]], ern.acc.tail(3).tolist())
put('F224',today(ERN)); put('F225',days_since(ERN,70))
# estimated next earnings: last earnings + median spacing (trading days) of prior earnings known at t
ep=np.sort(pos[pd.DatetimeIndex(ERN)].values); est=[]
for x in idx:
    i=pos[x]; prior=ep[ep<=i]
    if len(prior)<3: est.append(np.nan); continue
    sp_=int(np.median(np.diff(prior[-8:]))); nxt=prior[-1]+sp_; est.append(max(nxt-i,0))
est=pd.Series(est,index=idx,dtype=float); put('F226',est,'ESTIMATE: last earnings + median spacing; confirmed-date announcements not archived')
put('F227',(est<=5).astype(float).where(est.notna())); put('F228',today(sorted(set(k8.t.dropna()))))
q=ed[ed.form.isin(['10-Q','10-K'])]; QD=sorted(set(q.t.dropna())); put('F229',today(QD)); put('F230',days_since(QD,100))
f4=ed[ed.form.isin(['4','4/A'])]; cnt=f4.groupby('t').size().reindex(idx).fillna(0)
put('F231',cnt.rolling(5).sum()); put('F232',((F['F225']<=3)|(est<=3)).astype(float).where(F['F225'].notna()))
put('F233',days_since(sorted(set(k8.t.dropna())),60))
put('F234',days_since_true(cnt.rolling(5).sum()>=5,120),'cluster: >=5 Form 4 in 5 sessions, N=5 fixed a priori')

# ---------- Calendar ----------
nxt=pd.DatetimeIndex([TD[pos[x]+1] for x in idx]); prv=pd.DatetimeIndex([TD[pos[x]-1] for x in idx])
put('F235',idx.dayofweek,'integer 0-4; one-hot in models'); put('F236',(nxt-idx).days)
put('F237',np.sin(2*np.pi*idx.month/12),'sin component; cos added as F237_cos'); F['F237_cos']=pd.Series(np.cos(2*np.pi*idx.month/12),index=idx)
put('F238',(nxt.month!=idx.month).astype(float)); put('F239',(nxt.month!=idx.month).astype(float),'identical to F238 by construction')
nn=pd.DatetimeIndex([TD[pos[x]+2] for x in idx])
hol=lambda a,b: ((b-a).days>1)&~(((a.dayofweek==4)&((b-a).days==3)))
put('F240',hol(nxt,nn).astype(float)); put('F241',pd.Series([float(x in early) for x in nxt],index=idx))
def opex_dates():
    out=[]
    for yv in range(2021,2028):
        for m in range(1,13):
            d=pd.Timestamp(yv,m,1); fr=d+pd.Timedelta(days=(4-d.dayofweek)%7+14)
            j=TD.searchsorted(fr,side='right')-1; out.append(TD[j])
    return out
OP=opex_dates(); put('F242',nextday(OP)); put('F243',days_to(OP,30)); put('F244',days_to([d for d in OP if d.month in (3,6,9,12)],65))
me=[TD[i] for i in range(len(TD)-1) if TD[i+1].month!=TD[i].month]
def days_to_incl(dates,cap):
    p=np.sort(pos[pd.DatetimeIndex(dates)].values); return pd.Series([min(p[np.searchsorted(p,pos[x])]-pos[x],cap) for x in idx],index=idx,dtype=float)
put('F245',days_to_incl(me,30)); put('F246',days_to_incl([d for d in me if d.month in (3,6,9,12)],70)); put('F247',days_to_incl([d for d in me if d.month==12],60))
preh=[TD[i] for i in range(len(TD)-1) if hol(pd.DatetimeIndex([TD[i]]),pd.DatetimeIndex([TD[i+1]]))[0]]
posth=[TD[i+1] for i in range(len(TD)-1) if hol(pd.DatetimeIndex([TD[i]]),pd.DatetimeIndex([TD[i+1]]))[0]]
put('F248',days_to_incl(preh,30)); put('F249',days_since(posth,30))
# F216 event density next 5 sessions
evs=[CPI_D,PPI_D,JOB_D,JOL_D,PCE_D,fomc,mins,OP,preh]
dens=pd.Series(0.0,index=idx)
for E in evs:
    pe=set(pos[pd.DatetimeIndex([x for x in E if x in pos.index])].values)
    dens+=pd.Series([sum((pos[x]+k) in pe for k in range(1,6)) for x in idx],index=idx)
estpos=pd.Series([pos[x]+v for x,v in est.items()],index=idx)
dens+=((est>=1)&(est<=5)).astype(float)
put('F216',dens,'includes estimated NVDA earnings; holidays counted via pre-holiday session')
for f in ['F250','F251','F252','F253','F254','F255']: excl(f,"Optional: source NOT in inventory; no point-in-time announcement archive (R24) - excluded, not faked")

# ---------- target ----------
Cn=px('NVDA'); nextC=y['NVDA']['Close'].reindex(TD).shift(-1).reindex(idx)
X=pd.DataFrame(F); X['y']=(nextC>Cn).astype(float).where(nextC.notna()); X['ret_next']=np.log(nextC/Cn)
X['early_close_today']=[float(x in early) for x in idx]
X.to_parquet(f'{B}/features.parquet')
st=pd.DataFrame([(k,v[0],v[1]) for k,v in status.items()],columns=['id','status','note'])
st=spec[['Feature ID','Feature Name','Category','Priority']].rename(columns={'Feature ID':'id'}).merge(st,on='id',how='left')
st['status']=st.status.fillna('MISSING'); st.to_csv(f'{B}/build_status.csv',index=False)
print(st.status.value_counts()); print(X.loc['2024-01-02':].iloc[:, :255].isna().mean().sort_values().tail(15))
