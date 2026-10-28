"""Download raw data for NVDA feature-selection study. Raw files -> raw/. Retrieval timestamps logged."""
import os, io, json, time, datetime as dt, re
import pandas as pd, numpy as np, requests, yfinance as yf
from paths import RAW_DIR
RAW=str(RAW_DIR); os.makedirs(RAW,exist_ok=True)
_CONTACT=os.environ.get('NVDA_USER_AGENT_EMAIL','').strip()
if not _CONTACT:
    _CONTACT='contact-not-set@example.invalid'
    print('Set NVDA_USER_AGENT_EMAIL to your own contact email before downloading. SEC EDGAR requires that contact in the User-Agent; the fallback is not a real address and does not meet the policy.')
UA={'User-Agent':f'NVDA feature research (personal study) {_CONTACT}'}
START='2022-01-01'; END='2026-10-23'   # warm-up from 2022; modeling from 2024-01-02
log=[]
def note(src,**k): k['source']=src; k['retrieved']=dt.datetime.now().isoformat(timespec='seconds'); log.append(k); print(k)

# 1-7 Yahoo
tick=['NVDA','SPY','QQQ','DIA','IWM','SMH','SOXX','AMD','AVGO','TSM','INTC','MU','ARM','MSFT','GOOGL','AMZN','META','AAPL','TLT','IEF','HYG','LQD','GLD','USO','UUP','^IXIC']
if not os.path.exists(f'{RAW}/yahoo.parquet'):
    d=yf.download(tick,start=START,end=END,auto_adjust=True,actions=False,progress=False,group_by='ticker',threads=True)
    d.to_parquet(f'{RAW}/yahoo.parquet')
d=pd.read_parquet(f'{RAW}/yahoo.parquet')
for t in tick:
    s=d[t]['Close'].dropna(); note('yahoo',series=t,start=str(s.index.min().date()),end=str(s.index.max().date()),n=len(s))

# FRED daily + monthly current
fred=['SP500','DFII10','DGS2','DGS10','BAMLH0A0HYM2','BAMLC0A0CM','DTWEXBGS','DFEDTARU','DGS3MO']
for s in fred:
    p=f'{RAW}/fred_{s}.csv'
    if not os.path.exists(p):
        r=requests.get(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}',headers=UA,timeout=60); open(p,'w').write(r.text)
    x=pd.read_csv(p); x=x[pd.to_numeric(x.iloc[:,1],errors='coerce').notna()]
    note('fred',series=s,start=x.iloc[0,0],end=x.iloc[-1,0],n=len(x))

# ALFRED vintages for monthly macro (point-in-time) + release-date proxies
alf=['CPIAUCSL','CPILFESL','PAYEMS','UNRATE','PCEPI','PCEPILFE','PPIFIS','JTSJOL']
for s in alf:
    p=f'{RAW}/alfred_{s}_vintages.json'
    if not os.path.exists(p):
        h=requests.get(f'https://alfred.stlouisfed.org/series/downloaddata?seid={s}',headers=UA,timeout=60).text
        vd=sorted(set(re.findall(r'\d{4}-\d{2}-\d{2}',h)))
        json.dump(vd,open(p,'w'))
    vd=[v for v in json.load(open(p)) if '2021-06-01'<=v<=END]
    os.makedirs(f'{RAW}/alfred_{s}',exist_ok=True)
    for v in vd:
        q=f'{RAW}/alfred_{s}/{v}.csv'
        if os.path.exists(q): continue
        r=requests.get(f'https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={s}&vintage_date={v}',headers=UA,timeout=60)
        if r.ok and r.text.startswith('observation_date'): open(q,'w').write(r.text)
        time.sleep(0.2)
    note('alfred',series=s,vintages=len(vd),first=vd[0] if vd else None,last=vd[-1] if vd else None)

# Cboe vol indices
for s in ['VIX','VIX9D','VVIX','OVX','GVZ']:
    p=f'{RAW}/cboe_{s}.csv'
    if not os.path.exists(p):
        r=requests.get(f'https://cdn.cboe.com/api/global/us_indices/daily_prices/{s}_History.csv',headers=UA,timeout=60); open(p,'w').write(r.text)
    x=pd.read_csv(p); note('cboe',series=s,start=x.iloc[0,0],end=x.iloc[-1,0],n=len(x))

# Cboe daily options market statistics (put/call), 2023-08 onward
os.makedirs(f'{RAW}/cboe_pc',exist_ok=True)
import pandas_market_calendars as mcal
days=mcal.get_calendar('XNYS').valid_days('2023-08-01','2026-10-22').strftime('%Y-%m-%d')
miss=[]
for day in days:
    q=f'{RAW}/cboe_pc/{day}.json'
    if os.path.exists(q): continue
    try:
        r=requests.get(f'https://cdn.cboe.com/data/us/options/market_statistics/daily/{day}_daily_options',headers=UA,timeout=30)
        if r.ok and r.text.strip().startswith('{'): open(q,'w').write(r.text)
        else: miss.append(day)
    except Exception as e: miss.append(day)
    time.sleep(0.1)
note('cboe_pc',days_requested=len(days),missing=len(miss),missing_list=miss[:30])

# Treasury par yield curve
fr=[]
for y in range(2022,2027):
    p=f'{RAW}/treasury_{y}.csv'
    if not os.path.exists(p):
        r=requests.get(f'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{y}/all?type=daily_treasury_yield_curve&field_tdr_date_value={y}&page&_format=csv',headers=UA,timeout=60); open(p,'w').write(r.text)
    fr.append(pd.read_csv(p))
t=pd.concat(fr); note('treasury',start=str(pd.to_datetime(t.Date).min().date()),end=str(pd.to_datetime(t.Date).max().date()),n=len(t))

# Coinbase hourly candles
for prod in ['BTC-USD','ETH-USD']:
    p=f'{RAW}/coinbase_{prod}_1h.csv'
    if not os.path.exists(p):
        rows=[]; s=pd.Timestamp('2022-01-01',tz='UTC'); e=pd.Timestamp('2026-10-23 06:00',tz='UTC')
        while s<e:
            en=min(s+pd.Timedelta(hours=300),e)
            for k in range(5):
                r=requests.get(f'https://api.exchange.coinbase.com/products/{prod}/candles',params={'granularity':3600,'start':s.isoformat(),'end':en.isoformat()},headers=UA,timeout=30)
                if r.ok: rows+=r.json(); break
                time.sleep(2)
            s=en; time.sleep(0.15)
        c=pd.DataFrame(rows,columns=['time','low','high','open','close','volume']).drop_duplicates('time').sort_values('time')
        c.to_csv(p,index=False)
    c=pd.read_csv(p); note('coinbase',series=prod,start=str(pd.to_datetime(c.time.min(),unit='s')),end=str(pd.to_datetime(c.time.max(),unit='s')),n=len(c))

# SEC EDGAR
p=f'{RAW}/edgar_nvda.json'
if not os.path.exists(p):
    open(p,'w').write(requests.get('https://data.sec.gov/submissions/CIK0001045810.json',headers=UA,timeout=60).text)
note('edgar',file='submissions recent (1000 filings, back to 2020-09)')

# FOMC calendar + BLS schedules (HTML snapshots)
for name,url in [('fomc_calendar','https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm'),
                 ('bls_sched_cpi','https://www.bls.gov/schedule/news_release/cpi.htm'),
                 ('bls_sched_empsit','https://www.bls.gov/schedule/news_release/empsit.htm'),
                 ('bls_sched_ppi','https://www.bls.gov/schedule/news_release/ppi.htm'),
                 ('bls_sched_jolts','https://www.bls.gov/schedule/news_release/jolts.htm'),
                 ('bea_schedule','https://www.bea.gov/news/schedule')]:
    p=f'{RAW}/{name}.html'
    if not os.path.exists(p):
        r=requests.get(url,headers={'User-Agent':f'Mozilla/5.0 (research script; {_CONTACT})'},timeout=60); open(p,'w').write(r.text)
        note('html',name=name,status=r.status_code,bytes=len(r.text))
json.dump(log,open(f'{RAW}/download_log.json','a'),indent=1)
