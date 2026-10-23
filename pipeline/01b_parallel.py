import os,requests,time,pandas_market_calendars as mcal
from concurrent.futures import ThreadPoolExecutor
from paths import RAW_DIR
RAW=str(RAW_DIR); os.makedirs(f'{RAW}/cboe_pc',exist_ok=True); UA={'User-Agent':'NVDA feature research (personal study) research@example.com'}
days=mcal.get_calendar('XNYS').valid_days('2023-08-01','2026-10-22').strftime('%Y-%m-%d')
def get(day):
    q=f'{RAW}/cboe_pc/{day}.json'
    if os.path.exists(q): return 1
    for k in range(3):
        try:
            r=requests.get(f'https://cdn.cboe.com/data/us/options/market_statistics/daily/{day}_daily_options',headers=UA,timeout=30)
            if r.ok and r.text.strip().startswith('{'): open(q,'w').write(r.text); return 1
            if r.status_code in (403,404): return 0
        except Exception: time.sleep(1)
    return 0
with ThreadPoolExecutor(6) as ex: res=list(ex.map(get,days))
print('cboe ok',sum(res),'of',len(days)); print([d for d,r in zip(days,res) if not r])
