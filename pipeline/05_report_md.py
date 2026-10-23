import pickle,json,pandas as pd,numpy as np
from paths import DATA_DIR, SPEC_DIR
B=str(DATA_DIR); o=pickle.load(open(f'{B}/results.pkl','rb')); s=pickle.load(open(f'{B}/summary.pkl','rb')); cfg=json.load(open(SPEC_DIR / 'windows.json'))
st=pd.read_csv(f'{B}/build_status.csv')
def tbl(df): 
    cols=list(df.columns); L=['| '+' | '.join(map(str,cols))+' |','|'+'---|'*len(cols)]
    for _,r in df.iterrows(): L.append('| '+' | '.join('' if (isinstance(v,float) and np.isnan(v)) else (f'{v:.3f}' if isinstance(v,float) else str(v)) for v in r.values)+' |')
    return '\n'.join(L)
nm=lambda w,f:o[w][0].at[f,'name']
md=[]
md.append('# NVDA Feature Selection Report\n\nNext-day direction (y = 1 if NVDA close[t+1] > close[t]); features as of 18:00 ET on day t. Selection study only: no production model and no daily routine. Prepared 2026-10-23 (CT). Code: `pipeline/` (01_download.py, 01b_parallel.py, 02_features.py, 03_select_eval.py, 04_deliverables.py, 05_report_md.py). Data and outputs: `data/` (override with NVDA_DATA_DIR).\n')
md.append('## 1. Bottom line\n')
md.append('- **No window shows a usable edge on the locked holdout.** On the 64-session holdout (2026-07-23 to 2026-10-21), no model beats the base rate with confidence. Every 95% bootstrap CI for holdout AUC includes 0.5 or sits below it. The best holdout figure, full-set logistic in window A (AUC 0.52, accuracy 57.8%), also has the worst log loss (0.86 vs 0.70 for the base rate), so it is miscalibrated noise and not an edge.\n- **The reduced sets look good on dev walk-forward, but they fail out of sample.** Dev walk-forward AUC was 0.59-0.74. Those numbers are optimistic because selection used the same validation folds. On the holdout, the reduced sets score AUC 0.25-0.48. In windows B and C the holdout AUC is **significantly below 0.5** (e.g., C logistic 0.25, CI 0.14-0.36). Relationships fitted on short recent windows reversed out of sample. That is overfitting plus regime instability. It is not a contrarian signal, and flipping it would mean iterating on the holdout.\n- **The selected features barely overlap across windows.** Jaccard overlap is A-B 0.28, A-C 0.03, B-C 0.08. Only **hyg_lqd_ratio_chg_5d** is in all three final sets, 10 more are shared by A and B, and 2 by B and C. That instability is what you expect when the target is mostly noise.\n- **Recommendation:** if work continues, use **window A** (longest, smallest detectable edge, least-bad holdout). Treat the 11 features common to A and B as the candidate core list, and expect next-day direction to be close to unpredictable with this catalog.\n')
md.append('## 2. Sample design\n')
md.append(f"- Warm-up data from 2022-01 is used **only** to compute lookbacks (SMA200, 252-day highs, z60, z252). No pre-window row is used to train or evaluate.\n- Windows (development periods):\n  - **A** 2024-01-02 to 2026-07-21 (639 rows)\n  - **B** 2024-11-06 (first session after the 2024 election) to 2026-07-21 (425 rows)\n  - **C** 2026-03-31 to 2026-07-21 (77 rows)\n- **Window C start date, chosen from the data:** {cfg['c_start_evidence']} There was a clear selloff, so the 2026-04-01 default was not used.\n- **Locked holdout, shared by all windows:** 2026-07-23 to 2026-10-21 (64 sessions; the 10-21 label uses the 10-22 close). 2026-07-22 is purged as an embargo. No selection or tuning touched the holdout, and each model was scored on it once.\n- **Walk-forward:** expanding folds with a 1-session gap. A: min train 250, validation blocks of 65 (6 folds, 388 OOS rows). B: 200/50 (4 folds, 200 OOS rows). C: 40/12 (3 folds, 36 OOS rows).\n- **Preprocessing inside each training fold:** winsorize at 1/99%, impute with the median, standardize.\n- **Models:**\n  - Base rate: training-fold up-frequency.\n  - Logistic regression: L2, C=0.05 (strong regularization).\n  - LightGBM: depth 2, 4 leaves, 150 trees at lr 0.02, L2=5, row and feature subsampling.\n- **Confidence intervals:** 95% moving-block bootstrap (block 5, 1,000 reps) for AUC and accuracy.\n")
md.append('## 3. Statistical power: what edge could be detected at all\n')
pw=pd.DataFrame([dict(window=w,OOS_rows=o[w][6]['val_rows'],min_detectable_acc_edge=f"+{o[w][6]['mde_acc_dev']*100:.1f} pts",min_detectable_AUC=f"{0.5+o[w][6]['mde_auc_dev']:.3f}") for w in 'ABC']+[dict(window='Holdout (all)',OOS_rows=64,min_detectable_acc_edge=f"+{o['A'][6]['mde_acc_holdout']*100:.1f} pts",min_detectable_AUC=f"{0.5+o['A'][6]['mde_auc_holdout']:.3f}")])
md.append(tbl(pw)+'\n\nMethod: two-sided 5% test with 80% power. Accuracy uses a binomial SE around 50%; AUC uses the Hanley-McNeil SE. Realistic daily-direction edges are 1-3 accuracy points (AUC about 0.51-0.53), far below anything this sample can detect. Even window A would need about a 7-point accuracy edge, and the 64-day holdout about 17.5 points. **None of these samples can confirm or reject a realistic edge.** Read every result here as a screen for instability, not as proof of signal.\n')
md.append('## 4. Data coverage and gaps\n')
cov=s['cov']; md.append(f"Feature build status: **{(st.status=='built').sum()} of 255 built** (plus a cos term for month_of_year, giving 247 model columns). **9 excluded:**\n- F128, F129, F131: Nasdaq Trader consolidated volume. No free, verifiable historical archive was obtained.\n- F250-F255: optional calendars with sources not in the inventory and no point-in-time announcement archive (rule R24).\n\nNo data was invented.\n")
md.append('''| Source | Coverage used | Notes |
|---|---|---|
| Yahoo Finance (25 tickers + ^IXIC) | 2022-01-03 to 2026-10-22 | Split/dividend adjusted. ARM starts at its 2023-09-14 IPO, so its z60 is valid well before 2024. |
| FRED daily (DFII10, DGS2, DGS10, DGS3MO, SP500, DFEDTARU) | full | Lagged 1 session (R03/R23). FRED SP500 only goes back to 2016-10. |
| FRED ICE BofA OAS (HY, IG) | 2023-10-24 onward only | FRED license limit. hy_oas_z252 is missing 29% of window A rows, so it is cut in A. |
| FRED DTWEXBGS | to 2026-10-16 | Usable from the Monday after the observation week (weekly H.10 release). |
| ALFRED vintages (CPI, core CPI, PAYEMS, UNRATE, PCE, core PCE, PPIFIS, JTSJOL) | vintages 2021-06 to 2026-10 | True point-in-time, first-release values. Release dates = vintages that add a new month. |
| Cboe VIX, VIX9D, VVIX, OVX, GVZ | full | Same-day official close. |
| Cboe put/call (daily market statistics JSON) | 2023-08-01 to 2026-10-22, 811/811 days | Legacy CSV archive ends in 2019. Lagged 1 session because historical posting time vs 18:00 ET is unverified (R09). |
| U.S. Treasury par curve | 2022 to 2026-10-23 | Lagged 1 session (R10). |
| Coinbase BTC/ETH hourly | 2022-01 to 2026-10-23 | Exact 18:00 ET and 16:00 ET prices from hourly candle closes, DST-aware. No approximation needed. |
| SEC EDGAR NVDA submissions | 2020-09 onward | acceptanceDateTime converted to ET. Earnings 8-Ks are accepted about 20:20-21:30 ET, after the cutoff, so earnings_8k_today flags the reaction day t+1 (R12). |
| FOMC calendar (federalreserve.gov) | 2021-2027, 56 decisions | Aug-2025 notation vote excluded. Minutes date derived as decision + 21 days. |
| BLS schedules / BEA | future dates for CPI, jobs, PPI, JOLTS | BEA future PCE dates were not parsed, so days_to_next_pce is NaN for the last ~17 holdout rows (imputed). |
| Nasdaq Trader volume | **not obtained** | F128, F129, F131 excluded. |
| Peer/hyperscaler earnings, NVIDIA events, index rebalances | **not obtained** | F250-F255 excluded. |
''')
md.append('**Point-in-time caveats:**\n- Yahoo adjusted history is restated after splits. This does not affect returns or ratios.\n- Scheduled-release "next day" flags use actual release dates. In autumn 2025 the shutdown-delayed BLS releases were not known in advance, so a few of those flags are mildly forward-looking.\n- NVDA next-earnings dates are an estimate (last date + median spacing), because no archive of confirmed-date announcements exists. The estimate runs about 5 sessions late.\n')
md.append('## 5. Selection funnel\n')
md.append('''**Stages:**
- **a. Data quality:** cut if >10% missing in the window, constant, a binary flag with fewer than 10 events, top value above 95%, or a monthly macro series with fewer than 12 distinct values (a time proxy). F239 is identical to F238 by construction.
- **b. Redundancy:**
  - Clusters at |Spearman| > 0.85 (average linkage). Keep one member, preferring higher spreadsheet priority, then fewer missing values, then the simpler formula.
  - Then iterative VIF < 10.
- **c. Univariate:**
  - Walk-forward AUC, with the sign fixed on the training fold.
  - Spearman IC vs next-day return and its t-stat.
  - Sign consistency across quarters (months in C) and high vs low VIX.
  - Cut if there is no WF signal (AUC ≤ 0.5 and |IC t| < 1), or if AUC < 0.5 and the feature is unstable.
- **d. Multivariate, inside training folds:**
  - Elastic-net logistic stability selection (l1_ratio 0.7, C=0.1; 30 half-subsamples per fold, 50 in C).
  - LightGBM permutation importance (log-loss increase) and mean |SHAP|, both on validation folds.
  - Cut if selection frequency < 0.2, permutation importance ≤ 0, and WF AUC < 0.52.
- **e. Economic logic:** each survivor gets a plain-language rationale. Slow levels and calendar counters are flagged "weak" and cut unless they are in the top quartile of the combined score. Statistical-only survivors never enter a final set; at most 3 go to reserve.
- **Final set:** ranked by the combined score (mean percentile rank of selection frequency, permutation importance, |SHAP| and WF AUC), with a penalty for stability flags. Final size is 25 for A/B and 15 for C, plus a reserve list.
''')
fc=pd.DataFrame({w:o[w][1] for w in 'ABC'}).rename(index={'built':'built (model columns)','a_quality':'after a. data quality','b_redundancy':'after b. redundancy','c_univariate':'after c. univariate','d_multivariate':'after d. multivariate','e_econ':'after e. economic logic','final':'final set','reserve':'reserve'}).reset_index().rename(columns={'index':'stage'})
md.append(tbl(fc)+'\n\nAll 255 catalog features and their per-window fates are in the Sheet\'s Feature_Fates tab. In window C, 77 rows make most correlations look high, so redundancy cuts 174 features.\n')
md.append('## 6. Final feature sets\n')
for w,lab in [('A','Window A (2024-01-02+), recommended'),('B','Window B (2024-11-06+)'),('C','Window C (2026-03-31+), indicative only: 77 dev rows')]:
    log=o[w][0]; f=o[w][2]
    d=log.loc[f,['name','category','wf_auc','ic','stab_freq','econ_logic']].reset_index().rename(columns={'index':'id','stab_freq':'enet_freq'})
    d['econ_logic']=d.econ_logic.str.slice(0,110)
    md.append(f'### {lab}\n\n'+tbl(d)+f"\n\nReserve: {', '.join(nm(w,x) for x in o[w][3])}\n")
ov=s['ov']; md.append('### Overlap across windows (the most robust candidates)\n\n'+tbl(ov[ov.n_windows_final>=2][['id','name','category','in_A','in_B','in_C']])+'\n\nJaccard overlap of the final sets: A-B 0.28, A-C 0.03, B-C 0.08. Only hyg_lqd_ratio_chg_5d (5-day change in the HYG/LQD ratio, i.e. high-yield vs investment-grade credit risk appetite) is final in all three windows.\n\nThe A∩B themes are economically sensible:\n- short-horizon NVDA direction and volatility persistence (ret_1d_sign, up_frac_5d, rvol_5d_chg_5d)\n- semiconductor peer read-through (AVGO, INTC 1d; TSM 5d)\n- dollar (UUP 1d)\n- options sentiment (index and VIX put/call, equity P/C change)\n\nNone of them survives as an out-of-sample edge.\n')
md.append('## 7. Model results\n\nAUC and accuracy show 95% block-bootstrap CIs. Base rate = training up-frequency. On the holdout it predicts "up" every day, which scores 48.4% because the holdout up-rate was 48.4% (majority class "down" = 51.6%).\n')
r=s['res'].copy(); r['AUC [95% CI]']=r.apply(lambda x:f"{x.auc:.3f} [{x.auc_lo:.2f}, {x.auc_hi:.2f}]",axis=1); r['Acc [95% CI]']=r.apply(lambda x:f"{x.acc:.3f} [{x.acc_lo:.2f}, {x.acc_hi:.2f}]",axis=1)
r['Acc - base']=r.acc_minus_base.map(lambda v:f'{v:+.3f}'); r['LogLoss']=r.logloss.map(lambda v:f'{v:.3f}'); r['Brier']=r.brier.map(lambda v:f'{v:.3f}')
for sp in ['dev walk-forward','holdout']:
    md.append(f'### {sp}\n\n'+tbl(r[r.split==sp][['window','model','n_feats','n','AUC [95% CI]','Acc [95% CI]','Acc - base','LogLoss','Brier']])+'\n')
md.append('''**How to read these results:**
- **Full feature sets:** in all windows these are worse than the base rate on log loss and Brier (e.g., A logistic log loss 0.90 dev / 0.86 holdout vs 0.69-0.70). Even with C=0.05, 247 features overwhelm 250-600 training rows, and the probabilities are overconfident.
- **Reduced sets on dev:** these look like an edge (A: AUC 0.59, accuracy +4.9 pts). This is selection bias, because features were chosen using the same validation folds.
- **Reduced sets on the holdout:** the edge disappears or inverts. Shorter windows invert more:
  - A: AUC 0.48 / 0.44
  - B: 0.41 / 0.34
  - C: 0.25 / 0.26
- **Calibration:** the Calibration tab has quintile tables. On the holdout, quintile-mean predicted probabilities span about 0.38-0.65 for LightGBM and A's reduced logistic, but 0.19-0.94 for the full logistic models, which are overconfident. Realized up-rates per quintile do not rise with predicted probability.
- **Base rate dev AUC below 0.5:** this is an artifact of pooling per-fold constant predictions. It is not signal.
''')
md.append('## 8. Window comparison and recommendation\n')
md.append('''| | A (2024-01-02+) | B (2024-11-06+) | C (2026-03-31+) |
|---|---|---|---|
| Dev rows / OOS rows | 639 / 388 | 425 / 200 | 77 / 36 |
| Min detectable edge (dev) | +7.1 pts acc, AUC 0.58 | +9.9 pts, AUC 0.61 | +23 pts, AUC 0.77 |
| Survivors before final | 61 | 45 | 22 |
| Best dev AUC (reduced) | 0.59 | 0.62 | 0.74 |
| Holdout AUC, logit / LGBM (reduced) | 0.48 / 0.44 | 0.41 / 0.34 | 0.25 / 0.26 |
| Holdout log loss, logit reduced (base about 0.70) | 0.718 | 0.798 | 0.947 |

**Recommendation: window A.**
- It has the most data and therefore the only remotely usable power.
- Its holdout degradation is the smallest; its CIs include 0.5, so it shows no edge, but it does not invert.
- Its selection is the least idiosyncratic.

**Why not B or C:**
- B and C fit transient relationships that reversed in the holdout.
- **C's selection and results are indicative only and highly uncertain**: 77 dev rows, 36 OOS validation rows, a minimum detectable AUC of about 0.77, and features chosen largely by chance. The very strong dev numbers in C (AUC 0.74) next to a holdout AUC of 0.25 show this directly.

To favour the recent regime as asked, prefer A-selected features that B also selects (the A∩B list). Those are the ones stable through the post-election period.
''')
md.append('## 9. Caveats\n\n- Sample sizes are tiny for a nearly-random binary target. The holdout (64 days) can only detect a huge edge.\n- Dev walk-forward scores for reduced sets are optimistically biased. Nested selection (re-running the funnel inside each outer fold) was not run because of compute time. The holdout is the honest number.\n- Put/call features are lagged a day out of caution. If Cboe posting before 18:00 ET can be verified, same-day values could be tested.\n- Earnings-day handling follows the 8-K acceptance rule. The press release (~16:20 ET) actually precedes the 18:00 cutoff, so a scheduled-announcement-date flag could be a legitimate refinement.\n- The 2025 government-shutdown release delays make a handful of "next-day release" flags mildly forward-looking.\n- Threshold choices (0.85 correlation, VIF 10, missing above 10%, regularization strengths) were fixed a priori and not tuned. Different reasonable choices would change the specific lists, which is itself evidence that the lists are fragile.\n')
md.append('## 10. Recommended next steps\n\n1. Do not build a production direction model on this evidence. If proceeding, treat it as a research baseline and use window A with the 11-feature A∩B core list (which includes hyg_lqd_ratio_chg_5d).\n2. Get more data. Extending the sample back (e.g., 2019+ or 2015+) is the only way to reach the power needed for 1-3-point edges. The 2024+ restriction makes it statistically impossible to validate a realistic edge.\n3. Consider less noisy targets from the Target tab, then re-run this funnel unchanged:\n   - excess return vs SMH or QQQ\n   - thresholded moves (|ret| > k·σ)\n   - next-day volatility or absolute return (usually far more predictable than direction)\n4. Run fully nested selection so dev estimates are unbiased, and extend the holdout forward as new days arrive. Evaluate once per quarter and never iterate on it.\n5. Close data gaps: verify Cboe P/C posting times, add BEA future PCE dates and NVDA confirmed earnings-date announcements, and look for a Nasdaq volume archive.\n')
open(f'{B}/NVDA_Feature_Selection_Report.md','w').write('\n'.join(md)); print(len('\n'.join(md)))
