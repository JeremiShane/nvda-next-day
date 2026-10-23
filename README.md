# NVDA next-day direction

Research project: predict whether NVDA closes higher on the next trading day, using only information available by **6:00 p.m. ET** on day *t*, and more broadly find which signals actually drive NVDA price action.

This repository is the feature specification and the selection study that was already run. It builds about 246 point-in-time features (catalog F001–F255, with a few excluded for missing sources) and runs a selection funnel on three development windows plus one locked holdout.

## Layout

| Path | Role |
|---|---|
| `pipeline/01_download.py` | Download free raw series into `data/raw/`. |
| `pipeline/01b_parallel.py` | Parallel helper for the Cboe daily put/call files (same destination as step 01). |
| `pipeline/02_features.py` | Build point-in-time features and the next-day target. |
| `pipeline/03_select_eval.py` | Quality, redundancy, univariate, multivariate, and economic-logic funnel, then model evaluation. |
| `pipeline/04_deliverables.py` | Write the results spreadsheet. |
| `pipeline/05_report_md.py` | Write the markdown report. |
| `pipeline/paths.py` | Data-root resolution (`data/` by default, `NVDA_DATA_DIR` to override). |
| `spec/` | Feature spec, data sources, leakage rules, target, notes, and `windows.json`. |

Market data and model outputs stay under `data/` and are gitignored. Yahoo Finance and Cboe series have licensing restrictions and must not be committed.

## Data sources

Inventory and URLs are in `spec/Data_Sources.csv`. The downloaders pull:

- **Yahoo Finance** (yfinance): NVDA and the market, sector, peer, mega-cap, bond, credit, commodity, and dollar ETFs, plus the Nasdaq Composite (`^IXIC`).
- **FRED** public graph CSVs: SP500, real and nominal yields, HY and IG OAS, the broad dollar index, and the fed funds target. No FRED API key.
- **ALFRED** vintages: CPI, core CPI, payrolls, unemployment, PCE, core PCE, PPI, and JOLTS, so monthly macro is first-release / as-of the vintage date.
- **Cboe**: VIX, VIX9D, VVIX, OVX, GVZ, and daily options market-statistics put/call ratios.
- **U.S. Treasury** daily par yield curve.
- **Coinbase Exchange** hourly candles for BTC-USD and ETH-USD, sampled at 18:00 ET.
- **SEC EDGAR** NVDA submissions (CIK 0001045810), timed on `acceptanceDateTime`.
- **FOMC, BLS, and BEA** HTML release calendars.

Nasdaq Trader consolidated volume (F128, F129, F131) and the optional peer-earnings / NVIDIA-events / index-rebalance calendars (F250–F255) were not obtained. Those features are excluded rather than filled in.

The scripts call public HTTP endpoints. They do not read an API key. SEC fair-access rules want a descriptive User-Agent; the downloaders send one. Nothing in the repo is a credential.

## Point-in-time and leakage rules

Full text: `spec/Leakage_and_Timing_Rules.csv`. The rule that everything else follows:

**Every feature for date *t* may use only information timestamped at or before 18:00 ET on trading day *t*.** The label is `1` if NVDA's next-session close is above the close on *t* (ties are 0). The last row has no label and is dropped. Features never include a *t+1* field.

Other rules the pipeline implements:

- Align on the NYSE trading calendar. Forward-fill non-equity series. Do not back-fill.
- FRED daily series lag one business day. The broad dollar index (weekly H.10) is usable from the following Monday. Monthly macro is usable only from its release date, using ALFRED vintages rather than today's revised history.
- Cboe put/call statistics are lagged one session because the historical posting time versus 18:00 ET was not verified. Treasury par yields are lagged one session for the same reason.
- BTC and ETH are the close of the hourly candle ending at 18:00 ET, not a UTC daily candle.
- EDGAR filings for day *t* are those accepted in `(18:00 ET on t−1, 18:00 ET on t]`.
- Forward-looking features are limited to dates known at *t* (FOMC, BLS schedules, holidays, options expiration). The next NVDA earnings date is an explicit estimate (last earnings plus median spacing), not a future announcement.
- Rolling z-scores, betas, and moving averages use trailing windows ending at *t*. Vol-scaled moves (`*_z60`) divide by the standard deviation over the trailing 60 observations ending at *t−1*.
- Fixed thresholds (5% move, VIX 30, Form 4 cluster size) are set in advance. Missing values stay missing.

Window definitions are in `spec/windows.json`.

| Window | Development start | Development end |
|---|---|---|
| A | 2024-01-02 | 2026-07-21 |
| B | 2024-11-06 (first session after the 2024 election) | 2026-07-21 |
| C | 2026-03-31 (session after the Mar 30, 2026 closing low in SPY, QQQ, NVDA, and SMH) | 2026-07-21 |

2026-07-22 is purged. The locked holdout, shared by all three windows, is 2026-07-23 through 2026-10-21 (64 sessions). Warm-up history from 2022 is used only to compute lookbacks.

## Setup

Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the direct packages from the environment that produced the study. Outputs go to `<repo>/data`. To put them somewhere else:

```bash
export NVDA_DATA_DIR=/path/to/data
```

A relative `NVDA_DATA_DIR` is resolved from the repo root.

## Run

From the repo root, in order. Steps 02–05 expect the previous step's files under `data/`.

```bash
python pipeline/01_download.py
python pipeline/01b_parallel.py
python pipeline/02_features.py
python pipeline/03_select_eval.py
python pipeline/04_deliverables.py
python pipeline/05_report_md.py
```

`01_download.py` already fetches Cboe put/call files one day at a time and skips anything on disk. `01b_parallel.py` is the same fetch with a thread pool, for when that loop is the slow part. Either script can run first; both leave files in `data/raw/cboe_pc/`.

The download walks several years of ALFRED vintages and Cboe daily files, and the selection step refits elastic-net and LightGBM models inside each walk-forward fold (with SHAP). Both steps take a long time.

## Outputs

All of these are written under `data/` (or `NVDA_DATA_DIR`) and are gitignored:

| File | Contents |
|---|---|
| `raw/` | Downloaded series and `download_log.json` (retrieval timestamps). |
| `features.parquet` | Feature matrix, `y`, `ret_next`, and an early-close flag. |
| `build_status.csv` | Built vs excluded feature, with notes. |
| `results.pkl` | Per-window funnel log, selected sets, metrics, calibration. |
| `summary.pkl` | Tables used by the report. |
| `NVDA_Feature_Selection_Results.xlsx` | Selected features, overlap, model comparison, fates, coverage. |
| `NVDA_Feature_Selection_Report.md` | Narrative report. |

## Status / findings so far

The study was run on 2026-10-23. None of the three windows (A from 2024-01-02, B from 2024-11-06, C from 2026-03-31; common 64-session holdout 2026-07-23 to 2026-10-21) showed a holdout edge over the base rate. Every holdout AUC interval includes 0.5 or sits below it. Reduced-set dev walk-forward scores (AUC about 0.59–0.74) are optimistically biased because selection was not nested within folds. On the holdout those reduced sets score about 0.25–0.48. The shorter windows invert.

The 11 features shared by windows A and B are the tentative core. The run describes them as:

- `hyg_lqd_ratio_chg_5d` (the only feature in all three final sets)
- short-horizon NVDA direction and volatility persistence (`ret_1d_sign`, `up_frac_5d`, `rvol_5d_chg_5d`)
- semiconductor peer read-through (AVGO and INTC 1-day; TSM 5-day)
- the dollar (UUP 1-day)
- options sentiment (index put/call, VIX put/call, equity put/call change)

Final sets barely overlap otherwise (Jaccard about 0.28 for A–B, 0.03 for A–C, 0.08 for B–C). Window A is the least-bad sample if work continues. It has the most rows and its holdout does not invert, and it still has no confirmed edge. These samples can detect only a large accuracy gap, so the results are a screen for instability.

Next ideas:

- A factor decomposition (market, semis, rates/dollar, credit) to separate drivers from symptoms.
- Less noisy targets: excess return versus SMH, next-day volatility or move size, and 3–5 day volatility-scaled returns. Alternatives are specified in `spec/Target.csv`.
- Nested selection, so development scores are not fit on the same folds used to pick features.
- A per-month rank information coefficient.
