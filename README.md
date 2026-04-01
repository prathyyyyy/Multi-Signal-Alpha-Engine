# Multi-Signal Alpha Engine

**Institutional-grade quantitative trading system built on Azure Databricks**

An end-to-end ML/data engineering pipeline that ingests raw market data, engineers 100+ alpha features, trains three complementary machine learning models, fuses their signals into a regime-weighted ensemble, and validates the strategy across a 33-year backtest spanning 520 US equities.

| Net Sharpe | Annual Return | Max Drawdown | Sortino | ICIR | Hit Rate |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **6.15** | **19.4%** | **-10.3%** | **11.53** | **0.623** | **73.3%** |

> 33 years · 520 equities · 100% positive calendar years · Production score 7/7

---

## Architecture

<img width="2000" height="1200" alt="architecture_diagram" src="https://github.com/user-attachments/assets/75aef070-9ccb-45b1-a1dc-6000d429364b" />

---

## Project Structure

```
├── Bronze/                  # Raw ingestion notebooks
│   ├── bronze_ohlcv
│   ├── bronze_macro
│   ├── bronze_options
│   ├── bronze_sentiment
│   └── bronze_sec_filings
├── Silver/                  # Data cleaning & alignment
│   └── silver_clean_all
├── Gold/                    # Feature engineering
│   ├── gold_01_price_factors       # 104 features: momentum, vol, RSI, Sharpe
│   ├── gold_02_vol_surface         # 72 features: ATM IV, skew, term structure, VRP
│   ├── gold_03_macro_regime        # HMM regime labels from macro factors
│   ├── gold_04_sentiment           # News sentiment, 59 tickers, rolling aggregations
│   └── gold_05_pairs               # 13 cointegrated pairs from 100 tested
├── ML/                      # Machine learning models
│   ├── ml_01_hmm_regime            # 3-state Gaussian HMM regime detection
│   ├── ml_02_lgbm_alpha            # LightGBM GPU + Optuna HPO (50 trials)
│   ├── ml_03_patchtst_vol          # PatchTST Transformer volatility forecaster
│   └── ml_04_ensemble              # Regime-weighted signal fusion
├── Backtest/                # Strategy validation
│   ├── bt_01_daily_returns         # Quintile L/S portfolio backtest
│   ├── bt_02_risk_summary          # VaR, CVaR, factor exposure, drawdown analysis
│   └── bt_03_signal_attribution    # IC decay, regime decomposition, signal contrib
├── EDA/                     # Exploratory data analysis
│   ├── eda_01_universe_profiling
│   └── eda_02_stationarity_cointegration
└── Report/
    └── MultiSignal_Alpha_Engine_Report.pdf
```

---

## Lakehouse Architecture

The system follows a strict **medallion architecture** (Bronze → Silver → Gold → ML → Backtest), persisted as Delta Lake tables on ADLS Gen2.

| Layer | Purpose | Key Outputs |
|-------|---------|-------------|
| **Bronze** | Raw ingestion | 9 tables — OHLCV, macro, options, sentiment, SEC filings, earnings, CBOE IV |
| **Silver** | Cleaning & alignment | 3.5M rows — winsorised, survivorship-bias-free, partitioned Delta tables |
| **Gold** | Feature engineering | 100+ features across price (104 cols), vol surface (72 cols), macro regime, sentiment, pairs |
| **ML** | Model training & inference | HMM regime labels, LightGBM predictions, PatchTST vol forecasts, ensemble signals |
| **Backtest** | Strategy validation | 15+ result tables — daily returns, risk metrics, signal attribution |

---

## Machine Learning Models

### ML01 — Hidden Markov Model (Regime Detection)
A 3-state Gaussian HMM trained on macro and volatility features identifies Bull, HighVol, and Bear market regimes with 99% average confidence and 2.39 Sharpe separation between states. Regime probabilities drive soft position weighting in the ensemble.

### ML02 — LightGBM (Cross-Sectional Alpha)
A 5-seed LightGBM ensemble with GPU acceleration and Optuna HPO (50 trials, TPE sampler) produces cross-sectional alpha rankings. Walk-forward validation across 8 folds ensures time-series integrity. Achieves IC of +0.093 and net Sharpe of 7.31.

### ML03 — PatchTST (Volatility Forecasting)
A 3-seed PatchTST Transformer ensemble forecasts 21-day forward volatility with 0.81 correlation. Uses RevIN normalisation, cross-sectional attention, percentile-weighted Huber loss, and AMP mixed precision on T4 GPU. Lazy dataset loading reduced memory from 57 GB to ~4 GB.

### ML04 — Regime-Weighted Ensemble
Fuses all three model outputs through alpha blending (0.70 LightGBM + 0.30 vol signal), soft HMM regime weighting (Bull 1.0 / HighVol 0.6 / Bear 0.3), and vol-adjusted position sizing. Ensemble improves ICIR by +11% over standalone LightGBM.

---

## Backtest Results

**Full portfolio** — quintile long/short across 520 stocks, 1993–2026, with 5 bps turnover-based transaction costs.

| Metric | L/S Net | Long Only |
|--------|---------|-----------|
| Ann. Return | 19.4% | 25.7% |
| Sharpe Ratio | 6.15 | 5.28 |
| Sortino Ratio | 11.53 | 6.86 |
| Max Drawdown | -10.3% | -35.1% |

**Regime attribution** — the strategy performs best in Bear markets (Sharpe 8.11, IC +0.112), demonstrating genuine downside alpha generation.

| Regime | Sharpe | Ann. Return | Mean IC |
|--------|--------|-------------|---------|
| Bull | 5.61 | 13.8% | +0.079 |
| HighVol | 4.98 | 15.2% | +0.070 |
| Bear | 8.11 | 30.6% | +0.112 |

**Signal half-life** of ~10 days places the strategy in the medium-frequency statistical arbitrage space. IC remains positive at 63-day horizon.

<img width="1049" height="500" alt="newplot(20)" src="https://github.com/user-attachments/assets/c3e46049-6a48-47e5-bea9-bd15a727ea03" />
<img width="1049" height="748" alt="newplot(21)" src="https://github.com/user-attachments/assets/7cd18049-fc20-45bd-8f18-6c60758d7854" />
<img width="1049" height="900" alt="newplot(1)" src="https://github.com/user-attachments/assets/e1562b56-d03b-41fb-9bb0-26624544386e" />
<img width="1049" height="800" alt="newplot(2)" src="https://github.com/user-attachments/assets/d9e4525a-be81-46c5-ba7b-55272477d748" />
<img width="1049" height="700" alt="newplot(3)" src="https://github.com/user-attachments/assets/1c3c2273-5216-42f4-b676-2d75698c2003" />
<img width="1049" height="500" alt="newplot(4)" src="https://github.com/user-attachments/assets/378d4c7e-9093-4cbf-8ef7-edf9db4d95d3" />
<img width="1049" height="748" alt="newplot(5)" src="https://github.com/user-attachments/assets/40827e20-c187-4bd2-b78d-aca6d403457c" />
<img width="1049" height="700" alt="newplot(6)" src="https://github.com/user-attachments/assets/aa4497fc-ccc9-483a-9594-683965b5761e" />
<img width="1049" height="700" alt="newplot(7)" src="https://github.com/user-attachments/assets/e101c6c4-5208-40fb-829f-64242500a044" />
<img width="1049" height="750" alt="newplot(8)" src="https://github.com/user-attachments/assets/730b2286-a5a8-4c00-bdae-b0dfbfafd9cb" />
<img width="1049" height="800" alt="newplot(9)" src="https://github.com/user-attachments/assets/1c9530b4-dd99-4491-81ef-7467aca04576" />
<img width="1049" height="700" alt="newplot(10)" src="https://github.com/user-attachments/assets/02a8ff5c-3036-4a3d-8413-56c58c7b18a5" />
<img width="1049" height="800" alt="newplot(11)" src="https://github.com/user-attachments/assets/95c4c733-6185-4f97-9863-944a2dceaa99" />
<img width="1049" height="750" alt="newplot(12)" src="https://github.com/user-attachments/assets/504429c0-ff0e-4670-9d3b-699bf8dfba56" />
<img width="1049" height="700" alt="newplot(13)" src="https://github.com/user-attachments/assets/a6b28893-7e5b-4a1c-905b-cb4395a01532" />
<img width="1049" height="700" alt="newplot(14)" src="https://github.com/user-attachments/assets/698783ac-2e68-47c1-a9fc-5f704f39d937" />
<img width="1049" height="500" alt="newplot(15)" src="https://github.com/user-attachments/assets/5ec8948a-efc8-46a9-879f-2e73a1d9fa1f" />
<img width="1049" height="700" alt="newplot(16)" src="https://github.com/user-attachments/assets/f6e5336f-bbf4-481a-ae89-dc941451de18" />
<img width="1049" height="700" alt="newplot(17)" src="https://github.com/user-attachments/assets/5b01cac9-2a8d-4b07-bf2d-821369356f0c" />
<img width="1049" height="500" alt="newplot(18)" src="https://github.com/user-attachments/assets/0d1a9bcb-19cf-4b59-ae36-cc4b661bdb90" />
<img width="1049" height="500" alt="newplot(19)" src="https://github.com/user-attachments/assets/558bc7dd-c1a7-4399-8757-c210439571f9" />


---

## Key Engineering Decisions

**Performance optimisations:**
- Feature engineering speedup: per-date groupby (508s) → global median/percentile operations (30s) — **17x faster**
- PatchTST memory: eager tensor storage (57 GB crash) → lazy dataset with on-demand windowing (~4 GB)
- LightGBM: CPU → T4 GPU training — **>5x faster**
- Batch inference: per-ticker Python loop → vectorised stride tricks — **~100x faster**
- Mixed precision: FP32 → AMP (FP16 + FP32) — **~2x throughput**

**Critical bugs resolved:**
- Optuna silent parameter ignoring (short aliases like `lr` vs full `learning_rate`)
- Transaction cost model (full-position daily TC gave Sharpe -1.46 → turnover-based TC restored to 7.31)
- HMM early stopping collapse (iter=1 destroying regime separation)
- PatchTST CUDA/CPU device mismatch in loss computation
- OOM on 2.6M samples (57 GB tensor allocation)

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Storage | Delta Lake on ADLS Gen2 |
| Compute | Azure Databricks (Spark 3.x) |
| Gradient Boosting | LightGBM GPU + Optuna HPO |
| Deep Learning | PyTorch + PatchTST Transformer |
| Regime Detection | hmmlearn (3-state Gaussian HMM) |
| Feature Engineering | PySpark + pandas |
| Mixed Precision | torch.cuda.amp |
| Visualisation | Plotly (dark theme) |
| Cluster — CPU | D4s_v3 |
| Cluster — Photon | D8s_v3 |
| Cluster — GPU | NC16as T4 v3 (110 GB) |

---

## Risk Metrics

| Metric | Value |
|--------|-------|
| VaR 95% | -0.153% |
| CVaR 95% | -0.231% |
| VaR 99% | -0.264% |
| Market Beta | 0.3175 |
| Alpha/yr | 11.64% |
| Information Ratio | 1.095 |
| Tail Ratio | 2.186 |
| Positive Years | 33/33 (100%) |

---

## Limitations & Future Work

- **RAPIDS cuDF** integration deferred due to Numba/pynvjitlink conflicts
- **Pairs trading signals** (13 identified pairs) not yet integrated into ML04 ensemble
- **Walk-forward** covers 8 folds; monthly rebalancing would stress-test further
- **Options flow** and **earnings-event overlays** could improve IC around catalyst dates
- **Live trading** integration (OMS, execution, real-time feeds) is the next engineering milestone

---

## Author

**Prathy P**
- Email: csprathyy@gmail.com
- LinkedIn: [Prathy P](https://www.linkedin.com/in/prathy-p)
- GitHub: [prathyyyyy](https://github.com/prathyyyyy)

---

<p align="center"><i>Built on Azure Databricks · Delta Lake · PyTorch · LightGBM</i></p>
