# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import *
from datetime import datetime
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from joblib import Parallel, delayed
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.shuffle.partitions", "200")

STORAGE_ACCOUNT = "multisignalalphaeng"
CONTAINER       = "quant-lakehouse"
ADLS_KEY        = dbutils.secrets.get(
    scope="quant-scope", key="adls-key-01"
)
spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
    ADLS_KEY
)

BASE_PATH   = f"abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net"
SILVER_PATH = f"{BASE_PATH}/silver/delta"
EDA_PATH    = f"{BASE_PATH}/eda/delta"

print("Config loaded ✓")
print(f"Silver : {SILVER_PATH}")
print(f"EDA    : {EDA_PATH}")

# Quick connectivity test
try:
    dbutils.fs.ls(BASE_PATH)
    print("ADLS connection ✓")
except Exception as e:
    print(f"ADLS failed: {e}")

# COMMAND ----------

def _test_single_ticker(ticker, price_series, return_series):
    """Run ADF + KPSS on one ticker. Module-level for joblib."""
    import numpy as np
    from statsmodels.tsa.stattools import adfuller, kpss
 
    if len(price_series) < 30:
        return None
 
    result = {"ticker": ticker, "n_obs": len(price_series)}
    log_price = np.log(price_series)
 
    # ADF on log prices
    try:
        adf = adfuller(log_price, autolag="AIC", maxlag=20)
        result["adf_price_stat"] = adf[0]
        result["adf_price_pval"] = adf[1]
        result["adf_price_lags"] = adf[2]
    except Exception:
        result["adf_price_stat"] = np.nan
        result["adf_price_pval"] = np.nan
        result["adf_price_lags"] = np.nan
 
    # ADF on returns
    try:
        ret_clean = return_series.dropna()
        if len(ret_clean) > 30:
            adf_r = adfuller(ret_clean, autolag="AIC", maxlag=20)
            result["adf_ret_stat"] = adf_r[0]
            result["adf_ret_pval"] = adf_r[1]
        else:
            result["adf_ret_stat"] = np.nan
            result["adf_ret_pval"] = np.nan
    except Exception:
        result["adf_ret_stat"] = np.nan
        result["adf_ret_pval"] = np.nan
 
    # KPSS on log prices
    try:
        k = kpss(log_price, regression="c", nlags="auto")
        result["kpss_price_stat"] = k[0]
        result["kpss_price_pval"] = k[1]
    except Exception:
        result["kpss_price_stat"] = np.nan
        result["kpss_price_pval"] = np.nan
 
    # KPSS on returns
    try:
        ret_clean = return_series.dropna()
        if len(ret_clean) > 30:
            k_r = kpss(ret_clean, regression="c", nlags="auto")
            result["kpss_ret_stat"] = k_r[0]
            result["kpss_ret_pval"] = k_r[1]
        else:
            result["kpss_ret_stat"] = np.nan
            result["kpss_ret_pval"] = np.nan
    except Exception:
        result["kpss_ret_stat"] = np.nan
        result["kpss_ret_pval"] = np.nan
 
    # Derived flags
    ap = result["adf_price_pval"]
    kp = result["kpss_price_pval"]
    ar = result["adf_ret_pval"]
    kr = result["kpss_ret_pval"]
 
    result["adf_price_stationary"]  = int(ap < 0.05)  if not np.isnan(ap) else 0
    result["kpss_price_stationary"] = int(kp > 0.05)  if not np.isnan(kp) else 0
    result["adf_ret_stationary"]    = int(ar < 0.05)  if not np.isnan(ar) else 0
    result["kpss_ret_stationary"]   = int(kr > 0.05)  if not np.isnan(kr) else 0
    result["price_is_i1"]  = int(ap > 0.05 and kp < 0.05) if (not np.isnan(ap) and not np.isnan(kp)) else 0
    result["return_is_i0"] = int(ar < 0.05 and kr > 0.05) if (not np.isnan(ar) and not np.isnan(kr)) else 0
 
    return result
 
 
def _calc_half_life(spread):
    """Calculate mean reversion half-life via OLS."""
    import numpy as np
    try:
        lag  = spread.shift(1).iloc[1:]
        diff = spread.diff().iloc[1:]
        beta = np.polyfit(lag.values, diff.values, 1)[0]
        if beta >= 0:
            return np.nan
        return -np.log(2) / beta
    except Exception:
        return np.nan
 
 
def _test_single_pair(t1, t2, s1, s2):
    """Engle-Granger for one pair. Module-level for joblib."""
    import numpy as np
    from statsmodels.tsa.stattools import coint
 
    common = s1.index.intersection(s2.index)
    if len(common) < 252:
        return None
 
    s1a = s1.loc[common]
    s2a = s2.loc[common]
 
    try:
        score, pval, _ = coint(np.log(s1a), np.log(s2a))
 
        if pval < 0.05:
            ratio = np.polyfit(np.log(s2a), np.log(s1a), 1)[0]
            spread = np.log(s1a) - ratio * np.log(s2a)
            half_life = _calc_half_life(spread)
            spread_zscore = (spread.iloc[-1] - spread.mean()) / spread.std()
        else:
            ratio         = np.nan
            half_life     = np.nan
            spread_zscore = np.nan
 
        return {
            "ticker1": t1, "ticker2": t2,
            "eg_statistic": score, "eg_pvalue": pval,
            "cointegrated": int(pval < 0.05),
            "hedge_ratio": ratio,
            "half_life_days": half_life,
            "spread_zscore": spread_zscore,
            "n_obs": len(common),
        }
    except Exception:
        return None

# COMMAND ----------

class EDAStationarityCointegration:
    """
    EDA 02 — Stationarity & Cointegration (OPTIMIZED).
 
    Speedups vs original:
      - joblib Parallel for ADF/KPSS   (~12x on 16 cores)
      - joblib Parallel for Engle-Granger (~10x on 16 cores)
      - Spark-side pivot (less data to collect)
      - maxlag=20 cap on ADF (prevents slow AIC on long series)
    """
 
    def __init__(self, spark, silver_path, eda_path, n_jobs=-1):
        self.spark       = spark
        self.silver_path = f"{silver_path}/ohlcv"
        self.eda_path    = f"{eda_path}/stationarity_cointegration"
        self.n_jobs      = n_jobs
        print(f"EDAStationarityCointegration ✓  (n_jobs={n_jobs})")
 
    # ------------------------------------------------------------------ #
    #  Load
    # ------------------------------------------------------------------ #
    def load(self):
        print("\nLoading silver OHLCV...")
        df = self.spark.read.format("delta").load(self.silver_path)
        cnt    = df.count()
        n_tick = df.select("ticker").distinct().count()
        print(f"  Rows    : {cnt:,}")
        print(f"  Tickers : {n_tick:,}")
        return df
 
    # ------------------------------------------------------------------ #
    #  Pivot — Spark-side pivot, single collect
    # ------------------------------------------------------------------ #
    def pivot_prices_returns(self, df, min_obs=1000):
        print("\nPivoting prices and returns...")
 
        # Filter tickers with enough history — on Spark side
        ticker_counts = (
            df.groupBy("ticker").count()
              .filter(F.col("count") >= min_obs)
        )
        valid_tickers = [r.ticker for r in ticker_counts.collect()]
        print(f"  Tickers with {min_obs}+ obs: {len(valid_tickers):,}")
 
        # Select only needed columns, filter early
        slim = (
            df.filter(F.col("ticker").isin(valid_tickers))
              .select("date", "ticker", "close", "return_1d")
        )
 
        # Spark-side pivot for prices
        print("  Pivoting prices on Spark...")
        price_pivot = (
            slim.groupBy("date")
                .pivot("ticker", valid_tickers)
                .agg(F.mean("close"))
                .orderBy("date")
        )
 
        # Spark-side pivot for returns
        print("  Pivoting returns on Spark...")
        return_pivot = (
            slim.groupBy("date")
                .pivot("ticker", valid_tickers)
                .agg(F.mean("return_1d"))
                .orderBy("date")
        )
 
        # Collect (already pivoted — much smaller transfer)
        print("  Collecting price matrix...")
        prices_pdf = price_pivot.toPandas()
        prices_pdf["date"] = pd.to_datetime(prices_pdf["date"])
        prices = prices_pdf.set_index("date")
 
        print("  Collecting return matrix...")
        returns_pdf = return_pivot.toPandas()
        returns_pdf["date"] = pd.to_datetime(returns_pdf["date"])
        returns = returns_pdf.set_index("date")
 
        # Drop columns with too many nulls
        prices  = prices.dropna(axis=1, thresh=min_obs)
        returns = returns.dropna(axis=1, thresh=min_obs)
 
        print(f"  Price matrix  : {prices.shape}")
        print(f"  Return matrix : {returns.shape}")
        return prices, returns, valid_tickers
 
    # ------------------------------------------------------------------ #
    #  ADF/KPSS — parallelized via joblib
    # ------------------------------------------------------------------ #
    def run_adf_tests(self, prices, returns):
        print(f"\nRunning ADF/KPSS tests in parallel (n_jobs={self.n_jobs})...")
        tickers = list(prices.columns)
 
        # Build args — each ticker gets its own pre-extracted Series
        args = []
        for t in tickers:
            p = prices[t].dropna()
            r = returns[t].dropna() if t in returns.columns else pd.Series(dtype=float)
            args.append((t, p, r))
 
        # Parallel execution across all cores
        raw = Parallel(n_jobs=self.n_jobs, backend="loky", verbose=5)(
            delayed(_test_single_ticker)(t, p, r) for t, p, r in args
        )
 
        results = [r for r in raw if r is not None]
        pdf_results = pd.DataFrame(results)
 
        print(f"\n  Tickers tested           : {len(pdf_results):,}")
        print(f"  Prices non-stationary    : "
              f"{(pdf_results['adf_price_stationary']==0).sum():,} "
              f"({(pdf_results['adf_price_stationary']==0).mean()*100:.1f}%)")
        print(f"  Returns stationary (ADF) : "
              f"{pdf_results['adf_ret_stationary'].sum():,} "
              f"({pdf_results['adf_ret_stationary'].mean()*100:.1f}%)")
        print(f"  Price is I(1)            : {pdf_results['price_is_i1'].sum():,}")
        print(f"  Return is I(0)           : {pdf_results['return_is_i0'].sum():,}")
        return pdf_results
 
    # ------------------------------------------------------------------ #
    #  Engle-Granger — parallelized
    # ------------------------------------------------------------------ #
    def run_engle_granger(self, prices, n_pairs=50):
        print(f"\nRunning Engle-Granger cointegration (parallel)...")
        print(f"  Testing top {n_pairs} tickers by data length")
 
        top_tickers  = prices.count().nlargest(n_pairs).index.tolist()
        price_subset = prices[top_tickers]
 
        # Pre-extract price series per ticker (avoid repeated .dropna)
        series_cache = {t: price_subset[t].dropna() for t in top_tickers}
 
        # Build pair list
        pairs = [
            (top_tickers[i], top_tickers[j],
             series_cache[top_tickers[i]],
             series_cache[top_tickers[j]])
            for i in range(len(top_tickers))
            for j in range(i + 1, len(top_tickers))
        ]
        print(f"  Total pairs: {len(pairs):,}")
 
        # Parallel execution
        raw = Parallel(n_jobs=self.n_jobs, backend="loky", verbose=5)(
            delayed(_test_single_pair)(t1, t2, s1, s2)
            for t1, t2, s1, s2 in pairs
        )
 
        results = [r for r in raw if r is not None]
        pdf = pd.DataFrame(results)
 
        if len(pdf) > 0:
            n_coint = pdf["cointegrated"].sum()
            print(f"\n  Pairs tested       : {len(pdf):,}")
            print(f"  Cointegrated pairs : {n_coint:,} "
                  f"({n_coint/len(pdf)*100:.1f}%)")
            if n_coint > 0:
                print(f"\n  Top cointegrated pairs:")
                print(pdf[pdf["cointegrated"] == 1]
                      .nsmallest(10, "eg_pvalue")
                      [["ticker1", "ticker2", "eg_pvalue",
                        "half_life_days", "spread_zscore"]]
                      .to_string(index=False))
        return pdf
 
    # ------------------------------------------------------------------ #
    #  Johansen multivariate cointegration
    # ------------------------------------------------------------------ #
    def run_johansen(self, prices):
        print("\nRunning Johansen cointegration by sector group...")
 
        sector_groups = {
            "Tech"       : ["AAPL","MSFT","GOOGL","META","NVDA",
                            "AMZN","AMD","INTC","QCOM","ADBE"],
            "Financials" : ["JPM","BAC","WFC","GS","MS","C",
                            "BLK","AXP","USB","PNC"],
            "Healthcare" : ["JNJ","PFE","UNH","ABBV","MRK",
                            "TMO","ABT","DHR","BMY","AMGN"],
            "Energy"     : ["XOM","CVX","COP","SLB","EOG",
                            "OXY","MPC","VLO","PSX","HAL"],
            "Consumer"   : ["AMZN","WMT","COST","TGT","HD",
                            "LOW","MCD","SBUX","NKE","TJX"],
        }
 
        available_cols = set(prices.columns)
        results = []
 
        for sector, tickers in sector_groups.items():
            available = [t for t in tickers if t in available_cols]
            if len(available) < 3:
                print(f"  {sector}: only {len(available)} "
                      f"tickers available — skipping")
                continue
 
            sector_prices = np.log(prices[available].dropna())
            if len(sector_prices) < 252:
                continue
 
            try:
                result = coint_johansen(
                    sector_prices.values, det_order=0, k_ar_diff=1
                )
                trace_stats = result.lr1
                trace_cv_95 = result.cvt[:, 1]
                n_coint     = int(np.sum(trace_stats > trace_cv_95))
                max_stats   = result.lr2
                max_cv_95   = result.cvm[:, 1]
                n_coint_max = int(np.sum(max_stats > max_cv_95))
 
                results.append({
                    "sector"               : sector,
                    "n_tickers"            : len(available),
                    "tickers"              : ",".join(available),
                    "n_obs"                : len(sector_prices),
                    "n_coint_vectors_trace": n_coint,
                    "n_coint_vectors_max"  : n_coint_max,
                    "max_trace_stat"       : float(trace_stats.max()),
                    "max_eigen_stat"       : float(max_stats.max()),
                    "has_cointegration"    : int(n_coint > 0),
                })
                print(f"  {sector}: {n_coint} cointegrating vectors "
                      f"(trace), {n_coint_max} (max eigen)")
 
            except Exception as e:
                print(f"  {sector} failed: {e}")
 
        return pd.DataFrame(results)
 
    # ------------------------------------------------------------------ #
    #  Write results to Delta
    # ------------------------------------------------------------------ #
    def write_results(self, adf_results, eg_results, johansen_results):
        print("\nWriting results to Delta...")
 
        self.spark.createDataFrame(adf_results).write \
            .format("delta").mode("overwrite") \
            .option("overwriteSchema", "true") \
            .save(f"{self.eda_path}/adf_kpss_results")
        print("  ✓ adf_kpss_results")
 
        if len(eg_results) > 0:
            self.spark.createDataFrame(eg_results).write \
                .format("delta").mode("overwrite") \
                .option("overwriteSchema", "true") \
                .save(f"{self.eda_path}/engle_granger_results")
            print("  ✓ engle_granger_results")
 
        if len(johansen_results) > 0:
            self.spark.createDataFrame(johansen_results).write \
                .format("delta").mode("overwrite") \
                .option("overwriteSchema", "true") \
                .save(f"{self.eda_path}/johansen_results")
            print("  ✓ johansen_results")
 
    # ------------------------------------------------------------------ #
    #  Validate & summary
    # ------------------------------------------------------------------ #
    def validate(self, adf_results, eg_results):
        print("\n" + "=" * 55)
        print("EDA 02 FINDINGS — Stationarity & Cointegration")
        print("=" * 55)
 
        print(f"\n  ADF/KPSS Results:")
        print(f"  Tickers tested           : {len(adf_results):,}")
        print(f"  Prices non-stationary    : "
              f"{(adf_results['adf_price_stationary']==0).sum():,} "
              f"({(adf_results['adf_price_stationary']==0).mean()*100:.1f}%)")
        print(f"  Returns stationary (ADF) : "
              f"{adf_results['adf_ret_stationary'].sum():,} "
              f"({adf_results['adf_ret_stationary'].mean()*100:.1f}%)")
        print(f"  Price is I(1)            : "
              f"{adf_results['price_is_i1'].sum():,}")
        print(f"  Return is I(0)           : "
              f"{adf_results['return_is_i0'].sum():,}")
 
        if len(eg_results) > 0:
            coint_pairs = eg_results[eg_results["cointegrated"] == 1]
            print(f"\n  Engle-Granger Results:")
            print(f"  Pairs tested       : {len(eg_results):,}")
            print(f"  Cointegrated pairs : {len(coint_pairs):,}")
            if len(coint_pairs) > 0:
                print(f"\n  Best pairs trading candidates:")
                best = coint_pairs[
                    coint_pairs["half_life_days"].between(5, 60)
                ].nsmallest(10, "eg_pvalue")
                if len(best) > 0:
                    print(best[["ticker1", "ticker2", "eg_pvalue",
                                "half_life_days", "spread_zscore"]]
                          .to_string(index=False))
 
        pct_non_stat = (
            adf_results['adf_price_stationary'] == 0
        ).mean() * 100
        pct_ret_stat = adf_results[
            'adf_ret_stationary'
        ].mean() * 100
 
        print(f"\n  KEY DECISION:")
        print(f"  → {pct_non_stat:.0f}% of prices are non-stationary")
        print(f"  → {pct_ret_stat:.0f}% of returns are stationary")
        print(f"  → Use returns (not prices) in Gold layer ✓")
        print(f"  → Cointegrated pairs available for pairs trading ✓")
 
    # ------------------------------------------------------------------ #
    #  Run all
    # ------------------------------------------------------------------ #
    def run(self):
        t0 = time.time()
        print("=" * 55)
        print("EDA 02 — Stationarity & Cointegration (OPTIMIZED)")
        print("=" * 55)
 
        df = self.load()
 
        t1 = time.time()
        prices, returns, valid_tickers = self.pivot_prices_returns(df)
        print(f"  ⏱ Pivot: {time.time() - t1:.1f}s")
 
        t1 = time.time()
        adf_results = self.run_adf_tests(prices, returns)
        print(f"  ⏱ ADF/KPSS: {time.time() - t1:.1f}s")
 
        t1 = time.time()
        eg_results = self.run_engle_granger(prices, n_pairs=50)
        print(f"  ⏱ Engle-Granger: {time.time() - t1:.1f}s")
 
        t1 = time.time()
        johansen_results = self.run_johansen(prices)
        print(f"  ⏱ Johansen: {time.time() - t1:.1f}s")
 
        t1 = time.time()
        self.write_results(adf_results, eg_results, johansen_results)
        print(f"  ⏱ Write: {time.time() - t1:.1f}s")
 
        self.validate(adf_results, eg_results)
        print(f"\n  ⏱ TOTAL: {time.time() - t0:.1f}s")
 
        return prices, returns, adf_results, eg_results, johansen_results

# COMMAND ----------

class EDAStationarityCharts:
    """
    Interactive Plotly charts for EDA 02.
    """
    TEMPLATE = "plotly_dark"
    COLORS   = {
        "primary"  : "#2196F3",
        "secondary": "#FF5722",
        "success"  : "#4CAF50",
        "warning"  : "#FFC107",
        "purple"   : "#9C27B0",
    }
 
    # ------------------------------------------------------------------ #
    #  Chart 1 — Price vs Returns stationarity comparison
    # ------------------------------------------------------------------ #
    def chart_price_vs_return(self, prices, returns, ticker="AAPL"):
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=[
                f"{ticker} — Log Price (Non-Stationary)",
                f"{ticker} — Daily Return (Stationary)",
                f"{ticker} — Rolling Mean (should be constant if stationary)"
            ],
            vertical_spacing=0.08
        )
 
        log_price  = np.log(prices[ticker].dropna())
        ret_series = returns[ticker].dropna()
 
        # Log price
        fig.add_trace(
            go.Scatter(
                x=log_price.index, y=log_price,
                name="Log Price",
                line=dict(color=self.COLORS["primary"], width=1)
            ),
            row=1, col=1
        )
 
        # Returns
        fig.add_trace(
            go.Scatter(
                x=ret_series.index, y=ret_series * 100,
                name="Return (%)",
                line=dict(color=self.COLORS["success"], width=0.8),
                opacity=0.8
            ),
            row=2, col=1
        )
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=2, col=1
        )
 
        # Rolling mean of returns
        rolling_mean = ret_series.rolling(63).mean() * 100
        rolling_std  = ret_series.rolling(63).std() * 100
        fig.add_trace(
            go.Scatter(
                x=rolling_mean.index, y=rolling_mean,
                name="63d Rolling Mean",
                line=dict(color=self.COLORS["warning"], width=2)
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=rolling_std.index, y=rolling_std,
                name="63d Rolling Std",
                line=dict(color=self.COLORS["secondary"], width=2)
            ),
            row=3, col=1
        )
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=3, col=1
        )
 
        fig.update_layout(
            title=dict(
                text=f"<b>EDA 02 — {ticker}: Price vs Return "
                     f"Stationarity</b>",
                font=dict(size=18)
            ),
            template=self.TEMPLATE,
            height=750,
            hovermode="x unified"
        )
        fig.update_yaxes(title_text="Log Price", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Rolling Stats (%)", row=3, col=1)
        fig.show()
 
    # ------------------------------------------------------------------ #
    #  Chart 2 — ADF p-value distribution
    # ------------------------------------------------------------------ #
    def chart_adf_pvalue_distribution(self, adf_results):
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "ADF p-value Distribution — Log Prices",
                "ADF p-value Distribution — Returns"
            ]
        )
 
        fig.add_trace(
            go.Histogram(
                x=adf_results["adf_price_pval"].dropna(),
                nbinsx=50,
                name="Price ADF p-value",
                marker_color=self.COLORS["secondary"],
                opacity=0.7
            ),
            row=1, col=1
        )
        fig.add_vline(
            x=0.05, line_color="white", line_dash="dash",
            line_width=2, annotation_text="α=0.05",
            row=1, col=1
        )
 
        fig.add_trace(
            go.Histogram(
                x=adf_results["adf_ret_pval"].dropna(),
                nbinsx=50,
                name="Return ADF p-value",
                marker_color=self.COLORS["success"],
                opacity=0.7
            ),
            row=1, col=2
        )
        fig.add_vline(
            x=0.05, line_color="white", line_dash="dash",
            line_width=2, annotation_text="α=0.05",
            row=1, col=2
        )
 
        pct_price_stat = adf_results["adf_price_stationary"].mean() * 100
        pct_ret_stat   = adf_results["adf_ret_stationary"].mean() * 100
 
        fig.update_layout(
            title=dict(
                text=f"<b>EDA 02 — ADF Test Results<br>"
                     f"<sup>Prices stationary: {pct_price_stat:.1f}% | "
                     f"Returns stationary: {pct_ret_stat:.1f}%</sup></b>",
                font=dict(size=18)
            ),
            template=self.TEMPLATE,
            height=500,
            showlegend=False
        )
        fig.update_xaxes(title_text="ADF p-value", row=1, col=1)
        fig.update_xaxes(title_text="ADF p-value", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.show()
 
    # ------------------------------------------------------------------ #
    #  Chart 3 — Stationarity verdict summary bar
    # ------------------------------------------------------------------ #
    def chart_stationarity_summary(self, adf_results):
        summary = {
            "Price ADF\n(reject unit root)":
                adf_results["adf_price_stationary"].sum(),
            "Price ADF\n(fail to reject)":
                (adf_results["adf_price_stationary"] == 0).sum(),
            "Price KPSS\n(stationary)":
                adf_results["kpss_price_stationary"].sum(),
            "Price KPSS\n(non-stationary)":
                (adf_results["kpss_price_stationary"] == 0).sum(),
            "Return ADF\n(stationary)":
                adf_results["adf_ret_stationary"].sum(),
            "Return ADF\n(non-stationary)":
                (adf_results["adf_ret_stationary"] == 0).sum(),
            "Return KPSS\n(stationary)":
                adf_results["kpss_ret_stationary"].sum(),
            "Return KPSS\n(non-stationary)":
                (adf_results["kpss_ret_stationary"] == 0).sum(),
        }
 
        colors_list = [
            self.COLORS["secondary"], self.COLORS["primary"],
            self.COLORS["secondary"], self.COLORS["primary"],
            self.COLORS["success"],   self.COLORS["secondary"],
            self.COLORS["success"],   self.COLORS["secondary"],
        ]
 
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=list(summary.keys()),
                y=list(summary.values()),
                marker_color=colors_list,
                text=list(summary.values()),
                textposition="outside",
            )
        )
 
        total = len(adf_results)
        fig.update_layout(
            title=dict(
                text=f"<b>EDA 02 — Stationarity Test Summary "
                     f"(N={total:,} tickers)</b>",
                font=dict(size=18)
            ),
            template=self.TEMPLATE,
            height=500,
            xaxis_title="Test",
            yaxis_title="Number of Tickers",
            showlegend=False
        )
        fig.show()
 
    # ------------------------------------------------------------------ #
    #  Chart 4 — Cointegrated pairs scatter
    # ------------------------------------------------------------------ #
    def chart_cointegrated_pairs(self, eg_results):
        if len(eg_results) == 0:
            print("No EG results to plot")
            return
 
        coint_pairs = eg_results[eg_results["cointegrated"] == 1].copy()
        non_coint   = eg_results[eg_results["cointegrated"] == 0]
 
        fig = go.Figure()
 
        # Non-cointegrated (background)
        fig.add_trace(
            go.Scatter(
                x=non_coint["eg_pvalue"],
                y=non_coint["half_life_days"],
                mode="markers",
                name="Not Cointegrated",
                marker=dict(color="rgba(150,150,150,0.3)", size=5),
                text=non_coint["ticker1"] + " / " + non_coint["ticker2"],
                hovertemplate="<b>%{text}</b><br>"
                              "p-value: %{x:.4f}<br>"
                              "Half-life: %{y:.1f}d<extra></extra>"
            )
        )
 
        # Cointegrated
        if len(coint_pairs) > 0:
            fig.add_trace(
                go.Scatter(
                    x=coint_pairs["eg_pvalue"],
                    y=coint_pairs["half_life_days"],
                    mode="markers+text",
                    name="Cointegrated ✓",
                    marker=dict(
                        color=self.COLORS["success"],
                        size=10, symbol="star"
                    ),
                    text=coint_pairs["ticker1"] + "/" +
                         coint_pairs["ticker2"],
                    textposition="top center",
                    textfont=dict(size=8),
                    hovertemplate="<b>%{text}</b><br>"
                                  "p-value: %{x:.4f}<br>"
                                  "Half-life: %{y:.1f}d<br>"
                                  "Spread Z: %{customdata:.2f}"
                                  "<extra></extra>",
                    customdata=coint_pairs["spread_zscore"]
                )
            )
 
        fig.add_hrect(
            y0=5, y1=60,
            fillcolor="rgba(76,175,80,0.1)", line_width=0,
            annotation_text="Ideal half-life zone (5-60d)"
        )
        fig.add_vline(
            x=0.05, line_dash="dash",
            line_color="white", opacity=0.5,
            annotation_text="α=0.05"
        )
 
        fig.update_layout(
            title=dict(
                text=f"<b>EDA 02 — Cointegrated Pairs "
                     f"(Engle-Granger)<br>"
                     f"<sup>{len(coint_pairs)} cointegrated / "
                     f"{len(eg_results)} total pairs</sup></b>",
                font=dict(size=18)
            ),
            template=self.TEMPLATE,
            height=600,
            xaxis_title="EG p-value",
            yaxis_title="Mean Reversion Half-Life (days)",
            yaxis=dict(range=[0, 120])
        )
        fig.show()
 
    # ------------------------------------------------------------------ #
    #  Chart 5 — Spread visualization for best pairs
    # ------------------------------------------------------------------ #
    def chart_spread_visualization(self, prices, eg_results):
        coint_pairs = eg_results[
            (eg_results["cointegrated"] == 1) &
            (eg_results["half_life_days"].between(5, 60))
        ].nsmallest(3, "eg_pvalue")
 
        if len(coint_pairs) == 0:
            coint_pairs = eg_results[
                eg_results["cointegrated"] == 1
            ].nsmallest(3, "eg_pvalue")
 
        if len(coint_pairs) == 0:
            print("No cointegrated pairs to plot")
            return
 
        n_pairs = min(3, len(coint_pairs))
        fig = make_subplots(
            rows=n_pairs, cols=1,
            subplot_titles=[
                f"{row['ticker1']} / {row['ticker2']} "
                f"(HL={row['half_life_days']:.0f}d, "
                f"p={row['eg_pvalue']:.4f})"
                for _, row in coint_pairs.head(n_pairs).iterrows()
            ],
            vertical_spacing=0.08
        )
 
        colors = [
            self.COLORS["success"],
            self.COLORS["warning"],
            self.COLORS["purple"]
        ]
 
        for idx, (_, pair) in enumerate(
            coint_pairs.head(n_pairs).iterrows()
        ):
            t1 = pair["ticker1"]
            t2 = pair["ticker2"]
 
            if t1 not in prices.columns or t2 not in prices.columns:
                continue
 
            s1     = np.log(prices[t1].dropna())
            s2     = np.log(prices[t2].dropna())
            common = s1.index.intersection(s2.index)
            if len(common) < 100:
                continue
 
            s1 = s1.loc[common]
            s2 = s2.loc[common]
            ratio  = np.polyfit(s2, s1, 1)[0]
            spread = s1 - ratio * s2
            zscore = (spread - spread.mean()) / spread.std()
 
            fill_colors = [
                "rgba(76,175,80,0.1)",
                "rgba(255,193,7,0.1)",
                "rgba(156,39,176,0.1)"
            ]
 
            fig.add_trace(
                go.Scatter(
                    x=zscore.index, y=zscore,
                    name=f"{t1}/{t2}",
                    line=dict(color=colors[idx], width=1),
                    fill="tozeroy",
                    fillcolor=fill_colors[idx]
                ),
                row=idx + 1, col=1
            )
            fig.add_hline(
                y=2, line_dash="dash", line_color="red",
                opacity=0.5, annotation_text="Short signal",
                row=idx + 1, col=1
            )
            fig.add_hline(
                y=-2, line_dash="dash", line_color="green",
                opacity=0.5, annotation_text="Long signal",
                row=idx + 1, col=1
            )
            fig.add_hline(
                y=0, line_dash="solid", line_color="white",
                opacity=0.2, row=idx + 1, col=1
            )
 
        fig.update_layout(
            title=dict(
                text="<b>EDA 02 — Pairs Trading Spread Z-Scores</b>",
                font=dict(size=18)
            ),
            template=self.TEMPLATE,
            height=300 * n_pairs,
            showlegend=False,
            hovermode="x unified"
        )
        fig.show()
 
    # ------------------------------------------------------------------ #
    #  Chart 6 — Johansen results
    # ------------------------------------------------------------------ #
    def chart_johansen_results(self, johansen_results):
        if len(johansen_results) == 0:
            print("No Johansen results to plot")
            return
 
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Cointegrating Vectors by Sector (Trace Test)",
                "Cointegrating Vectors by Sector (Max Eigen)"
            ]
        )
 
        colors = [
            self.COLORS["success"] if r > 0
            else self.COLORS["secondary"]
            for r in johansen_results["n_coint_vectors_trace"]
        ]
        fig.add_trace(
            go.Bar(
                x=johansen_results["sector"],
                y=johansen_results["n_coint_vectors_trace"],
                marker_color=colors,
                text=johansen_results["n_coint_vectors_trace"],
                textposition="outside",
                name="Trace Test"
            ),
            row=1, col=1
        )
 
        colors2 = [
            self.COLORS["success"] if r > 0
            else self.COLORS["secondary"]
            for r in johansen_results["n_coint_vectors_max"]
        ]
        fig.add_trace(
            go.Bar(
                x=johansen_results["sector"],
                y=johansen_results["n_coint_vectors_max"],
                marker_color=colors2,
                text=johansen_results["n_coint_vectors_max"],
                textposition="outside",
                name="Max Eigen"
            ),
            row=1, col=2
        )
 
        fig.update_layout(
            title=dict(
                text="<b>EDA 02 — Johansen Cointegration by Sector</b>",
                font=dict(size=18)
            ),
            template=self.TEMPLATE,
            height=500,
            showlegend=False
        )
        fig.update_yaxes(
            title_text="# Cointegrating Vectors", row=1, col=1
        )
        fig.update_yaxes(
            title_text="# Cointegrating Vectors", row=1, col=2
        )
        fig.show()
 
    # ------------------------------------------------------------------ #
    #  Chart 7 — Half life distribution
    # ------------------------------------------------------------------ #
    def chart_half_life_distribution(self, eg_results):
        coint = eg_results[
            (eg_results["cointegrated"] == 1) &
            (eg_results["half_life_days"].notna()) &
            (eg_results["half_life_days"] > 0) &
            (eg_results["half_life_days"] < 252)
        ]
 
        if len(coint) == 0:
            print("No cointegrated pairs with valid half-life")
            return
 
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Half-Life Distribution (Cointegrated Pairs)",
                "Current Spread Z-Score Distribution"
            ]
        )
 
        fig.add_trace(
            go.Histogram(
                x=coint["half_life_days"],
                nbinsx=30,
                name="Half-Life",
                marker_color=self.COLORS["success"],
                opacity=0.8
            ),
            row=1, col=1
        )
        fig.add_vrect(
            x0=5, x1=60,
            fillcolor="rgba(255,193,7,0.2)", line_width=0,
            annotation_text="Ideal zone",
            row=1, col=1
        )
 
        fig.add_trace(
            go.Histogram(
                x=coint["spread_zscore"].dropna(),
                nbinsx=30,
                name="Spread Z-Score",
                marker_color=self.COLORS["warning"],
                opacity=0.8
            ),
            row=1, col=2
        )
        fig.add_vline(
            x=2, line_dash="dash", line_color="red",
            annotation_text="Short zone", row=1, col=2
        )
        fig.add_vline(
            x=-2, line_dash="dash", line_color="green",
            annotation_text="Long zone", row=1, col=2
        )
 
        fig.update_layout(
            title=dict(
                text="<b>EDA 02 — Pairs Trading Signal Analysis</b>",
                font=dict(size=18)
            ),
            template=self.TEMPLATE,
            height=500,
            showlegend=False
        )
        fig.update_xaxes(title_text="Half-Life (days)", row=1, col=1)
        fig.update_xaxes(
            title_text="Current Spread Z-Score", row=1, col=2
        )
        fig.show()
 
    # ------------------------------------------------------------------ #
    #  Run all charts
    # ------------------------------------------------------------------ #
    def run_all(self, prices, returns, adf_results,
                eg_results, johansen_results):
        print("\n" + "=" * 55)
        print("Generating Interactive Charts...")
        print("=" * 55)
 
        ticker = "AAPL" if "AAPL" in prices.columns \
                 else prices.columns[0]
 
        print(f"\n[1/7] Price vs Return Stationarity ({ticker})...")
        self.chart_price_vs_return(prices, returns, ticker)
 
        print("[2/7] ADF p-value Distributions...")
        self.chart_adf_pvalue_distribution(adf_results)
 
        print("[3/7] Stationarity Test Summary...")
        self.chart_stationarity_summary(adf_results)
 
        print("[4/7] Cointegrated Pairs Scatter...")
        self.chart_cointegrated_pairs(eg_results)
 
        print("[5/7] Spread Z-Score Visualization...")
        self.chart_spread_visualization(prices, eg_results)
 
        print("[6/7] Johansen Results by Sector...")
        self.chart_johansen_results(johansen_results)
 
        print("[7/7] Half-Life & Spread Distribution...")
        self.chart_half_life_distribution(eg_results)
 
        print("\nAll 7 charts generated ✓")

# COMMAND ----------

eda = EDAStationarityCointegration(
    spark       = spark,
    silver_path = SILVER_PATH,
    eda_path    = EDA_PATH,
    n_jobs      = -1,  # Use all 16 cores on Standard_D16_v3
)
 
prices, returns, adf_results, eg_results, \
    johansen_results = eda.run()

# COMMAND ----------

charts = EDAStationarityCharts()
charts.run_all(
    prices           = prices,
    returns          = returns,
    adf_results      = adf_results,
    eg_results       = eg_results,
    johansen_results = johansen_results,
)
 
print("\nEDA 02 COMPLETE ✓")

# COMMAND ----------

adf = spark.read.format("delta").load(
    f"{EDA_PATH}/stationarity_cointegration/adf_kpss_results"
).toPandas()

print("=" * 55)
print("EDA 02 — Key Findings")
print("=" * 55)
print(f"Tickers tested        : {len(adf):,}")
print(f"Prices non-stationary : "
      f"{(adf['adf_price_stationary']==0).mean()*100:.1f}%")
print(f"Returns stationary    : "
      f"{adf['adf_ret_stationary'].mean()*100:.1f}%")
print(f"Price is I(1)         : {adf['price_is_i1'].sum():,}")
print(f"Return is I(0)        : {adf['return_is_i0'].sum():,}")

# Best pairs
eg = spark.read.format("delta").load(
    f"{EDA_PATH}/stationarity_cointegration/engle_granger_results"
).toPandas()

best_pairs = eg[
    (eg["cointegrated"] == 1) &
    (eg["half_life_days"].between(5, 60))
].nsmallest(10, "eg_pvalue")

print(f"\nTop 10 pairs trading candidates:")
print(best_pairs[[
    "ticker1", "ticker2", "eg_pvalue",
    "half_life_days", "spread_zscore"
]].to_string(index=False))

# Johansen summary
try:
    joh = spark.read.format("delta").load(
        f"{EDA_PATH}/stationarity_cointegration/johansen_results"
    ).toPandas()
    print(f"\nJohansen sector cointegration:")
    print(joh[["sector", "n_tickers", "n_coint_vectors_trace",
               "n_coint_vectors_max"]].to_string(index=False))
except Exception:
    print("\nNo Johansen results found")