# Databricks notebook source
# MAGIC %pip install plotly scipy pandas numpy --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import SparkSession
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.shuffle.partitions","200")
spark.conf.set("spark.sql.ansi.enabled","false")

STORAGE_ACCOUNT = "multisignalalphaeng"
CONTAINER       = "quant-lakehouse"
ADLS_KEY        = dbutils.secrets.get(
    scope="quant-scope", key="adls-key-01"
)
spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}"
    f".dfs.core.windows.net",
    ADLS_KEY
)
BASE_PATH = (
    f"abfss://{CONTAINER}@"
    f"{STORAGE_ACCOUNT}.dfs.core.windows.net"
)
GOLD_PATH = f"{BASE_PATH}/gold/delta"
ML_PATH   = f"{BASE_PATH}/ml/delta"
BT_PATH   = f"{BASE_PATH}/backtest/delta"

print("="*55)
print("Backtest 02 — Risk Analysis")
print("="*55)
print(f"BT_PATH : {BT_PATH}")

# COMMAND ----------

class Backtest02:
    """
    Backtest 02 — Comprehensive Risk Analysis.

    Sections:
      1. Tail risk      : VaR, CVaR, stress tests
      2. Drawdown       : full DD analysis + recovery
      3. Factor exposure: market, size, vol, momentum
      4. Correlation    : regime, rolling, stability
      5. Volatility     : vol clustering, GARCH-like
      6. Concentration  : position + sector risk
      7. Liquidity      : turnover, market impact est
      8. Stability      : rolling metrics, IC decay
    """

    ANNUAL_FACTOR = 252
    RISK_FREE     = 0.04
    CONFIDENCE    = [0.90, 0.95, 0.99]

    def __init__(self, spark, gold_path,
                 ml_path, bt_path):
        self.spark     = spark
        self.gold_path = gold_path
        self.ml_path   = ml_path
        self.bt_path   = bt_path

        print("Backtest02 ✓")
        print(f"  Confidence levels: "
              f"{self.CONFIDENCE}")

    # ─────────────────────────────────────────────────
    #  Step 1 — Load data
    # ─────────────────────────────────────────────────
    def load_data(self):
        print("\nStep 1: Loading backtest data...")
        start = datetime.now()

        # BT01 daily returns
        daily = self.spark.read.format(
            "delta"
        ).load(
            f"{self.bt_path}/bt01_daily_returns"
        ).toPandas()
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values(
            "date"
        ).reset_index(drop=True)

        # BT01 IC series
        ic_df = self.spark.read.format(
            "delta"
        ).load(
            f"{self.bt_path}/bt01_ic_series"
        ).toPandas()
        ic_df["date"] = pd.to_datetime(ic_df["date"])
        ic_df = ic_df.sort_values("date")

        # Ensemble predictions
        pred = self.spark.read.format(
            "delta"
        ).load(
            f"{self.ml_path}/ensemble_predictions"
        ).toPandas()
        pred["date"] = pd.to_datetime(pred["date"])

        # Gold price factors (market returns)
        sdf     = self.spark.read.format(
            "delta"
        ).load(f"{self.gold_path}/price_factors")
        avail   = set(sdf.columns)
        mk_cols = [
            c for c in [
                "date","ticker",
                "fwd_return_21d","fwd_return_1d",
                "vol_21d","vol_63d",
                "mom_21d","mom_252d",
                "regime_label",
            ] if c in avail
        ]
        mkt = sdf.select(*mk_cols).toPandas()
        mkt["date"] = pd.to_datetime(mkt["date"])

        # Market (equal-weight universe)
        mkt_ret = mkt.groupby("date")[
            "fwd_return_21d"
        ].mean().reset_index()
        mkt_ret.columns = ["date","mkt_ret"]

        elapsed = (datetime.now()-start).seconds
        print(f"  Daily rows : {len(daily):,}")
        print(f"  IC rows    : {len(ic_df):,}")
        print(f"  Pred rows  : {len(pred):,}")
        print(f"  Date range : "
              f"{daily['date'].min().date()} → "
              f"{daily['date'].max().date()}")
        print(f"  Elapsed    : {elapsed}s")
        return daily, ic_df, pred, mkt, mkt_ret

    # ─────────────────────────────────────────────────
    #  Section 1 — Tail Risk
    # ─────────────────────────────────────────────────
    def analyze_tail_risk(self, daily):
        """
        VaR, CVaR, Expected Shortfall,
        stress tests, tail ratio.
        """
        print("\nSection 1: Tail Risk Analysis...")

        r = daily["ls_net"].dropna()

        results = {}

        # Parametric VaR (normal assumption)
        mu    = r.mean()
        sigma = r.std()
        for conf in self.CONFIDENCE:
            z    = norm.ppf(1 - conf)
            pvar = mu + z * sigma
            results[f"pvar_{int(conf*100)}"] = float(
                pvar
            )

        # Historical VaR + CVaR
        for conf in self.CONFIDENCE:
            alpha = 1 - conf
            hvar  = float(np.percentile(r, alpha*100))
            cvar  = float(
                r[r <= hvar].mean()
            )
            results[f"hvar_{int(conf*100)}"] = hvar
            results[f"cvar_{int(conf*100)}"] = cvar

        # Cornish-Fisher VaR (adjusts for skew/kurt)
        sk = float(stats.skew(r))
        ku = float(stats.kurtosis(r))
        for conf in self.CONFIDENCE:
            z   = norm.ppf(1 - conf)
            z_cf = (
                z +
                (z**2 - 1) * sk / 6 +
                (z**3 - 3*z) * ku / 24 -
                (2*z**3 - 5*z) * sk**2 / 36
            )
            cfvar = float(mu + z_cf * sigma)
            results[f"cfvar_{int(conf*100)}"] = cfvar

        # Tail ratio (right tail / left tail)
        p95 = abs(np.percentile(r, 95))
        p5  = abs(np.percentile(r, 5))
        results["tail_ratio"] = float(
            p95 / (p5 + 1e-10)
        )

        # Gain-to-Pain ratio
        results["gain_pain"] = float(
            r[r > 0].sum() /
            abs(r[r <= 0].sum() + 1e-10)
        )

        # Stress tests
        stress = {}

        # Worst N-day periods
        for n in [1, 5, 21, 63]:
            roll = r.rolling(n).sum()
            stress[f"worst_{n}d"] = float(
                roll.min()
            )
            stress[f"best_{n}d"]  = float(
                roll.max()
            )

        # Worst year
        daily_copy = daily.copy()
        daily_copy["year"] = daily_copy["date"].dt.year
        yr_ret = daily_copy.groupby(
            "year"
        )["ls_net"].apply(
            lambda x: float((1+x).prod()-1)
        )
        stress["worst_year"] = float(yr_ret.min())
        stress["best_year"]  = float(yr_ret.max())
        stress["worst_year_label"] = str(
            yr_ret.idxmin()
        )
        stress["best_year_label"]  = str(
            yr_ret.idxmax()
        )

        # 2008 crisis performance
        crisis_2008 = daily_copy[
            daily_copy["year"] == 2008
        ]["ls_net"]
        if len(crisis_2008) > 0:
            stress["crisis_2008"] = float(
                (1+crisis_2008).prod()-1
            )

        # COVID crash (2020)
        crisis_2020 = daily_copy[
            (daily_copy["date"] >= "2020-01-01") &
            (daily_copy["date"] <= "2020-12-31")
        ]["ls_net"]
        if len(crisis_2020) > 0:
            stress["covid_2020"] = float(
                (1+crisis_2020).prod()-1
            )

        # GFC (2007-2009)
        gfc = daily_copy[
            (daily_copy["year"] >= 2007) &
            (daily_copy["year"] <= 2009)
        ]["ls_net"]
        if len(gfc) > 0:
            stress["gfc_2007_2009"] = float(
                (1+gfc).prod()-1
            )

        results["stress"] = stress

        print(f"\n  Historical VaR:")
        for conf in self.CONFIDENCE:
            hv = results[f"hvar_{int(conf*100)}"]
            cv = results[f"cvar_{int(conf*100)}"]
            print(f"    {int(conf*100)}%: "
                  f"VaR={hv*100:.3f}% "
                  f"CVaR={cv*100:.3f}%")

        print(f"\n  Tail Ratio   : "
              f"{results['tail_ratio']:.3f}")
        print(f"  Gain/Pain    : "
              f"{results['gain_pain']:.3f}")

        print(f"\n  Stress Tests:")
        for k, v in stress.items():
            if isinstance(v, float):
                print(f"    {k:20}: {v*100:.2f}%")

        return results

    # ─────────────────────────────────────────────────
    #  Section 2 — Drawdown Analysis
    # ─────────────────────────────────────────────────
    def analyze_drawdowns(self, daily):
        """
        Full drawdown analysis:
        DD duration, recovery time, underwater pct.
        """
        print("\nSection 2: Drawdown Analysis...")

        r   = daily["ls_net"].fillna(0)
        cum = (1 + r).cumprod()
        dd  = cum / cum.cummax() - 1

        # Find all drawdown periods
        dd_periods = []
        in_dd      = False
        dd_start   = None
        dd_peak    = None
        dd_peak_val= None
        dd_trough  = None
        dd_trough_val = None

        dates = daily["date"].values

        for i, (d, v) in enumerate(
            zip(dates, dd.values)
        ):
            if not in_dd and v < -0.001:
                in_dd      = True
                dd_start   = dates[
                    max(0, i-1)
                ]
                dd_peak    = dates[
                    max(0, i-1)
                ]
                dd_peak_val= float(cum.iloc[
                    max(0, i-1)
                ])
                dd_trough  = d
                dd_trough_val = float(v)

            elif in_dd:
                if v < dd_trough_val:
                    dd_trough     = d
                    dd_trough_val = float(v)

                if v >= -0.001:
                    in_dd = False
                    recovery   = d
                    duration   = (
                        pd.Timestamp(dd_trough) -
                        pd.Timestamp(dd_start)
                    ).days
                    rec_days   = (
                        pd.Timestamp(recovery) -
                        pd.Timestamp(dd_trough)
                    ).days
                    dd_periods.append({
                        "start"       : dd_start,
                        "trough"      : dd_trough,
                        "recovery"    : recovery,
                        "max_dd"      : dd_trough_val,
                        "duration_days": duration,
                        "recovery_days": rec_days,
                        "total_days"  : duration + rec_days,
                    })

        dd_df = pd.DataFrame(dd_periods)

        # Summary stats
        results = {
            "max_dd"       : float(dd.min()),
            "avg_dd"       : float(
                dd[dd < -0.001].mean()
            ),
            "pct_time_in_dd": float(
                (dd < -0.001).mean()
            ),
            "n_drawdowns"  : len(dd_df),
        }

        if len(dd_df) > 0:
            results.update({
                "avg_dd_depth"  : float(
                    dd_df["max_dd"].mean()
                ),
                "avg_dd_dur"    : float(
                    dd_df["duration_days"].mean()
                ),
                "avg_rec_time"  : float(
                    dd_df["recovery_days"].mean()
                ),
                "max_dd_dur"    : int(
                    dd_df["duration_days"].max()
                ),
                "max_rec_time"  : int(
                    dd_df["recovery_days"].max()
                ),
            })

        print(f"\n  Max Drawdown  : "
              f"{results['max_dd']*100:.2f}%")
        print(f"  Avg Drawdown  : "
              f"{results['avg_dd']*100:.2f}%")
        print(f"  % Time in DD  : "
              f"{results['pct_time_in_dd']:.1%}")
        print(f"  N Drawdowns   : "
              f"{results['n_drawdowns']}")

        if len(dd_df) > 0:
            print(f"  Avg DD Depth  : "
                  f"{results['avg_dd_depth']*100:.2f}%")
            print(f"  Avg DD Duration: "
                  f"{results['avg_dd_dur']:.0f} days")
            print(f"  Avg Recovery  : "
                  f"{results['avg_rec_time']:.0f} days")

            print(f"\n  Top 5 Drawdowns:")
            top5 = dd_df.nsmallest(
                5, "max_dd"
            )[[
                "start","trough","recovery",
                "max_dd","duration_days",
                "recovery_days"
            ]]
            for _, row in top5.iterrows():
                print(f"    {str(row['trough'])[:10]}: "
                      f"{row['max_dd']*100:.2f}% "
                      f"({row['duration_days']:.0f}d DD "
                      f"{row['recovery_days']:.0f}d rec)")

        return results, dd_df, dd

    # ─────────────────────────────────────────────────
    #  Section 3 — Factor Exposure
    # ─────────────────────────────────────────────────
    def analyze_factor_exposure(self, daily,
                                  mkt_ret):
        """
        Market beta, factor loadings,
        alpha estimation.
        """
        print("\nSection 3: Factor Exposure...")

        # Merge with market returns
        df = daily.merge(
            mkt_ret, on="date", how="left"
        )
        df["mkt_ret"] = df["mkt_ret"].fillna(0)

        r   = df["ls_net"].fillna(0).values
        mkt = df["mkt_ret"].fillna(0).values

        results = {}

        # Market beta (full period)
        if np.std(mkt) > 1e-10:
            beta_full = float(
                np.cov(r, mkt)[0,1] /
                (np.var(mkt) + 1e-10)
            )
            corr_mkt  = float(np.corrcoef(r, mkt)[0,1])
        else:
            beta_full = 0.0
            corr_mkt  = 0.0

        results["beta_market"]   = beta_full
        results["corr_market"]   = corr_mkt

        # Alpha (CAPM)
        rf_daily = self.RISK_FREE / self.ANNUAL_FACTOR
        alpha_ann = (
            np.mean(r) - rf_daily -
            beta_full * (np.mean(mkt) - rf_daily)
        ) * self.ANNUAL_FACTOR
        results["alpha_ann"] = float(alpha_ann)

        # Information ratio vs market
        active_r = r - mkt
        te       = np.std(active_r)
        ir       = (
            np.mean(active_r) /
            (te + 1e-10) *
            np.sqrt(self.ANNUAL_FACTOR)
        )
        results["tracking_error"] = float(
            te * np.sqrt(self.ANNUAL_FACTOR)
        )
        results["information_ratio"] = float(ir)

        # Rolling beta (252d)
        r_s   = pd.Series(r)
        mkt_s = pd.Series(mkt)
        roll_beta = []
        for i in range(252, len(r_s)):
            r_w   = r_s.iloc[i-252:i].values
            m_w   = mkt_s.iloc[i-252:i].values
            if np.std(m_w) > 1e-10:
                b = float(
                    np.cov(r_w,m_w)[0,1] /
                    np.var(m_w)
                )
            else:
                b = 0.0
            roll_beta.append(b)
        results["rolling_beta"] = roll_beta
        results["avg_roll_beta"] = float(
            np.mean(roll_beta)
        ) if roll_beta else 0.0

        # Up/Down capture
        up_idx  = mkt > 0
        dn_idx  = mkt <= 0
        results["up_capture"] = float(
            np.mean(r[up_idx]) /
            (np.mean(mkt[up_idx]) + 1e-10)
        ) if up_idx.sum() > 0 else 0.0
        results["dn_capture"] = float(
            np.mean(r[dn_idx]) /
            (np.mean(mkt[dn_idx]) + 1e-10)
        ) if dn_idx.sum() > 0 else 0.0

        print(f"\n  Market Beta  : "
              f"{results['beta_market']:.4f} "
              f"({'✅ market neutral' if abs(results['beta_market'])<0.1 else '⚠️'})")
        print(f"  Corr(Market) : "
              f"{results['corr_market']:.4f}")
        print(f"  Alpha (ann)  : "
              f"{results['alpha_ann']*100:.2f}%")
        print(f"  Track Error  : "
              f"{results['tracking_error']*100:.2f}%")
        print(f"  Info Ratio   : "
              f"{results['information_ratio']:.4f}")
        print(f"  Up Capture   : "
              f"{results['up_capture']:.4f}")
        print(f"  Down Capture : "
              f"{results['dn_capture']:.4f}")

        return results, df

    # ─────────────────────────────────────────────────
    #  Section 4 — Correlation Analysis
    # ─────────────────────────────────────────────────
    def analyze_correlations(self, daily,
                               mkt_ret):
        """
        Rolling correlation with market,
        correlation stability, crisis correlation.
        """
        print("\nSection 4: Correlation Analysis...")

        df = daily.merge(
            mkt_ret, on="date", how="left"
        )
        r   = df["ls_net"].fillna(0)
        mkt = df["mkt_ret"].fillna(0)

        results = {}

        # Rolling correlations
        roll_corr_63  = r.rolling(63).corr(mkt)
        roll_corr_252 = r.rolling(252).corr(mkt)

        results["roll_corr_63"]  = (
            roll_corr_63.dropna().values.tolist()
        )
        results["roll_corr_252"] = (
            roll_corr_252.dropna().values.tolist()
        )
        results["avg_corr_63"]   = float(
            roll_corr_63.mean()
        )
        results["avg_corr_252"]  = float(
            roll_corr_252.mean()
        )
        results["corr_stability"] = float(
            1 - roll_corr_63.std()
        )

        # Regime correlations
        regime_corr = {}
        for regime in ["Bull","HighVol","Bear"]:
            mask = df["regime"] == regime
            if mask.sum() > 30:
                rc = float(
                    r[mask].corr(mkt[mask])
                )
                regime_corr[regime] = rc

        results["regime_corr"] = regime_corr

        # Crisis beta (tail correlation)
        pct5 = np.percentile(mkt, 5)
        tail_mask = mkt <= pct5
        if tail_mask.sum() > 10:
            results["tail_beta"] = float(
                np.cov(
                    r[tail_mask].values,
                    mkt[tail_mask].values
                )[0,1] /
                (np.var(mkt[tail_mask].values) + 1e-10)
            )
        else:
            results["tail_beta"] = 0.0

        print(f"\n  Avg Corr (63d)  : "
              f"{results['avg_corr_63']:.4f}")
        print(f"  Avg Corr (252d) : "
              f"{results['avg_corr_252']:.4f}")
        print(f"  Corr Stability  : "
              f"{results['corr_stability']:.4f}")
        print(f"  Tail Beta       : "
              f"{results['tail_beta']:.4f} "
              f"({'✅ low' if abs(results['tail_beta'])<0.2 else '⚠️'})")
        print(f"\n  Regime Corr:")
        for r_name, rc in regime_corr.items():
            print(f"    {r_name:8}: {rc:.4f}")

        return results, df

    # ─────────────────────────────────────────────────
    #  Section 5 — Volatility Analysis
    # ─────────────────────────────────────────────────
    def analyze_volatility(self, daily):
        """
        Volatility clustering, regime,
        vol-of-vol analysis.
        """
        print("\nSection 5: Volatility Analysis...")

        r = daily["ls_net"].fillna(0)

        results = {}

        # Rolling volatility
        roll_vol_21  = (
            r.rolling(21).std() *
            np.sqrt(self.ANNUAL_FACTOR)
        )
        roll_vol_63  = (
            r.rolling(63).std() *
            np.sqrt(self.ANNUAL_FACTOR)
        )
        roll_vol_252 = (
            r.rolling(252).std() *
            np.sqrt(self.ANNUAL_FACTOR)
        )

        results["vol_21_mean"]  = float(
            roll_vol_21.mean()
        )
        results["vol_63_mean"]  = float(
            roll_vol_63.mean()
        )
        results["vol_252_mean"] = float(
            roll_vol_252.mean()
        )
        results["vol_of_vol"]   = float(
            roll_vol_21.std()
        )

        # Vol percentiles
        results["vol_p25"] = float(
            roll_vol_21.quantile(0.25)
        )
        results["vol_p75"] = float(
            roll_vol_21.quantile(0.75)
        )
        results["vol_p95"] = float(
            roll_vol_21.quantile(0.95)
        )

        # Autocorrelation of |returns| (vol clustering)
        abs_r  = r.abs()
        autocorr = {
            f"autocorr_lag{lag}": float(
                abs_r.autocorr(lag=lag)
            )
            for lag in [1, 5, 21]
        }
        results["vol_clustering"] = autocorr

        # High/low vol regime split
        vol_median = roll_vol_21.median()
        hi_vol = r[roll_vol_21 > vol_median]
        lo_vol = r[roll_vol_21 <= vol_median]

        results["hi_vol_sharpe"] = float(
            hi_vol.mean() / (hi_vol.std() + 1e-10) *
            np.sqrt(self.ANNUAL_FACTOR)
        )
        results["lo_vol_sharpe"] = float(
            lo_vol.mean() / (lo_vol.std() + 1e-10) *
            np.sqrt(self.ANNUAL_FACTOR)
        )

        results["roll_vol_21"]  = roll_vol_21
        results["roll_vol_63"]  = roll_vol_63
        results["roll_vol_252"] = roll_vol_252

        print(f"\n  Ann Vol (21d)  : "
              f"{results['vol_21_mean']*100:.2f}%")
        print(f"  Ann Vol (63d)  : "
              f"{results['vol_63_mean']*100:.2f}%")
        print(f"  Vol of Vol     : "
              f"{results['vol_of_vol']*100:.2f}%")
        print(f"  Vol P25/P75/P95: "
              f"{results['vol_p25']*100:.2f}% / "
              f"{results['vol_p75']*100:.2f}% / "
              f"{results['vol_p95']*100:.2f}%")
        print(f"\n  Vol Clustering (|r| autocorr):")
        for k, v in autocorr.items():
            print(f"    {k:20}: {v:.4f}")
        print(f"\n  High vol Sharpe: "
              f"{results['hi_vol_sharpe']:.2f}")
        print(f"  Low  vol Sharpe: "
              f"{results['lo_vol_sharpe']:.2f}")

        return results

    # ─────────────────────────────────────────────────
    #  Section 6 — IC Stability
    # ─────────────────────────────────────────────────
    def analyze_ic_stability(self, ic_df):
        """
        IC decay, stability across time,
        signal half-life estimation.
        """
        print("\nSection 6: IC Stability...")

        vals = ic_df["ic"].dropna()

        results = {}

        # Rolling IC stats
        for w in [63, 126, 252]:
            roll = vals.rolling(w)
            results[f"roll_ic_mean_{w}"] = float(
                roll.mean().mean()
            )
            results[f"roll_ic_std_{w}"]  = float(
                roll.mean().std()
            )
            results[f"roll_icir_{w}"]    = float(
                roll.mean().mean() /
                (roll.mean().std() + 1e-8)
            )

        # IC hit rate over time (stability)
        ic_yr = ic_df.copy()
        ic_yr["year"] = ic_yr["date"].dt.year
        yr_hit = ic_yr.groupby("year")["ic"].apply(
            lambda x: float((x > 0).mean())
        )
        results["yr_hit_rates"]  = yr_hit.to_dict()
        results["avg_yr_hit"]    = float(
            yr_hit.mean()
        )
        results["min_yr_hit"]    = float(
            yr_hit.min()
        )
        results["worst_yr_hit_year"] = str(
            yr_hit.idxmin()
        ) if len(yr_hit) > 0 else "N/A"

        # Autocorrelation of IC
        ic_autocorr = {
            f"ic_autocorr_lag{lag}": float(
                vals.autocorr(lag=lag)
            )
            for lag in [1, 5, 21]
        }
        results["ic_autocorr"] = ic_autocorr

        # Pct years with positive IC
        pos_yr = float((yr_hit > 0.5).mean())
        results["pct_pos_years"] = pos_yr

        print(f"\n  Rolling IC Mean (252d): "
              f"{results['roll_ic_mean_252']:+.4f}")
        print(f"  Rolling ICIR  (252d): "
              f"{results['roll_icir_252']:.4f}")
        print(f"  Avg Yr Hit Rate : "
              f"{results['avg_yr_hit']:.1%}")
        print(f"  Min Yr Hit Rate : "
              f"{results['min_yr_hit']:.1%} "
              f"({results['worst_yr_hit_year']})")
        print(f"  % Positive Years: "
              f"{results['pct_pos_years']:.1%}")
        print(f"\n  IC Autocorrelation:")
        for k, v in ic_autocorr.items():
            print(f"    {k:25}: {v:.4f}")

        return results

    # ─────────────────────────────────────────────────
    #  Section 7 — Stability Analysis
    # ─────────────────────────────────────────────────
    def analyze_stability(self, daily, ic_df):
        """
        Rolling Sharpe, Sortino, hit rate.
        Strategy consistency over time.
        """
        print("\nSection 7: Stability Analysis...")

        r   = daily["ls_net"].fillna(0)
        rf  = self.RISK_FREE / self.ANNUAL_FACTOR

        # Rolling Sharpe (multiple windows)
        roll_sharpe = {}
        for w in [63, 126, 252]:
            roll_s = (
                (r.rolling(w).mean() - rf) /
                r.rolling(w).std() *
                np.sqrt(self.ANNUAL_FACTOR)
            )
            roll_sharpe[w] = roll_s
            pct_pos = float(
                (roll_s.dropna() > 1.0).mean()
            )
            print(f"\n  Rolling {w}d Sharpe:")
            print(f"    Mean    : "
                  f"{roll_s.mean():.2f}")
            print(f"    Std     : "
                  f"{roll_s.std():.2f}")
            print(f"    Min     : "
                  f"{roll_s.min():.2f}")
            print(f"    Max     : "
                  f"{roll_s.max():.2f}")
            print(f"    % > 1.0 : {pct_pos:.1%}")

        # Annual returns
        daily_copy = daily.copy()
        daily_copy["year"] = (
            daily_copy["date"].dt.year
        )
        ann_rets = daily_copy.groupby("year")[
            "ls_net"
        ].apply(
            lambda x: float((1+x).prod()-1)
        )
        n_pos = int((ann_rets > 0).sum())
        n_tot = len(ann_rets)
        print(f"\n  Annual Returns:")
        print(f"    Positive years: "
              f"{n_pos}/{n_tot} "
              f"({n_pos/n_tot:.1%})")
        print(f"    Best year     : "
              f"{ann_rets.max()*100:.1f}% "
              f"({ann_rets.idxmax()})")
        print(f"    Worst year    : "
              f"{ann_rets.min()*100:.1f}% "
              f"({ann_rets.idxmin()})")

        results = {
            "roll_sharpe"  : roll_sharpe,
            "ann_rets"     : ann_rets.to_dict(),
            "pct_pos_years": float(n_pos / n_tot),
            "best_year"    : float(ann_rets.max()),
            "worst_year"   : float(ann_rets.min()),
        }
        return results

    # ─────────────────────────────────────────────────
    #  Step 5 — Write results
    # ─────────────────────────────────────────────────
    def write_results(self, tail_risk, dd_results,
                       dd_df, factor, corr,
                       vol, ic_stab, stability):
        print("\nWriting results...")

        def _write(df, path, partition=False):
            df   = df.copy()
            nums = df.select_dtypes(
                include=[np.number]
            ).columns
            df[nums] = df[nums].fillna(0)
            (
                self.spark.createDataFrame(df)
                .write.format("delta")
                .mode("overwrite")
                .option("overwriteSchema","true")
                .save(path)
            )

        # Drawdown periods
        if len(dd_df) > 0:
            _write(
                dd_df[[
                    c for c in dd_df.columns
                    if c not in ["start","trough",
                                 "recovery"]
                ] + [
                    c for c in [
                        "start","trough","recovery"
                    ] if c in dd_df.columns
                ]],
                f"{self.bt_path}/bt02_dd_periods"
            )
            print("  ✓ bt02_dd_periods")

        # Summary metrics
        summary_rows = []

        # Tail risk
        for k, v in tail_risk.items():
            if k == "stress":
                continue
            if isinstance(v, (int, float)):
                summary_rows.append({
                    "section": "tail_risk",
                    "metric" : k,
                    "value"  : float(v),
                })

        # Factor exposure
        for k, v in factor.items():
            if k in ["rolling_beta","roll_corr_63",
                     "roll_corr_252"]:
                continue
            if isinstance(v, (int, float)):
                summary_rows.append({
                    "section": "factor",
                    "metric" : k,
                    "value"  : float(v),
                })

        # Drawdown summary
        for k, v in dd_results.items():
            if isinstance(v, (int, float)):
                summary_rows.append({
                    "section": "drawdown",
                    "metric" : k,
                    "value"  : float(v),
                })

        if summary_rows:
            _write(
                pd.DataFrame(summary_rows),
                f"{self.bt_path}/bt02_risk_summary"
            )
            print("  ✓ bt02_risk_summary")

        # IC stability by year
        yr_hit = ic_stab.get("yr_hit_rates", {})
        if yr_hit:
            _write(
                pd.DataFrame([
                    {"year": k, "ic_hit_rate": v}
                    for k, v in yr_hit.items()
                ]),
                f"{self.bt_path}/bt02_ic_by_year"
            )
            print("  ✓ bt02_ic_by_year")

        # Annual returns
        ann_rets = stability.get("ann_rets", {})
        if ann_rets:
            _write(
                pd.DataFrame([
                    {"year": k, "ann_ret": v}
                    for k, v in ann_rets.items()
                ]),
                f"{self.bt_path}/bt02_annual_returns"
            )
            print("  ✓ bt02_annual_returns")

    # ─────────────────────────────────────────────────
    #  Run
    # ─────────────────────────────────────────────────
    def run(self):
        print("="*55)
        print("Backtest 02 — Risk Analysis")
        print("="*55)
        start = datetime.now()

        daily, ic_df, pred, mkt, mkt_ret = (
            self.load_data()
        )
        tail_risk = self.analyze_tail_risk(daily)
        dd_results, dd_df, dd_series = (
            self.analyze_drawdowns(daily)
        )
        factor, df_mkt = (
            self.analyze_factor_exposure(
                daily, mkt_ret
            )
        )
        corr, _ = self.analyze_correlations(
            daily, mkt_ret
        )
        vol  = self.analyze_volatility(daily)
        ic_s = self.analyze_ic_stability(ic_df)
        stab = self.analyze_stability(daily, ic_df)

        self.write_results(
            tail_risk, dd_results, dd_df,
            factor, corr, vol, ic_s, stab
        )

        elapsed = (
            datetime.now()-start
        ).seconds / 60
        print(f"\nTotal time : {elapsed:.1f} min")
        print("Backtest 02 COMPLETE ✓")
        return {
            "daily"     : daily,
            "ic_df"     : ic_df,
            "tail_risk" : tail_risk,
            "dd_results": dd_results,
            "dd_df"     : dd_df,
            "dd_series" : dd_series,
            "factor"    : factor,
            "df_mkt"    : df_mkt,
            "corr"      : corr,
            "vol"       : vol,
            "ic_stab"   : ic_s,
            "stability" : stab,
        }

# COMMAND ----------

class Backtest02Charts:
    TEMPLATE = "plotly_dark"
    C = {
        "primary"  : "#2196F3",
        "secondary": "#FF5722",
        "success"  : "#4CAF50",
        "warning"  : "#FFC107",
        "purple"   : "#9C27B0",
        "teal"     : "#00BCD4",
    }
    REGIME_C = {
        "Bull"   : "#4CAF50",
        "Bear"   : "#FF5722",
        "HighVol": "#FFC107",
    }

    # ─────────────────────────────────────────────────
    #  Chart 1 — Tail Risk
    # ─────────────────────────────────────────────────
    def chart_tail_risk(self, daily, tail_risk):
        r  = daily["ls_net"].dropna() * 100
        st = tail_risk.get("stress", {})

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Return Distribution + VaR/CVaR",
                "QQ-Plot vs Normal",
                "Stress Test Results (%)",
                "Rolling VaR 95% (63d)",
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
            specs=[
                [{"type":"xy"},{"type":"xy"}],
                [{"type":"xy"},{"type":"xy"}],
            ]
        )

        # Distribution + VaR lines
        fig.add_trace(go.Histogram(
            x=r, nbinsx=80,
            marker_color=self.C["primary"],
            opacity=0.8, showlegend=False
        ), row=1, col=1)
        for key, label, color in [
            ("hvar_95","VaR 95%",
             self.C["warning"]),
            ("cvar_95","CVaR 95%",
             self.C["secondary"]),
            ("hvar_99","VaR 99%",
             self.C["purple"]),
        ]:
            v = tail_risk.get(key, 0) * 100
            fig.add_vline(
                x=v, line_dash="dash",
                line_color=color, opacity=0.8,
                annotation_text=(
                    f"{label}={v:.2f}%"
                ),
                row=1, col=1
            )
        fig.update_xaxes(
            title_text="Daily Return (%)",
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="Count",
            row=1, col=1
        )

        # QQ-plot
        from scipy import stats as sp
        (osm, osr),(slope,intercept,_) = (
            sp.probplot(
                daily["ls_net"].dropna().values
            )
        )
        fig.add_trace(go.Scatter(
            x=osm, y=osr, mode="markers",
            marker=dict(
                color=self.C["primary"],
                size=2, opacity=0.4
            ),
            showlegend=False
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=[min(osm), max(osm)],
            y=[
                slope*min(osm)+intercept,
                slope*max(osm)+intercept
            ],
            mode="lines",
            line=dict(color="white", dash="dash"),
            showlegend=False
        ), row=1, col=2)
        fig.update_xaxes(
            title_text="Theoretical Quantile",
            row=1, col=2
        )
        fig.update_yaxes(
            title_text="Observed Quantile",
            row=1, col=2
        )

        # Stress tests bar chart
        st_keys = [
            k for k, v in st.items()
            if isinstance(v, float)
        ]
        st_vals = [st[k]*100 for k in st_keys]
        bar_c   = [
            self.C["success"] if v > 0
            else self.C["secondary"]
            for v in st_vals
        ]
        fig.add_trace(go.Bar(
            x=st_keys, y=st_vals,
            marker_color=bar_c,
            text=[f"{v:.1f}%" for v in st_vals],
            textposition="outside",
            showlegend=False
        ), row=2, col=1)
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=2, col=1
        )
        fig.update_yaxes(
            title_text="Return (%)",
            row=2, col=1
        )

        # Rolling VaR
        r_s = daily["ls_net"].fillna(0)
        daily_c = daily.copy()
        daily_c["date"] = pd.to_datetime(
            daily_c["date"]
        )
        daily_c = daily_c.sort_values("date")
        roll_var = (
            r_s.rolling(63).quantile(0.05) * 100
        )
        fig.add_trace(go.Scatter(
            x=daily_c["date"].values,
            y=roll_var.values,
            mode="lines",
            line=dict(
                color=self.C["secondary"],
                width=1.5
            ),
            fill="tozeroy",
            fillcolor="rgba(255,87,34,0.2)",
            showlegend=False
        ), row=2, col=2)
        fig.update_xaxes(
            title_text="Date", row=2, col=2
        )
        fig.update_yaxes(
            title_text="VaR (%)", row=2, col=2
        )

        hv  = tail_risk.get("hvar_95", 0) * 100
        cv  = tail_risk.get("cvar_95", 0) * 100
        tr  = tail_risk.get("tail_ratio", 0)
        gp  = tail_risk.get("gain_pain", 0)
        fig.update_layout(
            title=(
                f"<b>BT02 — Tail Risk Analysis<br>"
                f"<sup>"
                f"VaR95={hv:.3f}% | "
                f"CVaR95={cv:.3f}% | "
                f"Tail Ratio={tr:.3f} | "
                f"Gain/Pain={gp:.3f}"
                f"</sup></b>"
            ),
            template=self.TEMPLATE,
            height=750,
            hovermode="x unified"
        )
        fig.show()

    # ─────────────────────────────────────────────────
    #  Chart 2 — Drawdown Analysis
    # ─────────────────────────────────────────────────
    def chart_drawdown_analysis(self, daily,
                                  dd_df, dd_series):
        daily = daily.copy()
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date")
        r     = daily["ls_net"].fillna(0)
        cum   = (1 + r).cumprod()

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=[
                "Cumulative Return + DD Periods",
                "Underwater Curve (%)",
                "Drawdown Depth Distribution",
            ],
            vertical_spacing=0.08,
            row_heights=[0.45, 0.35, 0.20],
            specs=[
                [{"type":"xy"}],
                [{"type":"xy"}],
                [{"type":"xy"}],
            ]
        )

        # Cumulative return
        fig.add_trace(go.Scatter(
            x=daily["date"], y=cum,
            name="Cum Return", mode="lines",
            line=dict(
                color=self.C["primary"], width=2
            )
        ), row=1, col=1)

        # Shade top-10 drawdown periods
        if len(dd_df) > 0:
            top_dd = dd_df.nsmallest(10, "max_dd")
            for _, row in top_dd.iterrows():
                try:
                    fig.add_vrect(
                        x0=str(row["start"])[:10],
                        x1=str(
                            row["recovery"]
                        )[:10],
                        fillcolor=(
                            "rgba(255,87,34,0.12)"
                        ),
                        layer="below",
                        line_width=0,
                        row=1, col=1
                    )
                except Exception:
                    pass

        fig.update_yaxes(
            title_text="Cum Return",
            row=1, col=1
        )

        # Underwater
        fig.add_trace(go.Scatter(
            x=daily["date"],
            y=dd_series.values * 100,
            fill="tozeroy",
            fillcolor="rgba(255,87,34,0.3)",
            line=dict(
                color=self.C["secondary"],
                width=1
            ),
            showlegend=False
        ), row=2, col=1)
        fig.update_yaxes(
            title_text="Drawdown (%)",
            row=2, col=1
        )

        # DD depth histogram
        if len(dd_df) > 0:
            fig.add_trace(go.Histogram(
                x=dd_df["max_dd"] * 100,
                nbinsx=30,
                marker_color=self.C["secondary"],
                opacity=0.8,
                showlegend=False
            ), row=3, col=1)
        fig.update_xaxes(
            title_text="DD Depth (%)",
            row=3, col=1
        )
        fig.update_yaxes(
            title_text="Count",
            row=3, col=1
        )

        max_dd  = float(dd_series.min()) \
                  if len(dd_series) > 0 else 0
        n_dd    = len(dd_df)
        avg_dur = (
            float(dd_df["duration_days"].mean())
            if len(dd_df) > 0 else 0
        )
        avg_rec = (
            float(dd_df["recovery_days"].mean())
            if len(dd_df) > 0 else 0
        )
        fig.update_layout(
            title=(
                f"<b>BT02 — Drawdown Analysis<br>"
                f"<sup>"
                f"Max DD={max_dd*100:.2f}% | "
                f"N={n_dd} | "
                f"Avg Dur={avg_dur:.0f}d | "
                f"Avg Rec={avg_rec:.0f}d"
                f"</sup></b>"
            ),
            template=self.TEMPLATE,
            height=800,
            hovermode="x unified"
        )
        fig.show()

    # ─────────────────────────────────────────────────
    #  Chart 3 — Factor Exposure (FIXED)
    # ─────────────────────────────────────────────────
    def chart_factor_exposure(self, data,
                               factor_res):
        df = data.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        roll_beta = factor_res.get(
            "rolling_beta", []
        )
        dates_b   = df["date"].iloc[
            252:252+len(roll_beta)
        ].values

        # KEY FIX: {"type":"domain"} for table
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Rolling 252d Market Beta",
                "Up vs Down Capture",
                "Strategy vs Market Returns",
                "Factor Summary",
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
            specs=[
                [{"type":"xy"},  {"type":"xy"}],
                [{"type":"xy"},  {"type":"domain"}],
            ]
        )

        # Rolling beta
        if len(roll_beta) > 0:
            fig.add_trace(go.Scatter(
                x=dates_b, y=roll_beta,
                mode="lines",
                line=dict(
                    color=self.C["teal"], width=1.5
                ),
                showlegend=False
            ), row=1, col=1)
            fig.add_hline(
                y=0, line_dash="dash",
                line_color="white", opacity=0.5,
                row=1, col=1
            )
            avg_b = float(
                factor_res.get("beta_market", 0)
            )
            fig.add_hline(
                y=avg_b, line_dash="dot",
                line_color=self.C["warning"],
                annotation_text=f"Avg={avg_b:.4f}",
                row=1, col=1
            )
        fig.update_xaxes(
            title_text="Date", row=1, col=1
        )
        fig.update_yaxes(
            title_text="Beta", row=1, col=1
        )

        # Up/Down capture
        up = factor_res.get("up_capture", 0)
        dn = factor_res.get("dn_capture", 0)
        fig.add_trace(go.Bar(
            x=["Up Capture","Down Capture"],
            y=[up, dn],
            marker_color=[
                self.C["success"],
                self.C["secondary"]
            ],
            text=[f"{up:.3f}", f"{dn:.3f}"],
            textposition="outside",
            showlegend=False
        ), row=1, col=2)
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=1, col=2
        )
        fig.update_yaxes(
            title_text="Capture Ratio",
            row=1, col=2
        )

        # Scatter: strategy vs market
        if ("mkt_ret" in df.columns and
                "ls_net" in df.columns):
            fig.add_trace(go.Scatter(
                x=df["mkt_ret"]*100,
                y=df["ls_net"]*100,
                mode="markers",
                marker=dict(
                    color=self.C["primary"],
                    size=2, opacity=0.3
                ),
                showlegend=False
            ), row=2, col=1)
        fig.update_xaxes(
            title_text="Market Ret (%)",
            row=2, col=1
        )
        fig.update_yaxes(
            title_text="Strategy Ret (%)",
            row=2, col=1
        )

        # Summary table (domain type)
        keys = [
            "beta_market",
            "corr_market",
            "alpha_ann",
            "information_ratio",
            "tracking_error",
            "up_capture",
            "dn_capture",
        ]
        display_keys = [
            "Beta (Market)",
            "Corr (Market)",
            "Alpha (Ann)",
            "Info Ratio",
            "Tracking Error",
            "Up Capture",
            "Down Capture",
        ]
        vals = [
            f"{factor_res.get(k,0):.4f}"
            for k in keys
        ]
        fig.add_trace(go.Table(
            header=dict(
                values=["Metric","Value"],
                fill_color="#1e1e2e",
                font=dict(color="white", size=12),
                align="center"
            ),
            cells=dict(
                values=[display_keys, vals],
                fill_color=[
                    ["#263238"] * len(display_keys)
                ],
                font=dict(color="white", size=11),
                align=["left","center"]
            )
        ), row=2, col=2)

        beta  = factor_res.get("beta_market", 0)
        alpha = factor_res.get("alpha_ann", 0)
        ir    = factor_res.get(
            "information_ratio", 0
        )
        fig.update_layout(
            title=(
                f"<b>BT02 — Factor Exposure<br>"
                f"<sup>"
                f"Beta={beta:.4f} | "
                f"Alpha={alpha*100:.2f}%/yr | "
                f"IR={ir:.3f}"
                f"</sup></b>"
            ),
            template=self.TEMPLATE,
            height=700
        )
        fig.show()

    # ─────────────────────────────────────────────────
    #  Chart 4 — Volatility
    # ─────────────────────────────────────────────────
    def chart_volatility(self, daily, vol_res):
        daily = daily.copy()
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date")

        r21  = vol_res["roll_vol_21"]  * 100
        r63  = vol_res["roll_vol_63"]  * 100
        r252 = vol_res["roll_vol_252"] * 100

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=[
                "Rolling Volatility (21d, 63d, 252d)",
                "Vol of Vol (63d rolling std of vol)",
                "Vol Regime — High vs Low",
            ],
            vertical_spacing=0.08,
            row_heights=[0.40, 0.30, 0.30],
            specs=[
                [{"type":"xy"}],
                [{"type":"xy"}],
                [{"type":"xy"}],
            ]
        )

        for series, name, color, width in [
            (r21,  "21d",  self.C["primary"],  1.5),
            (r63,  "63d",  self.C["teal"],     2.0),
            (r252, "252d", self.C["warning"],  2.0),
        ]:
            n = len(series)
            fig.add_trace(go.Scatter(
                x=daily["date"].values[:n],
                y=series.values,
                name=name, mode="lines",
                line=dict(color=color, width=width)
            ), row=1, col=1)
        fig.update_yaxes(
            title_text="Ann Vol (%)",
            row=1, col=1
        )

        # Vol-of-vol
        vol_of_vol = r21.rolling(63).std()
        n = len(vol_of_vol)
        fig.add_trace(go.Scatter(
            x=daily["date"].values[:n],
            y=vol_of_vol.values,
            mode="lines",
            line=dict(
                color=self.C["purple"], width=1.5
            ),
            fill="tozeroy",
            fillcolor="rgba(156,39,176,0.15)",
            showlegend=False
        ), row=2, col=1)
        fig.update_yaxes(
            title_text="Vol of Vol",
            row=2, col=1
        )

        # High/low vol bar
        vol_med = float(r21.median())
        hi_mask = r21 > vol_med
        bar_c   = [
            self.C["secondary"] if h
            else self.C["success"]
            for h in hi_mask
        ]
        fig.add_trace(go.Bar(
            x=daily["date"].values[:len(r21)],
            y=r21.values,
            marker_color=bar_c,
            showlegend=False
        ), row=3, col=1)
        fig.add_hline(
            y=vol_med, line_dash="dot",
            line_color=self.C["warning"],
            annotation_text=(
                f"Median={vol_med:.2f}%"
            ),
            row=3, col=1
        )
        fig.update_yaxes(
            title_text="Ann Vol (%)",
            row=3, col=1
        )

        v21  = vol_res.get("vol_21_mean", 0)*100
        vv   = vol_res.get("vol_of_vol", 0)*100
        hi_s = vol_res.get("hi_vol_sharpe", 0)
        lo_s = vol_res.get("lo_vol_sharpe", 0)
        fig.update_layout(
            title=(
                f"<b>BT02 — Volatility Analysis<br>"
                f"<sup>"
                f"Ann Vol={v21:.2f}% | "
                f"Vol-of-Vol={vv:.2f}% | "
                f"Hi-Vol Sharpe={hi_s:.2f} | "
                f"Lo-Vol Sharpe={lo_s:.2f}"
                f"</sup></b>"
            ),
            template=self.TEMPLATE,
            height=800,
            hovermode="x unified"
        )
        fig.show()

    # ─────────────────────────────────────────────────
    #  Chart 5 — IC Stability
    # ─────────────────────────────────────────────────
    def chart_ic_stability(self, ic_df, ic_stab):
        ic_df = ic_df.copy()
        ic_df["date"] = pd.to_datetime(ic_df["date"])
        ic_df = ic_df.sort_values("date")
        ic_df["year"] = ic_df["date"].dt.year

        yr_hit = ic_stab.get("yr_hit_rates", {})

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Rolling IC (63d, 126d, 252d)",
                "Annual IC Hit Rate (%)",
                "Annual Mean IC",
                "IC Autocorrelation",
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
            specs=[
                [{"type":"xy"},{"type":"xy"}],
                [{"type":"xy"},{"type":"xy"}],
            ]
        )

        vals = ic_df["ic"].fillna(0)

        # Rolling IC
        for w, color in [
            (63,  self.C["primary"]),
            (126, self.C["teal"]),
            (252, self.C["warning"]),
        ]:
            fig.add_trace(go.Scatter(
                x=ic_df["date"].values,
                y=vals.rolling(w).mean().values,
                name=f"{w}d", mode="lines",
                line=dict(color=color, width=2)
            ), row=1, col=1)
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="Rolling IC",
            row=1, col=1
        )

        # Annual hit rate
        if yr_hit:
            years = list(yr_hit.keys())
            hits  = list(yr_hit.values())
            bar_c = [
                self.C["success"] if h > 0.5
                else self.C["secondary"]
                for h in hits
            ]
            fig.add_trace(go.Bar(
                x=years,
                y=[h*100 for h in hits],
                marker_color=bar_c,
                text=[f"{h:.0%}" for h in hits],
                textposition="outside",
                showlegend=False
            ), row=1, col=2)
            fig.add_hline(
                y=50, line_dash="dot",
                line_color=self.C["warning"],
                annotation_text="50%",
                row=1, col=2
            )
        fig.update_yaxes(
            title_text="Hit Rate (%)",
            row=1, col=2
        )

        # Annual mean IC
        yr_ic = ic_df.groupby("year")["ic"].mean()
        yr_c  = [
            self.C["success"] if v > 0
            else self.C["secondary"]
            for v in yr_ic.values
        ]
        fig.add_trace(go.Bar(
            x=yr_ic.index.tolist(),
            y=yr_ic.values.tolist(),
            marker_color=yr_c,
            text=[
                f"{v:.3f}" for v in yr_ic.values
            ],
            textposition="outside",
            showlegend=False
        ), row=2, col=1)
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=2, col=1
        )
        fig.update_yaxes(
            title_text="Mean IC",
            row=2, col=1
        )

        # Autocorrelation
        ac      = ic_stab.get("ic_autocorr", {})
        lags    = [
            int(k.split("lag")[-1])
            for k in ac.keys()
        ]
        vals_ac = list(ac.values())
        if lags:
            fig.add_trace(go.Bar(
                x=lags, y=vals_ac,
                marker_color=self.C["teal"],
                text=[
                    f"{v:.3f}" for v in vals_ac
                ],
                textposition="outside",
                showlegend=False
            ), row=2, col=2)
            fig.add_hline(
                y=0, line_dash="dash",
                line_color="white", opacity=0.3,
                row=2, col=2
            )
        fig.update_xaxes(
            title_text="Lag (days)",
            row=2, col=2
        )
        fig.update_yaxes(
            title_text="Autocorrelation",
            row=2, col=2
        )

        avg_hit = ic_stab.get("avg_yr_hit", 0)
        pct_pos = ic_stab.get("pct_pos_years", 0)
        icir    = ic_stab.get("roll_icir_252", 0)
        fig.update_layout(
            title=(
                f"<b>BT02 — IC Stability<br>"
                f"<sup>"
                f"Avg Annual Hit={avg_hit:.1%} | "
                f"% Pos Years={pct_pos:.1%} | "
                f"Roll ICIR(252d)={icir:.3f}"
                f"</sup></b>"
            ),
            template=self.TEMPLATE,
            height=750
        )
        fig.show()

    # ─────────────────────────────────────────────────
    #  Chart 6 — Rolling Sharpe + Annual Returns
    # ─────────────────────────────────────────────────
    def chart_rolling_sharpe(self, daily,
                              stability):
        daily = daily.copy()
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date")

        roll_s = stability.get("roll_sharpe", {})
        ann_r  = stability.get("ann_rets", {})

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=[
                "Rolling Sharpe Ratio "
                "(63d, 126d, 252d)",
                "Annual Returns (%)",
            ],
            vertical_spacing=0.1,
            row_heights=[0.60, 0.40],
            specs=[
                [{"type":"xy"}],
                [{"type":"xy"}],
            ]
        )

        for w, color in [
            (63,  self.C["primary"]),
            (126, self.C["teal"]),
            (252, self.C["warning"]),
        ]:
            if w not in roll_s:
                continue
            rs = roll_s[w]
            n  = len(rs)
            fig.add_trace(go.Scatter(
                x=daily["date"].values[:n],
                y=rs.values,
                name=f"{w}d Sharpe",
                mode="lines",
                line=dict(color=color, width=1.5)
            ), row=1, col=1)

        for thresh, label, color in [
            (1.0, "Sharpe=1.0",
             self.C["success"]),
            (0.0, "Zero", "white"),
        ]:
            fig.add_hline(
                y=thresh, line_dash="dot",
                line_color=color, opacity=0.5,
                annotation_text=label,
                row=1, col=1
            )
        fig.update_yaxes(
            title_text="Sharpe Ratio",
            row=1, col=1
        )

        # Annual returns bars
        years  = list(ann_r.keys())
        rets   = [v*100 for v in ann_r.values()]
        bar_c  = [
            self.C["success"] if v > 0
            else self.C["secondary"]
            for v in rets
        ]
        fig.add_trace(go.Bar(
            x=years, y=rets,
            marker_color=bar_c,
            text=[f"{v:.1f}%" for v in rets],
            textposition="outside",
            showlegend=False
        ), row=2, col=1)
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=2, col=1
        )
        fig.update_xaxes(
            title_text="Year", row=2, col=1
        )
        fig.update_yaxes(
            title_text="Ann Ret (%)",
            row=2, col=1
        )

        pos_yr = stability.get("pct_pos_years", 0)
        best   = stability.get("best_year", 0)
        worst  = stability.get("worst_year", 0)
        fig.update_layout(
            title=(
                f"<b>BT02 — Stability Analysis<br>"
                f"<sup>"
                f"% Pos Years={pos_yr:.1%} | "
                f"Best={best*100:.1f}% | "
                f"Worst={worst*100:.1f}%"
                f"</sup></b>"
            ),
            template=self.TEMPLATE,
            height=700,
            hovermode="x unified"
        )
        fig.show()

    # ─────────────────────────────────────────────────
    #  Chart 7 — Correlation Analysis
    # ─────────────────────────────────────────────────
    def chart_correlation(self, data, corr_res):
        df = data.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        rc63  = corr_res.get("roll_corr_63", [])
        rc252 = corr_res.get("roll_corr_252", [])
        dates63  = df["date"].iloc[
            63:63+len(rc63)
        ].values if len(rc63) > 0 else []
        dates252 = df["date"].iloc[
            252:252+len(rc252)
        ].values if len(rc252) > 0 else []

        regime_c = corr_res.get("regime_corr", {})

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Rolling Correlation vs Market",
                "Correlation by Regime",
                "Correlation Distribution (63d)",
                "Tail Beta",
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
            specs=[
                [{"type":"xy"},{"type":"xy"}],
                [{"type":"xy"},{"type":"xy"}],
            ]
        )

        # Rolling correlation
        if len(rc63) > 0:
            fig.add_trace(go.Scatter(
                x=dates63, y=rc63,
                name="63d", mode="lines",
                line=dict(
                    color=self.C["primary"],
                    width=1.5
                )
            ), row=1, col=1)
        if len(rc252) > 0:
            fig.add_trace(go.Scatter(
                x=dates252, y=rc252,
                name="252d", mode="lines",
                line=dict(
                    color=self.C["warning"],
                    width=2
                )
            ), row=1, col=1)
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="Correlation",
            row=1, col=1
        )

        # Regime correlation
        if regime_c:
            regimes  = list(regime_c.keys())
            r_corrs  = list(regime_c.values())
            r_colors = [
                self.REGIME_C.get(r,"#9E9E9E")
                for r in regimes
            ]
            fig.add_trace(go.Bar(
                x=regimes, y=r_corrs,
                marker_color=r_colors,
                text=[f"{v:.3f}" for v in r_corrs],
                textposition="outside",
                showlegend=False
            ), row=1, col=2)
            fig.add_hline(
                y=0, line_dash="dash",
                line_color="white", opacity=0.3,
                row=1, col=2
            )
        fig.update_yaxes(
            title_text="Correlation",
            row=1, col=2
        )

        # Correlation distribution
        if len(rc63) > 0:
            fig.add_trace(go.Histogram(
                x=rc63, nbinsx=40,
                marker_color=self.C["teal"],
                opacity=0.8, showlegend=False
            ), row=2, col=1)
            avg_c = float(np.mean(rc63))
            fig.add_vline(
                x=avg_c, line_dash="dot",
                line_color=self.C["warning"],
                annotation_text=f"Avg={avg_c:.3f}",
                row=2, col=1
            )
        fig.update_xaxes(
            title_text="Correlation",
            row=2, col=1
        )
        fig.update_yaxes(
            title_text="Count",
            row=2, col=1
        )

        # Tail beta bar
        tail_b  = corr_res.get("tail_beta", 0)
        full_b  = corr_res.get(
            "avg_corr_63", 0
        ) * 0  # placeholder — show 0 vs tail
        fig.add_trace(go.Bar(
            x=["Normal Beta","Tail Beta"],
            y=[0.0, tail_b],
            marker_color=[
                self.C["primary"],
                self.C["secondary"]
            ],
            text=[f"{0.0:.4f}", f"{tail_b:.4f}"],
            textposition="outside",
            showlegend=False
        ), row=2, col=2)
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=2, col=2
        )
        fig.update_yaxes(
            title_text="Beta",
            row=2, col=2
        )

        avg63  = corr_res.get("avg_corr_63", 0)
        avg252 = corr_res.get("avg_corr_252", 0)
        stab   = corr_res.get("corr_stability", 0)
        fig.update_layout(
            title=(
                f"<b>BT02 — Correlation Analysis<br>"
                f"<sup>"
                f"Avg Corr(63d)={avg63:.4f} | "
                f"Avg Corr(252d)={avg252:.4f} | "
                f"Tail Beta={tail_b:.4f} | "
                f"Stability={stab:.4f}"
                f"</sup></b>"
            ),
            template=self.TEMPLATE,
            height=700
        )
        fig.show()

    # ─────────────────────────────────────────────────
    #  Run all charts
    # ─────────────────────────────────────────────────
    def run_all(self, results):
        print("\n" + "="*55)
        print("Backtest 02 Charts")
        print("="*55)

        daily     = results["daily"]
        ic_df     = results["ic_df"]
        dd_df     = results["dd_df"]
        dd_series = results["dd_series"]

        print("\n[1/7] Tail Risk...")
        self.chart_tail_risk(
            daily, results["tail_risk"]
        )

        print("[2/7] Drawdown Analysis...")
        self.chart_drawdown_analysis(
            daily, dd_df, dd_series
        )

        print("[3/7] Factor Exposure...")
        self.chart_factor_exposure(
            results["df_mkt"], results["factor"]
        )

        print("[4/7] Volatility...")
        self.chart_volatility(
            daily, results["vol"]
        )

        print("[5/7] IC Stability...")
        self.chart_ic_stability(
            ic_df, results["ic_stab"]
        )

        print("[6/7] Rolling Sharpe + Stability...")
        self.chart_rolling_sharpe(
            daily, results["stability"]
        )

        print("[7/7] Correlation...")
        self.chart_correlation(
            results["df_mkt"], results["corr"]
        )

        print("\nAll 7 charts ✓")

# COMMAND ----------

bt02 = Backtest02(
    spark     = spark,
    gold_path = GOLD_PATH,
    ml_path   = ML_PATH,
    bt_path   = BT_PATH,
)

results = bt02.run()

charts = Backtest02Charts()
charts.run_all(results)

print("\nBacktest 02 COMPLETE ✓")

# COMMAND ----------

risk = spark.read.format("delta").load(
    f"{BT_PATH}/bt02_risk_summary"
).toPandas()

ann  = spark.read.format("delta").load(
    f"{BT_PATH}/bt02_annual_returns"
).toPandas()

ic_yr = spark.read.format("delta").load(
    f"{BT_PATH}/bt02_ic_by_year"
).toPandas()

print("="*55)
print("Backtest 02 — Risk Summary")
print("="*55)

tr = results["tail_risk"]
dd = results["dd_results"]
fa = results["factor"]
vo = results["vol"]
ic = results["ic_stab"]
st = results["stability"]

print(f"\nTail Risk:")
print(f"  VaR 95%   : {tr.get('hvar_95',0)*100:.3f}%")
print(f"  CVaR 95%  : {tr.get('cvar_95',0)*100:.3f}%")
print(f"  VaR 99%   : {tr.get('hvar_99',0)*100:.3f}%")
print(f"  Tail Ratio: {tr.get('tail_ratio',0):.3f}")
print(f"  Gain/Pain : {tr.get('gain_pain',0):.3f}")

print(f"\nDrawdown:")
print(f"  Max DD    : {dd.get('max_dd',0)*100:.2f}%")
print(f"  Avg DD    : {dd.get('avg_dd',0)*100:.2f}%")
print(f"  % in DD   : "
      f"{dd.get('pct_time_in_dd',0):.1%}")
print(f"  Avg Dur   : "
      f"{dd.get('avg_dd_dur',0):.0f} days")
print(f"  N DDs     : "
      f"{dd.get('n_drawdowns',0)}")

print(f"\nFactor:")
print(f"  Beta      : {fa.get('beta_market',0):.4f} "
      f"({'✅ neutral' if abs(fa.get('beta_market',0))<0.1 else '⚠️'})")
print(f"  Alpha/yr  : "
      f"{fa.get('alpha_ann',0)*100:.2f}%")
print(f"  Info Ratio: "
      f"{fa.get('information_ratio',0):.4f}")

print(f"\nVolatility:")
print(f"  Ann Vol   : "
      f"{vo.get('vol_21_mean',0)*100:.2f}%")
print(f"  Vol-of-Vol: "
      f"{vo.get('vol_of_vol',0)*100:.2f}%")

print(f"\nIC Stability:")
print(f"  Avg Yr Hit: "
      f"{ic.get('avg_yr_hit',0):.1%}")
print(f"  % Pos Yrs : "
      f"{ic.get('pct_pos_years',0):.1%}")

print(f"\nAnnual Returns:")
ann_s = ann.sort_values("year")
print(ann_s[["year","ann_ret"]].round(4).to_string(
    index=False
))

print(f"\n{'='*55}")
print(f"Backtest Progress:")
print(f"  Backtest 01 ✅ Full portfolio backtest")
print(f"  Backtest 02 ✅ Risk analysis")
print(f"  Backtest 03 🔲 Performance attribution")
print(f"{'='*55}")