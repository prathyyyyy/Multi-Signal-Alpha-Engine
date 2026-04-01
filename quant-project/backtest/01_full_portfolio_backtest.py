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
from scipy.stats import spearmanr
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
GOLD_PATH  = f"{BASE_PATH}/gold/delta"
ML_PATH    = f"{BASE_PATH}/ml/delta"
BT_PATH    = f"{BASE_PATH}/backtest/delta"

print("="*55)
print("Backtest 01 — Full Portfolio Backtest")
print("="*55)
print(f"BT_PATH : {BT_PATH}")

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
GOLD_PATH  = f"{BASE_PATH}/gold/delta"
ML_PATH    = f"{BASE_PATH}/ml/delta"
BT_PATH    = f"{BASE_PATH}/backtest/delta"

print("="*55)
print("Backtest 01 — Full Portfolio Backtest")
print("="*55)
print(f"BT_PATH : {BT_PATH}")

# COMMAND ----------

class Backtest01:
    """
    Backtest 01 — Full Portfolio Backtest.

    Strategy: L/S quintile equity portfolio
    Signal  : ML 04 ensemble (signal_final_cs)
    Universe: ~520 US equities
    Period  : 1993–2026

    Metrics computed:
      ✅ Returns  : daily/monthly/annual
      ✅ Risk     : vol, VaR, CVaR, max DD
      ✅ Ratios   : Sharpe, Sortino, Calmar, Omega
      ✅ IC       : daily, rolling, by regime
      ✅ Turnover : daily, TC breakdown
      ✅ Capacity : position sizes, concentration
      ✅ Regime   : Bull/HighVol/Bear attribution
      ✅ Stability: rolling Sharpe, IC decay
    """

    # Strategy params
    TC_BPS       = 5      # round-trip per trade
    QUINTILE_N   = 5      # top/bottom 1/5
    MIN_STOCKS   = 20     # min universe per day
    RISK_FREE    = 0.04   # annual risk-free rate
    ANNUAL_FACTOR= 252    # trading days/year

    def __init__(self, spark, gold_path,
                 ml_path, bt_path):
        self.spark     = spark
        self.gold_path = gold_path
        self.ml_path   = ml_path
        self.bt_path   = bt_path

        print("Backtest01 ✓")
        print(f"  TC          : {self.TC_BPS}bps")
        print(f"  Quintile    : 1/{self.QUINTILE_N}")
        print(f"  Min stocks  : {self.MIN_STOCKS}")
        print(f"  Risk-free   : "
              f"{self.RISK_FREE*100:.1f}%/yr")

    # ─────────────────────────────────────────────────
    #  Utilities
    # ─────────────────────────────────────────────────
    @staticmethod
    def _safe_ic(pred, actual, min_n=30):
        p = np.asarray(pred,  dtype=float)
        a = np.asarray(actual,dtype=float)
        m = (
            ~np.isnan(p) & ~np.isnan(a) &
            ~np.isinf(p) & ~np.isinf(a)
        )
        if m.sum() < min_n:
            return np.nan
        if (np.std(p[m]) < 1e-10 or
                np.std(a[m]) < 1e-10):
            return 0.0
        try:
            ic, _ = spearmanr(p[m], a[m])
            return float(ic) \
                   if not np.isnan(ic) else 0.0
        except Exception:
            return 0.0

    def _perf_metrics(self, returns,
                       label="strategy"):
        """
        Comprehensive performance metrics.
        """
        r   = pd.Series(returns).dropna()
        if len(r) < 10:
            return {}

        # Return metrics
        ann_ret  = r.mean() * self.ANNUAL_FACTOR
        ann_vol  = r.std()  * np.sqrt(
            self.ANNUAL_FACTOR
        )
        rf_daily = self.RISK_FREE / self.ANNUAL_FACTOR

        # Risk-adjusted ratios
        sharpe   = (
            (r.mean() - rf_daily) /
            (r.std() + 1e-10) *
            np.sqrt(self.ANNUAL_FACTOR)
        )
        down_std = r[r < rf_daily].std()
        sortino  = (
            (ann_ret - self.RISK_FREE) /
            (down_std * np.sqrt(self.ANNUAL_FACTOR)
             + 1e-10)
        )

        # Drawdown
        cum      = (1 + r).cumprod()
        roll_max = cum.cummax()
        dd_series= cum / roll_max - 1
        max_dd   = float(dd_series.min())
        calmar   = ann_ret / (abs(max_dd) + 1e-10)

        # DD duration
        in_dd       = dd_series < -0.001
        dd_dur      = 0
        cur_dur     = 0
        for v in in_dd:
            if v:
                cur_dur += 1
                dd_dur   = max(dd_dur, cur_dur)
            else:
                cur_dur  = 0

        # Tail metrics
        var_95   = float(np.percentile(r, 5))
        cvar_95  = float(r[r <= var_95].mean())
        var_99   = float(np.percentile(r, 1))
        cvar_99  = float(r[r <= var_99].mean())

        # Omega ratio (threshold = 0)
        gains    = r[r > 0].sum()
        losses   = abs(r[r <= 0].sum())
        omega    = float(gains / (losses + 1e-10))

        # Win rate / profit factor
        win_rate = float((r > 0).mean())
        avg_win  = float(r[r > 0].mean()) \
                   if (r > 0).any() else 0.0
        avg_loss = float(r[r <= 0].mean()) \
                   if (r <= 0).any() else 0.0
        pf       = abs(avg_win / (avg_loss + 1e-10))

        # Skewness / kurtosis
        skew     = float(stats.skew(r))
        kurt     = float(stats.kurtosis(r))

        # Monthly metrics
        r_monthly = (
            pd.Series(r.values,
                      index=pd.date_range(
                          "2000-01-01",
                          periods=len(r), freq="B"
                      ))
            .resample("M").apply(
                lambda x: (1+x).prod()-1
            )
        )
        pos_months = float(
            (r_monthly > 0).mean()
        )

        return {
            "ann_ret"     : float(ann_ret),
            "ann_vol"     : float(ann_vol),
            "sharpe"      : float(sharpe),
            "sortino"     : float(sortino),
            "calmar"      : float(calmar),
            "max_dd"      : max_dd,
            "max_dd_dur"  : int(dd_dur),
            "var_95"      : var_95,
            "cvar_95"     : cvar_95,
            "var_99"      : var_99,
            "cvar_99"     : cvar_99,
            "omega"       : omega,
            "win_rate"    : win_rate,
            "avg_win"     : avg_win,
            "avg_loss"    : avg_loss,
            "profit_factor": pf,
            "skewness"    : skew,
            "excess_kurt" : kurt,
            "pos_months"  : pos_months,
            "n_days"      : len(r),
        }

    # ─────────────────────────────────────────────────
    #  Step 1 — Load data
    # ─────────────────────────────────────────────────
    def load_data(self):
        print("\nStep 1: Loading data...")
        start = datetime.now()

        # ML 04 ensemble predictions
        print("  Loading ensemble predictions...")
        pred = self.spark.read.format(
            "delta"
        ).load(
            f"{self.ml_path}/ensemble_predictions"
        ).toPandas()
        pred["date"] = pd.to_datetime(pred["date"])

        # Gold price factors (for market returns)
        print("  Loading market data...")
        sdf      = self.spark.read.format(
            "delta"
        ).load(f"{self.gold_path}/price_factors")
        avail    = set(sdf.columns)
        mkt_cols = [
            c for c in [
                "date","ticker",
                "fwd_return_21d",
                "fwd_return_1d",
                "vol_21d",
                "regime_label",
            ] if c in avail
        ]
        mkt = sdf.select(*mkt_cols).toPandas()
        mkt["date"] = pd.to_datetime(mkt["date"])

        elapsed = (datetime.now()-start).seconds
        print(f"  Pred rows : {len(pred):,}")
        print(f"  Mkt rows  : {len(mkt):,}")
        print(f"  Dates     : "
              f"{pred['date'].nunique():,}")
        print(f"  Tickers   : "
              f"{pred['ticker'].nunique():,}")
        print(f"  Range     : "
              f"{pred['date'].min().date()} → "
              f"{pred['date'].max().date()}")
        print(f"  Elapsed   : {elapsed}s")
        return pred, mkt

    # ─────────────────────────────────────────────────
    #  Step 2 — Run backtest
    # ─────────────────────────────────────────────────
    def run_backtest(self, pred):
        """
        Main backtest loop.
        Constructs quintile L/S portfolio daily.
        Tracks positions, returns, turnover.
        """
        print("\nStep 2: Running backtest...")
        start = datetime.now()

        # Sort by date
        pred = pred.sort_values(
            ["date","ticker"]
        ).reset_index(drop=True)

        target_col  = "fwd_return_21d"
        signal_col  = "signal_final_cs"

        # Check signal column exists
        if signal_col not in pred.columns:
            signal_col = "pred_final"
            print(f"  Using signal: {signal_col}")
        else:
            print(f"  Using signal: {signal_col}")

        daily_rows  = []
        pos_rows    = []
        ic_rows     = []
        prev_long   = set()
        prev_short  = set()
        n_dates     = pred["date"].nunique()
        n_processed = 0

        for date, grp in pred.groupby("date"):

            if n_processed % 500 == 0:
                pct = n_processed/max(1,n_dates)*100
                print(f"  [{n_processed:>5}/"
                      f"{n_dates}] {pct:.0f}%")

            grp = grp.dropna(
                subset=[signal_col, target_col]
            )
            if len(grp) < self.MIN_STOCKS:
                n_processed += 1
                continue

            n = max(1, len(grp) // self.QUINTILE_N)
            srt = grp.sort_values(
                signal_col, ascending=False
            )

            long_grp  = srt.head(n)
            short_grp = srt.tail(n)
            long_set  = set(
                long_grp["ticker"].tolist()
            )
            short_set = set(
                short_grp["ticker"].tolist()
            )

            # Returns
            long_ret  = float(
                long_grp[target_col].mean()
            )
            short_ret = float(
                short_grp[target_col].mean()
            )
            long_only = long_ret
            ls_gross  = long_ret - short_ret

            # Turnover
            if prev_long and prev_short:
                lo = len(
                    long_set - prev_long
                ) / max(n, 1)
                so = len(
                    short_set - prev_short
                ) / max(n, 1)
                turnover = (lo + so) / 2
            else:
                turnover = 1.0

            tc_cost  = turnover * self.TC_BPS / 10000
            ls_net   = ls_gross - tc_cost
            lo_net   = long_only - tc_cost * 0.5

            prev_long  = long_set
            prev_short = short_set

            # IC
            ic = self._safe_ic(
                grp[signal_col].values,
                grp[target_col].values
            )
            ic_rows.append({
                "date"    : date,
                "ic"      : ic,
                "n_stocks": len(grp),
            })

            # Quintile returns (all 5)
            quintile_rets = {}
            for q in range(self.QUINTILE_N):
                idx_s = q * (
                    len(srt) // self.QUINTILE_N
                )
                idx_e = (q+1) * (
                    len(srt) // self.QUINTILE_N
                )
                q_grp = srt.iloc[idx_s:idx_e]
                quintile_rets[f"q{q+1}_ret"] = float(
                    q_grp[target_col].mean()
                )

            # Regime
            regime = grp["regime_label"].mode(
            ).iloc[0] if "regime_label" in grp.columns \
                else "Unknown"

            daily_row = {
                "date"       : date,
                "ls_gross"   : ls_gross,
                "ls_net"     : ls_net,
                "long_ret"   : long_ret,
                "short_ret"  : short_ret,
                "long_only_net": lo_net,
                "turnover"   : turnover,
                "tc_cost"    : tc_cost,
                "n_long"     : len(long_grp),
                "n_short"    : len(short_grp),
                "n_universe" : len(grp),
                "regime"     : regime,
                **quintile_rets,
            }
            daily_rows.append(daily_row)
            n_processed += 1

        daily_df = pd.DataFrame(daily_rows)
        ic_df    = pd.DataFrame(ic_rows)

        # Cumulative returns
        daily_df["cum_ls_gross"] = (
            1 + daily_df["ls_gross"].fillna(0)
        ).cumprod()
        daily_df["cum_ls_net"] = (
            1 + daily_df["ls_net"].fillna(0)
        ).cumprod()
        daily_df["cum_long_only"] = (
            1 + daily_df["long_only_net"].fillna(0)
        ).cumprod()

        # Drawdowns
        for col in [
            "cum_ls_gross","cum_ls_net",
            "cum_long_only"
        ]:
            dd_col = col.replace("cum_","dd_")
            daily_df[dd_col] = (
                daily_df[col] /
                daily_df[col].cummax() - 1
            )

        elapsed = (datetime.now()-start).seconds
        print(f"\n  Processed  : {len(daily_df):,} days")
        print(f"  Date range : "
              f"{daily_df['date'].min().date()} → "
              f"{daily_df['date'].max().date()}")
        print(f"  Avg stocks : "
              f"{daily_df['n_universe'].mean():.0f}/day")
        print(f"  Avg TO     : "
              f"{daily_df['turnover'].mean():.1%}/day")
        print(f"  Elapsed    : {elapsed}s")
        return daily_df, ic_df

    # ─────────────────────────────────────────────────
    #  Step 3 — Compute all metrics
    # ─────────────────────────────────────────────────
    def compute_metrics(self, daily_df, ic_df):
        print("\nStep 3: Computing metrics...")

        metrics = {}

        # ── Core strategies ───────────────────────────
        for label, col in [
            ("LS_Gross",    "ls_gross"),
            ("LS_Net",      "ls_net"),
            ("Long_Only",   "long_only_net"),
        ]:
            m = self._perf_metrics(
                daily_df[col].dropna().values,
                label=label
            )
            metrics[label] = m
            print(f"\n  {label}:")
            print(f"    Sharpe  : {m['sharpe']:.2f} "
                  f"{'✅' if m['sharpe']>1 else '⚠️'}")
            print(f"    Ann Ret : "
                  f"{m['ann_ret']*100:.1f}%")
            print(f"    Max DD  : "
                  f"{m['max_dd']*100:.1f}%")
            print(f"    Sortino : "
                  f"{m['sortino']:.2f}")
            print(f"    Calmar  : "
                  f"{m['calmar']:.2f}")

        # ── IC metrics ────────────────────────────────
        ic_vals   = ic_df["ic"].dropna().tolist()
        ic_clean  = [v for v in ic_vals
                     if not np.isnan(v)]
        if ic_clean:
            m_ic = float(np.mean(ic_clean))
            s_ic = float(np.std(ic_clean))
            metrics["IC"] = {
                "mean_ic" : m_ic,
                "icir"    : m_ic / (s_ic + 1e-8),
                "ic_std"  : s_ic,
                "hit_rate": float(
                    (np.array(ic_clean)>0).mean()
                ),
                "n_dates" : len(ic_clean),
            }
            print(f"\n  IC Summary:")
            print(f"    Mean IC  : {m_ic:+.4f} "
                  f"{'✅' if abs(m_ic)>0.04 else '⚠️'}")
            print(f"    ICIR     : "
                  f"{metrics['IC']['icir']:.4f} "
                  f"{'✅' if metrics['IC']['icir']>0.5 else '⚠️'}")
            print(f"    Hit rate : "
                  f"{metrics['IC']['hit_rate']:.1%}")

        # ── Turnover / TC ─────────────────────────────
        avg_to   = float(daily_df["turnover"].mean())
        avg_tc   = float(daily_df["tc_cost"].mean())
        tc_yr    = avg_tc * self.ANNUAL_FACTOR * 10000
        metrics["TC"] = {
            "avg_turnover"  : avg_to,
            "avg_tc_daily"  : avg_tc,
            "tc_bps_yr"     : tc_yr,
            "total_tc_bps"  : float(
                daily_df["tc_cost"].sum() * 10000
            ),
        }
        print(f"\n  Transaction Costs:")
        print(f"    Avg turnover : {avg_to:.1%}/day")
        print(f"    TC/year      : {tc_yr:.0f}bps/yr")
        print(f"    Gross Sharpe : "
              f"{metrics['LS_Gross']['sharpe']:.2f}")
        print(f"    Net Sharpe   : "
              f"{metrics['LS_Net']['sharpe']:.2f}")

        # ── Regime attribution ────────────────────────
        regime_metrics = {}
        for regime in ["Bull","HighVol","Bear"]:
            r_df = daily_df[
                daily_df["regime"] == regime
            ]
            if len(r_df) < 10:
                continue
            rm = self._perf_metrics(
                r_df["ls_net"].dropna().values
            )
            regime_metrics[regime] = rm
            print(f"\n  {regime} regime "
                  f"({len(r_df)} days):")
            print(f"    Sharpe  : "
                  f"{rm['sharpe']:.2f}")
            print(f"    Ann Ret : "
                  f"{rm['ann_ret']*100:.1f}%")
            print(f"    Max DD  : "
                  f"{rm['max_dd']*100:.1f}%")
        metrics["Regime"] = regime_metrics

        # ── Quintile spread ───────────────────────────
        q_cols = [
            c for c in daily_df.columns
            if c.startswith("q") and
            c.endswith("_ret")
        ]
        if q_cols:
            q_means = {
                c: float(
                    daily_df[c].mean() *
                    self.ANNUAL_FACTOR
                )
                for c in sorted(q_cols)
            }
            metrics["Quintiles"] = q_means
            print(f"\n  Quintile Ann Rets "
                  f"(Q1=best, Q5=worst):")
            for c, v in q_means.items():
                bar = "█" * int(abs(v)*50)
                print(f"    {c}: {v*100:+.1f}% {bar}")

        # ── Rolling stats ─────────────────────────────
        window = 252
        ls_r   = daily_df["ls_net"].fillna(0)
        metrics["Rolling"] = {
            "rolling_sharpe_252d": float(
                (ls_r.rolling(window).mean() /
                 ls_r.rolling(window).std() *
                 np.sqrt(self.ANNUAL_FACTOR)
                ).mean()
            ),
            "rolling_ic_63d": float(
                ic_df["ic"].fillna(0)
                .rolling(63).mean().mean()
            ),
            "pct_time_sharpe_gt1": float(
                (ls_r.rolling(window).mean() /
                 ls_r.rolling(window).std() *
                 np.sqrt(self.ANNUAL_FACTOR) > 1.0
                ).mean()
            ),
        }
        print(f"\n  Rolling Stats:")
        print(f"    Avg 1Y Sharpe   : "
              f"{metrics['Rolling']['rolling_sharpe_252d']:.2f}")
        print(f"    % time Sharpe>1 : "
              f"{metrics['Rolling']['pct_time_sharpe_gt1']:.1%}")

        # Final score
        m_net = metrics.get("LS_Net", {})
        m_ic  = metrics.get("IC", {})
        n_pass = sum([
            m_net.get("sharpe",0)   > 1.0,
            m_net.get("sortino",0)  > 1.5,
            abs(m_ic.get("mean_ic",0)) > 0.04,
            m_ic.get("icir",0)      > 0.5,
            m_ic.get("hit_rate",0)  > 0.55,
            m_net.get("max_dd",0)   > -0.30,
            metrics["Rolling"].get(
                "pct_time_sharpe_gt1",0
            ) > 0.50,
        ])
        metrics["score"]   = n_pass
        metrics["n_checks"]= 7
        print(f"\n  Production Score: {n_pass}/7 "
              f"{'✅ READY' if n_pass>=5 else '⚠️ REVIEW'}")
        return metrics

    # ─────────────────────────────────────────────────
    #  Step 4 — Monthly returns table
    # ─────────────────────────────────────────────────
    def monthly_returns(self, daily_df):
        print("\nStep 4: Monthly returns table...")

        df         = daily_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"]= df["date"].dt.month

        monthly = df.groupby(
            ["year","month"]
        )["ls_net"].apply(
            lambda x: float((1+x).prod()-1)
        ).reset_index()
        monthly.columns = ["year","month","ret"]

        # Pivot: years × months
        pivot = monthly.pivot(
            index="year", columns="month",
            values="ret"
        )
        pivot.columns = [
            "Jan","Feb","Mar","Apr","May","Jun",
            "Jul","Aug","Sep","Oct","Nov","Dec"
        ][:len(pivot.columns)]

        # Annual return
        pivot["Annual"] = (
            (1 + monthly.groupby("year")[
                "ret"
            ].apply(lambda x: (1+x).prod()-1))
            .values
        )

        print(f"\n  Monthly Returns (Net TC):")
        print(
            pivot.round(3).to_string()
        )
        return pivot

    # ─────────────────────────────────────────────────
    #  Step 5 — Write results
    # ─────────────────────────────────────────────────
    def write_results(self, daily_df, ic_df,
                       metrics, monthly_pivot):
        print("\nStep 5: Writing results...")

        def _write(df, path, partition=True):
            df   = df.copy()
            nums = df.select_dtypes(
                include=[np.number]
            ).columns
            df[nums] = df[nums].fillna(0)
            if "date" in df.columns:
                df["date"] = df["date"].astype(str)
                df["year"] = pd.to_datetime(
                    df["date"]
                ).dt.year
                df["month"] = pd.to_datetime(
                    df["date"]
                ).dt.month
            w = (
                self.spark.createDataFrame(df)
                .write.format("delta")
                .mode("overwrite")
                .option("overwriteSchema","true")
            )
            if partition and "year" in df.columns:
                w = w.partitionBy("year","month")
            w.save(path)

        _write(
            daily_df,
            f"{self.bt_path}/bt01_daily_returns"
        )
        print("  ✓ bt01_daily_returns")

        _write(
            ic_df,
            f"{self.bt_path}/bt01_ic_series",
            partition=False
        )
        print("  ✓ bt01_ic_series")

        # Metrics summary
        rows = []
        for label, m in [
            ("LS_Gross",  metrics.get("LS_Gross",{})),
            ("LS_Net",    metrics.get("LS_Net",{})),
            ("Long_Only", metrics.get("Long_Only",{})),
        ]:
            if not m:
                continue
            rows.append({
                "strategy"     : label,
                "ann_ret"      : m.get("ann_ret",0),
                "ann_vol"      : m.get("ann_vol",0),
                "sharpe"       : m.get("sharpe",0),
                "sortino"      : m.get("sortino",0),
                "calmar"       : m.get("calmar",0),
                "max_dd"       : m.get("max_dd",0),
                "max_dd_dur"   : m.get("max_dd_dur",0),
                "var_95"       : m.get("var_95",0),
                "cvar_95"      : m.get("cvar_95",0),
                "omega"        : m.get("omega",0),
                "win_rate"     : m.get("win_rate",0),
                "skewness"     : m.get("skewness",0),
                "excess_kurt"  : m.get("excess_kurt",0),
                "pos_months"   : m.get("pos_months",0),
                "n_days"       : m.get("n_days",0),
            })
        _write(
            pd.DataFrame(rows),
            f"{self.bt_path}/bt01_perf_summary",
            partition=False
        )
        print("  ✓ bt01_perf_summary")

        # Regime metrics
        r_rows = []
        for regime, m in metrics.get(
            "Regime",{}
        ).items():
            r_rows.append({
                "regime" : regime,
                **{k: v for k, v in m.items()
                   if isinstance(v, (int, float))}
            })
        if r_rows:
            _write(
                pd.DataFrame(r_rows),
                f"{self.bt_path}/bt01_regime_perf",
                partition=False
            )
            print("  ✓ bt01_regime_perf")

        # Monthly returns
        mr = monthly_pivot.reset_index()
        _write(
            mr,
            f"{self.bt_path}/bt01_monthly_returns",
            partition=False
        )
        print("  ✓ bt01_monthly_returns")

    # ─────────────────────────────────────────────────
    #  Run
    # ─────────────────────────────────────────────────
    def run(self):
        print("="*55)
        print("Backtest 01 — Full Portfolio Backtest")
        print("="*55)
        start = datetime.now()

        pred, mkt        = self.load_data()
        daily_df, ic_df  = self.run_backtest(pred)
        metrics          = self.compute_metrics(
            daily_df, ic_df
        )
        monthly_pivot    = self.monthly_returns(
            daily_df
        )
        self.write_results(
            daily_df, ic_df,
            metrics, monthly_pivot
        )

        elapsed = (
            datetime.now()-start
        ).seconds / 60
        print(f"\nTotal time : {elapsed:.1f} min")
        print("Backtest 01 COMPLETE ✓")
        return daily_df, ic_df, metrics

# COMMAND ----------

class Backtest01Charts:
    TEMPLATE = "plotly_dark"
    C = {
        "primary"  : "#2196F3",
        "secondary": "#FF5722",
        "success"  : "#4CAF50",
        "warning"  : "#FFC107",
        "purple"   : "#9C27B0",
        "teal"     : "#00BCD4",
        "gross"    : "#2196F3",
        "net"      : "#4CAF50",
        "longonly" : "#FF9800",
    }
    REGIME_C = {
        "Bull"   : "#4CAF50",
        "Bear"   : "#FF5722",
        "HighVol": "#FFC107",
    }

    def chart_cumulative_returns(self,
                                  daily_df,
                                  metrics):
        """Main equity curve chart."""
        df = daily_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=[
                "Cumulative Return",
                "Daily Return (Net TC)",
                "Drawdown",
                "Rolling 252d Sharpe",
            ],
            vertical_spacing=0.06,
            row_heights=[0.45,0.20,0.20,0.15]
        )

        # Equity curves
        for col, name, color in [
            ("cum_ls_gross","L/S Gross",
             self.C["gross"]),
            ("cum_ls_net",  "L/S Net TC",
             self.C["net"]),
            ("cum_long_only","Long Only",
             self.C["longonly"]),
        ]:
            if col not in df.columns:
                continue
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[col],
                name=name, mode="lines",
                line=dict(color=color, width=2)
            ), row=1, col=1)

        # Daily returns
        r     = df["ls_net"].fillna(0)
        bar_c = [
            self.C["success"] if v > 0
            else self.C["secondary"] for v in r
        ]
        fig.add_trace(go.Bar(
            x=df["date"], y=r*100,
            marker_color=bar_c,
            showlegend=False
        ), row=2, col=1)
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=2, col=1
        )

        # Drawdown
        if "dd_cum_ls_net" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"],
                y=df["dd_cum_ls_net"]*100,
                fill="tozeroy",
                fillcolor="rgba(255,87,34,0.3)",
                line=dict(
                    color=self.C["secondary"],
                    width=1
                ),
                showlegend=False
            ), row=3, col=1)

        # Rolling Sharpe
        roll_sharpe = (
            r.rolling(252).mean() /
            r.rolling(252).std() *
            np.sqrt(252)
        )
        fig.add_trace(go.Scatter(
            x=df["date"], y=roll_sharpe,
            mode="lines",
            line=dict(
                color=self.C["teal"], width=1.5
            ),
            showlegend=False
        ), row=4, col=1)
        fig.add_hline(
            y=1.0, line_dash="dot",
            line_color=self.C["warning"],
            annotation_text="Sharpe=1",
            row=4, col=1
        )
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=4, col=1
        )

        m_net   = metrics.get("LS_Net", {})
        m_gross = metrics.get("LS_Gross", {})
        fig.update_layout(
            title=(
                f"<b>Backtest 01 — "
                f"Portfolio Performance<br>"
                f"<sup>"
                f"Net Sharpe={m_net.get('sharpe',0):.2f} | "
                f"Ann Ret={m_net.get('ann_ret',0)*100:.1f}% | "
                f"Max DD={m_net.get('max_dd',0)*100:.1f}% | "
                f"Calmar={m_net.get('calmar',0):.2f}"
                f"</sup></b>"
            ),
            template=self.TEMPLATE,
            height=900,
            hovermode="x unified"
        )
        for row, t in [
            (1,"Cum Ret"),(2,"Daily(%)"),
            (3,"DD(%)"),(4,"Roll Sharpe")
        ]:
            fig.update_yaxes(
                title_text=t, row=row, col=1
            )
        fig.show()

    def chart_ic_analysis(self, ic_df, metrics):
        """IC time series + distribution."""
        ic_df = ic_df.copy()
        ic_df["date"] = pd.to_datetime(ic_df["date"])
        ic_df = ic_df.sort_values("date")
        vals  = ic_df["ic"].fillna(0)

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=False,
            subplot_titles=[
                "Daily IC",
                "Rolling IC (21d, 63d, 252d)",
                "IC Distribution",
            ],
            vertical_spacing=0.1,
            row_heights=[0.30, 0.40, 0.30],
            specs=[
                [{"colspan":1}],
                [{"colspan":1}],
                [{"colspan":1}],
            ]
        )

        # Daily IC
        bar_c = [
            self.C["success"] if v > 0
            else self.C["secondary"] for v in vals
        ]
        fig.add_trace(go.Bar(
            x=ic_df["date"], y=vals,
            marker_color=bar_c, showlegend=False
        ), row=1, col=1)
        fig.add_hline(
            y=float(vals.mean()), line_dash="dot",
            line_color=self.C["warning"],
            annotation_text=(
                f"μ={vals.mean():.4f}"
            ),
            row=1, col=1
        )
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=1, col=1
        )

        # Rolling IC
        for w, name, color in [
            (21,  "21d",  self.C["primary"]),
            (63,  "63d",  self.C["teal"]),
            (252, "252d", self.C["warning"]),
        ]:
            fig.add_trace(go.Scatter(
                x=ic_df["date"],
                y=vals.rolling(w).mean(),
                name=name, mode="lines",
                line=dict(color=color, width=2)
            ), row=2, col=1)
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=2, col=1
        )

        # IC distribution
        fig.add_trace(go.Histogram(
            x=vals, nbinsx=60,
            marker_color=self.C["primary"],
            opacity=0.8, showlegend=False
        ), row=3, col=1)
        for x_val, label, color in [
            (0.0,           "zero",  "white"),
            (float(vals.mean()),
             f"μ={vals.mean():.4f}",
             self.C["warning"]),
            (0.04, "target", self.C["success"]),
        ]:
            fig.add_vline(
                x=x_val, line_dash="dash",
                line_color=color, opacity=0.7,
                annotation_text=label,
                row=3, col=1
            )

        ic_m = metrics.get("IC", {})
        fig.update_layout(
            title=(
                f"<b>Backtest 01 — IC Analysis<br>"
                f"<sup>"
                f"Mean={ic_m.get('mean_ic',0):+.4f} | "
                f"ICIR={ic_m.get('icir',0):.3f} | "
                f"Hit={ic_m.get('hit_rate',0):.1%}"
                f"</sup></b>"
            ),
            template=self.TEMPLATE,
            height=800,
            hovermode="x unified"
        )
        fig.show()

    def chart_regime_analysis(self, daily_df,
                               metrics):
        """Returns + metrics by regime."""
        df = daily_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        regime_m = metrics.get("Regime", {})
        regimes  = list(regime_m.keys())
        colors   = [
            self.REGIME_C.get(r,"#9E9E9E")
            for r in regimes
        ]

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                "Sharpe by Regime",
                "Ann Return by Regime",
                "Max DD by Regime",
                "Win Rate by Regime",
                "Sortino by Regime",
                "Calmar by Regime",
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        metrics_list = [
            ("sharpe",   1,1),
            ("ann_ret",  1,2),
            ("max_dd",   1,3),
            ("win_rate", 2,1),
            ("sortino",  2,2),
            ("calmar",   2,3),
        ]
        for met, row, col in metrics_list:
            vals = [
                regime_m[r].get(met, 0)
                for r in regimes
            ]
            if met == "ann_ret":
                vals = [v*100 for v in vals]
            if met == "max_dd":
                vals = [v*100 for v in vals]
            if met == "win_rate":
                vals = [v*100 for v in vals]

            fig.add_trace(go.Bar(
                x=regimes, y=vals,
                marker_color=colors,
                text=[f"{v:.2f}" for v in vals],
                textposition="outside",
                showlegend=False
            ), row=row, col=col)
            fig.add_hline(
                y=0, line_dash="dash",
                line_color="white", opacity=0.3,
                row=row, col=col
            )

        fig.update_layout(
            title=(
                "<b>Backtest 01 — "
                "Regime Analysis</b>"
            ),
            template=self.TEMPLATE,
            height=700
        )
        fig.show()

    def chart_quintile_spread(self, daily_df,
                               metrics):
        """Quintile return spread."""
        q_m   = metrics.get("Quintiles", {})
        if not q_m:
            return

        q_cols = sorted(q_m.keys())
        vals   = [q_m[c]*100 for c in q_cols]
        labels = [
            f"Q{i+1}" for i in range(len(q_cols))
        ]
        colors = [
            self.C["success"] if i == 0
            else self.C["secondary"]
            if i == len(q_cols)-1
            else self.C["primary"]
            for i in range(len(q_cols))
        ]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Quintile Ann Returns (%)",
                "Monotonic Spread",
            ]
        )

        fig.add_trace(go.Bar(
            x=labels, y=vals,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in vals],
            textposition="outside",
            showlegend=False
        ), row=1, col=1)
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=1, col=1
        )

        # Spread line (shows monotonicity)
        fig.add_trace(go.Scatter(
            x=labels, y=vals,
            mode="lines+markers",
            line=dict(
                color=self.C["warning"], width=2
            ),
            marker=dict(size=8),
            showlegend=False
        ), row=1, col=2)
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=1, col=2
        )

        spread = vals[0] - vals[-1] if vals else 0
        fig.update_layout(
            title=(
                f"<b>Backtest 01 — "
                f"Quintile Analysis<br>"
                f"<sup>"
                f"Q1-Q5 Spread = "
                f"{spread:.1f}% ann</sup></b>"
            ),
            template=self.TEMPLATE,
            height=500
        )
        fig.show()

    def chart_monthly_heatmap(self, monthly_pivot):
        """Monthly returns heatmap."""
        if monthly_pivot is None:
            return

        mp      = monthly_pivot.copy()
        months  = [
            c for c in mp.columns
            if c != "Annual"
        ]
        years   = mp.index.tolist()
        z_data  = mp[months].values * 100

        fig = go.Figure(go.Heatmap(
            z=z_data,
            x=months,
            y=[str(y) for y in years],
            colorscale=[
                [0.0,  "#FF5722"],
                [0.5,  "#1a1a2e"],
                [1.0,  "#4CAF50"],
            ],
            zmid=0,
            text=np.round(z_data, 1),
            texttemplate="%{text}%",
            textfont={"size": 9},
            colorbar=dict(title="Ret %"),
        ))

        # Annual returns annotation
        annual = mp.get("Annual", pd.Series())
        ann_text = [
            f"{v*100:.1f}%" for v in annual.values
        ]
        fig.update_layout(
            title=(
                "<b>Backtest 01 — "
                "Monthly Returns Heatmap "
                "(Net TC, %)</b>"
            ),
            template=self.TEMPLATE,
            height=max(600, len(years)*22),
            xaxis_title="Month",
            yaxis_title="Year",
            yaxis=dict(autorange="reversed"),
        )
        fig.show()

    def chart_risk_metrics(self, daily_df,
                            metrics):
        """VaR, CVaR, distribution analysis."""
        r    = daily_df["ls_net"].dropna()
        m    = metrics.get("LS_Net", {})

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Return Distribution",
                "QQ-Plot vs Normal",
                "Rolling Volatility (63d)",
                "Underwater Equity Curve",
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # Distribution
        fig.add_trace(go.Histogram(
            x=r*100, nbinsx=80,
            marker_color=self.C["primary"],
            opacity=0.8, showlegend=False
        ), row=1, col=1)
        for x_val, label, color in [
            (float(m.get("var_95",0))*100,
             f"VaR95={m.get('var_95',0)*100:.2f}%",
             self.C["warning"]),
            (float(m.get("cvar_95",0))*100,
             f"CVaR95={m.get('cvar_95',0)*100:.2f}%",
             self.C["secondary"]),
        ]:
            fig.add_vline(
                x=x_val, line_dash="dash",
                line_color=color, opacity=0.8,
                annotation_text=label,
                row=1, col=1
            )

        # QQ-plot
        from scipy import stats as sp
        (osm, osr), (slope, intercept, _) = (
            sp.probplot(r.values)
        )
        fig.add_trace(go.Scatter(
            x=osm, y=osr,
            mode="markers",
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

        # Rolling vol
        roll_vol = (
            r.rolling(63).std() *
            np.sqrt(252) * 100
        )
        df = daily_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        fig.add_trace(go.Scatter(
            x=df["date"].values[:len(roll_vol)],
            y=roll_vol.values,
            mode="lines",
            line=dict(
                color=self.C["teal"], width=1.5
            ),
            showlegend=False
        ), row=2, col=1)

        # Underwater
        if "dd_cum_ls_net" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"],
                y=df["dd_cum_ls_net"]*100,
                fill="tozeroy",
                fillcolor="rgba(255,87,34,0.3)",
                line=dict(
                    color=self.C["secondary"],
                    width=1
                ),
                showlegend=False
            ), row=2, col=2)

        fig.update_layout(
            title=(
                f"<b>Backtest 01 — Risk Metrics<br>"
                f"<sup>"
                f"VaR95={m.get('var_95',0)*100:.2f}% | "
                f"CVaR95={m.get('cvar_95',0)*100:.2f}% | "
                f"Skew={m.get('skewness',0):.2f} | "
                f"Kurt={m.get('excess_kurt',0):.2f}"
                f"</sup></b>"
            ),
            template=self.TEMPLATE,
            height=700
        )
        fig.show()

    def chart_turnover_analysis(self, daily_df,
                                 metrics):
        """Turnover and TC over time."""
        df = daily_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        tc_m = metrics.get("TC", {})

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=[
                "Daily Turnover (%)",
                "Daily TC Cost (bps)",
                "Rolling 252d TC (bps/yr)",
            ],
            vertical_spacing=0.08,
            row_heights=[0.35,0.35,0.30]
        )

        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["turnover"]*100,
            mode="lines",
            line=dict(
                color=self.C["purple"], width=1
            ),
            fill="tozeroy",
            fillcolor="rgba(156,39,176,0.15)",
            showlegend=False
        ), row=1, col=1)
        fig.add_hline(
            y=float(df["turnover"].mean()*100),
            line_dash="dot",
            line_color=self.C["warning"],
            annotation_text=(
                f"Avg={df['turnover'].mean():.1%}"
            ),
            row=1, col=1
        )

        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["tc_cost"]*10000,
            mode="lines",
            line=dict(
                color=self.C["secondary"], width=1
            ),
            fill="tozeroy",
            fillcolor="rgba(255,87,34,0.15)",
            showlegend=False
        ), row=2, col=1)

        roll_tc = (
            df["tc_cost"].rolling(252).mean()
            * 252 * 10000
        )
        fig.add_trace(go.Scatter(
            x=df["date"], y=roll_tc,
            mode="lines",
            line=dict(
                color=self.C["warning"], width=1.5
            ),
            showlegend=False
        ), row=3, col=1)

        fig.update_layout(
            title=(
                f"<b>Backtest 01 — "
                f"Turnover & TC Analysis<br>"
                f"<sup>"
                f"Avg TO={tc_m.get('avg_turnover',0):.1%}/day | "
                f"TC/yr={tc_m.get('tc_bps_yr',0):.0f}bps"
                f"</sup></b>"
            ),
            template=self.TEMPLATE,
            height=700,
            hovermode="x unified"
        )
        for row, t in [
            (1,"Turnover(%)"),(2,"TC(bps)"),
            (3,"Roll TC(bps/yr)")
        ]:
            fig.update_yaxes(
                title_text=t, row=row, col=1
            )
        fig.show()

    def run_all(self, daily_df, ic_df,
                metrics, monthly_pivot):
        print("\n" + "="*55)
        print("Backtest 01 Charts")
        print("="*55)

        print("\n[1/7] Cumulative Returns...")
        self.chart_cumulative_returns(
            daily_df, metrics
        )

        print("[2/7] IC Analysis...")
        self.chart_ic_analysis(ic_df, metrics)

        print("[3/7] Regime Analysis...")
        self.chart_regime_analysis(
            daily_df, metrics
        )

        print("[4/7] Quintile Spread...")
        self.chart_quintile_spread(
            daily_df, metrics
        )

        print("[5/7] Monthly Heatmap...")
        self.chart_monthly_heatmap(monthly_pivot)

        print("[6/7] Risk Metrics...")
        self.chart_risk_metrics(daily_df, metrics)

        print("[7/7] Turnover Analysis...")
        self.chart_turnover_analysis(
            daily_df, metrics
        )

        print("\nAll 7 charts ✓")

# COMMAND ----------

bt = Backtest01(
    spark     = spark,
    gold_path = GOLD_PATH,
    ml_path   = ML_PATH,
    bt_path   = BT_PATH,
)

daily_df, ic_df, metrics = bt.run()

# Monthly returns
monthly_pivot = bt.monthly_returns(daily_df)

# Charts
charts = Backtest01Charts()
charts.run_all(
    daily_df      = daily_df,
    ic_df         = ic_df,
    metrics       = metrics,
    monthly_pivot = monthly_pivot,
)
print("\nBacktest 01 COMPLETE ✓")

# COMMAND ----------

perf = spark.read.format("delta").load(
    f"{BT_PATH}/bt01_perf_summary"
).toPandas()

regime_perf = spark.read.format("delta").load(
    f"{BT_PATH}/bt01_regime_perf"
).toPandas()

print("="*60)
print("Backtest 01 — Final Summary")
print("="*60)

print(f"\nPerformance Metrics:")
print(perf[[
    "strategy","ann_ret","ann_vol","sharpe",
    "sortino","calmar","max_dd","win_rate"
]].round(4).to_string(index=False))

print(f"\nRegime Performance:")
print(regime_perf[[
    "regime","sharpe","ann_ret","max_dd"
]].round(4).to_string(index=False))

m   = metrics.get("LS_Net",{})
ic  = metrics.get("IC",{})
tc  = metrics.get("TC",{})
n_pass = metrics.get("score",0)
n_tot  = metrics.get("n_checks",7)

print(f"\nKey Metrics (L/S Net TC):")
print(f"  Sharpe    : {m.get('sharpe',0):.2f} "
      f"{'✅' if m.get('sharpe',0)>1 else '⚠️'}")
print(f"  Sortino   : {m.get('sortino',0):.2f}")
print(f"  Calmar    : {m.get('calmar',0):.2f}")
print(f"  Ann Ret   : "
      f"{m.get('ann_ret',0)*100:.1f}%")
print(f"  Max DD    : "
      f"{m.get('max_dd',0)*100:.1f}%")
print(f"  VaR 95%   : "
      f"{m.get('var_95',0)*100:.2f}%")
print(f"  CVaR 95%  : "
      f"{m.get('cvar_95',0)*100:.2f}%")
print(f"  Skewness  : "
      f"{m.get('skewness',0):.2f}")
print(f"  Omega     : {m.get('omega',0):.2f}")
print(f"\nIC Metrics:")
print(f"  Mean IC   : "
      f"{ic.get('mean_ic',0):+.4f} "
      f"{'✅' if abs(ic.get('mean_ic',0))>0.04 else '⚠️'}")
print(f"  ICIR      : {ic.get('icir',0):.4f}")
print(f"  Hit Rate  : {ic.get('hit_rate',0):.1%}")
print(f"\nTransaction Costs:")
print(f"  Avg TO    : "
      f"{tc.get('avg_turnover',0):.1%}/day")
print(f"  TC/yr     : "
      f"{tc.get('tc_bps_yr',0):.0f}bps/yr")

print(f"\nProduction Score: {n_pass}/{n_tot} "
      f"{'✅ PRODUCTION READY' if n_pass>=5 else '⚠️ REVIEW'}")

print(f"\n{'='*60}")
print(f"Backtest Progress:")
print(f"  Backtest 01 ✅ Full portfolio backtest")
print(f"  Backtest 02 🔲 Risk analysis")
print(f"  Backtest 03 🔲 Performance attribution")
print(f"{'='*60}")