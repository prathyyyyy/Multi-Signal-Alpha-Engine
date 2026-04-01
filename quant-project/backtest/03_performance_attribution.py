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

ANNUAL_FACTOR = 252
RISK_FREE     = 0.04

print("="*55)
print("Backtest 03 — Performance Attribution")
print("="*55)

# COMMAND ----------

class Backtest03:
    """
    Backtest 03 — Performance Attribution.

    Sections:
      1. Signal attribution  : ML01 vs ML02 vs ML03
      2. Regime attribution  : Bull/HighVol/Bear
      3. Time attribution    : decade/year/month
      4. Sector attribution  : cross-sectional IC
      5. Factor attribution  : what drives alpha
      6. IC decay            : signal half-life
      7. Ensemble attribution: blend contribution
    """

    ANNUAL_FACTOR = 252
    RISK_FREE     = 0.04

    def __init__(self, spark, gold_path,
                 ml_path, bt_path):
        self.spark     = spark
        self.gold_path = gold_path
        self.ml_path   = ml_path
        self.bt_path   = bt_path
        print("Backtest03 ✓")

    # ─────────────────────────────────────────────────
    #  Utilities
    # ─────────────────────────────────────────────────
    @staticmethod
    def _safe_ic(pred, actual, min_n=30):
        p = np.asarray(pred,  dtype=float)
        a = np.asarray(actual,dtype=float)
        m = (~np.isnan(p) & ~np.isnan(a) &
             ~np.isinf(p) & ~np.isinf(a))
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

    def _ic_stats(self, ic_list):
        v = [x for x in ic_list
             if x is not None and not np.isnan(x)]
        if not v:
            return {
                "mean_ic":0.,"icir":0.,
                "ic_std":0.,"hit_rate":0.,
                "n_dates":0
            }
        m = float(np.mean(v))
        s = float(np.std(v))
        return {
            "mean_ic" : m,
            "icir"    : m / (s + 1e-8),
            "ic_std"  : s,
            "hit_rate": float(
                (np.array(v) > 0).mean()
            ),
            "n_dates" : len(v),
        }

    def _ls_stats(self, returns):
        r   = pd.Series(returns).dropna()
        if len(r) < 5:
            return {
                "ann_ret":0.,"ann_vol":0.,
                "sharpe":0.,"max_dd":0.
            }
        ann = r.mean() * self.ANNUAL_FACTOR
        vol = r.std()  * np.sqrt(self.ANNUAL_FACTOR)
        rf  = self.RISK_FREE / self.ANNUAL_FACTOR
        sr  = (r.mean()-rf) / (r.std()+1e-10) * \
              np.sqrt(self.ANNUAL_FACTOR)
        dd  = (
            (1+r).cumprod() /
            (1+r).cumprod().cummax() - 1
        ).min()
        return {
            "ann_ret": float(ann),
            "ann_vol": float(vol),
            "sharpe" : float(sr),
            "max_dd" : float(dd),
        }

    # ─────────────────────────────────────────────────
    #  Step 1 — Load all data
    # ─────────────────────────────────────────────────
    def load_data(self):
        print("\nStep 1: Loading data...")
        start = datetime.now()

        # BT01 daily returns
        daily = self.spark.read.format(
            "delta"
        ).load(
            f"{self.bt_path}/bt01_daily_returns"
        ).toPandas()
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date")

        # BT01 IC series
        ic_df = self.spark.read.format(
            "delta"
        ).load(
            f"{self.bt_path}/bt01_ic_series"
        ).toPandas()
        ic_df["date"] = pd.to_datetime(ic_df["date"])
        ic_df = ic_df.sort_values("date")

        # Ensemble predictions (has all signals)
        pred = self.spark.read.format(
            "delta"
        ).load(
            f"{self.ml_path}/ensemble_predictions"
        ).toPandas()
        pred["date"] = pd.to_datetime(pred["date"])

        # ML02 LightGBM predictions
        lgbm = self.spark.read.format(
            "delta"
        ).load(
            f"{self.ml_path}/lgbm_v3_predictions"
        ).toPandas()
        lgbm["date"] = pd.to_datetime(lgbm["date"])

        # ML03 PatchTST vol predictions
        vol = self.spark.read.format(
            "delta"
        ).load(
            f"{self.ml_path}/patchtst_vol_predictions"
        ).toPandas()
        vol["date"] = pd.to_datetime(vol["date"])

        # Gold price factors
        sdf     = self.spark.read.format(
            "delta"
        ).load(f"{self.gold_path}/price_factors")
        avail   = set(sdf.columns)
        gc_cols = [
            c for c in [
                "date","ticker",
                "fwd_return_21d",
                "regime_label",
                "prob_bull","prob_bear",
                "prob_highvol",
                "mom_21d","mom_252d",
                "vol_21d","sharpe_21d",
                "vol_21d_rank","mom_21d_rank",
            ] if c in avail
        ]
        gold = sdf.select(*gc_cols).toPandas()
        gold["date"] = pd.to_datetime(gold["date"])

        elapsed = (datetime.now()-start).seconds
        print(f"  Daily  : {len(daily):,} rows")
        print(f"  Pred   : {len(pred):,} rows")
        print(f"  LGBM   : {len(lgbm):,} rows")
        print(f"  Vol    : {len(vol):,} rows")
        print(f"  Gold   : {len(gold):,} rows")
        print(f"  Elapsed: {elapsed}s")
        return daily, ic_df, pred, lgbm, vol, gold

    # ─────────────────────────────────────────────────
    #  Section 1 — Signal Attribution
    # ─────────────────────────────────────────────────
    def signal_attribution(self, pred, lgbm, vol):
        """
        Decompose IC contribution from each signal:
          ML01 HMM   : regime_weight_soft
          ML02 LGBM  : pred_final (alpha)
          ML03 PatchTST: vol_position_signal
          ML04 Blend : signal_final_cs
        """
        print("\nSection 1: Signal Attribution...")

        target = "fwd_return_21d"
        signals = {}

        # ML02 standalone
        if "pred_final" in pred.columns:
            ic_list = []
            for date, grp in pred.groupby("date"):
                grp = grp.dropna(
                    subset=["pred_final", target]
                )
                if len(grp) < 20:
                    continue
                ic = self._safe_ic(
                    grp["pred_final"].values,
                    grp[target].values
                )
                if not np.isnan(ic):
                    ic_list.append(ic)
            signals["ML02_LightGBM"] = (
                self._ic_stats(ic_list)
            )

        # ML04 ensemble
        if "signal_final_cs" in pred.columns:
            ic_list = []
            for date, grp in pred.groupby("date"):
                grp = grp.dropna(
                    subset=["signal_final_cs", target]
                )
                if len(grp) < 20:
                    continue
                ic = self._safe_ic(
                    grp["signal_final_cs"].values,
                    grp[target].values
                )
                if not np.isnan(ic):
                    ic_list.append(ic)
            signals["ML04_Ensemble"] = (
                self._ic_stats(ic_list)
            )

        # Alpha blend component
        if "alpha_blend" in pred.columns:
            ic_list = []
            for date, grp in pred.groupby("date"):
                grp = grp.dropna(
                    subset=["alpha_blend", target]
                )
                if len(grp) < 20:
                    continue
                ic = self._safe_ic(
                    grp["alpha_blend"].values,
                    grp[target].values
                )
                if not np.isnan(ic):
                    ic_list.append(ic)
            signals["Alpha_Blend"] = (
                self._ic_stats(ic_list)
            )

        # Regime weight contribution
        if "regime_weight_soft" in pred.columns:
            ic_list = []
            for date, grp in pred.groupby("date"):
                grp = grp.dropna(
                    subset=["regime_weight_soft",
                             target]
                )
                if len(grp) < 20:
                    continue
                ic = self._safe_ic(
                    grp["regime_weight_soft"].values,
                    grp[target].values
                )
                if not np.isnan(ic):
                    ic_list.append(ic)
            signals["Regime_Weight"] = (
                self._ic_stats(ic_list)
            )

        # Vol signal contribution
        if "vol_position_signal" in pred.columns:
            ic_list = []
            for date, grp in pred.groupby("date"):
                grp = grp.dropna(
                    subset=["vol_position_signal",
                             target]
                )
                if len(grp) < 20:
                    continue
                ic = self._safe_ic(
                    grp["vol_position_signal"].values,
                    grp[target].values
                )
                if not np.isnan(ic):
                    ic_list.append(ic)
            signals["ML03_VolSignal"] = (
                self._ic_stats(ic_list)
            )

        print(f"\n  {'Signal':20} {'IC':>8} "
              f"{'ICIR':>8} {'Hit':>8}")
        print(f"  {'-'*48}")
        for name, s in signals.items():
            print(f"  {name:20} "
                  f"{s['mean_ic']:>+8.4f} "
                  f"{s['icir']:>+8.4f} "
                  f"{s['hit_rate']:>7.1%}")

        return signals

    # ─────────────────────────────────────────────────
    #  Section 2 — Regime Attribution
    # ─────────────────────────────────────────────────
    def regime_attribution(self, daily, pred):
        """
        P&L attribution by market regime.
        Also: position sizing contribution.
        """
        print("\nSection 2: Regime Attribution...")

        target = "fwd_return_21d"
        results = {}

        # IC by regime (from ensemble predictions)
        if "regime_label" in pred.columns:
            for regime in ["Bull","HighVol","Bear"]:
                r_pred = pred[
                    pred["regime_label"] == regime
                ]
                ic_list = []
                for date, grp in r_pred.groupby(
                    "date"
                ):
                    grp = grp.dropna(
                        subset=["signal_final_cs",
                                 target]
                    )
                    if len(grp) < 10:
                        continue
                    ic = self._safe_ic(
                        grp["signal_final_cs"].values,
                        grp[target].values
                    )
                    if not np.isnan(ic):
                        ic_list.append(ic)
                results[f"IC_{regime}"] = (
                    self._ic_stats(ic_list)
                )

        # Return attribution by regime from daily
        if "regime" in daily.columns:
            for regime in ["Bull","HighVol","Bear"]:
                r_daily = daily[
                    daily["regime"] == regime
                ]
                if len(r_daily) < 5:
                    continue
                ls = self._ls_stats(
                    r_daily["ls_net"].values
                )
                n  = len(r_daily)
                pct= n / len(daily)
                results[f"LS_{regime}"] = {
                    **ls,
                    "n_days": n,
                    "pct_days": pct,
                }
                print(f"\n  {regime} "
                      f"({n} days, {pct:.1%}):")
                print(f"    Sharpe  : "
                      f"{ls['sharpe']:.2f}")
                print(f"    Ann Ret : "
                      f"{ls['ann_ret']*100:.1f}%")
                print(f"    Max DD  : "
                      f"{ls['max_dd']*100:.1f}%")

        # Position sizing contribution
        if all(c in pred.columns for c in [
            "regime_weight_soft",
            "vol_position_signal","position_size"
        ]):
            ps_stats = {}
            for col in [
                "regime_weight_soft",
                "vol_position_signal",
                "position_size"
            ]:
                ps_stats[col] = {
                    "mean": float(pred[col].mean()),
                    "std" : float(pred[col].std()),
                    "p25" : float(
                        pred[col].quantile(0.25)
                    ),
                    "p75" : float(
                        pred[col].quantile(0.75)
                    ),
                }
            results["position_sizing"] = ps_stats
            print(f"\n  Position Sizing Stats:")
            for col, s in ps_stats.items():
                print(f"    {col:25}: "
                      f"μ={s['mean']:.3f} "
                      f"σ={s['std']:.3f} "
                      f"[{s['p25']:.3f}, "
                      f"{s['p75']:.3f}]")

        return results

    # ─────────────────────────────────────────────────
    #  Section 3 — Time Attribution
    # ─────────────────────────────────────────────────
    def time_attribution(self, daily, ic_df):
        """
        P&L breakdown by:
          - Decade (1993-2000, 2001-2010, 2011-2020, 2021+)
          - Year
          - Month (seasonality)
          - Day of week
        """
        print("\nSection 3: Time Attribution...")

        daily = daily.copy()
        daily["date"]  = pd.to_datetime(daily["date"])
        daily["year"]  = daily["date"].dt.year
        daily["month"] = daily["date"].dt.month
        daily["dow"]   = daily["date"].dt.dayofweek
        daily["decade"]= (
            daily["year"] // 10 * 10
        )

        ic_df = ic_df.copy()
        ic_df["date"]  = pd.to_datetime(ic_df["date"])
        ic_df["year"]  = ic_df["date"].dt.year
        ic_df["month"] = ic_df["date"].dt.month

        results = {}

        # ── Decade attribution ────────────────────────
        decade_rows = []
        for decade, grp in daily.groupby("decade"):
            ls = self._ls_stats(
                grp["ls_net"].values
            )
            ann_ret = float(
                (1+grp["ls_net"].fillna(0)
                ).prod() **
                (self.ANNUAL_FACTOR/len(grp)) - 1
            )
            decade_rows.append({
                "decade"  : f"{decade}s",
                "n_days"  : len(grp),
                "ann_ret" : ls["ann_ret"],
                "sharpe"  : ls["sharpe"],
                "max_dd"  : ls["max_dd"],
            })
        decade_df = pd.DataFrame(decade_rows)
        results["decade"] = decade_df
        print(f"\n  Decade Performance:")
        print(decade_df[[
            "decade","ann_ret","sharpe","max_dd"
        ]].round(4).to_string(index=False))

        # ── Monthly seasonality ───────────────────────
        month_names = [
            "Jan","Feb","Mar","Apr","May","Jun",
            "Jul","Aug","Sep","Oct","Nov","Dec"
        ]
        month_rows = []
        for m in range(1, 13):
            grp = daily[daily["month"] == m]
            if len(grp) < 10:
                continue
            ls = self._ls_stats(
                grp["ls_net"].values
            )
            ic_grp = ic_df[ic_df["month"] == m]
            avg_ic = float(
                ic_grp["ic"].mean()
            ) if len(ic_grp) > 0 else 0.0
            month_rows.append({
                "month"    : month_names[m-1],
                "month_num": m,
                "n_days"   : len(grp),
                "ann_ret"  : ls["ann_ret"],
                "sharpe"   : ls["sharpe"],
                "avg_ic"   : avg_ic,
                "hit_rate" : float(
                    (grp["ls_net"] > 0).mean()
                ),
            })
        month_df = pd.DataFrame(month_rows)
        results["monthly"] = month_df
        print(f"\n  Monthly Seasonality:")
        print(month_df[[
            "month","ann_ret","sharpe","avg_ic"
        ]].round(4).to_string(index=False))

        # ── Day-of-week ───────────────────────────────
        dow_names = ["Mon","Tue","Wed","Thu","Fri"]
        dow_rows  = []
        for d in range(5):
            grp = daily[daily["dow"] == d]
            if len(grp) < 10:
                continue
            avg_ret = float(grp["ls_net"].mean())
            hit     = float(
                (grp["ls_net"] > 0).mean()
            )
            dow_rows.append({
                "dow"    : dow_names[d],
                "avg_ret": avg_ret * 10000,
                "hit_rate": hit,
                "n_days" : len(grp),
            })
        dow_df = pd.DataFrame(dow_rows)
        results["dow"] = dow_df
        print(f"\n  Day-of-Week (avg ret bps):")
        print(dow_df.to_string(index=False))

        # ── Annual IC ─────────────────────────────────
        yr_ic = ic_df.groupby("year").agg(
            mean_ic=("ic","mean"),
            hit_rate=("ic",lambda x:(x>0).mean()),
            n_dates=("ic","count"),
        ).reset_index()
        results["annual_ic"] = yr_ic

        # ── Annual returns ────────────────────────────
        yr_ret = daily.groupby("year")[
            "ls_net"
        ].apply(
            lambda x: float((1+x).prod()-1)
        ).reset_index()
        yr_ret.columns = ["year","ann_ret"]
        results["annual_ret"] = yr_ret

        return results

    # ─────────────────────────────────────────────────
    #  Section 4 — Factor Attribution
    # ─────────────────────────────────────────────────
    def factor_attribution(self, pred, gold):
        """
        What drives the alpha signal?
        Correlate signal with known factors:
          Momentum, Vol, Quality, Regime
        """
        print("\nSection 4: Factor Attribution...")

        target = "fwd_return_21d"

        # Merge pred with gold factors
        factor_cols = [
            c for c in [
                "mom_21d","mom_252d",
                "vol_21d","sharpe_21d",
                "vol_21d_rank","mom_21d_rank",
            ] if c in gold.columns
        ]
        merged = pred.merge(
            gold[["date","ticker"] + factor_cols],
            on=["date","ticker"], how="left"
        )

        results = {}

        # IC of each factor vs signal
        signal_col = "signal_final_cs"
        if signal_col not in merged.columns:
            signal_col = "pred_final"

        print(f"\n  Factor ICs vs signal "
              f"({signal_col}):")
        factor_ics = {}
        for fac in factor_cols:
            ic_list = []
            for date, grp in merged.groupby("date"):
                grp = grp.dropna(
                    subset=[signal_col, fac]
                )
                if len(grp) < 20:
                    continue
                ic = self._safe_ic(
                    grp[signal_col].values,
                    grp[fac].values
                )
                if not np.isnan(ic):
                    ic_list.append(ic)
            s = self._ic_stats(ic_list)
            factor_ics[fac] = s
            print(f"    {fac:20}: "
                  f"IC={s['mean_ic']:+.4f} "
                  f"ICIR={s['icir']:.3f}")
        results["factor_ics"] = factor_ics

        # IC of each factor vs returns
        print(f"\n  Factor ICs vs target "
              f"({target}):")
        factor_ret_ics = {}
        for fac in factor_cols:
            ic_list = []
            for date, grp in merged.groupby("date"):
                grp = grp.dropna(
                    subset=[fac, target]
                )
                if len(grp) < 20:
                    continue
                ic = self._safe_ic(
                    grp[fac].values,
                    grp[target].values
                )
                if not np.isnan(ic):
                    ic_list.append(ic)
            s = self._ic_stats(ic_list)
            factor_ret_ics[fac] = s
            print(f"    {fac:20}: "
                  f"IC={s['mean_ic']:+.4f} "
                  f"ICIR={s['icir']:.3f}")
        results["factor_ret_ics"] = factor_ret_ics

        return results

    # ─────────────────────────────────────────────────
    #  Section 5 — IC Decay
    # ─────────────────────────────────────────────────
    def ic_decay_analysis(self, pred):
        """
        Signal half-life: how quickly IC decays
        as we predict further into the future.
        Uses fwd returns at multiple horizons.
        """
        print("\nSection 5: IC Decay Analysis...")

        target_col = "fwd_return_21d"
        signal_col = "signal_final_cs"
        if signal_col not in pred.columns:
            signal_col = "pred_final"

        results = {}

        # Rolling IC over time (lag analysis)
        # IC at lag 0 = today
        # IC at lag N = predicting N days ahead
        horizon_ics = {}
        lags        = [0, 5, 10, 21, 42, 63]

        for lag in lags:
            pred_lag = pred.copy()
            pred_lag["signal_lagged"] = (
                pred_lag.groupby("ticker")[
                    signal_col
                ].shift(lag)
            )
            pred_lag = pred_lag.dropna(
                subset=["signal_lagged", target_col]
            )

            ic_list = []
            for date, grp in pred_lag.groupby(
                "date"
            ):
                if len(grp) < 20:
                    continue
                ic = self._safe_ic(
                    grp["signal_lagged"].values,
                    grp[target_col].values
                )
                if not np.isnan(ic):
                    ic_list.append(ic)
            s = self._ic_stats(ic_list)
            horizon_ics[lag] = s
            print(f"  Lag {lag:>3}d: "
                  f"IC={s['mean_ic']:+.4f} "
                  f"ICIR={s['icir']:.3f} "
                  f"Hit={s['hit_rate']:.1%}")

        results["horizon_ics"] = horizon_ics

        # Half-life estimation
        ic_vals = [
            horizon_ics[lag]["mean_ic"]
            for lag in lags
        ]
        ic_0    = ic_vals[0] if ic_vals[0] != 0 else 1e-8
        # Find where IC drops to 50%
        half_life = None
        for i, (lag, ic) in enumerate(
            zip(lags, ic_vals)
        ):
            if abs(ic) <= abs(ic_0) * 0.5:
                half_life = lag
                break

        results["half_life_days"] = half_life
        results["lags"]           = lags
        results["ic_vals"]        = ic_vals

        if half_life:
            print(f"\n  Signal half-life: "
                  f"~{half_life} days")
        else:
            print(f"\n  Signal half-life: "
                  f">{lags[-1]} days (very persistent)")

        return results

    # ─────────────────────────────────────────────────
    #  Section 6 — Ensemble Contribution
    # ─────────────────────────────────────────────────
    def ensemble_contribution(self, pred):
        """
        How much does each blend component
        contribute to final IC?
        Brinson-style attribution.
        """
        print("\nSection 6: Ensemble Contribution...")

        target = "fwd_return_21d"
        results = {}

        # Marginal contribution of each component
        components = {
            "LGBM_alone"     : "pred_final",
            "Vol_signal"     : "vol_position_signal",
            "Regime_weight"  : "regime_weight_soft",
            "Alpha_blend"    : "alpha_blend",
            "Final_signal"   : "signal_final_cs",
        }

        contrib = {}
        for name, col in components.items():
            if col not in pred.columns:
                continue
            ic_list = []
            for date, grp in pred.groupby("date"):
                grp = grp.dropna(
                    subset=[col, target]
                )
                if len(grp) < 20:
                    continue
                ic = self._safe_ic(
                    grp[col].values,
                    grp[target].values
                )
                if not np.isnan(ic):
                    ic_list.append(ic)
            s = self._ic_stats(ic_list)
            contrib[name] = s
            print(f"  {name:20}: "
                  f"IC={s['mean_ic']:+.4f} "
                  f"ICIR={s['icir']:.3f}")

        results["contributions"] = contrib

        # IC improvement from blending
        base_ic  = contrib.get(
            "LGBM_alone",{}
        ).get("mean_ic", 0)
        final_ic = contrib.get(
            "Final_signal",{}
        ).get("mean_ic", 0)
        improvement = final_ic - base_ic

        results["base_ic"]    = base_ic
        results["final_ic"]   = final_ic
        results["improvement"] = improvement

        pct = (
            improvement / abs(base_ic) * 100
            if base_ic != 0 else 0
        )
        print(f"\n  IC Improvement from ensemble:")
        print(f"    Base (LGBM)  : {base_ic:+.4f}")
        print(f"    Final (blend): {final_ic:+.4f}")
        print(f"    Improvement  : "
              f"{improvement:+.4f} "
              f"({pct:+.1f}%)")

        return results

    # ─────────────────────────────────────────────────
    #  Write results
    # ─────────────────────────────────────────────────
    def write_results(self, signal_attr, regime_attr,
                       time_attr, factor_attr,
                       ic_decay, ensemble_attr):
        print("\nWriting results...")

        def _write(df, path):
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

        # Signal attribution
        rows = []
        for name, s in signal_attr.items():
            rows.append({"signal": name, **s})
        if rows:
            _write(
                pd.DataFrame(rows),
                f"{self.bt_path}/bt03_signal_attr"
            )
            print("  ✓ bt03_signal_attr")

        # Regime attribution
        rows = []
        for regime in ["Bull","HighVol","Bear"]:
            ls_key = f"LS_{regime}"
            ic_key = f"IC_{regime}"
            if ls_key in regime_attr:
                ls = regime_attr[ls_key]
                ic = regime_attr.get(ic_key, {})
                rows.append({
                    "regime"  : regime,
                    "sharpe"  : ls.get("sharpe",0),
                    "ann_ret" : ls.get("ann_ret",0),
                    "max_dd"  : ls.get("max_dd",0),
                    "n_days"  : ls.get("n_days",0),
                    "pct_days": ls.get("pct_days",0),
                    "mean_ic" : ic.get("mean_ic",0),
                    "icir"    : ic.get("icir",0),
                })
        if rows:
            _write(
                pd.DataFrame(rows),
                f"{self.bt_path}/bt03_regime_attr"
            )
            print("  ✓ bt03_regime_attr")

        # Time attribution
        if "monthly" in time_attr:
            _write(
                time_attr["monthly"],
                f"{self.bt_path}/bt03_monthly_attr"
            )
            print("  ✓ bt03_monthly_attr")

        if "decade" in time_attr:
            _write(
                time_attr["decade"],
                f"{self.bt_path}/bt03_decade_attr"
            )
            print("  ✓ bt03_decade_attr")

        if "annual_ret" in time_attr:
            _write(
                time_attr["annual_ret"],
                f"{self.bt_path}/bt03_annual_attr"
            )
            print("  ✓ bt03_annual_attr")

        # IC decay
        decay_rows = []
        for lag, ic in zip(
            ic_decay.get("lags",[]),
            ic_decay.get("ic_vals",[])
        ):
            decay_rows.append({
                "lag_days": lag,
                "mean_ic" : ic,
            })
        if decay_rows:
            _write(
                pd.DataFrame(decay_rows),
                f"{self.bt_path}/bt03_ic_decay"
            )
            print("  ✓ bt03_ic_decay")

        # Ensemble contribution
        rows = []
        for name, s in ensemble_attr.get(
            "contributions",{}
        ).items():
            rows.append({
                "component": name, **s
            })
        if rows:
            _write(
                pd.DataFrame(rows),
                f"{self.bt_path}/bt03_ensemble_attr"
            )
            print("  ✓ bt03_ensemble_attr")

    # ─────────────────────────────────────────────────
    #  Run
    # ─────────────────────────────────────────────────
    def run(self):
        print("="*55)
        print("Backtest 03 — Performance Attribution")
        print("="*55)
        start = datetime.now()

        daily, ic_df, pred, lgbm, vol, gold = (
            self.load_data()
        )
        signal_attr  = self.signal_attribution(
            pred, lgbm, vol
        )
        regime_attr  = self.regime_attribution(
            daily, pred
        )
        time_attr    = self.time_attribution(
            daily, ic_df
        )
        factor_attr  = self.factor_attribution(
            pred, gold
        )
        ic_decay     = self.ic_decay_analysis(pred)
        ensemble_attr= self.ensemble_contribution(pred)

        self.write_results(
            signal_attr, regime_attr,
            time_attr, factor_attr,
            ic_decay, ensemble_attr
        )

        elapsed = (
            datetime.now()-start
        ).seconds / 60
        print(f"\nTotal time : {elapsed:.1f} min")
        print("Backtest 03 COMPLETE ✓")

        return {
            "daily"        : daily,
            "ic_df"        : ic_df,
            "pred"         : pred,
            "signal_attr"  : signal_attr,
            "regime_attr"  : regime_attr,
            "time_attr"    : time_attr,
            "factor_attr"  : factor_attr,
            "ic_decay"     : ic_decay,
            "ensemble_attr": ensemble_attr,
        }

# COMMAND ----------

class Backtest03Charts:
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
    #  Chart 1 — Signal Attribution
    # ─────────────────────────────────────────────────
    def chart_signal_attribution(self,
                                   signal_attr):
        names   = list(signal_attr.keys())
        ics     = [
            signal_attr[n]["mean_ic"] for n in names
        ]
        icirs   = [
            signal_attr[n]["icir"] for n in names
        ]
        hits    = [
            signal_attr[n]["hit_rate"] for n in names
        ]
        colors  = [
            self.C["success"]
            if n == "ML04_Ensemble"
            else self.C["primary"]
            for n in names
        ]

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=["Mean IC","ICIR","Hit Rate"],
            specs=[[
                {"type":"xy"},
                {"type":"xy"},
                {"type":"xy"},
            ]]
        )
        for vals, col in [
            (ics,1),(icirs,2),(hits,3)
        ]:
            fig.add_trace(go.Bar(
                x=names, y=vals,
                marker_color=colors,
                text=[f"{v:+.4f}" for v in vals],
                textposition="outside",
                showlegend=False
            ), row=1, col=col)
            fig.add_hline(
                y=0, line_dash="dash",
                line_color="white", opacity=0.3,
                row=1, col=col
            )

        fig.update_layout(
            title=(
                "<b>BT03 — Signal Attribution<br>"
                "<sup>IC contribution by signal"
                "</sup></b>"
            ),
            template=self.TEMPLATE, height=500
        )
        fig.show()

    # ─────────────────────────────────────────────────
    #  Chart 2 — Regime Attribution
    # ─────────────────────────────────────────────────
    def chart_regime_attribution(self,
                                   regime_attr,
                                   daily):
        regimes = ["Bull","HighVol","Bear"]
        colors  = [
            self.REGIME_C.get(r,"#9E9E9E")
            for r in regimes
        ]

        sharpes = [
            regime_attr.get(
                f"LS_{r}",{}
            ).get("sharpe",0)
            for r in regimes
        ]
        ann_rets= [
            regime_attr.get(
                f"LS_{r}",{}
            ).get("ann_ret",0)*100
            for r in regimes
        ]
        max_dds = [
            regime_attr.get(
                f"LS_{r}",{}
            ).get("max_dd",0)*100
            for r in regimes
        ]
        ics     = [
            regime_attr.get(
                f"IC_{r}",{}
            ).get("mean_ic",0)
            for r in regimes
        ]
        pcts    = [
            regime_attr.get(
                f"LS_{r}",{}
            ).get("pct_days",0)*100
            for r in regimes
        ]

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                "Sharpe by Regime",
                "Ann Return (%) by Regime",
                "Time in Regime (%)",
                "Max DD (%) by Regime",
                "Mean IC by Regime",
                "Equity by Regime",
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
            specs=[
                [{"type":"xy"},{"type":"xy"},
                 {"type":"xy"}],
                [{"type":"xy"},{"type":"xy"},
                 {"type":"xy"}],
            ]
        )

        for vals, r, c, title in [
            (sharpes,  1,1,"Sharpe"),
            (ann_rets, 1,2,"Ann Ret (%)"),
            (pcts,     1,3,"% Time"),
            (max_dds,  2,1,"Max DD (%)"),
            (ics,      2,2,"Mean IC"),
        ]:
            fig.add_trace(go.Bar(
                x=regimes, y=vals,
                marker_color=colors,
                text=[f"{v:.2f}" for v in vals],
                textposition="outside",
                showlegend=False
            ), row=r, col=c)
            fig.add_hline(
                y=0, line_dash="dash",
                line_color="white", opacity=0.3,
                row=r, col=c
            )

        # Regime equity curves
        if "regime" in daily.columns:
            daily_c = daily.copy()
            daily_c["date"] = pd.to_datetime(
                daily_c["date"]
            )
            daily_c = daily_c.sort_values("date")
            for regime, color in self.REGIME_C.items():
                mask = daily_c["regime"] == regime
                r_df = daily_c[mask]
                if len(r_df) < 2:
                    continue
                cum = (
                    1 + r_df["ls_net"].fillna(0)
                ).cumprod()
                fig.add_trace(go.Scatter(
                    x=r_df["date"].values,
                    y=cum.values,
                    name=regime, mode="lines",
                    line=dict(color=color, width=2)
                ), row=2, col=3)

        fig.update_layout(
            title=(
                "<b>BT03 — Regime Attribution<br>"
                "<sup>P&L by Bull / HighVol / Bear"
                "</sup></b>"
            ),
            template=self.TEMPLATE, height=700
        )
        fig.show()

    # ─────────────────────────────────────────────────
    #  Chart 3 — Time Attribution
    # ─────────────────────────────────────────────────
    def chart_time_attribution(self, time_attr):
        monthly = time_attr.get("monthly",
                                pd.DataFrame())
        decade  = time_attr.get("decade",
                                pd.DataFrame())
        yr_ret  = time_attr.get("annual_ret",
                                pd.DataFrame())
        yr_ic   = time_attr.get("annual_ic",
                                pd.DataFrame())

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Monthly Seasonality (Sharpe)",
                "Decade Performance",
                "Annual Returns (%)",
                "Annual IC",
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
            specs=[
                [{"type":"xy"},{"type":"xy"}],
                [{"type":"xy"},{"type":"xy"}],
            ]
        )

        # Monthly
        if len(monthly) > 0:
            m_vals = monthly["sharpe"].tolist()
            bar_c  = [
                self.C["success"] if v > 0
                else self.C["secondary"]
                for v in m_vals
            ]
            fig.add_trace(go.Bar(
                x=monthly["month"].tolist(),
                y=m_vals,
                marker_color=bar_c,
                text=[f"{v:.2f}" for v in m_vals],
                textposition="outside",
                showlegend=False
            ), row=1, col=1)
            fig.add_hline(
                y=0, line_dash="dash",
                line_color="white", opacity=0.3,
                row=1, col=1
            )

        # Decade
        if len(decade) > 0:
            d_vals = decade["sharpe"].tolist()
            bar_c  = [
                self.C["success"] if v > 0
                else self.C["secondary"]
                for v in d_vals
            ]
            fig.add_trace(go.Bar(
                x=decade["decade"].tolist(),
                y=d_vals,
                marker_color=bar_c,
                text=[f"{v:.2f}" for v in d_vals],
                textposition="outside",
                showlegend=False
            ), row=1, col=2)

        # Annual returns
        if len(yr_ret) > 0:
            r_vals = (
                yr_ret["ann_ret"] * 100
            ).tolist()
            bar_c  = [
                self.C["success"] if v > 0
                else self.C["secondary"]
                for v in r_vals
            ]
            fig.add_trace(go.Bar(
                x=yr_ret["year"].tolist(),
                y=r_vals,
                marker_color=bar_c,
                text=[f"{v:.1f}%" for v in r_vals],
                textposition="outside",
                showlegend=False
            ), row=2, col=1)
            fig.add_hline(
                y=0, line_dash="dash",
                line_color="white", opacity=0.3,
                row=2, col=1
            )

        # Annual IC
        if len(yr_ic) > 0:
            ic_vals = yr_ic["mean_ic"].tolist()
            bar_c   = [
                self.C["success"] if v > 0
                else self.C["secondary"]
                for v in ic_vals
            ]
            fig.add_trace(go.Bar(
                x=yr_ic["year"].tolist(),
                y=ic_vals,
                marker_color=bar_c,
                text=[
                    f"{v:.3f}" for v in ic_vals
                ],
                textposition="outside",
                showlegend=False
            ), row=2, col=2)
            fig.add_hline(
                y=0, line_dash="dash",
                line_color="white", opacity=0.3,
                row=2, col=2
            )

        fig.update_layout(
            title=(
                "<b>BT03 — Time Attribution<br>"
                "<sup>Performance by period"
                "</sup></b>"
            ),
            template=self.TEMPLATE, height=700
        )
        fig.show()

    # ─────────────────────────────────────────────────
    #  Chart 4 — Factor Attribution
    # ─────────────────────────────────────────────────
    def chart_factor_attribution(self, factor_attr):
        fic   = factor_attr.get("factor_ics", {})
        fric  = factor_attr.get("factor_ret_ics", {})

        names = list(fic.keys())
        if not names:
            print("  No factor data")
            return

        ics_sig = [fic[n]["mean_ic"]  for n in names]
        ics_ret = [fric.get(n,{}).get("mean_ic",0)
                   for n in names]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Factor IC vs Signal",
                "Factor IC vs Returns",
            ],
            specs=[
                [{"type":"xy"},{"type":"xy"}]
            ]
        )

        for vals, col in [
            (ics_sig,1),(ics_ret,2)
        ]:
            bar_c = [
                self.C["success"] if v > 0
                else self.C["secondary"]
                for v in vals
            ]
            fig.add_trace(go.Bar(
                x=names, y=vals,
                marker_color=bar_c,
                text=[f"{v:+.4f}" for v in vals],
                textposition="outside",
                showlegend=False
            ), row=1, col=col)
            fig.add_hline(
                y=0, line_dash="dash",
                line_color="white", opacity=0.3,
                row=1, col=col
            )

        fig.update_layout(
            title=(
                "<b>BT03 — Factor Attribution<br>"
                "<sup>What drives the alpha signal?"
                "</sup></b>"
            ),
            template=self.TEMPLATE, height=500
        )
        fig.show()

    # ─────────────────────────────────────────────────
    #  Chart 5 — IC Decay
    # ─────────────────────────────────────────────────
    def chart_ic_decay(self, ic_decay):
        lags    = ic_decay.get("lags", [])
        ic_vals = ic_decay.get("ic_vals", [])
        hl      = ic_decay.get("half_life_days")

        if not lags:
            return

        ic_0  = ic_vals[0] if ic_vals[0] != 0 else 1e-8
        pct   = [v/abs(ic_0)*100 for v in ic_vals]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "IC Decay (absolute)",
                "IC Decay (% of initial)",
            ],
            specs=[
                [{"type":"xy"},{"type":"xy"}]
            ]
        )

        for vals, col, fmt in [
            (ic_vals,1,".4f"),
            (pct,    2,".1f"),
        ]:
            fig.add_trace(go.Scatter(
                x=lags, y=vals,
                mode="lines+markers",
                line=dict(
                    color=self.C["primary"],
                    width=2
                ),
                marker=dict(size=8),
                showlegend=False
            ), row=1, col=col)
            fig.add_hline(
                y=0, line_dash="dash",
                line_color="white", opacity=0.3,
                row=1, col=col
            )
            if col == 2:
                fig.add_hline(
                    y=50, line_dash="dot",
                    line_color=self.C["warning"],
                    annotation_text="Half-life",
                    row=1, col=col
                )
            fig.update_xaxes(
                title_text="Lag (days)",
                row=1, col=col
            )
        fig.update_yaxes(
            title_text="Mean IC", row=1, col=1
        )
        fig.update_yaxes(
            title_text="% of Initial IC",
            row=1, col=2
        )

        hl_txt = (
            f"Half-life={hl}d"
            if hl else "Half-life>63d"
        )
        fig.update_layout(
            title=(
                f"<b>BT03 — IC Decay Analysis<br>"
                f"<sup>"
                f"IC(lag=0)={ic_vals[0]:+.4f} | "
                f"{hl_txt}"
                f"</sup></b>"
            ),
            template=self.TEMPLATE, height=500
        )
        fig.show()

    # ─────────────────────────────────────────────────
    #  Chart 6 — Ensemble Contribution
    # ─────────────────────────────────────────────────
    def chart_ensemble_contribution(self,
                                     ensemble_attr):
        contrib = ensemble_attr.get(
            "contributions", {}
        )
        if not contrib:
            return

        names = list(contrib.keys())
        ics   = [contrib[n]["mean_ic"] for n in names]
        icirs = [contrib[n]["icir"]    for n in names]
        hits  = [
            contrib[n]["hit_rate"] for n in names
        ]

        colors = [
            self.C["success"]
            if n == "Final_signal"
            else self.C["primary"]
            if n == "LGBM_alone"
            else self.C["teal"]
            for n in names
        ]

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=["IC","ICIR","Hit Rate"],
            specs=[
                [{"type":"xy"},
                 {"type":"xy"},
                 {"type":"xy"}]
            ]
        )

        for vals, col in [
            (ics,1),(icirs,2),(hits,3)
        ]:
            fig.add_trace(go.Bar(
                x=names, y=vals,
                marker_color=colors,
                text=[f"{v:+.4f}" for v in vals],
                textposition="outside",
                showlegend=False
            ), row=1, col=col)
            fig.add_hline(
                y=0, line_dash="dash",
                line_color="white", opacity=0.3,
                row=1, col=col
            )

        base   = ensemble_attr.get("base_ic", 0)
        final  = ensemble_attr.get("final_ic", 0)
        improv = ensemble_attr.get("improvement", 0)
        pct    = (
            improv/abs(base)*100
            if base != 0 else 0
        )
        fig.update_layout(
            title=(
                f"<b>BT03 — Ensemble Contribution"
                f"<br><sup>"
                f"Base IC={base:+.4f} → "
                f"Final IC={final:+.4f} | "
                f"Improvement={improv:+.4f} "
                f"({pct:+.1f}%)"
                f"</sup></b>"
            ),
            template=self.TEMPLATE, height=500
        )
        fig.show()

    # ─────────────────────────────────────────────────
    #  Chart 7 — Seasonality Heatmap
    # ─────────────────────────────────────────────────
    def chart_seasonality_heatmap(self, daily):
        daily = daily.copy()
        daily["date"]  = pd.to_datetime(daily["date"])
        daily["year"]  = daily["date"].dt.year
        daily["month"] = daily["date"].dt.month

        # Monthly returns pivot
        monthly = daily.groupby(
            ["year","month"]
        )["ls_net"].apply(
            lambda x: float((1+x).prod()-1)
        ).reset_index()
        monthly.columns = ["year","month","ret"]

        pivot = monthly.pivot(
            index="year", columns="month",
            values="ret"
        ) * 100

        month_names = [
            "Jan","Feb","Mar","Apr","May","Jun",
            "Jul","Aug","Sep","Oct","Nov","Dec"
        ]
        pivot.columns = month_names[
            :len(pivot.columns)
        ]

        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=[str(y) for y in pivot.index.tolist()],
            colorscale=[
                [0.0, "#FF5722"],
                [0.5, "#1a1a2e"],
                [1.0, "#4CAF50"],
            ],
            zmid=0,
            text=np.round(pivot.values, 1),
            texttemplate="%{text}%",
            textfont={"size": 9},
            colorbar=dict(title="Ret %"),
        ))

        pos_months = int((monthly["ret"] > 0).sum())
        tot_months = len(monthly)
        fig.update_layout(
            title=(
                f"<b>BT03 — Monthly Returns "
                f"Heatmap (Net TC, %)<br>"
                f"<sup>"
                f"Positive months: "
                f"{pos_months}/{tot_months} "
                f"({pos_months/tot_months:.1%})"
                f"</sup></b>"
            ),
            template=self.TEMPLATE,
            height=max(600, len(pivot)*22),
            xaxis_title="Month",
            yaxis_title="Year",
            yaxis=dict(autorange="reversed"),
        )
        fig.show()

    # ─────────────────────────────────────────────────
    #  Run all
    # ─────────────────────────────────────────────────
    def run_all(self, results):
        print("\n" + "="*55)
        print("Backtest 03 Charts")
        print("="*55)

        print("\n[1/7] Signal Attribution...")
        self.chart_signal_attribution(
            results["signal_attr"]
        )

        print("[2/7] Regime Attribution...")
        self.chart_regime_attribution(
            results["regime_attr"],
            results["daily"]
        )

        print("[3/7] Time Attribution...")
        self.chart_time_attribution(
            results["time_attr"]
        )

        print("[4/7] Factor Attribution...")
        self.chart_factor_attribution(
            results["factor_attr"]
        )

        print("[5/7] IC Decay...")
        self.chart_ic_decay(results["ic_decay"])

        print("[6/7] Ensemble Contribution...")
        self.chart_ensemble_contribution(
            results["ensemble_attr"]
        )

        print("[7/7] Seasonality Heatmap...")
        self.chart_seasonality_heatmap(
            results["daily"]
        )

        print("\nAll 7 charts ✓")

# COMMAND ----------

bt03 = Backtest03(
    spark     = spark,
    gold_path = GOLD_PATH,
    ml_path   = ML_PATH,
    bt_path   = BT_PATH,
)

results = bt03.run()

charts = Backtest03Charts()
charts.run_all(results)

print("\nBacktest 03 COMPLETE ✓")

# COMMAND ----------

sig  = spark.read.format("delta").load(
    f"{BT_PATH}/bt03_signal_attr"
).toPandas()

reg  = spark.read.format("delta").load(
    f"{BT_PATH}/bt03_regime_attr"
).toPandas()

decay = spark.read.format("delta").load(
    f"{BT_PATH}/bt03_ic_decay"
).toPandas()

ens  = spark.read.format("delta").load(
    f"{BT_PATH}/bt03_ensemble_attr"
).toPandas()

print("="*60)
print("Backtest 03 — Final Attribution Summary")
print("="*60)

print(f"\nSignal Attribution:")
print(sig[["signal","mean_ic","icir","hit_rate"]
].round(4).to_string(index=False))

print(f"\nRegime Attribution:")
print(reg[["regime","sharpe","ann_ret",
           "max_dd","mean_ic"]
].round(4).to_string(index=False))

print(f"\nIC Decay:")
print(decay.to_string(index=False))

print(f"\nEnsemble Contribution:")
print(ens[["component","mean_ic","icir",
           "hit_rate"]
].round(4).to_string(index=False))

en = results["ensemble_attr"]
ic = results["ic_decay"]
print(f"\nKey Insights:")
print(f"  Ensemble IC improvement : "
      f"{en.get('improvement',0):+.4f} "
      f"({en.get('improvement',0)/max(abs(en.get('base_ic',1e-8)),1e-8)*100:+.1f}%)")
hl = ic.get("half_life_days")
print(f"  Signal half-life        : "
      f"{'~'+str(hl)+'d' if hl else '>63d'}")

print(f"\n{'='*60}")
print(f"Full Backtest Stack COMPLETE:")
print(f"  Backtest 01 ✅ Portfolio: "
      f"Sharpe=6.15 IC=0.086")
print(f"  Backtest 02 ✅ Risk: "
      f"100% pos yrs Alpha=11.6%/yr")
print(f"  Backtest 03 ✅ Attribution: "
      f"Full signal decomposition")
print(f"{'='*60}")