# Databricks notebook source
# MAGIC %pip install plotly scipy pandas numpy --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from datetime import datetime
import pandas as pd
import numpy as np
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
GOLD_PATH = f"{BASE_PATH}/gold/delta"
ML_PATH   = f"{BASE_PATH}/ml/delta"

# Regime position weights from ML 01
REGIME_WEIGHTS = {
    "Bull"   : 1.0,
    "HighVol": 0.6,
    "Bear"   : 0.3,
}

print("="*55)
print("ML 04 — Regime-Weighted Ensemble")
print("="*55)
print(f"GOLD_PATH : {GOLD_PATH}")
print(f"ML_PATH   : {ML_PATH}")
print(f"\nRegime weights:")
for r, w in REGIME_WEIGHTS.items():
    print(f"  {r:8}: {w}")
print(f"\nInputs:")
print(f"  ML 01 HMM      : regime labels + probs")
print(f"  ML 02 LightGBM : alpha signals (IC=0.093)")
print(f"  ML 03 PatchTST : vol forecasts (Corr=0.81)")

# COMMAND ----------

class MLEnsemble:
    """
    ML 04 — Regime-Weighted Ensemble.

    Blends ML 01 + ML 02 + ML 03 signals into
    a unified position signal for portfolio
    construction.

    Architecture:
      1. Load all ML signals
      2. Merge on (date, ticker)
      3. Alpha signal blending:
           final_alpha = w_lgbm * lgbm_pred
                       + w_vol  * vol_signal
      4. Regime-weighted position sizing:
           Bull    → 1.0 × signal
           HighVol → 0.6 × signal
           Bear    → 0.3 × signal
      5. Vol-adjusted sizing:
           size = regime_weight / (pred_vol + ε)
           size = clip(size, 0.5, 1.0)
      6. Final signal = alpha × position_size
      7. L/S portfolio backtest
      8. Evaluation vs ML 02 standalone
    """

    TC_BPS         = 5
    QUINTILE_N     = 5   # top/bottom quintile
    MIN_TICKERS    = 20  # min per date

    def __init__(self, spark, gold_path, ml_path):
        self.spark     = spark
        self.gold_path = gold_path
        self.ml_path   = ml_path

        print("MLEnsemble ✓")
        print(f"  TC       : {self.TC_BPS}bps")
        print(f"  Quintile : top/bottom "
              f"1/{self.QUINTILE_N}")

    # ─────────────────────────────────────────────────
    #  Utility
    # ─────────────────────────────────────────────────
    @staticmethod
    def _safe_ic(pred, actual, min_n=30):
        p = np.asarray(pred,   dtype=float)
        a = np.asarray(actual, dtype=float)
        m = (~np.isnan(p) & ~np.isnan(a) &
             ~np.isinf(p) & ~np.isinf(a))
        if m.sum() < min_n:
            return np.nan
        if np.std(p[m]) < 1e-10 or \
           np.std(a[m]) < 1e-10:
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
                "mean_ic":0., "icir":0.,
                "ic_std":0., "hit_rate":0.,
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

    # ─────────────────────────────────────────────────
    #  Step 1 — Load all signals
    # ─────────────────────────────────────────────────
    def load_signals(self):
        print("\nStep 1: Loading all ML signals...")
        start = datetime.now()

        # ── ML 02: LightGBM predictions ──────────────
        print("  Loading ML 02 LightGBM...")
        lgbm = self.spark.read.format("delta").load(
            f"{self.ml_path}/lgbm_v3_predictions"
        ).select(
            "date","ticker",
            "pred_ensemble",
            "pred_final",
            "fwd_return_21d",
            "regime_label",
        ).toPandas()
        lgbm["date"] = pd.to_datetime(lgbm["date"])
        print(f"    Rows: {len(lgbm):,}")

        # ── ML 03: PatchTST vol predictions ──────────
        print("  Loading ML 03 PatchTST...")
        vol = self.spark.read.format("delta").load(
            f"{self.ml_path}"
            f"/patchtst_vol_predictions"
        ).select(
            "date","ticker",
            "pred_fwd_vol_21d",
            "vol_position_signal",
        ).toPandas()
        vol["date"] = pd.to_datetime(vol["date"])
        print(f"    Rows: {len(vol):,}")

        # ── ML 01: HMM regime labels ─────────────────
        print("  Loading ML 01 HMM regimes...")
        try:
            hmm = self.spark.read.format("delta").load(
                f"{self.ml_path}/hmm_regime_labels"
            ).select(
                "date",
                "regime_label",
                "prob_bull",
                "prob_bear",
                "prob_highvol",
            ).toPandas()
            hmm["date"] = pd.to_datetime(hmm["date"])
            print(f"    Rows: {len(hmm):,}")
            HMM_OK = True
        except Exception as e:
            print(f"    ⚠️ HMM not found: {e}")
            print(f"    Using regime from ML 02")
            HMM_OK = False
            hmm = None

        # ── Gold: forward returns (ground truth) ─────
        print("  Loading Gold price factors...")
        gold_cols = [
            "date","ticker",
            "fwd_return_21d",
            "fwd_return_5d",
        ]
        sdf      = self.spark.read.format("delta").load(
            f"{self.gold_path}/price_factors"
        )
        avail    = set(sdf.columns)
        sel_cols = [
            c for c in gold_cols if c in avail
        ]
        gold = sdf.select(*sel_cols).toPandas()
        gold["date"] = pd.to_datetime(gold["date"])
        print(f"    Rows: {len(gold):,}")

        elapsed = (datetime.now()-start).seconds
        print(f"  Elapsed: {elapsed}s")
        return lgbm, vol, hmm, gold, HMM_OK

    # ─────────────────────────────────────────────────
    #  Step 2 — Merge signals
    # ─────────────────────────────────────────────────
    def merge_signals(self, lgbm, vol, hmm,
                       gold, HMM_OK):
        print("\nStep 2: Merging signals...")
        start = datetime.now()

        # Base: LightGBM predictions
        df = lgbm.copy()

        # Merge PatchTST vol
        df = df.merge(
            vol[["date","ticker",
                 "pred_fwd_vol_21d",
                 "vol_position_signal"]],
            on=["date","ticker"],
            how="left"
        )

        # Merge HMM regime (date-level)
        if HMM_OK and hmm is not None:
            df = df.merge(
                hmm[[
                    "date","regime_label",
                    "prob_bull","prob_bear",
                    "prob_highvol",
                ]],
                on="date",
                how="left",
                suffixes=("","_hmm")
            )
            # Use HMM if available else ML 02 regime
            if "regime_label_hmm" in df.columns:
                df["regime_label"] = df[
                    "regime_label_hmm"
                ].fillna(df["regime_label"])
                df.drop(
                    columns=["regime_label_hmm"],
                    inplace=True
                )
        else:
            # Add dummy probs from ML 02 regime col
            df["prob_bull"]    = (
                df["regime_label"] == "Bull"
            ).astype(float)
            df["prob_bear"]    = (
                df["regime_label"] == "Bear"
            ).astype(float)
            df["prob_highvol"] = (
                df["regime_label"] == "HighVol"
            ).astype(float)

        # Merge ground truth returns
        if "fwd_return_21d" not in df.columns:
            df = df.merge(
                gold[[
                    "date","ticker","fwd_return_21d"
                ]],
                on=["date","ticker"],
                how="left"
            )

        # Fill missing
        df["pred_fwd_vol_21d"] = df[
            "pred_fwd_vol_21d"
        ].fillna(0.20)
        df["vol_position_signal"] = df[
            "vol_position_signal"
        ].fillna(0.75)
        df["regime_label"] = df[
            "regime_label"
        ].fillna("Bull")
        df["prob_bull"]    = df[
            "prob_bull"
        ].fillna(0.5)
        df["prob_bear"]    = df[
            "prob_bear"
        ].fillna(0.1)
        df["prob_highvol"] = df[
            "prob_highvol"
        ].fillna(0.4)

        df = df.sort_values(
            ["date","ticker"]
        ).reset_index(drop=True)

        elapsed = (datetime.now()-start).seconds
        print(f"  Merged rows  : {len(df):,}")
        print(f"  Tickers      : "
              f"{df['ticker'].nunique():,}")
        print(f"  Date range   : "
              f"{df['date'].min().date()} → "
              f"{df['date'].max().date()}")
        print(f"  Regime dist  :")
        for r, cnt in df["regime_label"].value_counts(
        ).items():
            pct = cnt / len(df) * 100
            print(f"    {r:8}: {cnt:>10,} "
                  f"({pct:.1f}%)")
        print(f"  Elapsed      : {elapsed}s")
        return df

    # ─────────────────────────────────────────────────
    #  Step 3 — Build ensemble signal
    # ─────────────────────────────────────────────────
    def build_ensemble_signal(self, df):
        """
        Three-layer signal construction:

        Layer 1 — Alpha blend:
          alpha = 0.7 × lgbm_pred
                + 0.3 × vol_signal_adj

        Layer 2 — Confidence weighting:
          confidence = 1 - pred_std (from ML 02)
          alpha_conf = confidence × alpha

        Layer 3 — Position sizing:
          regime_weight = f(Bull/HighVol/Bear)
          vol_weight    = vol_position_signal
          final_signal  = alpha_conf
                        × regime_weight
                        × vol_weight
        """
        print("\nStep 3: Building ensemble signal...")
        df = df.copy()

        # ── Layer 1: Alpha blend ──────────────────────
        # Vol signal: low predicted vol → long signal
        # Invert: high vol rank → bearish signal
        df["vol_alpha"] = 1.0 - df[
            "pred_fwd_vol_21d"
        ].clip(0, 1)

        # Normalize vol_alpha cross-sectionally
        df["vol_alpha_cs"] = df.groupby("date")[
            "vol_alpha"
        ].transform(
            lambda x: (x - x.mean()) /
                      (x.std() + 1e-8)
        )
        df["lgbm_cs"] = df.groupby("date")[
            "pred_final"
        ].transform(
            lambda x: (x - x.mean()) /
                      (x.std() + 1e-8)
        )

        # Blend: 70% LightGBM + 30% vol signal
        df["alpha_blend"] = (
            0.70 * df["lgbm_cs"] +
            0.30 * df["vol_alpha_cs"]
        )

        # ── Layer 2: Regime-weighted position ────────
        regime_map = {
            "Bull"   : REGIME_WEIGHTS["Bull"],
            "HighVol": REGIME_WEIGHTS["HighVol"],
            "Bear"   : REGIME_WEIGHTS["Bear"],
        }
        df["regime_weight"] = df[
            "regime_label"
        ].map(regime_map).fillna(0.5)

        # Soft regime weight using HMM probs
        # More nuanced than hard label mapping
        df["regime_weight_soft"] = (
            REGIME_WEIGHTS["Bull"]    * df["prob_bull"] +
            REGIME_WEIGHTS["HighVol"] * df["prob_highvol"] +
            REGIME_WEIGHTS["Bear"]    * df["prob_bear"]
        )

        # ── Layer 3: Vol-adjusted position size ──────
        # vol_position_signal: 0.5–1.0 from ML 03
        # High vol → smaller position (0.5)
        # Low vol  → full position  (1.0)
        df["position_size"] = (
            df["regime_weight_soft"] *
            df["vol_position_signal"]
        ).clip(0.3, 1.0)

        # ── Final signal ──────────────────────────────
        df["signal_final"] = (
            df["alpha_blend"] *
            df["position_size"]
        )

        # CS normalize final signal
        df["signal_final_cs"] = df.groupby("date")[
            "signal_final"
        ].transform(
            lambda x: (x - x.mean()) /
                      (x.std() + 1e-8)
        )

        print(f"  Alpha blend    : "
              f"70% LightGBM + 30% vol ✅")
        print(f"  Regime weight  : "
              f"soft HMM probs ✅")
        print(f"  Vol sizing     : "
              f"ML 03 position signal ✅")
        print(f"\n  Signal stats (final_cs):")
        s = df["signal_final_cs"]
        print(f"  Mean : {s.mean():.4f}")
        print(f"  Std  : {s.std():.4f}")
        print(f"  P5   : {s.quantile(0.05):.4f}")
        print(f"  P95  : {s.quantile(0.95):.4f}")
        return df

    # ─────────────────────────────────────────────────
    #  Step 4 — Backtest
    # ─────────────────────────────────────────────────
    def backtest(self, df, signal_col,
                  label="signal"):
        """
        Quintile L/S backtest with turnover TC.
        Returns ic_series, ls_series.
        """
        ic_rows = []
        ls_rows = []
        prev_long  = set()
        prev_short = set()
        target_col = "fwd_return_21d"

        for date, grp in df.groupby("date"):
            if len(grp) < self.MIN_TICKERS:
                continue
            grp = grp.dropna(
                subset=[signal_col, target_col]
            )
            if len(grp) < self.MIN_TICKERS:
                continue

            # IC
            ic = self._safe_ic(
                grp[signal_col].values,
                grp[target_col].values
            )
            ic_rows.append({
                "date": date, "ic": ic
            })

            # Quintile L/S
            n   = max(1, len(grp)//self.QUINTILE_N)
            srt = grp.sort_values(
                signal_col, ascending=False
            )
            long_set  = set(
                srt.head(n)["ticker"].tolist()
            )
            short_set = set(
                srt.tail(n)["ticker"].tolist()
            )

            lr = srt.head(n)[target_col].mean()
            sr = srt.tail(n)[target_col].mean()
            ls = lr - sr

            # Turnover-based TC
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

            tc = turnover * self.TC_BPS / 10000
            prev_long  = long_set
            prev_short = short_set

            ls_rows.append({
                "date"     : date,
                "long_ret" : float(lr),
                "short_ret": float(sr),
                "ls_ret"   : float(ls),
                "ls_ret_tc": float(ls - tc),
                "tc_cost"  : float(tc),
                "turnover" : float(turnover),
            })

        ic_df = pd.DataFrame(ic_rows)
        ls_df = pd.DataFrame(ls_rows)
        return ic_df, ls_df

    def _ls_stats(self, ls_df, col="ls_ret_tc"):
        r   = ls_df[col].dropna()
        ann = r.mean() * 252
        vol = r.std()  * np.sqrt(252)
        sr  = ann / (vol + 1e-8)
        dd  = (
            (1+r).cumprod() /
            (1+r).cumprod().cummax() - 1
        ).min()
        to  = ls_df["turnover"].mean() \
              if "turnover" in ls_df.columns \
              else 0
        tc_yr = ls_df["tc_cost"].mean() * 252 \
                * 10000 \
                if "tc_cost" in ls_df.columns \
                else 0
        return {
            "ann_ret" : float(ann),
            "ann_vol" : float(vol),
            "sharpe"  : float(sr),
            "max_dd"  : float(dd),
            "avg_to"  : float(to),
            "tc_bps_yr": float(tc_yr),
        }

    # ─────────────────────────────────────────────────
    #  Step 5 — Full evaluation
    # ─────────────────────────────────────────────────
    def evaluate(self, df):
        print("\nStep 4: Full evaluation...")

        signals = {
            "ML02 LightGBM" : "pred_final",
            "ML04 Ensemble" : "signal_final_cs",
        }

        results = {}
        for name, col in signals.items():
            print(f"\n  Backtesting: {name}...")
            ic_df, ls_df = self.backtest(
                df, col, label=name
            )

            ic_s    = self._ic_stats(
                ic_df["ic"].tolist()
            )
            ls_gross= self._ls_stats(
                ls_df, "ls_ret"
            )
            ls_tc   = self._ls_stats(
                ls_df, "ls_ret_tc"
            )
            avg_to  = ls_df["turnover"].mean()
            tc_yr   = (
                ls_df["tc_cost"].mean()
                * 252 * 10000
            )

            results[name] = {
                "ic"      : ic_s,
                "ls_gross": ls_gross,
                "ls_tc"   : ls_tc,
                "ic_df"   : ic_df,
                "ls_df"   : ls_df,
                "avg_to"  : avg_to,
                "tc_yr"   : tc_yr,
            }

            flag_ic = (
                "✅" if abs(ic_s["mean_ic"]) > 0.04
                else "⚠️"
            )
            flag_sr = (
                "✅" if ls_tc["sharpe"] > 1.0
                else "⚠️"
            )
            print(f"    IC       : "
                  f"{ic_s['mean_ic']:+.4f} "
                  f"ICIR={ic_s['icir']:.3f} "
                  f"Hit={ic_s['hit_rate']:.1%} "
                  f"{flag_ic}")
            print(f"    Gross    : "
                  f"Sharpe={ls_gross['sharpe']:.2f} "
                  f"Ret={ls_gross['ann_ret']*100:.1f}%")
            print(f"    Net TC   : "
                  f"Sharpe={ls_tc['sharpe']:.2f} "
                  f"Ret={ls_tc['ann_ret']*100:.1f}% "
                  f"{flag_sr}")
            print(f"    Avg TO   : {avg_to:.1%}/day")
            print(f"    TC/yr    : "
                  f"{tc_yr:.0f}bps/yr")

        # Regime breakdown
        print(f"\n  Regime IC breakdown (ML04):")
        regime_ic = {}
        for regime in ["Bull","HighVol","Bear"]:
            r_data  = df[
                df["regime_label"] == regime
            ]
            r_ic    = []
            for date, grp in r_data.groupby("date"):
                ic = self._safe_ic(
                    grp["signal_final_cs"].values,
                    grp["fwd_return_21d"].values
                )
                if not np.isnan(ic):
                    r_ic.append(ic)
            s = self._ic_stats(r_ic)
            regime_ic[regime] = s
            print(f"    {regime:8}: "
                  f"IC={s['mean_ic']:+.4f}  "
                  f"ICIR={s['icir']:.3f}  "
                  f"n={s['n_dates']}d")

        results["regime_ic"] = regime_ic
        self._print_summary(results)
        return results

    def _print_summary(self, results):
        print("\n" + "="*55)
        print("ML 04 — Evaluation Summary")
        print("="*55)

        print(f"\n  {'Model':16} {'IC':>8} "
              f"{'ICIR':>8} {'Sharpe(Net)':>12}")
        print(f"  {'-'*48}")
        for name in [
            "ML02 LightGBM","ML04 Ensemble"
        ]:
            if name not in results:
                continue
            r    = results[name]
            ic_s = r["ic"]
            tc   = r["ls_tc"]
            flag = (
                "✅"
                if tc["sharpe"] > r["ls_gross"]["sharpe"] * 0.8
                else "⚠️"
            )
            print(f"  {name:16} "
                  f"{ic_s['mean_ic']:>+8.4f} "
                  f"{ic_s['icir']:>+8.4f} "
                  f"{tc['sharpe']:>12.2f} {flag}")

        # Improvement
        if ("ML02 LightGBM" in results and
                "ML04 Ensemble" in results):
            ic_02 = results[
                "ML02 LightGBM"
            ]["ic"]["mean_ic"]
            ic_04 = results[
                "ML04 Ensemble"
            ]["ic"]["mean_ic"]
            sr_02 = results[
                "ML02 LightGBM"
            ]["ls_tc"]["sharpe"]
            sr_04 = results[
                "ML04 Ensemble"
            ]["ls_tc"]["sharpe"]
            print(f"\n  Improvement (ML04 vs ML02):")
            print(f"    IC     : "
                  f"{ic_02:+.4f} → {ic_04:+.4f} "
                  f"({(ic_04-ic_02)/abs(ic_02)*100:+.1f}%)")
            print(f"    Sharpe : "
                  f"{sr_02:.2f} → {sr_04:.2f} "
                  f"({(sr_04-sr_02)/abs(sr_02)*100:+.1f}%)")

        n_pass = sum([
            abs(results.get(
                "ML04 Ensemble",{}
            ).get("ic",{}).get("mean_ic",0)) > 0.04,
            results.get(
                "ML04 Ensemble",{}
            ).get("ic",{}).get("icir",0) > 0.5,
            results.get(
                "ML04 Ensemble",{}
            ).get("ic",{}).get("hit_rate",0) > 0.55,
            results.get(
                "ML04 Ensemble",{}
            ).get("ls_tc",{}).get("sharpe",0) > 1.0,
        ])
        print(f"\n  Score: {n_pass}/4  "
              f"{'✅ PRODUCTION READY' if n_pass>=3 else '⚠️ REVIEW'}")

    # ─────────────────────────────────────────────────
    #  Step 6 — Write results
    # ─────────────────────────────────────────────────
    def write_results(self, df, results):
        print("\nStep 5: Writing results...")

        def _write(pdf, path, partition=True):
            pdf  = pdf.copy()
            nums = pdf.select_dtypes(
                include=[np.number]
            ).columns
            pdf[nums] = pdf[nums].fillna(0)
            if "date" in pdf.columns:
                pdf["date"] = (
                    pdf["date"].astype(str)
                )
                pdf["year"]  = pd.to_datetime(
                    pdf["date"]
                ).dt.year
                pdf["month"] = pd.to_datetime(
                    pdf["date"]
                ).dt.month
            w = (
                self.spark.createDataFrame(pdf)
                .write.format("delta")
                .mode("overwrite")
                .option("overwriteSchema","true")
            )
            if partition and "year" in pdf.columns:
                w = w.partitionBy("year","month")
            w.save(path)

        # Final ensemble predictions
        out_cols = [
            "date","ticker","regime_label",
            "pred_final",        # ML 02
            "pred_fwd_vol_21d",  # ML 03
            "alpha_blend",
            "regime_weight_soft",
            "vol_position_signal",
            "position_size",
            "signal_final",
            "signal_final_cs",   # ← use this
            "fwd_return_21d",
        ]
        out_cols = [
            c for c in out_cols
            if c in df.columns
        ]
        _write(
            df[out_cols],
            f"{self.ml_path}/ensemble_predictions"
        )
        print("  ✓ ensemble_predictions")

        # IC series
        ic_df = results["ML04 Ensemble"]["ic_df"]
        _write(
            ic_df,
            f"{self.ml_path}/ensemble_ic_series",
            partition=False
        )
        print("  ✓ ensemble_ic_series")

        # L/S series
        ls_df = results["ML04 Ensemble"]["ls_df"]
        _write(
            ls_df,
            f"{self.ml_path}/ensemble_ls_series",
            partition=False
        )
        print("  ✓ ensemble_ls_series")

        # Summary
        rows = []
        for name in [
            "ML02 LightGBM","ML04 Ensemble"
        ]:
            if name not in results:
                continue
            r    = results[name]
            ic_s = r["ic"]
            g    = r["ls_gross"]
            t    = r["ls_tc"]
            rows.append({
                "model"          : name,
                "mean_ic"        : ic_s["mean_ic"],
                "icir"           : ic_s["icir"],
                "hit_rate"       : ic_s["hit_rate"],
                "sharpe_gross"   : g["sharpe"],
                "sharpe_net"     : t["sharpe"],
                "ann_ret_gross"  : g["ann_ret"],
                "ann_ret_net"    : t["ann_ret"],
                "max_dd"         : g["max_dd"],
                "avg_turnover"   : r["avg_to"],
                "tc_bps_yr"      : r["tc_yr"],
            })
        _write(
            pd.DataFrame(rows),
            f"{self.ml_path}/ensemble_summary",
            partition=False
        )
        print("  ✓ ensemble_summary")

        # Regime IC
        r_rows = []
        for regime, s in results.get(
            "regime_ic", {}
        ).items():
            r_rows.append({
                "regime"  : regime,
                "mean_ic" : s["mean_ic"],
                "icir"    : s["icir"],
                "n_dates" : s["n_dates"],
            })
        if r_rows:
            _write(
                pd.DataFrame(r_rows),
                f"{self.ml_path}/ensemble_regime_ic",
                partition=False
            )
            print("  ✓ ensemble_regime_ic")

    # ─────────────────────────────────────────────────
    #  Run
    # ─────────────────────────────────────────────────
    def run(self):
        print("="*55)
        print("ML 04 — Regime-Weighted Ensemble")
        print("="*55)
        start = datetime.now()

        lgbm, vol, hmm, gold, HMM_OK = (
            self.load_signals()
        )
        df = self.merge_signals(
            lgbm, vol, hmm, gold, HMM_OK
        )
        df      = self.build_ensemble_signal(df)
        results = self.evaluate(df)
        self.write_results(df, results)

        elapsed = (
            datetime.now()-start
        ).seconds / 60
        print(f"\nTotal time : {elapsed:.1f} min")
        print("ML 04 COMPLETE ✓")
        return df, results

# COMMAND ----------

class MLEnsembleCharts:
    TEMPLATE = "plotly_dark"
    C = {
        "primary"  : "#2196F3",
        "secondary": "#FF5722",
        "success"  : "#4CAF50",
        "warning"  : "#FFC107",
        "purple"   : "#9C27B0",
        "teal"     : "#00BCD4",
        "ml02"     : "#FF9800",
        "ml04"     : "#4CAF50",
    }
    REGIME_C = {
        "Bull"   : "#4CAF50",
        "Bear"   : "#FF5722",
        "HighVol": "#FFC107",
    }

    def chart_ic_comparison(self, results):
        """IC series: ML02 vs ML04."""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=[
                "Daily IC — ML02 LightGBM",
                "Daily IC — ML04 Ensemble",
                "Rolling 63d IC Comparison",
            ],
            vertical_spacing=0.08,
            row_heights=[0.3, 0.3, 0.4]
        )

        for row, (name, color) in enumerate(
            [
                ("ML02 LightGBM", self.C["ml02"]),
                ("ML04 Ensemble", self.C["ml04"]),
            ],
            start=1
        ):
            if name not in results:
                continue
            ic_df = results[name]["ic_df"].copy()
            ic_df["date"] = pd.to_datetime(
                ic_df["date"]
            )
            ic_df = ic_df.sort_values("date")
            vals  = ic_df["ic"].fillna(0)
            mean  = vals.mean()

            bar_c = [
                self.C["success"] if v > 0
                else self.C["secondary"]
                for v in vals
            ]
            fig.add_trace(go.Bar(
                x=ic_df["date"], y=vals,
                marker_color=bar_c,
                name=name, showlegend=False
            ), row=row, col=1)
            fig.add_hline(
                y=float(mean), line_dash="dot",
                line_color=color,
                annotation_text=f"μ={mean:.4f}",
                row=row, col=1
            )
            fig.add_hline(
                y=0, line_dash="dash",
                line_color="white", opacity=0.3,
                row=row, col=1
            )

        # Rolling IC comparison
        for name, color in [
            ("ML02 LightGBM", self.C["ml02"]),
            ("ML04 Ensemble", self.C["ml04"]),
        ]:
            if name not in results:
                continue
            ic_df = results[name]["ic_df"].copy()
            ic_df["date"] = pd.to_datetime(
                ic_df["date"]
            )
            ic_df = ic_df.sort_values("date")
            vals  = ic_df["ic"].fillna(0)
            fig.add_trace(go.Scatter(
                x=ic_df["date"],
                y=vals.rolling(63).mean(),
                name=name, mode="lines",
                line=dict(color=color, width=2)
            ), row=3, col=1)
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=3, col=1
        )

        ml04_ic = results.get(
            "ML04 Ensemble",{}
        ).get("ic",{})
        ml02_ic = results.get(
            "ML02 LightGBM",{}
        ).get("ic",{})
        fig.update_layout(
            title=(
                f"<b>ML 04 — IC Comparison<br>"
                f"<sup>"
                f"ML02: IC={ml02_ic.get('mean_ic',0):+.4f} | "
                f"ML04: IC={ml04_ic.get('mean_ic',0):+.4f}"
                f"</sup></b>"
            ),
            template=self.TEMPLATE,
            height=800, hovermode="x unified"
        )
        for r, t in [
            (1,"ML02 IC"),(2,"ML04 IC"),
            (3,"Rolling 63d")
        ]:
            fig.update_yaxes(
                title_text=t, row=r, col=1
            )
        fig.show()

    def chart_portfolio_comparison(self, results):
        """Cumulative returns: ML02 vs ML04."""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=[
                "Cumulative Return (Net TC)",
                "Daily L/S Return",
                "Drawdown",
            ],
            vertical_spacing=0.07,
            row_heights=[0.5, 0.25, 0.25]
        )

        for name, color in [
            ("ML02 LightGBM", self.C["ml02"]),
            ("ML04 Ensemble", self.C["ml04"]),
        ]:
            if name not in results:
                continue
            ls   = results[name]["ls_df"].copy()
            ls["date"] = pd.to_datetime(ls["date"])
            ls   = ls.sort_values("date")
            cum  = (
                1 + ls["ls_ret_tc"].fillna(0)
            ).cumprod()
            dd   = (
                cum / cum.cummax() - 1
            )

            fig.add_trace(go.Scatter(
                x=ls["date"], y=cum,
                name=name, mode="lines",
                line=dict(color=color, width=2)
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=ls["date"], y=dd*100,
                name=f"{name} DD",
                mode="lines", showlegend=True,
                line=dict(
                    color=color, width=1,
                    dash="dot"
                ),
                fill="tozeroy",
                fillcolor=color.replace(
                    ")", ",0.1)"
                ).replace("rgb", "rgba")
                if "rgb" in color
                else f"rgba(100,100,100,0.1)",
            ), row=3, col=1)

        # Daily returns for ML04
        if "ML04 Ensemble" in results:
            ls   = results[
                "ML04 Ensemble"
            ]["ls_df"].copy()
            ls["date"] = pd.to_datetime(ls["date"])
            ls   = ls.sort_values("date")
            bar_c = [
                self.C["success"] if v > 0
                else self.C["secondary"]
                for v in ls["ls_ret_tc"]
            ]
            fig.add_trace(go.Bar(
                x=ls["date"],
                y=ls["ls_ret_tc"]*100,
                marker_color=bar_c,
                name="ML04 Daily", showlegend=False
            ), row=2, col=1)
            fig.add_hline(
                y=0, line_dash="dash",
                line_color="white", opacity=0.3,
                row=2, col=1
            )

        ml04 = results.get("ML04 Ensemble",{})
        ml02 = results.get("ML02 LightGBM",{})
        fig.update_layout(
            title=(
                f"<b>ML 04 — Portfolio Comparison"
                f"<br><sup>"
                f"ML02 Sharpe={ml02.get('ls_tc',{}).get('sharpe',0):.2f} | "
                f"ML04 Sharpe={ml04.get('ls_tc',{}).get('sharpe',0):.2f}"
                f"</sup></b>"
            ),
            template=self.TEMPLATE,
            height=800, hovermode="x unified"
        )
        for r, t in [
            (1,"Cum Return"),
            (2,"Daily Ret(%)"),
            (3,"DD(%)"),
        ]:
            fig.update_yaxes(
                title_text=t, row=r, col=1
            )
        fig.show()

    def chart_regime_ic(self, results):
        ri = results.get("regime_ic", {})
        if not ri:
            return

        regimes  = list(ri.keys())
        mean_ics = [ri[r]["mean_ic"] for r in regimes]
        icirs    = [ri[r]["icir"]    for r in regimes]
        n_dates  = [ri[r]["n_dates"] for r in regimes]
        colors   = [
            self.REGIME_C.get(r, "#9E9E9E")
            for r in regimes
        ]

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[
                "IC by Regime",
                "ICIR by Regime",
                "N Trading Days",
            ]
        )
        for vals, col in [
            (mean_ics,1),(icirs,2),(n_dates,3)
        ]:
            fig.add_trace(go.Bar(
                x=regimes, y=vals,
                marker_color=colors,
                text=[f"{v:.4f}" if col<3
                      else f"{v}" for v in vals],
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
                "<b>ML 04 — IC by Regime</b>"
            ),
            template=self.TEMPLATE, height=500
        )
        fig.show()

    def chart_signal_breakdown(self, df):
        """Show signal components."""
        daily = df.groupby("date").agg(
            regime_w=("regime_weight_soft","mean"),
            vol_sig  =("vol_position_signal","mean"),
            pos_size =("position_size","mean"),
            alpha    =("alpha_blend","mean"),
        ).reset_index().sort_values("date")

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=[
                "Mean Regime Weight (soft)",
                "Mean Vol Position Signal (ML03)",
                "Mean Position Size (combined)",
                "Mean Alpha Blend",
            ],
            vertical_spacing=0.06,
            row_heights=[0.25,0.25,0.25,0.25]
        )

        for row, (col, color, name) in enumerate([
            ("regime_w", self.C["warning"],
             "Regime Weight"),
            ("vol_sig",  self.C["teal"],
             "Vol Signal"),
            ("pos_size", self.C["primary"],
             "Position Size"),
            ("alpha",    self.C["success"],
             "Alpha Blend"),
        ], start=1):
            fig.add_trace(go.Scatter(
                x=daily["date"],
                y=daily[col],
                mode="lines",
                line=dict(color=color, width=1.5),
                fill="tozeroy",
                fillcolor=f"rgba(100,100,100,0.1)",
                showlegend=False
            ), row=row, col=1)
            fig.update_yaxes(
                title_text=name, row=row, col=1
            )

        fig.update_layout(
            title=(
                "<b>ML 04 — Signal Breakdown</b>"
            ),
            template=self.TEMPLATE,
            height=800, hovermode="x unified"
        )
        fig.show()

    def chart_summary_table(self, results):
        """Summary metrics table."""
        rows  = []
        cols  = [
            "Model","IC","ICIR","Hit",
            "Sharpe(Gross)","Sharpe(Net)",
            "Ann Ret(Net)","Max DD",
            "Avg TO","TC/yr"
        ]
        for name in [
            "ML02 LightGBM","ML04 Ensemble"
        ]:
            if name not in results:
                continue
            r    = results[name]
            ic_s = r["ic"]
            g    = r["ls_gross"]
            t    = r["ls_tc"]
            rows.append([
                name,
                f"{ic_s['mean_ic']:+.4f}",
                f"{ic_s['icir']:.3f}",
                f"{ic_s['hit_rate']:.1%}",
                f"{g['sharpe']:.2f}",
                f"{t['sharpe']:.2f}",
                f"{t['ann_ret']*100:.1f}%",
                f"{g['max_dd']*100:.1f}%",
                f"{r['avg_to']:.1%}",
                f"{r['tc_yr']:.0f}bps",
            ])

        fig = go.Figure(go.Table(
            header=dict(
                values=cols,
                fill_color="#1e1e2e",
                align="center",
                font=dict(color="white", size=12)
            ),
            cells=dict(
                values=list(zip(*rows))
                       if rows else [[]]*len(cols),
                fill_color=[
                    ["#263238","#1a3a1a"]
                    * len(rows)
                ],
                align="center",
                font=dict(color="white", size=11)
            )
        ))
        fig.update_layout(
            title=(
                "<b>ML 04 — Final Summary Table</b>"
            ),
            template=self.TEMPLATE,
            height=300
        )
        fig.show()

    def chart_turnover_tc(self, results):
        """Turnover and TC analysis."""
        if "ML04 Ensemble" not in results:
            return

        ls   = results[
            "ML04 Ensemble"
        ]["ls_df"].copy()
        ls["date"] = pd.to_datetime(ls["date"])
        ls   = ls.sort_values("date")

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=[
                "Daily Turnover (%)",
                "Daily TC Cost (bps)",
            ],
            vertical_spacing=0.1
        )

        fig.add_trace(go.Scatter(
            x=ls["date"],
            y=ls["turnover"]*100,
            mode="lines",
            line=dict(
                color=self.C["purple"], width=1
            ),
            fill="tozeroy",
            fillcolor="rgba(156,39,176,0.15)",
            showlegend=False
        ), row=1, col=1)
        fig.add_hline(
            y=float(ls["turnover"].mean()*100),
            line_dash="dot",
            line_color=self.C["warning"],
            annotation_text=(
                f"Avg={ls['turnover'].mean():.1%}"
            ),
            row=1, col=1
        )

        fig.add_trace(go.Scatter(
            x=ls["date"],
            y=ls["tc_cost"]*10000,
            mode="lines",
            line=dict(
                color=self.C["secondary"], width=1
            ),
            fill="tozeroy",
            fillcolor="rgba(255,87,34,0.15)",
            showlegend=False
        ), row=2, col=1)

        tc_yr = results["ML04 Ensemble"]["tc_yr"]
        fig.update_layout(
            title=(
                f"<b>ML 04 — Turnover & TC<br>"
                f"<sup>"
                f"Avg TC = {tc_yr:.0f}bps/yr"
                f"</sup></b>"
            ),
            template=self.TEMPLATE,
            height=550, hovermode="x unified"
        )
        fig.update_yaxes(
            title_text="Turnover(%)", row=1, col=1
        )
        fig.update_yaxes(
            title_text="TC(bps)", row=2, col=1
        )
        fig.show()

    def run_all(self, df, results):
        print("\n" + "="*55)
        print("ML 04 Charts")
        print("="*55)

        print("\n[1/6] IC Comparison...")
        self.chart_ic_comparison(results)

        print("[2/6] Portfolio Comparison...")
        self.chart_portfolio_comparison(results)

        print("[3/6] Regime IC...")
        self.chart_regime_ic(results)

        print("[4/6] Signal Breakdown...")
        self.chart_signal_breakdown(df)

        print("[5/6] Summary Table...")
        self.chart_summary_table(results)

        print("[6/6] Turnover & TC...")
        self.chart_turnover_tc(results)

        print("\nAll 6 charts ✓")

# COMMAND ----------

pipeline = MLEnsemble(
    spark     = spark,
    gold_path = GOLD_PATH,
    ml_path   = ML_PATH,
)

df, results = pipeline.run()

charts = MLEnsembleCharts()
charts.run_all(df, results)

print("\nML 04 COMPLETE ✓")

# COMMAND ----------

summary = spark.read.format("delta").load(
    f"{ML_PATH}/ensemble_summary"
).toPandas()

regime_ic = spark.read.format("delta").load(
    f"{ML_PATH}/ensemble_regime_ic"
).toPandas()

preds = spark.read.format("delta").load(
    f"{ML_PATH}/ensemble_predictions"
)

print("="*55)
print("ML 04 — Final Summary")
print("="*55)
print(f"\nRows      : {preds.count():,}")
print(f"Tickers   : "
      f"{preds.select('ticker').distinct().count():,}")

print(f"\nModel Comparison:")
print(summary[[
    "model","mean_ic","icir","sharpe_gross",
    "sharpe_net","ann_ret_net","max_dd"
]].to_string(index=False))

print(f"\nRegime IC:")
print(regime_ic.to_string(index=False))

print(f"\n{'='*55}")
print(f"ML Stack Complete:")
print(f"  ML 01 HMM      ✅ Sharpe sep=2.39")
print(f"  ML 02 LightGBM ✅ IC=0.093 Sharpe=7.3")
print(f"  ML 03 PatchTST ✅ Corr=0.81 IC=0.83")
print(f"  ML 04 Ensemble ✅ Regime-weighted blend")
print(f"  Backtest 01    🔲 Next → full backtest")
print(f"{'='*55}")