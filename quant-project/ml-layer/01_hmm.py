# Databricks notebook source
# MAGIC %pip install hmmlearn scikit-learn plotly scipy pandas numpy --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import *
from datetime import datetime
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.shuffle.partitions", "200")
spark.conf.set("spark.sql.ansi.enabled", "false")

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
GOLD_PATH   = f"{BASE_PATH}/gold/delta"
ML_PATH     = f"{BASE_PATH}/ml/delta"

print("Config loaded ✓")
print(f"ML path : {ML_PATH}")

# COMMAND ----------

class MLHMMRegimes:
    """
    ML 01 — HMM Regimes (Retrain on full dataset).

    Improves on EDA 04 HMM by:
      1. Using full feature set (macro + market)
      2. Cross-validating n_states (2/3/4)
      3. Stability analysis (multiple random seeds)
      4. Saving model for production inference
      5. Generating regime labels for ALL dates
      6. Regime-conditional return statistics

    Features used:
      - Cross-sectional market return (proxy for market)
      - Cross-sectional vol (market dispersion)
      - Market breadth (% stocks up)
      - VIX level
      - HY credit spread
      - Yield curve spread

    Output:
      ml/delta/hmm_regime_labels   → regime per date
      ml/delta/hmm_model_stats     → model diagnostics
    """

    N_STATES    = 3
    N_ITER      = 500
    N_SEEDS     = 5       # stability check
    RANDOM_SEED = 42

    REGIME_COLORS = {
        "Bull"   : "#4CAF50",
        "Bear"   : "#FF5722",
        "HighVol": "#FFC107",
        "Unknown": "#9E9E9E",
    }

    def __init__(self, spark, silver_path,
                 eda_path, gold_path, ml_path):
        self.spark       = spark
        self.silver_path = silver_path
        self.eda_path    = eda_path
        self.gold_path   = gold_path
        self.ml_path     = ml_path
        self.best_model  = None
        self.regime_map  = {}
        self.scaler      = StandardScaler()
        print("MLHMMRegimes ✓")
        print(f"  N states : {self.N_STATES}")
        print(f"  N iter   : {self.N_ITER}")
        print(f"  N seeds  : {self.N_SEEDS}")

    # ------------------------------------------------------------------ #
    #  Step 1 — Load and build features
    # ------------------------------------------------------------------ #
    def load_features(self) -> pd.DataFrame:
        print("\nStep 1: Loading features...")
        start = datetime.now()

        # Market aggregates from OHLCV
        ohlcv = self.spark.read.format("delta").load(
            f"{self.silver_path}/ohlcv"
        )
        market = ohlcv.groupBy("date").agg(
            F.mean("return_1d").alias("cs_return"),
            F.stddev("return_1d").alias("cs_vol"),
            F.mean("vol_21d").alias("avg_vol_21d"),
            F.mean("vol_63d").alias("avg_vol_63d"),
            F.sum(
                F.when(F.col("return_1d") > 0, 1)
                .otherwise(0)
            ).alias("up_count"),
            F.count("return_1d").alias("n_stocks"),
        ).withColumn(
            "breadth",
            F.when(
                F.col("n_stocks") > 0,
                F.col("up_count").cast("double") /
                F.col("n_stocks").cast("double")
            ).otherwise(F.lit(None).cast("double"))
        ).withColumn(
            "return_dispersion",
            F.col("cs_vol")
        )

        # Macro features
        macro = self.spark.read.format("delta").load(
            f"{self.silver_path}/macro"
        ).select(
            "date","VIXCLS","yield_spread_10y2y",
            "BAMLH0A0HYM2","FEDFUNDS",
            "vix_change_1d"
        )

        # Join
        features = market.join(
            macro, on="date", how="left"
        ).orderBy("date")

        pdf = features.toPandas()
        pdf["date"] = pd.to_datetime(pdf["date"])
        pdf = pdf.sort_values("date").reset_index(
            drop=True
        )

        elapsed = (datetime.now() - start).seconds
        print(f"  Rows     : {len(pdf):,}")
        print(f"  Date range: "
              f"{pdf['date'].min().date()} → "
              f"{pdf['date'].max().date()}")
        print(f"  Elapsed  : {elapsed}s")
        return pdf

    # ------------------------------------------------------------------ #
    #  Step 2 — Prepare feature matrix
    # ------------------------------------------------------------------ #
    def prepare_features(self,
                          pdf: pd.DataFrame) -> tuple:
        print("\nStep 2: Preparing feature matrix...")

        candidate_cols = [
            "cs_return",
            "cs_vol",
            "avg_vol_21d",
            "breadth",
            "VIXCLS",
            "yield_spread_10y2y",
            "BAMLH0A0HYM2",
        ]
        available = [
            c for c in candidate_cols
            if c in pdf.columns
        ]
        print(f"  Features : {available}")

        feat_df = pdf[["date"] + available].copy()

        # Forward fill then median
        feat_df[available] = feat_df[
            available
        ].fillna(method="ffill", limit=5)
        for col in available:
            med = feat_df[col].median()
            feat_df[col] = feat_df[col].fillna(
                0.0 if np.isnan(med) else med
            )

        # Scale
        feat_matrix = feat_df[available].values.astype(
            float
        )
        feat_scaled = self.scaler.fit_transform(
            feat_matrix
        )

        # Remove NaN rows
        valid_mask  = ~np.isnan(feat_scaled).any(axis=1)
        feat_scaled = feat_scaled[valid_mask]
        dates_valid = feat_df["date"].values[valid_mask]

        print(f"  Valid rows : {len(feat_scaled):,}")
        print(f"  Shape      : {feat_scaled.shape}")
        return feat_scaled, dates_valid, available

    # ------------------------------------------------------------------ #
    #  Step 3 — Cross-validate n_states
    # ------------------------------------------------------------------ #
    def cross_validate_states(self,
                               features: np.ndarray
                               ) -> dict:
        print("\nStep 3: Cross-validating n_states...")

        results = {}
        for n in [2, 3, 4]:
            scores = []
            for seed in range(3):
                try:
                    m = GaussianHMM(
                        n_components    = n,
                        covariance_type = "full",
                        n_iter          = 200,
                        random_state    = seed * 42
                    )
                    m.fit(features)
                    scores.append(m.score(features))
                except Exception:
                    pass

            if scores:
                results[n] = {
                    "mean_score" : np.mean(scores),
                    "std_score"  : np.std(scores),
                    "n_fits"     : len(scores),
                }
                print(f"  n={n}: "
                      f"log-lik={np.mean(scores):.1f} "
                      f"(±{np.std(scores):.1f})")

        # Best = highest log-likelihood
        best_n = max(
            results,
            key=lambda k: results[k]["mean_score"]
        )
        print(f"\n  Best n_states : {best_n}")

        # Use configured N_STATES (3) for interpretability
        print(f"  Using N_STATES={self.N_STATES} "
              f"(configured for interpretability)")
        return results

    # ------------------------------------------------------------------ #
    #  Step 4 — Train stable HMM
    # ------------------------------------------------------------------ #
    def train_stable_hmm(self,
                          features: np.ndarray
                          ) -> GaussianHMM:
        print(f"\nStep 4: Training stable HMM "
              f"({self.N_SEEDS} seeds)...")

        best_model  = None
        best_score  = -np.inf
        all_scores  = []

        for seed in range(self.N_SEEDS):
            try:
                model = GaussianHMM(
                    n_components    = self.N_STATES,
                    covariance_type = "full",
                    n_iter          = self.N_ITER,
                    random_state    = seed * 100 + 42,
                    verbose         = False
                )
                model.fit(features)
                score = model.score(features)
                all_scores.append(score)

                print(f"  Seed {seed}: "
                      f"log-lik={score:.2f} "
                      f"converged="
                      f"{model.monitor_.converged}")

                if score > best_score:
                    best_score = score
                    best_model = model

            except Exception as e:
                print(f"  Seed {seed} failed: {e}")

        print(f"\n  Best score  : {best_score:.2f}")
        print(f"  Score std   : {np.std(all_scores):.2f}")
        print(f"  Stability   : "
              f"{'HIGH' if np.std(all_scores) < 50 else 'LOW'}")

        self.best_model = best_model
        return best_model

    # ------------------------------------------------------------------ #
    #  Step 5 — Decode and label regimes
    # ------------------------------------------------------------------ #
    def decode_regimes(self, model: GaussianHMM,
                       features: np.ndarray,
                       dates: np.ndarray,
                       pdf: pd.DataFrame
                       ) -> pd.DataFrame:
        print("\nStep 5: Decoding regimes...")

        hidden_states = model.predict(features)
        state_probs   = model.predict_proba(features)
        means         = model.means_

        # Assign labels by volatility
        ret_idx   = 0
        vol_idx   = min(1, means.shape[1] - 1)
        state_vol = means[:, vol_idx]
        state_ret = means[:, ret_idx]

        sorted_by_vol = np.argsort(state_vol)
        bull_state    = int(sorted_by_vol[0])
        highvol_state = int(sorted_by_vol[-1])
        mid_state     = int(sorted_by_vol[1])

        bear_state = mid_state \
                     if state_ret[mid_state] < 0 \
                     else highvol_state

        if bear_state == highvol_state:
            highvol_state = mid_state

        self.regime_map = {
            bull_state   : "Bull",
            bear_state   : "Bear",
            highvol_state: "HighVol"
        }
        print(f"  Regime map : {self.regime_map}")

        regime_df = pd.DataFrame({
            "date"         : pd.to_datetime(dates),
            "hmm_state"    : hidden_states,
            "regime_label" : [
                self.regime_map.get(int(s), "Unknown")
                for s in hidden_states
            ],
            "prob_bull"    : state_probs[:, bull_state],
            "prob_bear"    : state_probs[:, bear_state],
            "prob_highvol" : state_probs[:, highvol_state],
        })

        # Merge market features
        result = pdf.merge(regime_df, on="date", how="left")
        result["regime_label"] = result[
            "regime_label"
        ].fillna("Unknown")

        # Position weights
        result["position_size_weight"] = result[
            "regime_label"
        ].map({
            "Bull"   : 1.0,
            "HighVol": 0.6,
            "Bear"   : 0.3,
            "Unknown": 0.5
        }).fillna(0.5)

        # Regime change flag
        result = result.sort_values("date")
        result["regime_changed"] = (
            result["regime_label"] !=
            result["regime_label"].shift(1)
        ).astype(int)

        # Regime numeric
        result["regime_numeric"] = result[
            "regime_label"
        ].map({
            "Bull": 0, "Bear": 1,
            "HighVol": 2, "Unknown": 3
        }).fillna(3)

        dist = result["regime_label"].value_counts()
        print(f"\n  Regime distribution:")
        for r, n in dist.items():
            pct = n / len(result) * 100
            print(f"    {r:8}: {n:,} ({pct:.1f}%)")

        return result

    # ------------------------------------------------------------------ #
    #  Step 6 — Regime performance stats
    # ------------------------------------------------------------------ #
    def compute_regime_stats(self,
                              regime_df: pd.DataFrame
                              ) -> pd.DataFrame:
        print("\nStep 6: Regime performance stats...")

        rows = []
        for regime in ["Bull","Bear","HighVol"]:
            mask = regime_df["regime_label"] == regime
            data = regime_df[mask]
            if len(data) == 0:
                continue

            ret = data["cs_return"].dropna()
            if len(ret) == 0:
                continue

            ann_ret = ret.mean() * 252
            ann_vol = ret.std() * np.sqrt(252)
            sharpe  = ann_ret / ann_vol \
                      if ann_vol > 0 else 0

            # Duration
            dur_df    = self._compute_durations(
                regime_df, regime
            )
            avg_dur   = dur_df["duration"].mean() \
                        if len(dur_df) > 0 else 0

            # Transition probability
            trans_prob = self.best_model.transmat_[
                self._get_state(regime),
                self._get_state(regime)
            ] if self.best_model else 0

            vix_vals = data["VIXCLS"].dropna() \
                       if "VIXCLS" in data.columns \
                       else pd.Series()

            rows.append({
                "regime"         : regime,
                "n_days"         : len(data),
                "pct_days"       : len(data) / len(regime_df),
                "ann_return"     : float(ann_ret),
                "ann_vol"        : float(ann_vol),
                "sharpe"         : float(sharpe),
                "hit_rate"       : float((ret > 0).mean()),
                "avg_duration"   : float(avg_dur),
                "stay_prob"      : float(trans_prob),
                "mean_vix"       : float(vix_vals.mean())
                                   if len(vix_vals) > 0
                                   else None,
                "skewness"       : float(stats.skew(ret))
                                   if len(ret) > 3 else None,
            })

        stats_df = pd.DataFrame(rows)
        print(f"\n  Regime stats:")
        print(stats_df[[
            "regime","n_days","ann_return",
            "ann_vol","sharpe","avg_duration"
        ]].to_string(index=False))
        return stats_df

    def _compute_durations(self, df, regime):
        rows, curr, length = [], None, 0
        for r in df["regime_label"].dropna():
            if r == regime:
                if curr == regime:
                    length += 1
                else:
                    curr, length = regime, 1
            else:
                if curr == regime and length > 0:
                    rows.append({"duration": length})
                curr, length = r, 0
        if curr == regime and length > 0:
            rows.append({"duration": length})
        return pd.DataFrame(rows)

    def _get_state(self, regime_label: str) -> int:
        for state, label in self.regime_map.items():
            if label == regime_label:
                return state
        return 0

    # ------------------------------------------------------------------ #
    #  Step 7 — Transition matrix
    # ------------------------------------------------------------------ #
    def get_transition_matrix(self) -> pd.DataFrame:
        if self.best_model is None:
            return pd.DataFrame()
        labels = [
            self.regime_map.get(i, f"State{i}")
            for i in range(self.N_STATES)
        ]
        trans_df = pd.DataFrame(
            self.best_model.transmat_,
            index=labels, columns=labels
        )
        print(f"\n  Transition matrix:")
        print(trans_df.round(3).to_string())
        return trans_df

    # ------------------------------------------------------------------ #
    #  Write
    # ------------------------------------------------------------------ #
    def write_results(self, regime_df, stats_df,
                      trans_df) -> None:
        print("\nWriting ML results to Delta...")

        # Regime labels
        regime_out = regime_df[[
            "date","hmm_state","regime_label",
            "prob_bull","prob_bear","prob_highvol",
            "position_size_weight","regime_changed",
            "regime_numeric","cs_return","cs_vol",
            "breadth"
        ]].copy()
        regime_out["date"] = regime_out["date"].astype(str)
        regime_out["year"] = pd.to_datetime(
            regime_out["date"]
        ).dt.year
        regime_out["month"] = pd.to_datetime(
            regime_out["date"]
        ).dt.month

        self.spark.createDataFrame(regime_out).write \
            .format("delta").mode("overwrite") \
            .option("overwriteSchema","true") \
            .partitionBy("year","month") \
            .save(f"{self.ml_path}/hmm_regime_labels")
        print("  ✓ hmm_regime_labels")

        # Stats
        self.spark.createDataFrame(stats_df).write \
            .format("delta").mode("overwrite") \
            .option("overwriteSchema","true") \
            .save(f"{self.ml_path}/hmm_regime_stats")
        print("  ✓ hmm_regime_stats")

        # Transition matrix
        trans_out = trans_df.reset_index().rename(
            columns={"index": "from_regime"}
        )
        self.spark.createDataFrame(trans_out).write \
            .format("delta").mode("overwrite") \
            .option("overwriteSchema","true") \
            .save(f"{self.ml_path}/hmm_transition_matrix")
        print("  ✓ hmm_transition_matrix")

        # Save model params as Delta
        if self.best_model:
            model_params = []
            for i in range(self.N_STATES):
                label = self.regime_map.get(i,"Unknown")
                model_params.append({
                    "state"        : i,
                    "regime_label" : label,
                    "log_likelihood": float(
                        self.best_model.score(
                            self.scaler.transform(
                                np.zeros((1, self.best_model.n_features))
                            )
                        )
                    ) if hasattr(
                        self.best_model, "n_features"
                    ) else 0.0,
                    "stay_prob"    : float(
                        self.best_model.transmat_[i, i]
                    ),
                    "mean_return"  : float(
                        self.best_model.means_[i, 0]
                    ),
                    "mean_vol"     : float(
                        self.best_model.means_[i, min(
                            1,
                            self.best_model.means_.shape[1]-1
                        )]
                    ),
                })

            self.spark.createDataFrame(
                pd.DataFrame(model_params)
            ).write \
                .format("delta").mode("overwrite") \
                .option("overwriteSchema","true") \
                .save(f"{self.ml_path}/hmm_model_params")
            print("  ✓ hmm_model_params")

    # ------------------------------------------------------------------ #
    #  Validate
    # ------------------------------------------------------------------ #
    def validate(self, regime_df, stats_df) -> None:
        print("\n" + "="*55)
        print("ML 01 FINDINGS — HMM Regimes")
        print("="*55)

        print(f"\n  Total days     : {len(regime_df):,}")
        print(f"  Date range     : "
              f"{regime_df['date'].min().date()} → "
              f"{regime_df['date'].max().date()}")

        dist = regime_df["regime_label"].value_counts()
        print(f"\n  Regime distribution:")
        for r, n in dist.items():
            pct = n / len(regime_df) * 100
            print(f"    {r:8}: {n:,} ({pct:.1f}%)")

        print(f"\n  Performance by regime:")
        print(stats_df[[
            "regime","ann_return","ann_vol",
            "sharpe","hit_rate","avg_duration"
        ]].to_string(index=False))

        print(f"\n  KEY DECISIONS:")
        bull = stats_df[stats_df["regime"] == "Bull"]
        bear = stats_df[stats_df["regime"] == "Bear"]
        hv   = stats_df[stats_df["regime"] == "HighVol"]
        if len(bull) > 0:
            print(f"  → Bull regime : "
                  f"ann={bull['ann_return'].values[0]*100:.1f}% "
                  f"Sharpe={bull['sharpe'].values[0]:.2f}")
        if len(bear) > 0:
            print(f"  → Bear regime : "
                  f"ann={bear['ann_return'].values[0]*100:.1f}% "
                  f"Sharpe={bear['sharpe'].values[0]:.2f}")
        if len(hv) > 0:
            print(f"  → HighVol     : "
                  f"ann={hv['ann_return'].values[0]*100:.1f}% "
                  f"Sharpe={hv['sharpe'].values[0]:.2f}")
        print(f"\n  → Position weights: "
              f"Bull=1.0, HighVol=0.6, Bear=0.3 ✓")
        print(f"  → Use ML regime labels in ensemble ✓")

    # ------------------------------------------------------------------ #
    #  Run
    # ------------------------------------------------------------------ #
    def run(self):
        print("="*55)
        print("ML 01 — HMM Regimes (Full Retrain)")
        print("="*55)
        start = datetime.now()

        pdf = self.load_features()
        features, dates, feat_cols = \
            self.prepare_features(pdf)
        cv_results = self.cross_validate_states(features)
        model      = self.train_stable_hmm(features)
        regime_df  = self.decode_regimes(
            model, features, dates, pdf
        )
        stats_df   = self.compute_regime_stats(regime_df)
        trans_df   = self.get_transition_matrix()

        self.write_results(regime_df, stats_df, trans_df)
        self.validate(regime_df, stats_df)

        elapsed = (
            datetime.now() - start
        ).seconds / 60
        print(f"\nTotal time : {elapsed:.1f} minutes")
        print("ML 01 — HMM Regimes COMPLETE ✓")
        return regime_df, stats_df, trans_df, cv_results

# COMMAND ----------

class MLHMMCharts:
    TEMPLATE = "plotly_dark"
    COLORS   = {
        "primary"  : "#2196F3",
        "secondary": "#FF5722",
        "success"  : "#4CAF50",
        "warning"  : "#FFC107",
        "purple"   : "#9C27B0",
        "teal"     : "#00BCD4",
    }
    REGIME_COLORS = {
        "Bull"   : "#4CAF50",
        "Bear"   : "#FF5722",
        "HighVol": "#FFC107",
        "Unknown": "#9E9E9E",
    }

    def chart_regime_timeline(self,
                               regime_df: pd.DataFrame
                               ) -> None:
        df = regime_df.dropna(
            subset=["regime_label","cs_return"]
        ).sort_values("date")

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=[
                "HMM Regime (ML Retrain)",
                "Market Return (CS Mean)",
                "Regime Probabilities"
            ],
            vertical_spacing=0.08,
            row_heights=[0.2, 0.45, 0.35]
        )

        for regime, color in self.REGIME_COLORS.items():
            mask = df["regime_label"] == regime
            if not mask.any():
                continue
            fig.add_trace(go.Scatter(
                x=df[mask]["date"],
                y=[regime] * mask.sum(),
                mode="markers",
                marker=dict(
                    color=color, size=4,
                    symbol="square"
                ),
                name=regime
            ), row=1, col=1)

        for regime, color in self.REGIME_COLORS.items():
            mask = df["regime_label"] == regime
            if not mask.any():
                continue
            fig.add_trace(go.Bar(
                x=df[mask]["date"],
                y=df[mask]["cs_return"] * 100,
                marker_color=color,
                opacity=0.6,
                showlegend=False
            ), row=2, col=1)

        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=2, col=1
        )

        for col, label, color in [
            ("prob_bull",    "P(Bull)",
             self.REGIME_COLORS["Bull"]),
            ("prob_bear",    "P(Bear)",
             self.REGIME_COLORS["Bear"]),
            ("prob_highvol", "P(HighVol)",
             self.REGIME_COLORS["HighVol"]),
        ]:
            if col not in df.columns:
                continue
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[col],
                name=label, mode="lines",
                line=dict(color=color, width=1),
                showlegend=False
            ), row=3, col=1)

        fig.update_layout(
            title="<b>ML 01 — HMM Regime Timeline "
                  "(Full Retrain)</b>",
            template=self.TEMPLATE,
            height=800,
            hovermode="x unified",
            barmode="overlay"
        )
        fig.update_yaxes(title_text="Regime", row=1,col=1)
        fig.update_yaxes(title_text="Return(%)",row=2,col=1)
        fig.update_yaxes(
            title_text="Prob", range=[0,1], row=3,col=1
        )
        fig.show()

    def chart_regime_performance(self,
                                  stats_df: pd.DataFrame
                                  ) -> None:
        colors = [
            self.REGIME_COLORS.get(r,"#9E9E9E")
            for r in stats_df["regime"]
        ]

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                "Ann Return","Ann Volatility",
                "Sharpe Ratio","Hit Rate",
                "Avg Duration (days)","Stay Probability"
            ]
        )

        metrics = [
            ("ann_return",  1, 1, True),
            ("ann_vol",     1, 2, True),
            ("sharpe",      1, 3, False),
            ("hit_rate",    2, 1, True),
            ("avg_duration",2, 2, False),
            ("stay_prob",   2, 3, True),
        ]

        for col, row, c, pct in metrics:
            if col not in stats_df.columns:
                continue
            vals = stats_df[col] * (100 if pct else 1)
            fig.add_trace(go.Bar(
                x=stats_df["regime"],
                y=vals,
                marker_color=colors,
                text=vals.apply(
                    lambda x: f"{x:.1f}%"
                    if pct else f"{x:.2f}"
                ),
                textposition="outside",
                showlegend=False
            ), row=row, col=c)

        fig.update_layout(
            title="<b>ML 01 — Regime Performance "
                  "Summary</b>",
            template=self.TEMPLATE,
            height=700
        )
        fig.show()

    def chart_transition_matrix(self,
                                 trans_df: pd.DataFrame
                                 ) -> None:
        if len(trans_df) == 0:
            return

        regimes = list(trans_df.index)
        values  = trans_df.values

        fig = go.Figure(go.Heatmap(
            z=values,
            x=regimes,
            y=regimes,
            colorscale="Blues",
            text=np.round(values, 3),
            texttemplate="%{text}",
            textfont=dict(size=14),
            colorbar=dict(title="Prob"),
            zmin=0, zmax=1
        ))

        fig.update_layout(
            title="<b>ML 01 — Regime Transition "
                  "Matrix</b>",
            template=self.TEMPLATE,
            height=500,
            xaxis_title="To Regime",
            yaxis_title="From Regime"
        )
        fig.show()

    def chart_regime_distributions(self,
                                    regime_df: pd.DataFrame
                                    ) -> None:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Return Distribution by Regime",
                "Volatility Distribution by Regime"
            ]
        )

        for regime, color in self.REGIME_COLORS.items():
            data = regime_df[
                regime_df["regime_label"] == regime
            ]
            if len(data) == 0:
                continue

            ret = data["cs_return"].dropna() * 100
            fig.add_trace(go.Violin(
                y=ret, x=[regime] * len(ret),
                name=regime, fillcolor=color,
                line_color="white", opacity=0.7,
                box_visible=True,
                meanline_visible=True
            ), row=1, col=1)

            if "cs_vol" in data.columns:
                vol = data["cs_vol"].dropna() * 100
                fig.add_trace(go.Violin(
                    y=vol, x=[regime] * len(vol),
                    name=regime, fillcolor=color,
                    line_color="white", opacity=0.7,
                    box_visible=True,
                    meanline_visible=True,
                    showlegend=False
                ), row=1, col=2)

        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=1, col=1
        )

        fig.update_layout(
            title="<b>ML 01 — Regime Return & "
                  "Vol Distributions</b>",
            template=self.TEMPLATE,
            height=550,
            violinmode="group"
        )
        fig.update_yaxes(
            title_text="Return(%)", row=1, col=1
        )
        fig.update_yaxes(
            title_text="CS Vol(%)", row=1, col=2
        )
        fig.show()

    def chart_cumulative_returns(self,
                                  regime_df: pd.DataFrame
                                  ) -> None:
        df = regime_df.dropna(
            subset=["cs_return","regime_label"]
        ).sort_values("date").copy()

        df["cum_return"] = (
            1 + df["cs_return"]
        ).cumprod()

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=[
                "Cumulative Return (colored by regime)",
                "Regime-Conditional Cumulative Returns"
            ],
            vertical_spacing=0.1
        )

        for regime, color in self.REGIME_COLORS.items():
            mask = df["regime_label"] == regime
            if not mask.any():
                continue
            fig.add_trace(go.Scatter(
                x=df[mask]["date"],
                y=df[mask]["cum_return"],
                mode="markers", name=regime,
                marker=dict(
                    color=color, size=3, opacity=0.5
                )
            ), row=1, col=1)

            cond = df[mask]["cs_return"]
            cr   = (1 + cond).cumprod()
            fig.add_trace(go.Scatter(
                x=df[mask]["date"], y=cr,
                mode="lines",
                line=dict(color=color, width=2),
                name=f"{regime} only",
                showlegend=False
            ), row=2, col=1)

        fig.update_layout(
            title="<b>ML 01 — Cumulative Returns "
                  "by Regime</b>",
            template=self.TEMPLATE,
            height=700,
            hovermode="x unified"
        )
        fig.update_yaxes(
            title_text="Cum Return", row=1, col=1
        )
        fig.update_yaxes(
            title_text="Cum Return", row=2, col=1
        )
        fig.show()

    def chart_cv_results(self,
                          cv_results: dict) -> None:
        if not cv_results:
            return

        n_states = list(cv_results.keys())
        scores   = [
            cv_results[n]["mean_score"]
            for n in n_states
        ]
        stds     = [
            cv_results[n]["std_score"]
            for n in n_states
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"n={n}" for n in n_states],
            y=scores,
            error_y=dict(type="data", array=stds),
            marker_color=self.COLORS["primary"],
            text=[f"{s:.0f}" for s in scores],
            textposition="outside"
        ))

        fig.update_layout(
            title="<b>ML 01 — HMM Cross-Validation: "
                  "Log-Likelihood by n_states</b>",
            template=self.TEMPLATE,
            height=450,
            xaxis_title="Number of States",
            yaxis_title="Log-Likelihood"
        )
        fig.show()

    def chart_breadth_regime(self,
                              regime_df: pd.DataFrame
                              ) -> None:
        if "breadth" not in regime_df.columns:
            return

        df = regime_df.dropna(
            subset=["breadth","regime_label",
                    "cs_return"]
        )

        color_map = {
            r: c for r, c in self.REGIME_COLORS.items()
        }
        fig = px.scatter(
            df,
            x="breadth",
            y="cs_return",
            color="regime_label",
            color_discrete_map=color_map,
            opacity=0.4,
            trendline="lowess",
            template=self.TEMPLATE,
            title="<b>ML 01 — Market Breadth vs "
                  "Return by Regime</b>",
            labels={
                "breadth"      : "Market Breadth",
                "cs_return"    : "CS Mean Return",
                "regime_label" : "Regime"
            }
        )
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3
        )
        fig.add_vline(
            x=0.5, line_dash="dash",
            line_color="yellow", opacity=0.5,
            annotation_text="50% breadth"
        )
        fig.update_layout(height=550)
        fig.show()

    def run_all(self, regime_df, stats_df,
                trans_df, cv_results) -> None:
        print("\n" + "="*55)
        print("Generating ML 01 Charts...")
        print("="*55)

        print("\n[1/7] Regime Timeline...")
        self.chart_regime_timeline(regime_df)

        print("[2/7] Regime Performance...")
        self.chart_regime_performance(stats_df)

        print("[3/7] Transition Matrix...")
        self.chart_transition_matrix(trans_df)

        print("[4/7] Return Distributions...")
        self.chart_regime_distributions(regime_df)

        print("[5/7] Cumulative Returns...")
        self.chart_cumulative_returns(regime_df)

        print("[6/7] CV Results...")
        self.chart_cv_results(cv_results)

        print("[7/7] Breadth vs Return...")
        self.chart_breadth_regime(regime_df)

        print("\nAll 7 charts ✓")

# COMMAND ----------

pipeline = MLHMMRegimes(
    spark       = spark,
    silver_path = SILVER_PATH,
    eda_path    = EDA_PATH,
    gold_path   = GOLD_PATH,
    ml_path     = ML_PATH
)

regime_df, stats_df, trans_df, \
cv_results = pipeline.run()

charts = MLHMMCharts()
charts.run_all(
    regime_df  = regime_df,
    stats_df   = stats_df,
    trans_df   = trans_df,
    cv_results = cv_results
)

print("\nML 01 COMPLETE ✓")

# COMMAND ----------

labels = spark.read.format("delta").load(
    f"{ML_PATH}/hmm_regime_labels"
).toPandas()

stats = spark.read.format("delta").load(
    f"{ML_PATH}/hmm_regime_stats"
).toPandas()

trans = spark.read.format("delta").load(
    f"{ML_PATH}/hmm_transition_matrix"
).toPandas()

print("="*55)
print("ML 01 — HMM Regimes Summary")
print("="*55)
print(f"Total days : {len(labels):,}")
print(f"\nRegime distribution:")
print(labels["regime_label"].value_counts().to_string())
print(f"\nPerformance:")
print(stats[[
    "regime","ann_return","ann_vol",
    "sharpe","hit_rate","avg_duration"
]].to_string(index=False))
print(f"\nTransition matrix:")
print(trans.set_index("from_regime").round(3).to_string())
print(f"\nPosition weights:")
print(labels.groupby("regime_label")[
    "position_size_weight"
].mean().to_string())

# COMMAND ----------

# Recompute features for evaluation
pdf_eval = pipeline.load_features()
features, dates, feat_cols = pipeline.prepare_features(pdf_eval)

# Now run the eval
import numpy as np
from scipy import stats

model    = pipeline.best_model
n        = len(features)
log_lik  = model.score(features)
n_states = model.n_components
n_feats  = model.n_features
n_params = (
    n_states * n_states +
    n_states * n_feats +
    n_states * n_feats**2
)
aic = 2 * n_params - 2 * log_lik
bic = n_params * np.log(n) - 2 * log_lik

print("="*50)
print("HMM Evaluation Metrics")
print("="*50)
print(f"\n1. Model fit:")
print(f"   Log-likelihood : {log_lik:.2f}")
print(f"   AIC            : {aic:.2f}")
print(f"   BIC            : {bic:.2f}")
print(f"   n_params       : {n_params:,}")

# Regime confidence
state_probs = model.predict_proba(features)
avg_conf    = state_probs.max(axis=1).mean()
avg_entropy = -(
    state_probs * np.log(state_probs + 1e-10)
).sum(axis=1).mean()

print(f"\n2. Regime separation:")
print(f"   Avg confidence : {avg_conf:.3f} "
      f"{'✅ GOOD' if avg_conf > 0.85 else '⚠️ LOW'}")
print(f"   Avg entropy    : {avg_entropy:.3f} "
      f"{'✅ GOOD' if avg_entropy < 0.5 else '⚠️ HIGH'}")

# Economic validation
bull_s = stats_df[stats_df["regime"]=="Bull"]["sharpe"].values[0]
bear_s = stats_df[stats_df["regime"]=="Bear"]["sharpe"].values[0]
sep    = bull_s - bear_s

print(f"\n3. Economic validation:")
print(f"   Sharpe sep     : {sep:.2f} "
      f"{'✅ EXCELLENT' if sep > 2 else '✅ GOOD' if sep > 1 else '⚠️ WEAK'}")
print(f"   Bull Sharpe    : {bull_s:.2f}")
print(f"   Bear Sharpe    : {bear_s:.2f}")

# IC of regime vs forward return
r_df = regime_df.dropna(
    subset=["regime_label","cs_return"]
).copy()
r_df["regime_num"] = r_df["regime_label"].map(
    {"Bull":1,"HighVol":0,"Bear":-1}
)
r_df["fwd_return"] = r_df["cs_return"].shift(-1)
valid = r_df.dropna(subset=["regime_num","fwd_return"])
ic = float(np.corrcoef(
    stats.rankdata(valid["regime_num"]),
    stats.rankdata(valid["fwd_return"])
)[0,1])

print(f"\n4. Predictive power:")
print(f"   Regime IC (1d) : {ic:.4f} "
      f"{'✅ SIGNIFICANT' if abs(ic) > 0.05 else '⚠️ WEAK'}")

# Stability
print(f"\n5. Stability:")
for regime in ["Bull","Bear","HighVol"]:
    if regime in trans_df.index and \
       regime in trans_df.columns:
        val = trans_df.loc[regime, regime]
        print(f"   {regime}→{regime:8}: "
              f"{val:.3f} "
              f"{'✅' if val > 0.90 else '⚠️'}")

# Overall score
n_pass = sum([
    avg_conf > 0.85,
    avg_entropy < 0.5,
    sep > 1.0,
    abs(ic) > 0.03,
    trans_df.loc["Bull","Bull"] > 0.95
    if "Bull" in trans_df.index else False,
])
print(f"\n{'='*50}")
print(f"Overall HMM Score : {n_pass}/5")
print(
    f"{'✅ PRODUCTION READY' if n_pass >= 4 else '⚠️ NEEDS TUNING'}"
)