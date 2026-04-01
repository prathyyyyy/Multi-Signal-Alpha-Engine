# Databricks notebook source
# MAGIC %pip install hmmlearn plotly scipy pandas numpy --quiet

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
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.shuffle.partitions", "200")
spark.conf.set("spark.sql.ansi.enabled", "false")  # allow div/0 → NULL

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

# COMMAND ----------

class EDARegimeAnalysis:
    """
    EDA 04 — Regime Analysis (Optimized).
    3-state Gaussian HMM on market features.
    States: Bull / Bear / HighVol
    Expected time: 3-5 minutes
    """

    N_STATES    = 3
    N_ITER      = 200
    RANDOM_SEED = 42

    REGIME_COLORS = {
        "Bull"   : "#4CAF50",
        "Bear"   : "#FF5722",
        "HighVol": "#FFC107",
        "Unknown": "#9E9E9E"
    }

    def __init__(self, spark, silver_path, eda_path):
        self.spark       = spark
        self.silver_path = silver_path
        self.eda_path    = f"{eda_path}/regime_analysis"
        self.hmm_model   = None
        self.regime_map  = {}
        print("EDARegimeAnalysis ✓")
        print(f"  States : {self.N_STATES} "
              f"(Bull / Bear / HighVol)")
        print(f"  N iter : {self.N_ITER}")

    # ------------------------------------------------------------------ #
    #  Step 1 — Load market features
    # ------------------------------------------------------------------ #
    def load_market_features(self) -> pd.DataFrame:
        print("\nStep 1: Loading market features...")
        start = datetime.now()

        # Disable ANSI to prevent div/0 errors
        self.spark.conf.set(
            "spark.sql.ansi.enabled", "false"
        )

        # Load OHLCV silver
        ohlcv = self.spark.read.format("delta").load(
            f"{self.silver_path}/ohlcv"
        )

        # Cross-sectional market aggregates
        cs_stats = ohlcv.groupBy("date").agg(
            F.mean("return_1d").alias("return_1d"),
            F.mean("log_return_1d").alias("log_return_1d"),
            F.mean("vol_21d").alias("vol_21d"),
            F.mean("vol_63d").alias("vol_63d"),
            F.stddev("return_1d").alias("cs_std_return"),
            F.mean("dollar_volume").alias("dollar_volume"),
            F.count("return_1d").alias("n_stocks"),
            F.sum(
                F.when(F.col("return_1d") > 0, 1)
                .otherwise(0)
            ).alias("up_count"),
            F.sum(
                F.when(F.col("return_1d") < 0, 1)
                .otherwise(0)
            ).alias("down_count"),
            F.percentile_approx(
                "return_1d", 0.1
            ).alias("cs_p10"),
            F.percentile_approx(
                "return_1d", 0.9
            ).alias("cs_p90"),
        ).withColumn(
            "breadth",
            F.when(
                F.col("n_stocks") > 0,
                F.col("up_count").cast("double") /
                F.col("n_stocks").cast("double")
            ).otherwise(F.lit(None).cast("double"))
        ).withColumn(
            "return_dispersion",
            F.col("cs_p90") - F.col("cs_p10")
        )

        # Load macro silver
        macro = self.spark.read.format("delta").load(
            f"{self.silver_path}/macro"
        ).select(
            "date",
            "VIXCLS",
            "yield_spread_10y2y",
            "FEDFUNDS",
            "BAMLH0A0HYM2",
            "vix_change_1d",
            "hy_spread_change"
        )

        # Join
        features = cs_stats.join(
            macro, on="date", how="left"
        ).orderBy("date")

        # Convert to pandas
        pdf = features.toPandas()
        pdf["date"] = pd.to_datetime(pdf["date"])
        pdf = pdf.sort_values("date").reset_index(drop=True)

        elapsed = (datetime.now() - start).seconds
        print(f"  Rows loaded  : {len(pdf):,}")
        print(f"  Date range   : "
              f"{pdf['date'].min().date()} → "
              f"{pdf['date'].max().date()}")
        print(f"  Time elapsed : {elapsed}s")
        return pdf

    # ------------------------------------------------------------------ #
    #  Step 2 — Prepare HMM feature matrix
    # ------------------------------------------------------------------ #
    def prepare_hmm_features(self,
                              pdf: pd.DataFrame) -> tuple:
        print("\nStep 2: Preparing HMM features...")

        candidate_cols = [
            "return_1d",
            "vol_21d",
            "cs_std_return",
            "breadth",
            "VIXCLS",
            "yield_spread_10y2y",
            "BAMLH0A0HYM2",
            "return_dispersion",
        ]

        available = [
            c for c in candidate_cols
            if c in pdf.columns
        ]
        print(f"  Features used : {available}")

        feat_df = pdf[["date"] + available].copy()

        # Forward fill then median fill
        feat_df[available] = feat_df[available].fillna(
            method="ffill", limit=5
        )
        for col in available:
            median = feat_df[col].median()
            if np.isnan(median):
                median = 0.0
            feat_df[col] = feat_df[col].fillna(median)

        # Standardize
        feat_matrix = feat_df[available].values.astype(float)
        feat_mean   = np.nanmean(feat_matrix, axis=0)
        feat_std    = np.nanstd(feat_matrix, axis=0) + 1e-8
        feat_scaled = (feat_matrix - feat_mean) / feat_std

        # Remove NaN rows
        valid_mask  = ~np.isnan(feat_scaled).any(axis=1)
        feat_scaled = feat_scaled[valid_mask]
        dates_valid = feat_df["date"].values[valid_mask]

        print(f"  Valid rows    : {len(feat_scaled):,}")
        print(f"  Feature matrix: {feat_scaled.shape}")

        if len(feat_scaled) == 0:
            raise ValueError(
                "No valid rows — check silver OHLCV data."
            )

        return feat_scaled, dates_valid, available, valid_mask

    # ------------------------------------------------------------------ #
    #  Step 3 — Fit HMM
    # ------------------------------------------------------------------ #
    def fit_hmm(self, features: np.ndarray) -> GaussianHMM:
        print(f"\nStep 3: Fitting {self.N_STATES}-state HMM...")
        start = datetime.now()

        model = GaussianHMM(
            n_components    = self.N_STATES,
            covariance_type = "full",
            n_iter          = self.N_ITER,
            random_state    = self.RANDOM_SEED,
            verbose         = False
        )
        model.fit(features)

        elapsed = (datetime.now() - start).seconds
        print(f"  Fitted        : {elapsed}s")
        print(f"  Log likelihood: {model.score(features):.2f}")
        print(f"  Converged     : {model.monitor_.converged}")
        self.hmm_model = model
        return model

    # ------------------------------------------------------------------ #
    #  Step 4 — Decode regimes
    # ------------------------------------------------------------------ #
    def decode_regimes(self, model: GaussianHMM,
                       features: np.ndarray,
                       dates: np.ndarray,
                       pdf: pd.DataFrame) -> pd.DataFrame:
        print("\nStep 4: Decoding regimes (Viterbi)...")

        hidden_states = model.predict(features)
        state_probs   = model.predict_proba(features)
        means         = model.means_

        # Assign labels based on vol (index 1) and return (index 0)
        ret_idx = 0
        vol_idx = min(1, means.shape[1] - 1)

        state_vol     = means[:, vol_idx]
        state_ret     = means[:, ret_idx]
        sorted_by_vol = np.argsort(state_vol)

        bull_state    = int(sorted_by_vol[0])   # lowest vol
        highvol_state = int(sorted_by_vol[-1])  # highest vol
        mid_state     = int(sorted_by_vol[1])

        # Middle state: bear if negative return
        if state_ret[mid_state] < 0:
            bear_state    = mid_state
        else:
            bear_state    = highvol_state
            highvol_state = mid_state

        self.regime_map = {
            bull_state   : "Bull",
            bear_state   : "Bear",
            highvol_state: "HighVol"
        }
        print(f"  Regime map    : {self.regime_map}")

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

        result = pdf.merge(regime_df, on="date", how="left")

        dist = result["regime_label"].value_counts()
        print(f"\n  Regime distribution:")
        for label, count in dist.items():
            pct = count / len(result) * 100
            print(f"    {label:8} : "
                  f"{count:,} days ({pct:.1f}%)")

        return result

    # ------------------------------------------------------------------ #
    #  Step 5 — Regime stats
    # ------------------------------------------------------------------ #
    def compute_regime_stats(self,
                              regime_df: pd.DataFrame
                              ) -> pd.DataFrame:
        print("\nStep 5: Computing regime stats...")
        stats_list = []

        for regime in ["Bull","Bear","HighVol"]:
            mask = regime_df["regime_label"] == regime
            data = regime_df[mask]
            if len(data) == 0:
                continue

            ret = data["return_1d"].dropna()
            vol = data["vol_21d"].dropna() \
                  if "vol_21d" in data.columns \
                  else pd.Series(dtype=float)
            vix = data["VIXCLS"].dropna() \
                  if "VIXCLS" in data.columns \
                  else pd.Series(dtype=float)

            ann_ret = ret.mean() * 252
            ann_vol = ret.std() * np.sqrt(252)

            stats_list.append({
                "regime"         : regime,
                "n_days"         : len(data),
                "pct_days"       : len(data)/len(regime_df),
                "mean_return"    : float(ret.mean()),
                "std_return"     : float(ret.std()),
                "ann_return"     : float(ann_ret),
                "ann_vol"        : float(ann_vol),
                "sharpe"         : float(
                    ann_ret/ann_vol if ann_vol > 0 else 0
                ),
                "hit_rate"       : float((ret > 0).mean()),
                "max_drawdown"   : float(
                    self._max_drawdown(ret)
                ),
                "mean_vix"       : float(vix.mean())
                                   if len(vix) > 0
                                   else np.nan,
                "mean_vol_21d"   : float(vol.mean())
                                   if len(vol) > 0
                                   else np.nan,
                "skewness"       : float(stats.skew(ret))
                                   if len(ret) > 3
                                   else np.nan,
                "excess_kurtosis": float(
                    stats.kurtosis(ret)
                ) if len(ret) > 3 else np.nan,
            })

        stats_df = pd.DataFrame(stats_list)
        print(f"\n  Regime performance:")
        print(stats_df[[
            "regime","n_days","ann_return",
            "ann_vol","sharpe","hit_rate"
        ]].to_string(index=False))
        return stats_df

    def _max_drawdown(self, returns: pd.Series) -> float:
        try:
            cumret  = (1 + returns).cumprod()
            rolling = cumret.expanding().max()
            dd      = (cumret - rolling) / rolling
            return float(dd.min())
        except Exception:
            return np.nan

    # ------------------------------------------------------------------ #
    #  Step 6 — Transition matrix
    # ------------------------------------------------------------------ #
    def compute_transition_stats(self,
                                  model: GaussianHMM,
                                  regime_df: pd.DataFrame
                                  ) -> tuple:
        print("\nStep 6: Transition matrix...")

        trans_mat = model.transmat_
        labels    = [
            self.regime_map.get(i, f"State{i}")
            for i in range(self.N_STATES)
        ]

        trans_df = pd.DataFrame(
            trans_mat,
            index   = labels,
            columns = labels
        )
        print(f"\n  Transition probabilities:")
        print(trans_df.round(3).to_string())

        # Regime durations
        durations      = []
        current_regime = None
        current_len    = 0

        for regime in \
                regime_df["regime_label"].dropna().values:
            if regime == current_regime:
                current_len += 1
            else:
                if current_regime is not None:
                    durations.append({
                        "regime"  : current_regime,
                        "duration": current_len
                    })
                current_regime = regime
                current_len    = 1

        if current_regime is not None:
            durations.append({
                "regime"  : current_regime,
                "duration": current_len
            })

        dur_df = pd.DataFrame(durations)
        if len(dur_df) > 0:
            avg = dur_df.groupby(
                "regime"
            )["duration"].mean()
            print(f"\n  Avg duration (days):")
            print(avg.to_string())

        return trans_df, dur_df

    # ------------------------------------------------------------------ #
    #  Write results
    # ------------------------------------------------------------------ #
    def write_results(self, regime_df: pd.DataFrame,
                      stats_df: pd.DataFrame,
                      trans_df: pd.DataFrame,
                      dur_df: pd.DataFrame) -> None:
        print("\nWriting results...")

        base_cols = [
            "date","hmm_state","regime_label",
            "prob_bull","prob_bear","prob_highvol",
            "return_1d","breadth"
        ]
        optional = [
            "vol_21d","cs_std_return","VIXCLS",
            "yield_spread_10y2y","BAMLH0A0HYM2",
            "return_dispersion"
        ]
        keep = base_cols + [
            c for c in optional
            if c in regime_df.columns
        ]

        regime_out = regime_df[keep].copy()
        regime_out["date"]  = regime_out[
            "date"
        ].astype(str)
        regime_out["year"]  = pd.to_datetime(
            regime_out["date"]
        ).dt.year
        regime_out["month"] = pd.to_datetime(
            regime_out["date"]
        ).dt.month

        self.spark.createDataFrame(regime_out).write \
            .format("delta").mode("overwrite") \
            .option("overwriteSchema","true") \
            .partitionBy("year","month") \
            .save(f"{self.eda_path}/regime_labels")
        print("  ✓ regime_labels")

        self.spark.createDataFrame(stats_df).write \
            .format("delta").mode("overwrite") \
            .option("overwriteSchema","true") \
            .save(f"{self.eda_path}/regime_stats")
        print("  ✓ regime_stats")

        if len(dur_df) > 0:
            self.spark.createDataFrame(dur_df).write \
                .format("delta").mode("overwrite") \
                .option("overwriteSchema","true") \
                .save(f"{self.eda_path}/regime_durations")
            print("  ✓ regime_durations")

        trans_out = trans_df.reset_index().rename(
            columns={"index":"from_regime"}
        )
        self.spark.createDataFrame(trans_out).write \
            .format("delta").mode("overwrite") \
            .option("overwriteSchema","true") \
            .save(f"{self.eda_path}/transition_matrix")
        print("  ✓ transition_matrix")

    # ------------------------------------------------------------------ #
    #  Validate
    # ------------------------------------------------------------------ #
    def validate(self, regime_df: pd.DataFrame,
                 stats_df: pd.DataFrame) -> None:
        print("\n" + "="*55)
        print("EDA 04 FINDINGS — Regime Analysis")
        print("="*55)

        print(f"\n  Total days   : {len(regime_df):,}")
        dist = regime_df["regime_label"].value_counts()
        print(f"\n  Distribution:")
        for r, n in dist.items():
            print(f"    {r:8}: {n:,} days "
                  f"({n/len(regime_df)*100:.1f}%)")

        print(f"\n  Performance:")
        print(stats_df[[
            "regime","ann_return","ann_vol",
            "sharpe","hit_rate","max_drawdown"
        ]].to_string(index=False))

        bull = stats_df[stats_df["regime"] == "Bull"]
        bear = stats_df[stats_df["regime"] == "Bear"]
        hv   = stats_df[stats_df["regime"] == "HighVol"]

        print(f"\n  KEY DECISIONS:")
        if len(bull) > 0:
            print(f"  → Bull ann return  : "
                  f"{bull['ann_return'].values[0]*100:.1f}%")
        if len(bear) > 0:
            print(f"  → Bear ann return  : "
                  f"{bear['ann_return'].values[0]*100:.1f}%")
        if len(hv) > 0:
            print(f"  → HighVol sharpe   : "
                  f"{hv['sharpe'].values[0]:.2f}")
        print(f"  → Use 3-state HMM in ML layer ✓")
        print(f"  → Regime label as Gold feature ✓")
        print(f"  → Regime-adaptive position sizing ✓")

    # ------------------------------------------------------------------ #
    #  Run
    # ------------------------------------------------------------------ #
    def run(self):
        print("="*55)
        print("EDA 04 — Regime Analysis (Optimized)")
        print("="*55)
        start = datetime.now()

        pdf                = self.load_market_features()
        features, dates, feat_cols, mask = \
            self.prepare_hmm_features(pdf)
        model              = self.fit_hmm(features)
        regime_df          = self.decode_regimes(
            model, features, dates, pdf
        )
        stats_df           = self.compute_regime_stats(
            regime_df
        )
        trans_df, dur_df   = self.compute_transition_stats(
            model, regime_df
        )
        self.write_results(
            regime_df, stats_df, trans_df, dur_df
        )
        self.validate(regime_df, stats_df)

        elapsed = (datetime.now() - start).seconds / 60
        print(f"\nTotal time: {elapsed:.1f} minutes")
        print("EDA 04 COMPLETE ✓")
        return regime_df, stats_df, trans_df, dur_df, model

# COMMAND ----------

class EDARegimeCharts:
    TEMPLATE      = "plotly_dark"
    REGIME_COLORS = {
        "Bull"   : "#4CAF50",
        "Bear"   : "#FF5722",
        "HighVol": "#FFC107",
        "Unknown": "#9E9E9E"
    }

    def chart_regime_timeline(self, regime_df):
        df = regime_df.dropna(
            subset=["regime_label","return_1d"]
        ).sort_values("date")

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=[
                "Market Regime (3-State HMM)",
                "Cross-Sectional Mean Return (%)",
                "Regime Probability"
            ],
            vertical_spacing=0.08,
            row_heights=[0.25, 0.45, 0.30]
        )

        for regime, color in self.REGIME_COLORS.items():
            mask = df["regime_label"] == regime
            if not mask.any():
                continue
            fig.add_trace(go.Scatter(
                x=df[mask]["date"],
                y=[regime] * mask.sum(),
                mode="markers",
                marker=dict(color=color,size=4,symbol="square"),
                name=regime
            ), row=1, col=1)

        for regime, color in self.REGIME_COLORS.items():
            mask = df["regime_label"] == regime
            if not mask.any():
                continue
            fig.add_trace(go.Bar(
                x=df[mask]["date"],
                y=df[mask]["return_1d"] * 100,
                name=regime,
                marker_color=color,
                opacity=0.7,
                showlegend=False
            ), row=2, col=1)

        for prob_col, label, color in [
            ("prob_bull",   "P(Bull)",    "#4CAF50"),
            ("prob_bear",   "P(Bear)",    "#FF5722"),
            ("prob_highvol","P(HighVol)", "#FFC107"),
        ]:
            if prob_col not in df.columns:
                continue
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[prob_col],
                name=label, mode="lines",
                line=dict(color=color, width=1),
                showlegend=False
            ), row=3, col=1)

        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=2, col=1
        )
        fig.update_layout(
            title="<b>EDA 04 — Market Regime Timeline</b>",
            template=self.TEMPLATE, height=800,
            hovermode="x unified", barmode="overlay"
        )
        fig.update_yaxes(title_text="Regime",     row=1,col=1)
        fig.update_yaxes(title_text="Return(%)",  row=2,col=1)
        fig.update_yaxes(title_text="Prob",
                         range=[0,1],             row=3,col=1)
        fig.show()

    def chart_regime_distributions(self, regime_df):
        fig = make_subplots(rows=1, cols=2,
            subplot_titles=[
                "Return Distribution by Regime",
                "Volatility Distribution by Regime"
            ])

        for regime, color in self.REGIME_COLORS.items():
            data = regime_df[
                regime_df["regime_label"]==regime
            ]
            if len(data) == 0:
                continue
            ret = data["return_1d"].dropna() * 100
            fig.add_trace(go.Violin(
                x=[regime]*len(ret), y=ret,
                name=regime, fillcolor=color,
                line_color="white", opacity=0.7,
                box_visible=True, meanline_visible=True
            ), row=1, col=1)

            if "vol_21d" in data.columns:
                vol = data["vol_21d"].dropna() * 100
                if len(vol) > 0:
                    fig.add_trace(go.Violin(
                        x=[regime]*len(vol), y=vol,
                        name=regime, fillcolor=color,
                        line_color="white", opacity=0.7,
                        box_visible=True,
                        meanline_visible=True,
                        showlegend=False
                    ), row=1, col=2)

        fig.update_layout(
            title="<b>EDA 04 — Regime Distributions</b>",
            template=self.TEMPLATE, height=550,
            violinmode="group"
        )
        fig.update_yaxes(title_text="Return(%)", row=1,col=1)
        fig.update_yaxes(title_text="Vol(%)",    row=1,col=2)
        fig.show()

    def chart_regime_performance(self, stats_df):
        colors = [
            self.REGIME_COLORS.get(r,"grey")
            for r in stats_df["regime"]
        ]
        fig = make_subplots(rows=2, cols=2,
            subplot_titles=[
                "Annualized Return","Annualized Vol",
                "Sharpe Ratio","Hit Rate"
            ])

        for col, row, c, pct in [
            ("ann_return",1,1,True),
            ("ann_vol",   1,2,True),
            ("sharpe",    2,1,False),
            ("hit_rate",  2,2,True),
        ]:
            vals = stats_df[col] * (100 if pct else 1)
            fig.add_trace(go.Bar(
                x=stats_df["regime"], y=vals,
                marker_color=colors,
                text=vals.apply(
                    lambda x: f"{x:.1f}%" if pct
                    else f"{x:.2f}"
                ),
                textposition="outside",
                showlegend=False
            ), row=row, col=c)

        fig.update_layout(
            title="<b>EDA 04 — Regime Performance</b>",
            template=self.TEMPLATE, height=650
        )
        fig.show()

    def chart_transition_matrix(self, trans_df):
        regimes = list(trans_df.index)
        values  = trans_df.values
        fig = go.Figure(go.Heatmap(
            z=values, x=regimes, y=regimes,
            colorscale="Blues",
            text=np.round(values,3),
            texttemplate="%{text}",
            textfont=dict(size=14),
            colorbar=dict(title="Probability"),
            zmin=0, zmax=1
        ))
        fig.update_layout(
            title="<b>EDA 04 — Transition Matrix</b>",
            template=self.TEMPLATE, height=500,
            xaxis_title="To Regime",
            yaxis_title="From Regime"
        )
        fig.show()

    def chart_regime_duration(self, dur_df):
        if len(dur_df) == 0:
            print("No duration data")
            return
        fig = go.Figure()
        for regime, color in self.REGIME_COLORS.items():
            mask = dur_df["regime"] == regime
            if not mask.any():
                continue
            fig.add_trace(go.Box(
                y=dur_df[mask]["duration"],
                name=regime, marker_color=color,
                boxpoints="outliers"
            ))
        fig.update_layout(
            title="<b>EDA 04 — Regime Duration (days)</b>",
            template=self.TEMPLATE, height=500,
            yaxis_title="Duration (days)"
        )
        fig.show()

    def chart_cumulative_returns(self, regime_df):
        df = regime_df.dropna(
            subset=["return_1d","regime_label"]
        ).sort_values("date").copy()
        df["cum_return"] = (1 + df["return_1d"]).cumprod()

        fig = make_subplots(rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=[
                "Cumulative Return (colored by regime)",
                "Regime-Conditional Cumulative Returns"
            ],
            vertical_spacing=0.1)

        for regime, color in self.REGIME_COLORS.items():
            mask = df["regime_label"] == regime
            if not mask.any():
                continue
            fig.add_trace(go.Scatter(
                x=df[mask]["date"],
                y=df[mask]["cum_return"],
                mode="markers", name=regime,
                marker=dict(color=color,size=3,opacity=0.5)
            ), row=1, col=1)

            cond = df[mask]["return_1d"]
            cr   = (1 + cond).cumprod()
            fig.add_trace(go.Scatter(
                x=df[mask]["date"], y=cr,
                name=f"{regime} only", mode="lines",
                line=dict(color=color, width=2),
                showlegend=False
            ), row=2, col=1)

        fig.update_layout(
            title="<b>EDA 04 — Cumulative Returns</b>",
            template=self.TEMPLATE, height=700,
            hovermode="x unified"
        )
        fig.update_yaxes(
            title_text="Cum Return", row=1, col=1
        )
        fig.update_yaxes(
            title_text="Cum Return", row=2, col=1
        )
        fig.show()

    def chart_breadth_regime(self, regime_df):
        df = regime_df.dropna(
            subset=["breadth","regime_label","return_1d"]
        )
        if len(df) == 0:
            print("No breadth data")
            return

        color_map = {
            r: c for r, c in self.REGIME_COLORS.items()
        }
        fig = px.scatter(
            df, x="breadth", y="return_1d",
            color="regime_label",
            color_discrete_map=color_map,
            opacity=0.4, trendline="lowess",
            template=self.TEMPLATE,
            title="<b>EDA 04 — Breadth vs Return</b>",
            labels={
                "breadth"     : "Breadth (% stocks up)",
                "return_1d"   : "Mean Daily Return",
                "regime_label": "Regime"
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
                trans_df, dur_df):
        print("\n" + "="*55)
        print("Generating Charts...")
        print("="*55)
        print("\n[1/7] Regime Timeline...")
        self.chart_regime_timeline(regime_df)
        print("[2/7] Distributions...")
        self.chart_regime_distributions(regime_df)
        print("[3/7] Performance...")
        self.chart_regime_performance(stats_df)
        print("[4/7] Transition Matrix...")
        self.chart_transition_matrix(trans_df)
        print("[5/7] Duration...")
        self.chart_regime_duration(dur_df)
        print("[6/7] Cumulative Returns...")
        self.chart_cumulative_returns(regime_df)
        print("[7/7] Breadth vs Return...")
        self.chart_breadth_regime(regime_df)
        print("\nAll 7 charts ✓")

# COMMAND ----------

eda = EDARegimeAnalysis(
    spark       = spark,
    silver_path = SILVER_PATH,
    eda_path    = EDA_PATH
)

regime_df, stats_df, trans_df, \
dur_df, model = eda.run()

charts = EDARegimeCharts()
charts.run_all(
    regime_df = regime_df,
    stats_df  = stats_df,
    trans_df  = trans_df,
    dur_df    = dur_df
)

print("\nEDA 04 COMPLETE ✓")

# COMMAND ----------

regime = spark.read.format("delta").load(
    f"{EDA_PATH}/regime_analysis/regime_labels"
).toPandas()

stats = spark.read.format("delta").load(
    f"{EDA_PATH}/regime_analysis/regime_stats"
).toPandas()

trans = spark.read.format("delta").load(
    f"{EDA_PATH}/regime_analysis/transition_matrix"
).toPandas()

print("="*55)
print("EDA 04 — Summary")
print("="*55)
print(f"Total days : {len(regime):,}")
print(f"\nRegime distribution:")
print(regime["regime_label"].value_counts().to_string())
print(f"\nPerformance:")
print(stats[[
    "regime","ann_return","ann_vol",
    "sharpe","hit_rate","max_drawdown"
]].to_string(index=False))
print(f"\nTransition matrix:")
print(trans.set_index("from_regime").to_string())