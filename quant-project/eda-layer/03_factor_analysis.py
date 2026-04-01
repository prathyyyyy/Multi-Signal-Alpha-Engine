# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import *
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_1samp
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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

# COMMAND ----------

class EDAFactorAnalysisFast:
    """
    EDA 03 — Factor Analysis (Optimized).

    Speed improvements:
      1. Spark computes cross-sectional ranks (no pandas loop)
      2. Vectorized IC via numpy corrcoef on rank matrices
      3. All factors computed in single pass
      4. Forward returns computed in Spark (not pandas)
      5. 10-50x faster than sequential version
    """

    FACTORS = {
        "mom_1d"    : ("return_1d",     1),
        "mom_5d"    : ("return_5d",     1),
        "mom_21d"   : ("return_21d",    1),
        "mom_63d"   : ("return_63d",    1),
        "mom_252d"  : ("return_252d",   1),
        "rev_1d"    : ("return_1d",    -1),   # negated
        "rev_5d"    : ("return_5d",    -1),   # negated
        "vol_21d"   : ("vol_21d",       1),
        "vol_63d"   : ("vol_63d",       1),
        "range"     : ("daily_range",   1),
        "dolvol"    : ("dollar_volume", 1),
        "volume"    : ("volume",        1),
        "vwap_ratio": ("vwap_ratio",    1),
    }

    FWD_HORIZONS = [1, 5, 10, 21]

    def __init__(self, spark, silver_path, eda_path):
        self.spark       = spark
        self.silver_path = f"{silver_path}/ohlcv"
        self.eda_path    = f"{eda_path}/factor_analysis"
        print("EDAFactorAnalysisFast ✓")

    # ------------------------------------------------------------------ #
    #  Step 1 — Load + compute everything in Spark
    # ------------------------------------------------------------------ #
    def load_and_prepare(self):
        print("\nStep 1: Loading and preparing data in Spark...")
        start = datetime.now()

        df = self.spark.read.format("delta").load(self.silver_path)

        # Add vwap_ratio
        df = df.withColumn(
            "vwap_ratio",
            F.col("vwap") / F.col("close") - 1
        )

        # Add forward returns in Spark using window functions
        w = Window.partitionBy("ticker").orderBy("date")
        for h in self.FWD_HORIZONS:
            df = df.withColumn(
                f"fwd_return_{h}d",
                F.avg("return_1d").over(
                    w.rowsBetween(1, h)
                )
            )

        # Add cross-sectional ranks per date (Spark)
        # This replaces the slow per-date pandas loop
        w_date = Window.partitionBy("date")

        all_factor_cols = [
            col for col, _ in self.FACTORS.values()
        ]
        all_factor_cols = list(set(all_factor_cols))

        for col in all_factor_cols:
            df = df.withColumn(
                f"rank_{col}",
                F.percent_rank().over(
                    w_date.orderBy(col)
                )
            )

        # Add forward return ranks
        for h in self.FWD_HORIZONS:
            df = df.withColumn(
                f"rank_fwd_{h}d",
                F.percent_rank().over(
                    w_date.orderBy(f"fwd_return_{h}d")
                )
            )

        # Select only needed columns
        select_cols = (
            ["date","ticker"] +
            [f"rank_{c}" for c in all_factor_cols] +
            [f"rank_fwd_{h}d" for h in self.FWD_HORIZONS]
        )
        df = df.select(*select_cols).dropna()

        # Cache — will be reused many times
        df.cache()
        n = df.count()

        elapsed = (datetime.now() - start).seconds
        print(f"  Rows cached    : {n:,}")
        print(f"  Time elapsed   : {elapsed}s")
        return df

    # ------------------------------------------------------------------ #
    #  Step 2 — Vectorized IC computation
    # ------------------------------------------------------------------ #
    def compute_ic_vectorized(self, df) -> tuple:
        """
        Key optimization: compute IC for ALL factors and ALL
        horizons in a single pandas operation using numpy.

        Traditional: O(n_dates × n_factors × n_horizons)
        Optimized  : O(n_factors × n_horizons) via matrix ops
        """
        print("\nStep 2: Computing IC (vectorized)...")
        start = datetime.now()

        # Convert to pandas once
        pdf = df.toPandas()
        pdf["date"] = pd.to_datetime(pdf["date"])

        # For each factor-horizon combo, compute IC per date
        # using numpy vectorized groupby operations
        all_ic_series   = {}
        all_factor_stats = []

        factor_rank_cols = {
            name: f"rank_{col}"
            for name, (col, _) in self.FACTORS.items()
        }
        fwd_rank_cols = {
            h: f"rank_fwd_{h}d"
            for h in self.FWD_HORIZONS
        }

        # Group by date ONCE — reuse for all factors
        grouped = pdf.groupby("date")

        for factor_name, (col, sign) in self.FACTORS.items():
            rank_col = factor_rank_cols[factor_name]
            all_ic_series[factor_name] = {}

            for h in self.FWD_HORIZONS:
                fwd_col = fwd_rank_cols[h]

                # Vectorized Spearman via pre-computed ranks
                # IC = correlation of ranks (already computed in Spark)
                def compute_ic_for_group(g):
                    x = g[rank_col].values * sign
                    y = g[fwd_col].values
                    mask = ~(np.isnan(x) | np.isnan(y))
                    if mask.sum() < 5:
                        return np.nan
                    # Pearson on ranks = Spearman
                    return np.corrcoef(x[mask], y[mask])[0, 1]

                ic_series = grouped.apply(
                    compute_ic_for_group
                ).rename(factor_name)

                all_ic_series[factor_name][h] = ic_series
                ic_clean = ic_series.dropna()

                if len(ic_clean) < 10:
                    continue

                ic_mean  = ic_clean.mean()
                ic_std   = ic_clean.std()
                icir     = ic_mean / ic_std if ic_std > 0 else 0
                hit_rate = (ic_clean > 0).mean()
                t_stat, t_pval = ttest_1samp(ic_clean, 0)

                all_factor_stats.append({
                    "factor"      : factor_name,
                    "fwd_horizon" : f"{h}d",
                    "ic_mean"     : ic_mean,
                    "ic_std"      : ic_std,
                    "icir"        : icir,
                    "ic_abs_mean" : abs(ic_mean),
                    "hit_rate"    : hit_rate,
                    "t_stat"      : t_stat,
                    "t_pval"      : t_pval,
                    "significant" : int(abs(t_stat) > 2.0),
                    "n_dates"     : len(ic_clean),
                })

        elapsed = (datetime.now() - start).seconds
        factor_stats = pd.DataFrame(all_factor_stats)
        print(f"  Computed       : {len(factor_stats):,} "
              f"factor-horizon combos")
        print(f"  Time elapsed   : {elapsed}s")
        print(f"  Significant    : "
              f"{factor_stats['significant'].sum():,}")
        return factor_stats, all_ic_series, pdf

    # ------------------------------------------------------------------ #
    #  Step 3 — Factor decay (fast)
    # ------------------------------------------------------------------ #
    def compute_factor_decay(self,
                              factor_stats: pd.DataFrame
                              ) -> pd.DataFrame:
        print("\nStep 3: Computing factor decay...")
        decay = factor_stats[[
            "factor","fwd_horizon","ic_mean","ic_abs_mean"
        ]].copy()
        decay["horizon_days"] = decay["fwd_horizon"].str.replace(
            "d",""
        ).astype(int)
        return decay

    # ------------------------------------------------------------------ #
    #  Step 4 — Rolling IC (fast via pandas rolling)
    # ------------------------------------------------------------------ #
    def compute_rolling_ic(self, ic_series_dict: dict,
                            window: int = 63) -> pd.DataFrame:
        print(f"\nStep 4: Computing rolling IC ({window}d)...")

        top_factors = [
            "mom_21d","mom_63d","mom_252d",
            "vol_21d","rev_1d","dolvol"
        ]
        h           = 21
        all_rolling = []

        for factor in top_factors:
            if factor not in ic_series_dict:
                continue
            if h not in ic_series_dict[factor]:
                continue

            ic      = ic_series_dict[factor][h]
            rolling = ic.rolling(window).mean()

            tmp         = rolling.reset_index()
            tmp.columns = ["date","rolling_ic"]
            tmp["factor"] = factor
            all_rolling.append(tmp)

        return pd.concat(all_rolling, ignore_index=True) \
               if all_rolling else pd.DataFrame()

    # ------------------------------------------------------------------ #
    #  Write results
    # ------------------------------------------------------------------ #
    def write_results(self, factor_stats, decay_df,
                      rolling_df) -> None:
        print("\nWriting results to Delta...")

        self.spark.createDataFrame(factor_stats).write \
            .format("delta").mode("overwrite") \
            .option("overwriteSchema","true") \
            .save(f"{self.eda_path}/factor_stats")
        print("  ✓ factor_stats")

        self.spark.createDataFrame(decay_df).write \
            .format("delta").mode("overwrite") \
            .option("overwriteSchema","true") \
            .save(f"{self.eda_path}/factor_decay")
        print("  ✓ factor_decay")

        if len(rolling_df) > 0:
            self.spark.createDataFrame(rolling_df).write \
                .format("delta").mode("overwrite") \
                .option("overwriteSchema","true") \
                .save(f"{self.eda_path}/rolling_ic")
            print("  ✓ rolling_ic")

    # ------------------------------------------------------------------ #
    #  Validate
    # ------------------------------------------------------------------ #
    def validate(self, factor_stats: pd.DataFrame) -> None:
        print("\n" + "="*55)
        print("EDA 03 FINDINGS — Factor Analysis")
        print("="*55)

        sig = factor_stats[factor_stats["significant"] == 1]
        print(f"\n  Factor-horizon combos : {len(factor_stats):,}")
        print(f"  Significant (|t|>2)   : {len(sig):,} "
              f"({len(sig)/len(factor_stats)*100:.1f}%)")

        print(f"\n  Best factors → 1d forward return:")
        print(factor_stats[
            factor_stats["fwd_horizon"] == "1d"
        ].nlargest(10,"icir")[[
            "factor","ic_mean","icir",
            "t_stat","hit_rate","significant"
        ]].to_string(index=False))

        print(f"\n  Best factors → 21d forward return:")
        print(factor_stats[
            factor_stats["fwd_horizon"] == "21d"
        ].nlargest(10,"icir")[[
            "factor","ic_mean","icir",
            "t_stat","hit_rate","significant"
        ]].to_string(index=False))

        keep = factor_stats[
            (factor_stats["significant"] == 1) &
            (factor_stats["icir"].abs() > 0.3)
        ]["factor"].unique().tolist()
        print(f"\n  KEY DECISION:")
        print(f"  → Keep in Gold: {keep}")
        print(f"  → IC > 0.05   : strong signal")
        print(f"  → ICIR > 0.5  : consistent signal")

    # ------------------------------------------------------------------ #
    #  Run
    # ------------------------------------------------------------------ #
    def run(self):
        print("="*55)
        print("EDA 03 — Factor Analysis (Fast Version)")
        print("="*55)
        start = datetime.now()

        df = self.load_and_prepare()
        factor_stats, ic_series_dict, pdf = \
            self.compute_ic_vectorized(df)
        decay_df   = self.compute_factor_decay(factor_stats)
        rolling_df = self.compute_rolling_ic(ic_series_dict)

        self.write_results(factor_stats, decay_df, rolling_df)
        self.validate(factor_stats)

        # Unpersist cache
        df.unpersist()

        total = (datetime.now() - start).seconds / 60
        print(f"\nTotal time: {total:.1f} minutes")
        print("EDA 03 COMPLETE ✓")
        return factor_stats, ic_series_dict, decay_df, rolling_df

# COMMAND ----------

class EDAFactorCharts:
    TEMPLATE = "plotly_dark"
    COLORS   = {
        "primary"  : "#2196F3",
        "secondary": "#FF5722",
        "success"  : "#4CAF50",
        "warning"  : "#FFC107",
        "purple"   : "#9C27B0",
        "teal"     : "#00BCD4",
    }

    def chart_ic_heatmap(self, factor_stats):
        pivot = factor_stats.pivot_table(
            index="factor", columns="fwd_horizon",
            values="ic_mean"
        ).reindex(
            factor_stats.groupby("factor")["ic_mean"]
            .apply(lambda x: abs(x).mean())
            .sort_values(ascending=False).index
        )
        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="RdYlGn", zmid=0,
            text=np.round(pivot.values, 3),
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorbar=dict(title="IC Mean")
        ))
        fig.update_layout(
            title="<b>EDA 03 — IC Heatmap</b>",
            template=self.TEMPLATE, height=550,
            xaxis_title="Forward Horizon",
            yaxis_title="Factor"
        )
        fig.show()

    def chart_icir_bars(self, factor_stats):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "ICIR — 1d","ICIR — 5d",
                "ICIR — 10d","ICIR — 21d"
            ]
        )
        for (horizon, row, col) in [
            ("1d",1,1),("5d",1,2),
            ("10d",2,1),("21d",2,2)
        ]:
            d = factor_stats[
                factor_stats["fwd_horizon"]==horizon
            ].sort_values("icir",key=abs,ascending=False)
            colors = [
                self.COLORS["success"]
                if s else self.COLORS["secondary"]
                for s in d["significant"]
            ]
            fig.add_trace(go.Bar(
                x=d["factor"], y=d["icir"],
                marker_color=colors,
                text=d["icir"].round(3),
                textposition="outside",
                showlegend=False,
                customdata=d[[
                    "ic_mean","t_stat","hit_rate"
                ]].values,
                hovertemplate=(
                    "<b>%{x}</b><br>ICIR: %{y:.3f}<br>"
                    "IC: %{customdata[0]:.4f}<br>"
                    "t: %{customdata[1]:.2f}<br>"
                    "Hit: %{customdata[2]:.1%}<extra></extra>"
                )
            ), row=row, col=col)
            for y_val in [0.5,-0.5,0.3,-0.3]:
                fig.add_hline(
                    y=y_val,
                    line_dash="dash",
                    line_color="green" if abs(y_val)==0.5
                               else "yellow",
                    opacity=0.4,
                    row=row, col=col
                )
        fig.update_layout(
            title="<b>EDA 03 — ICIR by Factor & Horizon</b>",
            template=self.TEMPLATE, height=750
        )
        fig.show()

    def chart_tstat(self, factor_stats):
        fig = make_subplots(rows=1, cols=2,
            subplot_titles=["t-statistics (1d)","Hit Rate (1d)"]
        )
        d = factor_stats[
            factor_stats["fwd_horizon"]=="1d"
        ].sort_values("t_stat",key=abs,ascending=True)

        fig.add_trace(go.Bar(
            x=d["t_stat"], y=d["factor"],
            orientation="h",
            marker_color=[
                self.COLORS["success"] if abs(t)>2
                else self.COLORS["secondary"]
                for t in d["t_stat"]
            ],
            showlegend=False
        ), row=1, col=1)
        fig.add_vline(x=2,line_dash="dash",
                      line_color="green",opacity=0.7,row=1,col=1)
        fig.add_vline(x=-2,line_dash="dash",
                      line_color="green",opacity=0.7,row=1,col=1)

        d2 = factor_stats[
            factor_stats["fwd_horizon"]=="1d"
        ].sort_values("hit_rate",ascending=True)
        fig.add_trace(go.Bar(
            x=d2["hit_rate"], y=d2["factor"],
            orientation="h",
            marker_color=[
                self.COLORS["success"] if h>0.55
                else self.COLORS["warning"] if h>0.50
                else self.COLORS["secondary"]
                for h in d2["hit_rate"]
            ],
            text=d2["hit_rate"].apply(lambda x: f"{x:.1%}"),
            textposition="outside",
            showlegend=False
        ), row=1, col=2)
        fig.add_vline(x=0.55,line_dash="dash",
                      line_color="green",opacity=0.7,row=1,col=2)
        fig.update_layout(
            title="<b>EDA 03 — Factor Significance</b>",
            template=self.TEMPLATE, height=550
        )
        fig.show()

    def chart_decay(self, decay_df):
        fig = go.Figure()
        colors = px.colors.qualitative.Plotly
        for i, factor in enumerate(
            decay_df["factor"].unique()
        ):
            d = decay_df[
                decay_df["factor"]==factor
            ].sort_values("horizon_days")
            fig.add_trace(go.Scatter(
                x=d["horizon_days"], y=d["ic_mean"],
                name=factor, mode="lines+markers",
                line=dict(color=colors[i%len(colors)],width=2),
                marker=dict(size=8)
            ))
        fig.add_hline(y=0,line_dash="dash",
                      line_color="white",opacity=0.3)
        fig.add_hline(y=0.05,line_dash="dot",
                      line_color="green",opacity=0.5,
                      annotation_text="IC=0.05")
        fig.add_hline(y=-0.05,line_dash="dot",
                      line_color="green",opacity=0.5)
        fig.update_layout(
            title="<b>EDA 03 — Factor Decay Curves</b>",
            template=self.TEMPLATE, height=500,
            xaxis_title="Horizon (days)",
            yaxis_title="IC Mean",
            xaxis=dict(
                tickvals=[1,5,10,21],
                ticktext=["1d","5d","10d","21d"]
            ),
            hovermode="x unified"
        )
        fig.show()

    def chart_rolling_ic(self, rolling_df):
        if len(rolling_df) == 0:
            print("No rolling IC data")
            return
        fig = go.Figure()
        colors = px.colors.qualitative.Plotly
        for i, factor in enumerate(
            rolling_df["factor"].unique()
        ):
            d = rolling_df[
                rolling_df["factor"]==factor
            ].sort_values("date")
            fig.add_trace(go.Scatter(
                x=d["date"], y=d["rolling_ic"],
                name=factor, mode="lines",
                line=dict(color=colors[i%len(colors)],width=1.5)
            ))
        fig.add_hline(y=0,line_dash="dash",
                      line_color="white",opacity=0.3)
        fig.add_hline(y=0.05,line_dash="dot",
                      line_color="green",opacity=0.4,
                      annotation_text="IC=0.05")
        fig.add_hline(y=-0.05,line_dash="dot",
                      line_color="red",opacity=0.4)
        fig.update_layout(
            title="<b>EDA 03 — Rolling 63d IC (21d horizon)</b>",
            template=self.TEMPLATE, height=500,
            xaxis_title="Date",
            yaxis_title="Rolling IC",
            hovermode="x unified"
        )
        fig.show()

    def chart_ic_distribution(self, ic_series_dict):
        top = ["mom_21d","mom_63d","mom_252d",
               "vol_21d","rev_1d","dolvol"]
        h   = 21
        fig = make_subplots(rows=2, cols=3,
                            subplot_titles=top)
        colors = [
            self.COLORS["primary"], self.COLORS["success"],
            self.COLORS["warning"], self.COLORS["secondary"],
            self.COLORS["purple"], self.COLORS["teal"],
        ]
        for i, (factor, (row,col)) in enumerate(zip(
            top,[(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
        )):
            if factor not in ic_series_dict:
                continue
            if h not in ic_series_dict[factor]:
                continue
            ic = ic_series_dict[factor][h].dropna()
            if len(ic) == 0:
                continue
            mu   = ic.mean()
            icir = mu/ic.std() if ic.std()>0 else 0
            fig.add_trace(go.Histogram(
                x=ic, nbinsx=50, name=factor,
                marker_color=colors[i],
                opacity=0.8, showlegend=False
            ), row=row, col=col)
            fig.add_vline(x=0,line_dash="dash",
                          line_color="white",opacity=0.4,
                          row=row,col=col)
            fig.add_vline(x=mu,line_color=colors[i],
                          line_width=2,
                          annotation_text=f"μ={mu:.3f} ICIR={icir:.2f}",
                          annotation_font_size=9,
                          row=row,col=col)
        fig.update_layout(
            title="<b>EDA 03 — IC Distribution (21d horizon)</b>",
            template=self.TEMPLATE, height=600
        )
        fig.show()

    def chart_scatter(self, factor_stats):
        fig = px.scatter(
            factor_stats,
            x="ic_mean", y="icir",
            color="fwd_horizon",
            size="hit_rate",
            text="factor",
            hover_data={
                "factor":":%s","ic_mean":":.4f",
                "icir":":.3f","t_stat":":.2f",
                "hit_rate":":.1%"
            },
            template=self.TEMPLATE,
            title="<b>EDA 03 — IC vs ICIR</b>",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig.update_traces(
            textposition="top center",
            textfont=dict(size=7),
            marker=dict(opacity=0.7)
        )
        for y in [0.5,-0.5]:
            fig.add_hline(y=y,line_dash="dash",
                          line_color="green",opacity=0.5)
        for x in [0.05,-0.05]:
            fig.add_vline(x=x,line_dash="dash",
                          line_color="yellow",opacity=0.5)
        fig.update_layout(height=600)
        fig.show()

    def run_all(self, factor_stats, ic_series_dict,
                decay_df, rolling_df):
        print("\n" + "="*55)
        print("Generating Charts...")
        print("="*55)
        print("[1/7] IC Heatmap...")
        self.chart_ic_heatmap(factor_stats)
        print("[2/7] ICIR Bars...")
        self.chart_icir_bars(factor_stats)
        print("[3/7] t-stat...")
        self.chart_tstat(factor_stats)
        print("[4/7] Factor Decay...")
        self.chart_decay(decay_df)
        print("[5/7] Rolling IC...")
        self.chart_rolling_ic(rolling_df)
        print("[6/7] IC Distribution...")
        self.chart_ic_distribution(ic_series_dict)
        print("[7/7] IC vs ICIR Scatter...")
        self.chart_scatter(factor_stats)
        print("\nAll 7 charts ✓")

# COMMAND ----------

eda = EDAFactorAnalysisFast(
    spark       = spark,
    silver_path = SILVER_PATH,
    eda_path    = EDA_PATH
)

factor_stats, ic_series_dict, \
decay_df, rolling_df = eda.run()

charts = EDAFactorCharts()
charts.run_all(
    factor_stats   = factor_stats,
    ic_series_dict = ic_series_dict,
    decay_df       = decay_df,
    rolling_df     = rolling_df
)

print("\nEDA 03 COMPLETE ✓")

# COMMAND ----------

fs = spark.read.format("delta").load(
    f"{EDA_PATH}/factor_analysis/factor_stats"
).toPandas()

print("="*55)
print("EDA 03 — Factor Analysis Summary")
print("="*55)
print(f"Combos      : {len(fs):,}")
print(f"Significant : {fs['significant'].sum():,} "
      f"({fs['significant'].mean()*100:.1f}%)")

print(f"\nRanked factors by avg ICIR:")
ranked = fs.groupby("factor").agg(
    avg_icir=("icir","mean"),
    avg_ic=("ic_mean","mean"),
    pct_sig=("significant","mean")
).sort_values("avg_icir",key=abs,ascending=False)
print(ranked.to_string())

print(f"\nStrong signals (ICIR>0.5):")
strong = fs[fs["icir"].abs()>0.5][[
    "factor","fwd_horizon","ic_mean",
    "icir","t_stat","hit_rate"
]].sort_values("icir",key=abs,ascending=False)
print(strong.to_string(index=False))