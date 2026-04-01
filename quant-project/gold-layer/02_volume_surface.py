# Databricks notebook source
# MAGIC %pip install scipy statsmodels==0.14.5 plotly pandas numpy --quiet

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
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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

print("Config loaded ✓")
print(f"Gold path : {GOLD_PATH}")

# COMMAND ----------

class GoldVolSurface:
    """
    Gold 02 — Volatility Surface Features.

    Note: Silver options has only 1 date (2026-03-21).
    Strategy:
      - Build all cross-sectional IV features per ticker
      - Join CBOE VIX term structure (multi-date)
      - Join realized vol from OHLCV silver
      - Compute vol risk premium per ticker

    Features built:
      ATM IV         : atm_iv, call/put split
      IV Skew        : put - call IV
      IV Term Struct : near vs far tenor
      Put/Call ratio : OI and IV based
      Vol Risk Prem  : IV - realized vol
      IV Surface     : moneyness buckets
      CBOE           : VIX/VIX3M/VIX6M/VIX9D
    """

    def __init__(self, spark, silver_path,
                 eda_path, gold_path):
        self.spark       = spark
        self.silver_path = silver_path
        self.eda_path    = eda_path
        self.gold_path   = f"{gold_path}/vol_surface"
        print("GoldVolSurface ✓")
        print(f"  Output : {self.gold_path}")

    # ------------------------------------------------------------------ #
    #  Step 1 — Load silver options
    # ------------------------------------------------------------------ #
    def load_options(self):
        print("\nStep 1: Loading silver options...")
        start = datetime.now()

        opts = self.spark.read.format("delta").load(
            f"{self.silver_path}/options"
        )

        total   = opts.count()
        tickers = opts.select("ticker").distinct().count()
        dates   = opts.select("date").distinct().count()

        elapsed = (datetime.now() - start).seconds
        print(f"  Rows    : {total:,}")
        print(f"  Tickers : {tickers:,}")
        print(f"  Dates   : {dates:,}")
        print(f"  Elapsed : {elapsed}s")
        return opts

    # ------------------------------------------------------------------ #
    #  Step 2 — ATM IV features per ticker/date
    # ------------------------------------------------------------------ #
    def build_atm_features(self, opts):
        print("\nStep 2: Building ATM IV features...")

        atm = opts.groupBy("ticker","date").agg(
            # ATM IV overall
            F.mean(
                F.when(F.col("is_atm"),
                       F.col("implied_vol"))
            ).alias("atm_iv"),

            # ATM by option type
            F.mean(
                F.when(
                    F.col("is_atm") &
                    (F.col("option_type") == "call"),
                    F.col("implied_vol")
                )
            ).alias("atm_iv_call"),

            F.mean(
                F.when(
                    F.col("is_atm") &
                    (F.col("option_type") == "put"),
                    F.col("implied_vol")
                )
            ).alias("atm_iv_put"),

            # Term structure buckets
            F.mean(
                F.when(
                    F.col("is_atm") &
                    (F.col("tte_days") <= 30),
                    F.col("implied_vol")
                )
            ).alias("atm_iv_near"),

            F.mean(
                F.when(
                    F.col("is_atm") &
                    (F.col("tte_days").between(31, 90)),
                    F.col("implied_vol")
                )
            ).alias("atm_iv_mid"),

            F.mean(
                F.when(
                    F.col("is_atm") &
                    (F.col("tte_days") > 90),
                    F.col("implied_vol")
                )
            ).alias("atm_iv_far"),

            # IV across all options
            F.mean("implied_vol").alias("avg_iv"),
            F.max("implied_vol").alias("max_iv"),
            F.min("implied_vol").alias("min_iv"),

            # Put vs call averages
            F.mean(
                F.when(
                    F.col("option_type") == "put",
                    F.col("implied_vol")
                )
            ).alias("put_iv_avg"),
            F.mean(
                F.when(
                    F.col("option_type") == "call",
                    F.col("implied_vol")
                )
            ).alias("call_iv_avg"),

            # OI features
            F.sum("open_interest").alias(
                "total_oi"
            ),
            F.sum(
                F.when(
                    F.col("option_type") == "put",
                    F.col("open_interest")
                ).otherwise(F.lit(0))
            ).alias("put_oi"),
            F.sum(
                F.when(
                    F.col("option_type") == "call",
                    F.col("open_interest")
                ).otherwise(F.lit(0))
            ).alias("call_oi"),

            # Option counts
            F.count("*").alias("n_options"),
            F.countDistinct("expiry").alias("n_expiries"),
            F.countDistinct("strike").alias("n_strikes"),

            # Weighted IV by OI
            F.sum(
                F.col("implied_vol") *
                F.col("open_interest")
            ).alias("iv_oi_weighted_sum"),

            # IV by moneyness bucket
            F.mean(
                F.when(
                    F.col("moneyness").between(-0.05, 0.05),
                    F.col("implied_vol")
                )
            ).alias("iv_atm_bucket"),
            F.mean(
                F.when(
                    F.col("moneyness") < -0.10,
                    F.col("implied_vol")
                )
            ).alias("iv_otm_put"),
            F.mean(
                F.when(
                    F.col("moneyness") > 0.10,
                    F.col("implied_vol")
                )
            ).alias("iv_otm_call"),

            # Liquidity
            F.mean("relative_spread").alias(
                "avg_bid_ask_spread"
            ),
            F.mean("mid_price").alias("avg_mid_price"),

            # TTE stats
            F.mean("tte_days").alias("avg_tte_days"),
            F.min("tte_days").alias("min_tte_days"),
            F.max("tte_days").alias("max_tte_days"),
        )

        # Derived features
        atm = atm.withColumn(
            "iv_spread",
            F.col("max_iv") - F.col("min_iv")
        ).withColumn(
            "iv_skew_computed",
            F.col("atm_iv_put") - F.col("atm_iv_call")
        ).withColumn(
            "iv_term_structure_computed",
            F.col("atm_iv_near") - F.col("atm_iv_far")
        ).withColumn(
            "put_call_iv_ratio",
            F.col("put_iv_avg") /
            (F.col("call_iv_avg") + F.lit(1e-8))
        ).withColumn(
            "put_call_oi_ratio",
            F.col("put_oi") /
            (F.col("call_oi") + F.lit(1.0))
        ).withColumn(
            "iv_oi_weighted",
            F.col("iv_oi_weighted_sum") /
            (F.col("total_oi") + F.lit(1.0))
        ).withColumn(
            "iv_smirk",
            F.col("iv_otm_put") - F.col("iv_otm_call")
        ).drop("iv_oi_weighted_sum")

        count = atm.count()
        print(f"  ATM features rows : {count:,}")
        return atm

    # ------------------------------------------------------------------ #
    #  Step 3 — Cross-sectional IV features
    # ------------------------------------------------------------------ #
    def add_cross_sectional(self, atm):
        print("\nStep 3: Cross-sectional IV features...")

        w_date = Window.partitionBy("date")

        cs_factors = [
            "atm_iv",
            "iv_skew_computed",
            "iv_term_structure_computed",
            "put_call_iv_ratio",
            "put_call_oi_ratio",
            "avg_iv",
            "iv_spread",
            "iv_smirk",
            "avg_bid_ask_spread",
        ]

        for factor in cs_factors:
            if factor not in atm.columns:
                continue
            # CS rank (0-1)
            atm = atm.withColumn(
                f"{factor}_rank",
                F.percent_rank().over(
                    w_date.orderBy(factor)
                )
            )
            # CS z-score
            atm = atm.withColumn(
                f"{factor}_zscore",
                (F.col(factor) -
                 F.mean(factor).over(w_date)) /
                (F.stddev(factor).over(w_date) +
                 F.lit(1e-8))
            )

        print("  CS IV features added ✓")
        return atm

    # ------------------------------------------------------------------ #
    #  Step 4 — Join realized vol (from OHLCV)
    # ------------------------------------------------------------------ #
    def add_rv_iv_spread(self, atm):
        print("\nStep 4: Joining realized vol...")

        # Load latest realized vol per ticker
        ohlcv = self.spark.read.format("delta").load(
            f"{self.silver_path}/ohlcv"
        )

        # Get latest row per ticker
        w_last = Window.partitionBy("ticker").orderBy(
            F.desc("date")
        )
        ohlcv_latest = ohlcv.withColumn(
            "rn", F.row_number().over(w_last)
        ).filter(F.col("rn") == 1).drop("rn") \
         .select(
             "ticker",
             F.col("vol_21d").alias("rv_21d"),
             F.col("vol_63d").alias("rv_63d"),
         )

        # Join to atm features
        atm = atm.join(
            ohlcv_latest, on="ticker", how="left"
        )

        # Vol risk premium
        atm = atm.withColumn(
            "rv_iv_spread",
            F.col("atm_iv") - F.col("rv_21d")
        ).withColumn(
            "vol_risk_premium",
            (F.col("atm_iv") - F.col("rv_21d")) /
            (F.col("atm_iv") + F.lit(1e-8))
        ).withColumn(
            "iv_rv_ratio",
            F.col("atm_iv") /
            (F.col("rv_21d") + F.lit(1e-8))
        )

        print("  RV-IV spread added ✓")
        return atm

    # ------------------------------------------------------------------ #
    #  Step 5 — CBOE features
    # ------------------------------------------------------------------ #
    def add_cboe_features(self, atm):
        print("\nStep 5: Adding CBOE features...")

        try:
            cboe = self.spark.read.format("delta").load(
                f"{BASE_PATH}/bronze/delta/cboe_iv"
            ).withColumn(
                "date", F.to_date(F.col("date"))
            )

            # Get latest CBOE values
            w_last = Window.partitionBy(
                "index_name"
            ).orderBy(F.desc("date"))
            cboe_latest = cboe.withColumn(
                "rn", F.row_number().over(w_last)
            ).filter(F.col("rn") == 1).drop("rn")

            # Pivot to wide
            cboe_vals = {}
            for row in cboe_latest.select(
                "index_name","close"
            ).collect():
                cboe_vals[row["index_name"]] = float(
                    row["close"]
                )

            print(f"  CBOE values : {cboe_vals}")

            # Add as constants
            for idx, val in cboe_vals.items():
                safe_name = idx.replace("/","_")
                atm = atm.withColumn(
                    f"cboe_{safe_name}",
                    F.lit(float(val))
                )

            # Derived CBOE features
            if "cboe_VIX" in atm.columns and \
               "cboe_VIX3M" in atm.columns:
                atm = atm.withColumn(
                    "vix_term_structure",
                    F.col("cboe_VIX3M") -
                    F.col("cboe_VIX")
                )

            if "cboe_VIX" in atm.columns and \
               "cboe_VIX9D" in atm.columns:
                atm = atm.withColumn(
                    "vix_9d_1m_spread",
                    F.col("cboe_VIX") -
                    F.col("cboe_VIX9D")
                )

            if "cboe_VIX3M" in atm.columns and \
               "cboe_VIX6M" in atm.columns:
                atm = atm.withColumn(
                    "vix_3m_6m_spread",
                    F.col("cboe_VIX6M") -
                    F.col("cboe_VIX3M")
                )

            # Ratio of stock IV to market IV
            if "cboe_VIX" in atm.columns:
                atm = atm.withColumn(
                    "iv_to_vix_ratio",
                    F.col("atm_iv") /
                    (F.col("cboe_VIX") / F.lit(100.0) +
                     F.lit(1e-8))
                )

            print("  CBOE features added ✓")
        except Exception as e:
            print(f"  CBOE skipped: {e}")

        return atm

    # ------------------------------------------------------------------ #
    #  Step 6 — Add regime labels
    # ------------------------------------------------------------------ #
    def add_regime(self, atm):
        print("\nStep 6: Adding regime labels...")

        try:
            # Get latest regime
            regimes = self.spark.read.format("delta").load(
                f"{self.eda_path}/regime_analysis"
                f"/regime_labels"
            ).withColumn(
                "date", F.to_date(F.col("date"))
            ).orderBy(F.desc("date")).limit(1).select(
                "regime_label",
                "prob_bull",
                "prob_bear",
                "prob_highvol"
            ).collect()

            if len(regimes) > 0:
                r = regimes[0]
                atm = atm.withColumn(
                    "regime_label",
                    F.lit(str(r["regime_label"]))
                ).withColumn(
                    "prob_bull",
                    F.lit(float(r["prob_bull"] or 0.0))
                ).withColumn(
                    "prob_bear",
                    F.lit(float(r["prob_bear"] or 0.0))
                ).withColumn(
                    "prob_highvol",
                    F.lit(
                        float(r["prob_highvol"] or 0.0)
                    )
                )
                print(f"  Latest regime : "
                      f"{r['regime_label']} ✓")
            else:
                atm = atm.withColumn(
                    "regime_label", F.lit("Unknown")
                )
        except Exception as e:
            print(f"  Regime skipped: {e}")
            atm = atm.withColumn(
                "regime_label", F.lit("Unknown")
            )

        return atm

    # ------------------------------------------------------------------ #
    #  Step 7 — Metadata + partition cols
    # ------------------------------------------------------------------ #
    def add_metadata(self, atm):
        atm = atm.withColumn(
            "year",  F.year(F.col("date"))
        ).withColumn(
            "month", F.month(F.col("date"))
        ).withColumn(
            "gold_created_at",
            F.lit(
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )
        )
        return atm

    # ------------------------------------------------------------------ #
    #  Write
    # ------------------------------------------------------------------ #
    def write(self, df) -> None:
        print(f"\nWriting Gold Vol Surface...")
        print(f"  Path : {self.gold_path}")

        (df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema",                  "true")
            .option("delta.autoOptimize.optimizeWrite", "true")
            .option("delta.autoOptimize.autoCompact",   "true")
            .partitionBy("year","month")
            .save(self.gold_path)
        )

        self.spark.sql(
            f"OPTIMIZE delta.`{self.gold_path}`"
        )
        self.spark.conf.set(
            "spark.databricks.delta.retentionDuration"
            "Check.enabled", "false"
        )
        self.spark.sql(
            f"VACUUM delta.`{self.gold_path}` "
            f"RETAIN 168 HOURS"
        )

        details = self.spark.sql(
            f"DESCRIBE DETAIL delta.`{self.gold_path}`"
        ).select("numFiles","sizeInBytes").collect()[0]
        print(f"  Files : {details['numFiles']}")
        print(f"  Size  : "
              f"{details['sizeInBytes']/1e6:.1f} MB")
        print("  Write complete ✓")

    # ------------------------------------------------------------------ #
    #  Validate
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        print("\n" + "="*55)
        print("VALIDATION — Gold Vol Surface")
        print("="*55)

        df      = self.spark.read.format("delta").load(
            self.gold_path
        )
        total   = df.count()
        tickers = df.select("ticker").distinct().count()

        print(f"\n  Total rows  : {total:,}")
        print(f"  Tickers     : {tickers:,}")
        print(f"  Columns     : {len(df.columns):,}")

        dr = df.agg(
            F.min("date").alias("min"),
            F.max("date").alias("max")
        ).collect()[0]
        print(f"  Date range  : "
              f"{dr['min']} → {dr['max']}")

        print(f"\n  Sample (AAPL):")
        key_cols = [
            "date","ticker",
            "atm_iv","atm_iv_call","atm_iv_put",
            "iv_skew_computed",
            "iv_term_structure_computed",
            "put_call_iv_ratio",
            "vol_risk_premium",
            "iv_to_vix_ratio",
            "regime_label"
        ]
        avail = [
            c for c in key_cols if c in df.columns
        ]
        df.filter(F.col("ticker") == "AAPL") \
          .select(*avail) \
          .show(3)

        print(f"\n  Null check:")
        for col in [
            "atm_iv","iv_skew_computed",
            "vol_risk_premium","put_call_iv_ratio"
        ]:
            if col in df.columns:
                n = df.filter(
                    F.col(col).isNull()
                ).count()
                print(f"    {col:30}: "
                      f"{n:,} nulls "
                      f"({n/total*100:.1f}%)")

        print(f"\n  IV stats (cross-sectional):")
        df.select(
            F.mean("atm_iv").alias("mean_atm_iv"),
            F.min("atm_iv").alias("min_atm_iv"),
            F.max("atm_iv").alias("max_atm_iv"),
            F.mean("iv_skew_computed").alias(
                "mean_skew"
            ),
            F.mean("vol_risk_premium").alias(
                "mean_vrp"
            ),
        ).show()

        assert total > 0, "FAIL — empty table"
        print(f"\nValidation PASSED ✓")

    # ------------------------------------------------------------------ #
    #  Run
    # ------------------------------------------------------------------ #
    def run(self):
        print("="*55)
        print("Gold 02 — Vol Surface Pipeline")
        print("="*55)
        start = datetime.now()

        opts = self.load_options()
        atm  = self.build_atm_features(opts)
        atm  = self.add_cross_sectional(atm)
        atm  = self.add_rv_iv_spread(atm)
        atm  = self.add_cboe_features(atm)
        atm  = self.add_regime(atm)
        atm  = self.add_metadata(atm)

        self.write(atm)
        self.validate()

        elapsed = (
            datetime.now() - start
        ).seconds / 60
        print(f"\nTotal time : {elapsed:.1f} minutes")
        print("Gold 02 — Vol Surface COMPLETE ✓")
        return atm

# COMMAND ----------

class GoldVolSurfaceCharts:
    TEMPLATE = "plotly_dark"
    COLORS   = {
        "primary"  : "#2196F3",
        "secondary": "#FF5722",
        "success"  : "#4CAF50",
        "warning"  : "#FFC107",
        "purple"   : "#9C27B0",
        "teal"     : "#00BCD4",
    }

    def _load(self, spark, gold_path, filters=None):
        df = spark.read.format("delta").load(gold_path)
        if filters:
            for col, val in filters.items():
                df = df.filter(F.col(col) == val)
        return df.toPandas()

    def chart_iv_surface_3d(self, spark) -> None:
        """Chart 1 — 3D IV surface from silver options."""
        try:
            opts = spark.read.format("delta").load(
                f"{SILVER_PATH}/options"
            ).filter(
                F.col("ticker") == "AAPL"
            ).select(
                "moneyness","tte_days","implied_vol",
                "option_type"
            ).dropna().toPandas()

            if len(opts) < 10:
                print("  [Skipped] Insufficient data")
                return

            fig = make_subplots(
                rows=1, cols=2,
                specs=[
                    [{"type":"scatter3d"},
                     {"type":"scatter"}]
                ],
                subplot_titles=[
                    "AAPL IV Surface (3D)",
                    "IV vs Moneyness by Tenor"
                ]
            )

            for ot, color in [
                ("call", self.COLORS["success"]),
                ("put",  self.COLORS["secondary"])
            ]:
                mask = opts["option_type"] == ot
                fig.add_trace(go.Scatter3d(
                    x=opts[mask]["moneyness"],
                    y=opts[mask]["tte_days"],
                    z=opts[mask]["implied_vol"],
                    mode="markers",
                    marker=dict(
                        size=3, color=color,
                        opacity=0.6
                    ),
                    name=f"{ot.title()} IV"
                ), row=1, col=1)

            # 2D scatter by expiry bucket
            for tte_bucket, label, color in [
                ((0,  30),  "Near (≤30d)",
                 self.COLORS["secondary"]),
                ((31, 90),  "Mid (31-90d)",
                 self.COLORS["warning"]),
                ((91, 999), "Far (>90d)",
                 self.COLORS["success"]),
            ]:
                mask = opts["tte_days"].between(
                    tte_bucket[0], tte_bucket[1]
                )
                sub = opts[mask].sort_values("moneyness")
                fig.add_trace(go.Scatter(
                    x=sub["moneyness"],
                    y=sub["implied_vol"],
                    mode="markers",
                    name=label,
                    marker=dict(
                        color=color, size=4, opacity=0.6
                    )
                ), row=1, col=2)

            fig.update_layout(
                title="<b>Gold 02 — AAPL IV Surface</b>",
                template=self.TEMPLATE,
                height=600
            )
            fig.show()
        except Exception as e:
            print(f"  [Skipped] 3D surface: {e}")

    def chart_iv_overview(self, spark,
                           gold_path) -> None:
        """Chart 2 — IV overview across tickers."""
        pdf = self._load(spark, gold_path)
        if len(pdf) == 0:
            return

        pdf = pdf.sort_values("atm_iv", ascending=False)
        top50 = pdf.head(50)

        fig = go.Figure(go.Bar(
            x=top50["ticker"],
            y=top50["atm_iv"],
            marker=dict(
                color=top50["atm_iv"],
                colorscale="RdYlGn_r",
                colorbar=dict(title="ATM IV"),
                cmid=pdf["atm_iv"].median()
            ),
            text=top50["atm_iv"].round(3),
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "ATM IV: %{y:.3f}<br>"
                "Skew: %{customdata[0]:.3f}<br>"
                "VRP: %{customdata[1]:.3f}"
                "<extra></extra>"
            ),
            customdata=top50[[
                "iv_skew_computed","vol_risk_premium"
            ]].values
        ))

        fig.add_hline(
            y=pdf["atm_iv"].mean(),
            line_dash="dash",
            line_color="white", opacity=0.5,
            annotation_text=f"Mean="
                             f"{pdf['atm_iv'].mean():.3f}"
        )

        fig.update_layout(
            title="<b>Gold 02 — ATM IV by Ticker "
                  "(Top 50)</b>",
            template=self.TEMPLATE,
            height=500,
            xaxis_title="Ticker",
            yaxis_title="ATM Implied Vol"
        )
        fig.show()

    def chart_iv_skew_distribution(self, spark,
                                    gold_path) -> None:
        """Chart 3 — IV skew + put/call distributions."""
        pdf = self._load(spark, gold_path)
        if len(pdf) == 0:
            return

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "IV Skew Distribution (Put - Call)",
                "Put/Call IV Ratio Distribution"
            ]
        )

        skew = pdf["iv_skew_computed"].dropna()
        fig.add_trace(go.Histogram(
            x=skew, nbinsx=40,
            name="IV Skew",
            marker_color=self.COLORS["warning"],
            opacity=0.8,
            showlegend=False
        ), row=1, col=1)
        fig.add_vline(
            x=0, line_dash="dash",
            line_color="white", opacity=0.4,
            row=1, col=1
        )
        fig.add_vline(
            x=skew.mean(),
            line_color=self.COLORS["secondary"],
            line_width=2,
            annotation_text=f"Mean={skew.mean():.3f}",
            row=1, col=1
        )

        if "put_call_iv_ratio" in pdf.columns:
            pc = pdf["put_call_iv_ratio"].dropna()
            pc = pc.clip(0, 3)
            fig.add_trace(go.Histogram(
                x=pc, nbinsx=40,
                name="P/C IV Ratio",
                marker_color=self.COLORS["primary"],
                opacity=0.8,
                showlegend=False
            ), row=1, col=2)
            fig.add_vline(
                x=1, line_dash="dash",
                line_color="white", opacity=0.4,
                annotation_text="P/C=1",
                row=1, col=2
            )
            fig.add_vline(
                x=pc.mean(),
                line_color=self.COLORS["warning"],
                line_width=2,
                annotation_text=f"Mean={pc.mean():.2f}",
                row=1, col=2
            )

        fig.update_layout(
            title="<b>Gold 02 — IV Skew Analysis</b>",
            template=self.TEMPLATE,
            height=500
        )
        fig.show()

    def chart_vol_risk_premium(self, spark,
                                gold_path) -> None:
        """Chart 4 — Vol risk premium distribution."""
        pdf = self._load(spark, gold_path)
        if len(pdf) == 0:
            return

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Vol Risk Premium (IV - RV21d)",
                "ATM IV vs Realized Vol Scatter"
            ]
        )

        if "vol_risk_premium" in pdf.columns:
            vrp = pdf["vol_risk_premium"].dropna()
            vrp = vrp.clip(-1, 2)
            fig.add_trace(go.Histogram(
                x=vrp, nbinsx=50,
                name="VRP",
                marker_color=self.COLORS["teal"],
                opacity=0.8,
                showlegend=False
            ), row=1, col=1)
            fig.add_vline(
                x=0, line_dash="dash",
                line_color="white", opacity=0.4,
                row=1, col=1
            )
            fig.add_vline(
                x=vrp.mean(),
                line_color=self.COLORS["warning"],
                line_width=2,
                annotation_text=f"Mean={vrp.mean():.3f}",
                row=1, col=1
            )

        if "rv_21d" in pdf.columns and \
           "atm_iv" in pdf.columns:
            d = pdf[["ticker","atm_iv","rv_21d"]].dropna()
            fig.add_trace(go.Scatter(
                x=d["rv_21d"],
                y=d["atm_iv"],
                mode="markers",
                text=d["ticker"],
                marker=dict(
                    color=self.COLORS["primary"],
                    size=6, opacity=0.6
                ),
                showlegend=False,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "RV: %{x:.3f}<br>"
                    "IV: %{y:.3f}<extra></extra>"
                )
            ), row=1, col=2)

            # 45° line (IV = RV)
            max_val = max(
                d["rv_21d"].max(),
                d["atm_iv"].max()
            )
            fig.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                name="IV = RV",
                line=dict(
                    color="white",
                    dash="dash", width=1
                )
            ), row=1, col=2)

        fig.update_layout(
            title="<b>Gold 02 — Vol Risk Premium</b>",
            template=self.TEMPLATE,
            height=500
        )
        fig.update_xaxes(
            title_text="VRP", row=1, col=1
        )
        fig.update_xaxes(
            title_text="Realized Vol", row=1, col=2
        )
        fig.update_yaxes(
            title_text="Count", row=1, col=1
        )
        fig.update_yaxes(
            title_text="Implied Vol", row=1, col=2
        )
        fig.show()

    def chart_term_structure(self, spark,
                              gold_path) -> None:
        """Chart 5 — IV term structure by ticker."""
        pdf = self._load(spark, gold_path)
        if len(pdf) == 0:
            return

        if "atm_iv_near" not in pdf.columns or \
           "atm_iv_far" not in pdf.columns:
            print("  [Skipped] Term structure cols missing")
            return

        d = pdf[[
            "ticker","atm_iv_near",
            "atm_iv_mid","atm_iv_far",
            "iv_term_structure_computed"
        ]].dropna().sort_values(
            "iv_term_structure_computed",
            ascending=False
        ).head(40)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "IV by Tenor (Near/Mid/Far)",
                "IV Term Structure (Near-Far)"
            ]
        )

        for col, name, color in [
            ("atm_iv_near","Near (≤30d)",
             self.COLORS["secondary"]),
            ("atm_iv_mid", "Mid (31-90d)",
             self.COLORS["warning"]),
            ("atm_iv_far", "Far (>90d)",
             self.COLORS["success"]),
        ]:
            fig.add_trace(go.Bar(
                x=d["ticker"],
                y=d[col],
                name=name,
                opacity=0.8
            ), row=1, col=1)

        colors_ts = [
            self.COLORS["success"]
            if v > 0 else self.COLORS["secondary"]
            for v in d["iv_term_structure_computed"]
        ]
        fig.add_trace(go.Bar(
            x=d["ticker"],
            y=d["iv_term_structure_computed"],
            marker_color=colors_ts,
            name="Term Structure",
            showlegend=False
        ), row=1, col=2)

        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=1, col=2
        )

        fig.update_layout(
            title="<b>Gold 02 — IV Term Structure "
                  "by Ticker</b>",
            template=self.TEMPLATE,
            height=500,
            barmode="group"
        )
        fig.show()

    def chart_cboe_levels(self, spark,
                           gold_path) -> None:
        """Chart 6 — CBOE VIX term structure."""
        try:
            cboe = spark.read.format("delta").load(
                f"{BASE_PATH}/bronze/delta/cboe_iv"
            ).withColumn(
                "date", F.to_date(F.col("date"))
            ).orderBy("date").toPandas()

            if len(cboe) == 0:
                print("  [Skipped] No CBOE data")
                return

            cboe["date"] = pd.to_datetime(cboe["date"])
            wide = cboe.pivot_table(
                index="date",
                columns="index_name",
                values="close"
            ).reset_index()

            fig = go.Figure()
            colors = [
                self.COLORS["secondary"],
                self.COLORS["warning"],
                self.COLORS["success"],
                self.COLORS["primary"],
            ]
            vix_cols = [
                c for c in wide.columns if c != "date"
            ]
            for i, idx in enumerate(vix_cols):
                if idx not in wide.columns:
                    continue
                fig.add_trace(go.Scatter(
                    x=wide["date"],
                    y=wide[idx],
                    name=idx, mode="lines",
                    line=dict(
                        color=colors[i%len(colors)],
                        width=2
                    )
                ))

            # Latest values annotation
            latest = wide.dropna().tail(1)
            if len(latest) > 0:
                for idx in vix_cols:
                    if idx in latest.columns:
                        val = latest[idx].values[0]
                        fig.add_annotation(
                            x=latest["date"].values[0],
                            y=val,
                            text=f"{idx}={val:.1f}",
                            showarrow=True,
                            arrowhead=2,
                            font=dict(size=10)
                        )

            fig.update_layout(
                title="<b>Gold 02 — CBOE VIX "
                      "Term Structure History</b>",
                template=self.TEMPLATE,
                height=500,
                xaxis_title="Date",
                yaxis_title="VIX Level",
                hovermode="x unified"
            )
            fig.show()
        except Exception as e:
            print(f"  [Skipped] CBOE chart: {e}")

    def chart_iv_factor_correlation(self, spark,
                                     gold_path) -> None:
        """Chart 7 — IV factor correlation."""
        pdf = self._load(spark, gold_path)
        if len(pdf) == 0:
            return

        factor_cols = [
            "atm_iv","iv_skew_computed",
            "iv_term_structure_computed",
            "put_call_iv_ratio","vol_risk_premium",
            "iv_spread","iv_smirk",
            "put_call_oi_ratio",
        ]
        avail = [
            c for c in factor_cols if c in pdf.columns
        ]
        corr  = pdf[avail].dropna().corr()

        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdYlGn",
            zmid=0, zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorbar=dict(title="Corr")
        ))

        fig.update_layout(
            title="<b>Gold 02 — IV Feature "
                  "Correlation Matrix</b>",
            template=self.TEMPLATE,
            height=550,
            xaxis=dict(tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9))
        )
        fig.show()

    def run_all(self, spark, gold_path) -> None:
        print("\n" + "="*55)
        print("Generating Gold 02 Charts...")
        print("="*55)

        print("\n[1/7] IV Surface 3D...")
        self.chart_iv_surface_3d(spark)

        print("[2/7] IV Overview by Ticker...")
        self.chart_iv_overview(spark, gold_path)

        print("[3/7] IV Skew Distribution...")
        self.chart_iv_skew_distribution(spark, gold_path)

        print("[4/7] Vol Risk Premium...")
        self.chart_vol_risk_premium(spark, gold_path)

        print("[5/7] IV Term Structure...")
        self.chart_term_structure(spark, gold_path)

        print("[6/7] CBOE VIX History...")
        self.chart_cboe_levels(spark, gold_path)

        print("[7/7] IV Factor Correlation...")
        self.chart_iv_factor_correlation(
            spark, gold_path
        )

        print("\nAll 7 charts ✓")

# COMMAND ----------

pipeline = GoldVolSurface(
    spark       = spark,
    silver_path = SILVER_PATH,
    eda_path    = EDA_PATH,
    gold_path   = GOLD_PATH
)

atm = pipeline.run()

charts = GoldVolSurfaceCharts()
charts.run_all(
    spark     = spark,
    gold_path = f"{GOLD_PATH}/vol_surface"
)

print("\nGold 02 COMPLETE ✓")

# COMMAND ----------

df = spark.read.format("delta").load(
    f"{GOLD_PATH}/vol_surface"
)

print("="*55)
print("Gold 02 — Vol Surface Summary")
print("="*55)
print(f"Total rows  : {df.count():,}")
print(f"Tickers     : "
      f"{df.select('ticker').distinct().count():,}")
print(f"Columns     : {len(df.columns):,}")

print(f"\nAll columns:")
for i, c in enumerate(sorted(df.columns)):
    print(f"  {i+1:3}. {c}")

print(f"\nIV stats summary:")
df.select(
    F.mean("atm_iv").alias("mean_atm_iv"),
    F.mean("iv_skew_computed").alias("mean_skew"),
    F.mean("vol_risk_premium").alias("mean_vrp"),
    F.mean("put_call_iv_ratio").alias("mean_pc_ratio"),
    F.mean("iv_term_structure_computed").alias(
        "mean_term_struct"
    )
).show()

print(f"\nSample (AAPL):")
df.filter(F.col("ticker") == "AAPL") \
  .select(
    "ticker","atm_iv","atm_iv_call","atm_iv_put",
    "iv_skew_computed",
    "iv_term_structure_computed",
    "vol_risk_premium","put_call_iv_ratio",
    "regime_label"
  ).show(3)