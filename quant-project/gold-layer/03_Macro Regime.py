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

class GoldMacroRegime:
    """
    Gold 03 — Macro Regime Features.
    Sources: Silver macro, EDA regime labels, CBOE IV
    Output: gold/delta/macro_regime
    """

    def __init__(self, spark, silver_path,
                 eda_path, gold_path):
        self.spark       = spark
        self.silver_path = silver_path
        self.eda_path    = eda_path
        self.gold_path   = f"{gold_path}/macro_regime"
        print("GoldMacroRegime ✓")
        print(f"  Output : {self.gold_path}")

    # ------------------------------------------------------------------ #
    #  Step 1 — Load silver macro
    # ------------------------------------------------------------------ #
    def load_macro(self):
        print("\nStep 1: Loading silver macro...")
        start = datetime.now()

        macro = self.spark.read.format("delta").load(
            f"{self.silver_path}/macro"
        ).withColumn("date", F.to_date(F.col("date")))

        total   = macro.count()
        elapsed = (datetime.now() - start).seconds

        print(f"  Rows    : {total:,}")
        print(f"  Columns : {len(macro.columns)}")
        print(f"  Elapsed : {elapsed}s")

        dr = macro.agg(
            F.min("date").alias("min"),
            F.max("date").alias("max")
        ).collect()[0]
        print(f"  Dates   : {dr['min']} → {dr['max']}")
        return macro

    # ------------------------------------------------------------------ #
    #  Step 2 — Load EDA regime labels
    # ------------------------------------------------------------------ #
    def load_regime_labels(self):
        print("\nStep 2: Loading regime labels...")

        try:
            regimes = self.spark.read.format("delta").load(
                f"{self.eda_path}/regime_analysis/regime_labels"
            ).withColumn(
                "date", F.to_date(F.col("date"))
            ).select(
                "date","regime_label",
                "prob_bull","prob_bear",
                "prob_highvol","hmm_state"
            ).dropDuplicates(["date"])

            count = regimes.count()
            print(f"  Regime rows : {count:,}")
            regimes.groupBy("regime_label") \
                   .count() \
                   .orderBy("count", ascending=False) \
                   .show()
            return regimes
        except Exception as e:
            print(f"  Regime labels not found: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Step 3 — Rate features
    # ------------------------------------------------------------------ #
    def add_rate_features(self, df):
        print("\nStep 3: Rate features...")

        w     = Window.partitionBy(F.lit(1)).orderBy("date")
        w_21  = Window.partitionBy(F.lit(1)).orderBy("date").rowsBetween(-20, 0)
        w_63  = Window.partitionBy(F.lit(1)).orderBy("date").rowsBetween(-62, 0)
        w_252 = Window.partitionBy(F.lit(1)).orderBy("date").rowsBetween(-251, 0)

        # Use Kalman-smoothed if available
        rate_col  = "DGS10_kalman" if "DGS10_kalman" in df.columns else "DGS10"
        rate2_col = "DGS2_kalman"  if "DGS2_kalman"  in df.columns else "DGS2"
        fed_col   = "FEDFUNDS_kalman" if "FEDFUNDS_kalman" in df.columns else "FEDFUNDS"

        # Alias key columns
        if "DGS10" in df.columns:
            df = df.withColumn("rate_10y", F.col("DGS10"))
        if "DGS2" in df.columns:
            df = df.withColumn("rate_2y", F.col("DGS2"))
        if "FEDFUNDS" in df.columns:
            df = df.withColumn("fed_funds_rate", F.col("FEDFUNDS"))
        if "yield_spread_10y2y" in df.columns:
            df = df.withColumn("yield_curve", F.col("yield_spread_10y2y"))
        if "real_rate_10y" in df.columns:
            df = df.withColumn("real_rate", F.col("real_rate_10y"))
        if "cpi_yoy" in df.columns:
            df = df.withColumn("inflation", F.col("cpi_yoy"))

        # Rate momentum
        rate_momentum_map = [
            ("rate_10y",      rate_col),
            ("yield_curve",   "yield_spread_10y2y"),
            ("fed_funds_rate", fed_col),
        ]
        for alias, col in rate_momentum_map:
            if col not in df.columns:
                continue
            df = df.withColumn(
                f"{alias}_chg_1d",
                F.col(col) - F.lag(col, 1).over(w)
            ).withColumn(
                f"{alias}_chg_5d",
                F.col(col) - F.lag(col, 5).over(w)
            ).withColumn(
                f"{alias}_chg_21d",
                F.col(col) - F.lag(col, 21).over(w)
            )

        # Rolling z-scores
        zscore_cols = [
            "DGS10","DGS2","yield_spread_10y2y",
            "FEDFUNDS","BAMLH0A0HYM2"
        ]
        for col in zscore_cols:
            if col not in df.columns:
                continue
            df = df.withColumn(
                f"{col}_zscore_63d",
                (F.col(col) - F.mean(col).over(w_63)) /
                (F.stddev(col).over(w_63) + F.lit(1e-8))
            ).withColumn(
                f"{col}_zscore_252d",
                (F.col(col) - F.mean(col).over(w_252)) /
                (F.stddev(col).over(w_252) + F.lit(1e-8))
            )

        print("  Rate features added ✓")
        return df

    # ------------------------------------------------------------------ #
    #  Step 4 — VIX features
    # ------------------------------------------------------------------ #
    def add_vix_features(self, df):
        print("\nStep 4: VIX features...")

        if "VIXCLS" not in df.columns:
            print("  VIX not found — skipping")
            return df

        w     = Window.partitionBy(F.lit(1)).orderBy("date")
        w_21  = Window.partitionBy(F.lit(1)).orderBy("date").rowsBetween(-20, 0)
        w_63  = Window.partitionBy(F.lit(1)).orderBy("date").rowsBetween(-62, 0)
        w_252 = Window.partitionBy(F.lit(1)).orderBy("date").rowsBetween(-251, 0)

        df = df.withColumn("vix", F.col("VIXCLS"))

        # VIX momentum
        df = df.withColumn(
            "vix_chg_1d",
            F.col("VIXCLS") - F.lag("VIXCLS", 1).over(w)
        ).withColumn(
            "vix_chg_5d",
            F.col("VIXCLS") - F.lag("VIXCLS", 5).over(w)
        ).withColumn(
            "vix_chg_21d",
            F.col("VIXCLS") - F.lag("VIXCLS", 21).over(w)
        )

        # VIX rolling
        df = df.withColumn(
            "vix_mean_21d", F.mean("VIXCLS").over(w_21)
        ).withColumn(
            "vix_mean_63d", F.mean("VIXCLS").over(w_63)
        ).withColumn(
            "vix_std_21d", F.stddev("VIXCLS").over(w_21)
        )

        # VIX z-score
        df = df.withColumn(
            "vix_zscore_63d",
            (F.col("VIXCLS") - F.col("vix_mean_63d")) /
            (F.col("vix_std_21d") + F.lit(1e-8))
        )

        # VIX percentile (unbounded window — no rowsBetween)
        w_pct = Window.partitionBy(F.lit(1)).orderBy("VIXCLS")
        df = df.withColumn(
            "vix_pct_rank",
            F.percent_rank().over(w_pct)
        )

        # VIX regime
        if "vix_regime" not in df.columns:
            df = df.withColumn(
                "vix_regime",
                F.when(F.col("VIXCLS") < 15, F.lit(1))
                 .when(F.col("VIXCLS") < 20, F.lit(2))
                 .when(F.col("VIXCLS") < 30, F.lit(3))
                 .otherwise(F.lit(4))
            )

        # VIX above mean
        df = df.withColumn(
            "vix_above_mean",
            F.col("VIXCLS") - F.col("vix_mean_21d")
        )

        print("  VIX features added ✓")
        return df

    # ------------------------------------------------------------------ #
    #  Step 5 — Spread / FX / Oil features
    # ------------------------------------------------------------------ #
    def add_spread_features(self, df):
        print("\nStep 5: Spread/FX/Oil features...")

        w    = Window.partitionBy(F.lit(1)).orderBy("date")
        w_21 = Window.partitionBy(F.lit(1)).orderBy("date").rowsBetween(-20, 0)
        w_63 = Window.partitionBy(F.lit(1)).orderBy("date").rowsBetween(-62, 0)

        # HY spread
        if "BAMLH0A0HYM2" in df.columns:
            df = df.withColumn(
                "hy_spread", F.col("BAMLH0A0HYM2")
            ).withColumn(
                "hy_spread_chg_1d",
                F.col("BAMLH0A0HYM2") - F.lag("BAMLH0A0HYM2", 1).over(w)
            ).withColumn(
                "hy_spread_chg_5d",
                F.col("BAMLH0A0HYM2") - F.lag("BAMLH0A0HYM2", 5).over(w)
            ).withColumn(
                "hy_spread_mean_21d",
                F.mean("BAMLH0A0HYM2").over(w_21)
            ).withColumn(
                "hy_spread_zscore_63d",
                (F.col("BAMLH0A0HYM2") - F.mean("BAMLH0A0HYM2").over(w_63)) /
                (F.stddev("BAMLH0A0HYM2").over(w_63) + F.lit(1e-8))
            )

        # Oil
        if "DCOILWTICO" in df.columns:
            df = df.withColumn(
                "oil_price", F.col("DCOILWTICO")
            ).withColumn(
                "oil_chg_1d",
                F.col("DCOILWTICO") - F.lag("DCOILWTICO", 1).over(w)
            ).withColumn(
                "oil_chg_5d",
                F.col("DCOILWTICO") - F.lag("DCOILWTICO", 5).over(w)
            ).withColumn(
                "oil_zscore_63d",
                (F.col("DCOILWTICO") - F.mean("DCOILWTICO").over(w_63)) /
                (F.stddev("DCOILWTICO").over(w_63) + F.lit(1e-8))
            )

        # USD
        if "DTWEXBGS" in df.columns:
            df = df.withColumn(
                "usd_index", F.col("DTWEXBGS")
            ).withColumn(
                "usd_chg_1d",
                F.col("DTWEXBGS") - F.lag("DTWEXBGS", 1).over(w)
            ).withColumn(
                "usd_chg_5d",
                F.col("DTWEXBGS") - F.lag("DTWEXBGS", 5).over(w)
            ).withColumn(
                "usd_zscore_63d",
                (F.col("DTWEXBGS") - F.mean("DTWEXBGS").over(w_63)) /
                (F.stddev("DTWEXBGS").over(w_63) + F.lit(1e-8))
            )

        # Unemployment
        if "UNRATE" in df.columns:
            df = df.withColumn(
                "unemployment", F.col("UNRATE")
            ).withColumn(
                "unemployment_chg",
                F.col("UNRATE") - F.lag("UNRATE", 21).over(w)
            )

        print("  Spread/FX/Oil features added ✓")
        return df

    # ------------------------------------------------------------------ #
    #  Step 6 — Regime features
    # ------------------------------------------------------------------ #
    def add_regime_features(self, df, regimes):
        print("\nStep 6: Regime features...")

        if regimes is None:
            print("  No regime labels — adding defaults")
            df = df.withColumn("regime_label", F.lit("Unknown")) \
                   .withColumn("prob_bull",    F.lit(None).cast("double")) \
                   .withColumn("prob_bear",    F.lit(None).cast("double")) \
                   .withColumn("prob_highvol", F.lit(None).cast("double"))
            return df

        # Join
        df = df.join(regimes, on="date", how="left")
        df = df.withColumn(
            "regime_label",
            F.coalesce(F.col("regime_label"), F.lit("Unknown"))
        )

        # Binary indicators
        df = df.withColumn(
            "is_bull",
            F.when(F.col("regime_label") == "Bull", F.lit(1)).otherwise(F.lit(0))
        ).withColumn(
            "is_bear",
            F.when(F.col("regime_label") == "Bear", F.lit(1)).otherwise(F.lit(0))
        ).withColumn(
            "is_highvol",
            F.when(F.col("regime_label") == "HighVol", F.lit(1)).otherwise(F.lit(0))
        )

        # Numeric regime
        df = df.withColumn(
            "regime_numeric",
            F.when(F.col("regime_label") == "Bull",    F.lit(0))
             .when(F.col("regime_label") == "Bear",    F.lit(1))
             .when(F.col("regime_label") == "HighVol", F.lit(2))
             .otherwise(F.lit(3))
        )

        # Regime change
        w = Window.partitionBy(F.lit(1)).orderBy("date")
        df = df.withColumn(
            "regime_changed",
            F.when(
                F.col("regime_label") != F.lag("regime_label", 1).over(w),
                F.lit(1)
            ).otherwise(F.lit(0))
        )

        # Position sizing weights
        df = df.withColumn(
            "position_size_weight",
            F.when(F.col("regime_label") == "Bull",    F.lit(1.0))
             .when(F.col("regime_label") == "HighVol", F.lit(0.6))
             .when(F.col("regime_label") == "Bear",    F.lit(0.3))
             .otherwise(F.lit(0.5))
        )

        # Regime confidence
        df = df.withColumn(
            "regime_confidence",
            F.greatest(
                F.coalesce(F.col("prob_bull"),    F.lit(0.0)),
                F.coalesce(F.col("prob_bear"),    F.lit(0.0)),
                F.coalesce(F.col("prob_highvol"), F.lit(0.0))
            )
        )

        count = df.filter(F.col("regime_label") != "Unknown").count()
        print(f"  Regime rows joined : {count:,}")
        print("  Regime features added ✓")
        return df

    # ------------------------------------------------------------------ #
    #  Step 7 — Composite indicators
    # ------------------------------------------------------------------ #
    def add_composite_indicators(self, df):
        print("\nStep 7: Composite indicators...")

        w = Window.partitionBy(F.lit(1)).orderBy("date")

        # Risk-on score components
        risk_on_components = []

        if "vix_zscore_63d" in df.columns:
            risk_on_components.append(
                -F.col("vix_zscore_63d") / F.lit(10.0)
            )
        elif "VIXCLS" in df.columns:
            risk_on_components.append(
                -F.col("VIXCLS") / F.lit(100.0)
            )

        if "hy_spread_zscore_63d" in df.columns:
            risk_on_components.append(
                -F.col("hy_spread_zscore_63d") / F.lit(10.0)
            )
        elif "BAMLH0A0HYM2" in df.columns:
            risk_on_components.append(
                -F.col("BAMLH0A0HYM2") / F.lit(20.0)
            )

        if "yield_spread_10y2y" in df.columns:
            risk_on_components.append(
                F.col("yield_spread_10y2y") / F.lit(5.0)
            )

        if len(risk_on_components) > 0:
            n     = float(len(risk_on_components))
            score = risk_on_components[0]
            for c in risk_on_components[1:]:
                score = score + c
            df = df.withColumn(
                "risk_on_score", score / F.lit(n)
            )

        # Financial stress
        stress_components = []
        if "VIXCLS" in df.columns:
            stress_components.append(F.col("VIXCLS") / F.lit(100.0))
        if "BAMLH0A0HYM2" in df.columns:
            stress_components.append(F.col("BAMLH0A0HYM2") / F.lit(20.0))

        if len(stress_components) > 0:
            n      = float(len(stress_components))
            stress = stress_components[0]
            for c in stress_components[1:]:
                stress = stress + c
            df = df.withColumn(
                "financial_stress", stress / F.lit(n)
            )

        # Yield curve inversion
        if "yield_spread_10y2y" in df.columns:
            df = df.withColumn(
                "yield_curve_inverted",
                F.when(F.col("yield_spread_10y2y") < 0, F.lit(1))
                 .otherwise(F.lit(0))
            )

        # Rate hike signal
        if "FEDFUNDS" in df.columns:
            df = df.withColumn(
                "rate_hike_signal",
                F.when(
                    F.col("FEDFUNDS") > F.lag("FEDFUNDS", 21).over(w),
                    F.lit(1)
                ).when(
                    F.col("FEDFUNDS") < F.lag("FEDFUNDS", 21).over(w),
                    F.lit(-1)
                ).otherwise(F.lit(0))
            )

        print("  Composite indicators added ✓")
        return df

    # ------------------------------------------------------------------ #
    #  Step 8 — Metadata
    # ------------------------------------------------------------------ #
    def add_metadata(self, df):
        df = df.withColumn(
            "gold_created_at",
            F.lit(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        if "year" not in df.columns:
            df = df.withColumn("year", F.year(F.col("date")))
        if "month" not in df.columns:
            df = df.withColumn("month", F.month(F.col("date")))
        return df

    # ------------------------------------------------------------------ #
    #  Write
    # ------------------------------------------------------------------ #
    def write(self, df) -> None:
        print(f"\nWriting Gold Macro Regime...")
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

        self.spark.sql(f"OPTIMIZE delta.`{self.gold_path}`")
        self.spark.conf.set(
            "spark.databricks.delta.retentionDurationCheck.enabled",
            "false"
        )
        self.spark.sql(
            f"VACUUM delta.`{self.gold_path}` RETAIN 168 HOURS"
        )

        details = self.spark.sql(
            f"DESCRIBE DETAIL delta.`{self.gold_path}`"
        ).select("numFiles","sizeInBytes").collect()[0]
        print(f"  Files : {details['numFiles']}")
        print(f"  Size  : {details['sizeInBytes']/1e6:.1f} MB")
        print("  Write complete ✓")

    # ------------------------------------------------------------------ #
    #  Validate
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        print("\n" + "="*55)
        print("VALIDATION — Gold Macro Regime")
        print("="*55)

        df    = self.spark.read.format("delta").load(self.gold_path)
        total = df.count()

        print(f"\n  Total rows  : {total:,}")
        print(f"  Columns     : {len(df.columns):,}")

        dr = df.agg(
            F.min("date").alias("min"),
            F.max("date").alias("max")
        ).collect()[0]
        print(f"  Date range  : {dr['min']} → {dr['max']}")

        if "regime_label" in df.columns:
            print(f"\n  Regime distribution:")
            df.groupBy("regime_label").count() \
              .orderBy("count", ascending=False).show()

        key_cols = [
            "date","VIXCLS","DGS10","yield_spread_10y2y",
            "BAMLH0A0HYM2","regime_label",
            "position_size_weight","risk_on_score",
            "financial_stress"
        ]
        avail = [c for c in key_cols if c in df.columns]
        print(f"\n  Latest values:")
        df.orderBy(F.desc("date")).select(*avail).show(5)

        print(f"\n  Null check:")
        for col in [
            "VIXCLS","DGS10","yield_spread_10y2y",
            "regime_label","risk_on_score"
        ]:
            if col in df.columns:
                n = df.filter(F.col(col).isNull()).count()
                print(f"    {col:28}: "
                      f"{n:,} nulls ({n/total*100:.1f}%)")

        assert total > 0, "FAIL — empty"
        print(f"\nValidation PASSED ✓")

    # ------------------------------------------------------------------ #
    #  Run
    # ------------------------------------------------------------------ #
    def run(self):
        print("="*55)
        print("Gold 03 — Macro Regime Pipeline")
        print("="*55)
        start = datetime.now()

        macro   = self.load_macro()
        regimes = self.load_regime_labels()
        macro   = self.add_rate_features(macro)
        macro   = self.add_vix_features(macro)
        macro   = self.add_spread_features(macro)
        macro   = self.add_regime_features(macro, regimes)
        macro   = self.add_composite_indicators(macro)
        macro   = self.add_metadata(macro)

        self.write(macro)
        self.validate()

        elapsed = (datetime.now() - start).seconds / 60
        print(f"\nTotal time : {elapsed:.1f} minutes")
        print("Gold 03 — Macro Regime COMPLETE ✓")
        return macro

# COMMAND ----------

class GoldMacroRegimeCharts:
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

    def _load_pdf(self, spark, gold_path):
        return spark.read.format("delta") \
                    .load(gold_path).toPandas()

    def chart_macro_dashboard(self, spark,
                               gold_path) -> None:
        pdf = self._load_pdf(spark, gold_path)
        pdf["date"] = pd.to_datetime(pdf["date"])
        pdf = pdf.sort_values("date")

        fig = make_subplots(
            rows=3, cols=2,
            shared_xaxes=True,
            subplot_titles=[
                "10Y & 2Y Treasury Yields",
                "Yield Curve Spread (10Y-2Y)",
                "VIX Level",
                "HY Credit Spread",
                "Oil Price (WTI)",
                "USD Index",
            ],
            vertical_spacing=0.08
        )

        for col, name, color in [
            ("DGS10","10Y", self.COLORS["secondary"]),
            ("DGS2", "2Y",  self.COLORS["success"]),
        ]:
            if col not in pdf.columns:
                continue
            fig.add_trace(go.Scatter(
                x=pdf["date"], y=pdf[col],
                name=name, mode="lines",
                line=dict(color=color, width=1.5)
            ), row=1, col=1)

        if "yield_spread_10y2y" in pdf.columns:
            yc = pdf["yield_spread_10y2y"]
            bar_colors = [
                self.COLORS["success"] if v > 0
                else self.COLORS["secondary"]
                for v in yc
            ]
            fig.add_trace(go.Bar(
                x=pdf["date"], y=yc,
                marker_color=bar_colors,
                name="Yield Curve",
                showlegend=False
            ), row=1, col=2)
            fig.add_hline(
                y=0, line_dash="dash",
                line_color="white", opacity=0.3,
                row=1, col=2
            )

        if "VIXCLS" in pdf.columns:
            fig.add_trace(go.Scatter(
                x=pdf["date"], y=pdf["VIXCLS"],
                name="VIX",
                line=dict(
                    color=self.COLORS["secondary"],
                    width=1.5
                ),
                fill="tozeroy",
                fillcolor="rgba(255,87,34,0.15)",
                showlegend=False
            ), row=2, col=1)
            for y_val, color, label in [
                (20,"yellow","VIX=20"),
                (30,"red",   "VIX=30"),
            ]:
                fig.add_hline(
                    y=y_val, line_dash="dash",
                    line_color=color, opacity=0.5,
                    annotation_text=label,
                    row=2, col=1
                )

        if "BAMLH0A0HYM2" in pdf.columns:
            fig.add_trace(go.Scatter(
                x=pdf["date"],
                y=pdf["BAMLH0A0HYM2"],
                name="HY Spread",
                line=dict(
                    color=self.COLORS["warning"],
                    width=1.5
                ),
                fill="tozeroy",
                fillcolor="rgba(255,193,7,0.15)",
                showlegend=False
            ), row=2, col=2)

        if "DCOILWTICO" in pdf.columns:
            fig.add_trace(go.Scatter(
                x=pdf["date"],
                y=pdf["DCOILWTICO"],
                name="WTI Oil",
                line=dict(
                    color=self.COLORS["teal"], width=1.5
                ),
                showlegend=False
            ), row=3, col=1)

        if "DTWEXBGS" in pdf.columns:
            fig.add_trace(go.Scatter(
                x=pdf["date"],
                y=pdf["DTWEXBGS"],
                name="USD Index",
                line=dict(
                    color=self.COLORS["purple"], width=1.5
                ),
                showlegend=False
            ), row=3, col=2)

        fig.update_layout(
            title="<b>Gold 03 — Macro Dashboard</b>",
            template=self.TEMPLATE,
            height=900,
            hovermode="x unified"
        )
        fig.show()

    def chart_regime_timeline(self, spark,
                               gold_path) -> None:
        pdf = self._load_pdf(spark, gold_path)
        if "regime_label" not in pdf.columns:
            return

        pdf["date"] = pd.to_datetime(pdf["date"])
        pdf = pdf.sort_values("date").dropna(
            subset=["regime_label"]
        )

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=[
                "HMM Regime Classification",
                "Regime Probabilities",
                "Position Size Weight"
            ],
            vertical_spacing=0.08,
            row_heights=[0.25, 0.45, 0.30]
        )

        for regime, color in self.REGIME_COLORS.items():
            mask = pdf["regime_label"] == regime
            if not mask.any():
                continue
            fig.add_trace(go.Scatter(
                x=pdf[mask]["date"],
                y=[regime] * mask.sum(),
                mode="markers",
                marker=dict(
                    color=color, size=4,
                    symbol="square"
                ),
                name=regime
            ), row=1, col=1)

        for col, label, color in [
            ("prob_bull",    "P(Bull)",
             self.REGIME_COLORS["Bull"]),
            ("prob_bear",    "P(Bear)",
             self.REGIME_COLORS["Bear"]),
            ("prob_highvol", "P(HighVol)",
             self.REGIME_COLORS["HighVol"]),
        ]:
            if col not in pdf.columns:
                continue
            fig.add_trace(go.Scatter(
                x=pdf["date"], y=pdf[col],
                name=label, mode="lines",
                line=dict(color=color, width=1),
                showlegend=False
            ), row=2, col=1)

        if "position_size_weight" in pdf.columns:
            bar_colors = [
                self.REGIME_COLORS.get(r, "#9E9E9E")
                for r in pdf["regime_label"]
            ]
            fig.add_trace(go.Bar(
                x=pdf["date"],
                y=pdf["position_size_weight"],
                marker_color=bar_colors,
                name="Position Weight",
                showlegend=False
            ), row=3, col=1)

        fig.update_layout(
            title="<b>Gold 03 — HMM Regime Timeline</b>",
            template=self.TEMPLATE,
            height=750,
            hovermode="x unified"
        )
        fig.update_yaxes(title_text="Regime",  row=1, col=1)
        fig.update_yaxes(title_text="Prob", range=[0,1], row=2, col=1)
        fig.update_yaxes(title_text="Weight", row=3, col=1)
        fig.show()

    def chart_regime_macro_profile(self, spark,
                                    gold_path) -> None:
        pdf = self._load_pdf(spark, gold_path)
        if "regime_label" not in pdf.columns:
            return

        factors = [
            "VIXCLS","yield_spread_10y2y",
            "BAMLH0A0HYM2","DGS10","DCOILWTICO"
        ]
        avail = [f for f in factors if f in pdf.columns]
        if len(avail) == 0:
            return

        profile = pdf.groupby("regime_label")[avail].mean()

        fig = go.Figure()
        for regime, color in self.REGIME_COLORS.items():
            if regime not in profile.index:
                continue
            vals   = profile.loc[regime, avail].values
            min_v  = profile[avail].min()
            max_v  = profile[avail].max()
            norm   = (vals - min_v) / (max_v - min_v + 1e-8)
            r_list = norm.tolist() + [norm[0]]
            t_list = avail + [avail[0]]
            fig.add_trace(go.Scatterpolar(
                r=r_list, theta=t_list,
                fill="toself", name=regime,
                line=dict(color=color)
            ))

        fig.update_layout(
            title="<b>Gold 03 — Macro Profile by Regime</b>",
            template=self.TEMPLATE,
            height=550,
            polar=dict(radialaxis=dict(visible=True, range=[0,1]))
        )
        fig.show()

    def chart_risk_on_score(self, spark,
                             gold_path) -> None:
        pdf = self._load_pdf(spark, gold_path)
        if "risk_on_score" not in pdf.columns:
            return

        pdf["date"] = pd.to_datetime(pdf["date"])
        pdf = pdf.sort_values("date").dropna(
            subset=["risk_on_score"]
        )

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=[
                "Risk-On/Off Score",
                "Financial Stress Indicator"
            ],
            vertical_spacing=0.1
        )

        bar_colors = [
            self.COLORS["success"] if v > 0
            else self.COLORS["secondary"]
            for v in pdf["risk_on_score"]
        ]
        fig.add_trace(go.Bar(
            x=pdf["date"],
            y=pdf["risk_on_score"],
            marker_color=bar_colors,
            name="Risk-On Score",
            showlegend=False
        ), row=1, col=1)
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=1, col=1
        )

        if "financial_stress" in pdf.columns:
            fig.add_trace(go.Scatter(
                x=pdf["date"],
                y=pdf["financial_stress"],
                name="Stress",
                line=dict(
                    color=self.COLORS["secondary"],
                    width=1.5
                ),
                fill="tozeroy",
                fillcolor="rgba(255,87,34,0.15)",
                showlegend=False
            ), row=2, col=1)

        fig.update_layout(
            title="<b>Gold 03 — Risk-On/Off & "
                  "Financial Stress</b>",
            template=self.TEMPLATE,
            height=600,
            hovermode="x unified"
        )
        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_yaxes(title_text="Stress", row=2, col=1)
        fig.show()

    def chart_rate_environment(self, spark,
                                gold_path) -> None:
        pdf = self._load_pdf(spark, gold_path)
        pdf["date"] = pd.to_datetime(pdf["date"])
        pdf = pdf.sort_values("date")

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Fed Funds Rate",
                "Real Rate (10Y - CPI)",
                "Yield Curve Regime",
                "Rate Hike Signal"
            ]
        )

        if "FEDFUNDS" in pdf.columns:
            fig.add_trace(go.Scatter(
                x=pdf["date"], y=pdf["FEDFUNDS"],
                line=dict(
                    color=self.COLORS["primary"],
                    width=1.5
                ),
                name="Fed Funds",
                showlegend=False
            ), row=1, col=1)

        if "real_rate_10y" in pdf.columns:
            real = pdf["real_rate_10y"].dropna()
            dates_real = pdf["date"].iloc[-len(real):]
            bar_colors = [
                self.COLORS["secondary"] if v < 0
                else self.COLORS["success"]
                for v in real
            ]
            fig.add_trace(go.Bar(
                x=dates_real, y=real,
                marker_color=bar_colors,
                name="Real Rate",
                showlegend=False
            ), row=1, col=2)
            fig.add_hline(
                y=0, line_dash="dash",
                line_color="white", opacity=0.3,
                row=1, col=2
            )

        if "yield_curve_inverted" in pdf.columns:
            fig.add_trace(go.Scatter(
                x=pdf["date"],
                y=pdf["yield_curve_inverted"],
                mode="lines",
                line=dict(
                    color=self.COLORS["secondary"],
                    width=1.5
                ),
                fill="tozeroy",
                fillcolor="rgba(255,87,34,0.3)",
                name="Inverted",
                showlegend=False
            ), row=2, col=1)

        if "rate_hike_signal" in pdf.columns:
            sig = pdf["rate_hike_signal"].dropna()
            dates_sig = pdf["date"].iloc[-len(sig):]
            bar_colors = [
                self.COLORS["secondary"] if v > 0
                else self.COLORS["success"] if v < 0
                else "#9E9E9E"
                for v in sig
            ]
            fig.add_trace(go.Bar(
                x=dates_sig, y=sig,
                marker_color=bar_colors,
                name="Hike Signal",
                showlegend=False
            ), row=2, col=2)
            fig.add_hline(
                y=0, line_dash="dash",
                line_color="white", opacity=0.3,
                row=2, col=2
            )

        fig.update_layout(
            title="<b>Gold 03 — Rate Environment</b>",
            template=self.TEMPLATE,
            height=650
        )
        fig.show()

    def chart_macro_correlations(self, spark,
                                  gold_path) -> None:
        pdf = self._load_pdf(spark, gold_path)

        factor_cols = [
            "VIXCLS","DGS10","DGS2",
            "yield_spread_10y2y","BAMLH0A0HYM2",
            "DCOILWTICO","DTWEXBGS","FEDFUNDS",
            "risk_on_score","financial_stress",
        ]
        avail = [c for c in factor_cols if c in pdf.columns]
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
            title="<b>Gold 03 — Macro Feature "
                  "Correlation Matrix</b>",
            template=self.TEMPLATE,
            height=600
        )
        fig.show()

    def chart_vix_analysis(self, spark,
                            gold_path) -> None:
        pdf = self._load_pdf(spark, gold_path)
        if "VIXCLS" not in pdf.columns:
            return

        pdf["date"] = pd.to_datetime(pdf["date"])
        pdf = pdf.sort_values("date").dropna(
            subset=["VIXCLS"]
        )

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "VIX Distribution by Regime",
                "VIX Percentile Over Time"
            ]
        )

        # Left — VIX violin by regime
        if "regime_label" in pdf.columns:
            for regime, color in self.REGIME_COLORS.items():
                mask = pdf["regime_label"] == regime
                if not mask.any():
                    continue
                fig.add_trace(go.Violin(
                    y=pdf[mask]["VIXCLS"],
                    name=regime,
                    fillcolor=color,
                    line_color="white",
                    opacity=0.7,
                    box_visible=True,
                    meanline_visible=True
                ), row=1, col=1)

        # Right — find percentile col
        pct_col = next(
            (c for c in [
                "vix_pct_rank","vix_pct_252d",
                "vix_pct","VIXCLS_pct"
            ] if c in pdf.columns),
            None
        )

        if pct_col:
            valid = pdf.dropna(subset=[pct_col])
            fig.add_trace(go.Scatter(
                x=valid["date"],
                y=valid[pct_col],
                mode="lines",
                line=dict(
                    color=self.COLORS["primary"],
                    width=1.5
                ),
                fill="tozeroy",
                fillcolor="rgba(33,150,243,0.15)",
                name="VIX Percentile",
                showlegend=False
            ), row=1, col=2)
            for y_val, color, label in [
                (0.75, "yellow", "75th pct"),
                (0.90, "red",    "90th pct"),
            ]:
                fig.add_hline(
                    y=y_val, line_dash="dash",
                    line_color=color, opacity=0.6,
                    annotation_text=label,
                    row=1, col=2
                )
            fig.update_yaxes(
                title_text="VIX Percentile",
                range=[0, 1], row=1, col=2
            )
        else:
            # Fallback — VIX level over time
            fig.add_trace(go.Scatter(
                x=pdf["date"],
                y=pdf["VIXCLS"],
                mode="lines",
                line=dict(
                    color=self.COLORS["secondary"],
                    width=1.5
                ),
                fill="tozeroy",
                fillcolor="rgba(255,87,34,0.15)",
                name="VIX Level",
                showlegend=False
            ), row=1, col=2)
            for y_val, color, label in [
                (20, "yellow", "VIX=20"),
                (30, "red",    "VIX=30"),
            ]:
                fig.add_hline(
                    y=y_val, line_dash="dash",
                    line_color=color, opacity=0.5,
                    annotation_text=label,
                    row=1, col=2
                )
            fig.update_yaxes(
                title_text="VIX Level",
                row=1, col=2
            )

        fig.update_layout(
            title="<b>Gold 03 — VIX Analysis</b>",
            template=self.TEMPLATE,
            height=550
        )
        fig.update_yaxes(title_text="VIX", row=1, col=1)
        fig.show()

    def run_all(self, spark, gold_path) -> None:
        print("\n" + "="*55)
        print("Generating Gold 03 Charts...")
        print("="*55)

        print("\n[1/7] Macro Dashboard...")
        self.chart_macro_dashboard(spark, gold_path)

        print("[2/7] Regime Timeline...")
        self.chart_regime_timeline(spark, gold_path)

        print("[3/7] Regime Macro Profile...")
        self.chart_regime_macro_profile(spark, gold_path)

        print("[4/7] Risk-On/Off Score...")
        self.chart_risk_on_score(spark, gold_path)

        print("[5/7] Rate Environment...")
        self.chart_rate_environment(spark, gold_path)

        print("[6/7] Macro Correlations...")
        self.chart_macro_correlations(spark, gold_path)

        print("[7/7] VIX Analysis...")
        self.chart_vix_analysis(spark, gold_path)

        print("\nAll 7 charts ✓")

# COMMAND ----------

pipeline = GoldMacroRegime(
    spark       = spark,
    silver_path = SILVER_PATH,
    eda_path    = EDA_PATH,
    gold_path   = GOLD_PATH
)

macro = pipeline.run()

charts = GoldMacroRegimeCharts()
charts.run_all(
    spark     = spark,
    gold_path = f"{GOLD_PATH}/macro_regime"
)

print("\nGold 03 COMPLETE ✓")

# COMMAND ----------

df = spark.read.format("delta").load(
    f"{GOLD_PATH}/macro_regime"
)

print("="*55)
print("Gold 03 — Macro Regime Summary")
print("="*55)
print(f"Total rows  : {df.count():,}")
print(f"Columns     : {len(df.columns):,}")

print(f"\nAll columns:")
for i, c in enumerate(sorted(df.columns)):
    print(f"  {i+1:3}. {c}")

print(f"\nRegime distribution:")
df.groupBy("regime_label").agg(
    F.count("*").alias("n_days"),
    F.mean("position_size_weight").alias(
        "avg_position_weight"
    )
).orderBy("n_days", ascending=False).show()

print(f"\nLatest macro values:")
df.orderBy(F.desc("date")).select(
    "date","VIXCLS","DGS10",
    "yield_spread_10y2y","BAMLH0A0HYM2",
    "regime_label","position_size_weight",
    "risk_on_score","financial_stress"
).show(5)