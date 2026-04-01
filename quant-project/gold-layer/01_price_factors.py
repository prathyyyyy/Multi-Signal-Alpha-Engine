# Databricks notebook source
# MAGIC %pip install scipy statsmodels==0.14.5 plotly pandas numpy scikit-learn --quiet

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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

class GoldPriceFactors:
    """
    Gold 01 — Price Factors.

    Builds the core alpha factor table used by
    LightGBM ranker and ensemble model.

    Factors built:
      Momentum   : 1d/5d/21d/63d/252d returns
      Reversal   : neg 1d/5d returns
      Volatility : 21d/63d realized vol (annualized)
      Volume     : dollar volume, volume rank
      Technical  : VWAP ratio, daily range,
                   price vs 20d/50d/200d MA
      Quality    : Sharpe (21d), Sortino (21d)
      Cross-sect : CS rank, CS z-score per factor

    Optimizations:
      - All features computed in Spark (no pandas loop)
      - Vectorized window functions
      - Single write pass
      - Photon-optimized partitioning
    """

    def __init__(self, spark, silver_path,
                 eda_path, gold_path):
        self.spark       = spark
        self.silver_path = f"{silver_path}/ohlcv"
        self.eda_path    = eda_path
        self.gold_path   = f"{gold_path}/price_factors"
        print("GoldPriceFactors ✓")
        print(f"  Input  : {self.silver_path}")
        print(f"  Output : {self.gold_path}")

    # ------------------------------------------------------------------ #
    #  Step 1 — Load silver OHLCV
    # ------------------------------------------------------------------ #
    def load(self):
        print("\nStep 1: Loading silver OHLCV...")
        start = datetime.now()

        df = self.spark.read.format("delta").load(
            self.silver_path
        )

        total  = df.count()
        tickers = df.select("ticker").distinct().count()

        elapsed = (datetime.now() - start).seconds
        print(f"  Rows    : {total:,}")
        print(f"  Tickers : {tickers:,}")
        print(f"  Elapsed : {elapsed}s")
        return df

    # ------------------------------------------------------------------ #
    #  Step 2 — Momentum factors
    # ------------------------------------------------------------------ #
    def add_momentum(self, df):
        print("\nStep 2: Momentum factors...")

        w = Window.partitionBy("ticker").orderBy("date")

        # Raw momentum
        for h, col in [
            (1,   "return_1d"),
            (5,   "return_5d"),
            (21,  "return_21d"),
            (63,  "return_63d"),
            (252, "return_252d"),
        ]:
            if col in df.columns:
                df = df.withColumn(f"mom_{h}d", F.col(col))
            else:
                df = df.withColumn(
                    f"mom_{h}d",
                    (F.col("close") -
                     F.lag("close", h).over(w)) /
                    F.lag("close", h).over(w)
                )

        # Reversal (sign flip)
        df = df.withColumn("rev_1d",  -F.col("mom_1d"))
        df = df.withColumn("rev_5d",  -F.col("mom_5d"))
        df = df.withColumn("rev_21d", -F.col("mom_21d"))

        # 52-week high ratio
        w_52 = Window.partitionBy("ticker") \
                     .orderBy("date") \
                     .rowsBetween(-252, 0)
        df = df.withColumn(
            "high_52w",
            F.max("close").over(w_52)
        ).withColumn(
            "price_to_52w_high",
            F.col("close") / F.col("high_52w")
        ).drop("high_52w")

        # 52-week low ratio
        df = df.withColumn(
            "low_52w",
            F.min("close").over(w_52)
        ).withColumn(
            "price_to_52w_low",
            F.col("close") / F.col("low_52w")
        ).drop("low_52w")

        # Momentum acceleration (mom change)
        df = df.withColumn(
            "mom_accel",
            F.col("mom_21d") - F.lag(
                "mom_21d", 21
            ).over(w)
        )

        print("  Momentum factors added ✓")
        return df

    # ------------------------------------------------------------------ #
    #  Step 3 — Volatility factors
    # ------------------------------------------------------------------ #
    def add_volatility(self, df):
        print("\nStep 3: Volatility factors...")

        w_21  = Window.partitionBy("ticker") \
                      .orderBy("date") \
                      .rowsBetween(-20, 0)
        w_63  = Window.partitionBy("ticker") \
                      .orderBy("date") \
                      .rowsBetween(-62, 0)
        w_252 = Window.partitionBy("ticker") \
                      .orderBy("date") \
                      .rowsBetween(-251, 0)
        w_lag = Window.partitionBy("ticker") \
                      .orderBy("date")

        # Realized vol (already in silver)
        # Add longer window if not present
        if "vol_21d" not in df.columns:
            df = df.withColumn(
                "vol_21d",
                F.stddev("log_return_1d").over(w_21) *
                F.sqrt(F.lit(252.0))
            )
        if "vol_63d" not in df.columns:
            df = df.withColumn(
                "vol_63d",
                F.stddev("log_return_1d").over(w_63) *
                F.sqrt(F.lit(252.0))
            )

        # 252d vol
        df = df.withColumn(
            "vol_252d",
            F.stddev("log_return_1d").over(w_252) *
            F.sqrt(F.lit(252.0))
        )

        # Vol ratio (short vs long)
        df = df.withColumn(
            "vol_ratio_21_63",
            F.col("vol_21d") /
            (F.col("vol_63d") + F.lit(1e-8))
        )

        # Vol of vol (vol regime)
        w_vol = Window.partitionBy("ticker") \
                      .orderBy("date") \
                      .rowsBetween(-20, 0)
        df = df.withColumn(
            "vol_of_vol",
            F.stddev("vol_21d").over(w_vol)
        )

        # Downside vol (sortino denominator)
        df = df.withColumn(
            "neg_return",
            F.when(
                F.col("log_return_1d") < 0,
                F.col("log_return_1d")
            ).otherwise(F.lit(0.0))
        )
        df = df.withColumn(
            "downside_vol_21d",
            F.stddev("neg_return").over(w_21) *
            F.sqrt(F.lit(252.0))
        ).drop("neg_return")

        # Vol change
        df = df.withColumn(
            "vol_change_1d",
            F.col("vol_21d") -
            F.lag("vol_21d", 1).over(w_lag)
        )

        print("  Volatility factors added ✓")
        return df

    # ------------------------------------------------------------------ #
    #  Step 4 — Volume factors
    # ------------------------------------------------------------------ #
    def add_volume(self, df):
        print("\nStep 4: Volume factors...")

        w_21  = Window.partitionBy("ticker") \
                      .orderBy("date") \
                      .rowsBetween(-20, 0)
        w_lag = Window.partitionBy("ticker") \
                      .orderBy("date")

        # Dollar volume (already in silver)
        if "dollar_volume" not in df.columns:
            df = df.withColumn(
                "dollar_volume",
                F.col("close") * F.col("volume")
            )

        # Rolling avg dollar volume
        df = df.withColumn(
            "avg_dolvol_21d",
            F.mean("dollar_volume").over(w_21)
        )

        # Volume ratio (today vs avg)
        df = df.withColumn(
            "volume_ratio",
            F.col("dollar_volume") /
            (F.col("avg_dolvol_21d") + F.lit(1e-8))
        )

        # Volume momentum
        df = df.withColumn(
            "volume_mom_5d",
            F.col("dollar_volume") /
            (F.lag("dollar_volume", 5).over(w_lag)
             + F.lit(1e-8))
        )

        # Amihud illiquidity (|return|/dollar_volume)
        df = df.withColumn(
            "amihud_illiquidity",
            F.abs(F.col("return_1d")) /
            (F.col("dollar_volume") + F.lit(1e-8))
        )

        # Rolling Amihud
        df = df.withColumn(
            "amihud_21d",
            F.mean("amihud_illiquidity").over(w_21)
        )

        print("  Volume factors added ✓")
        return df

    # ------------------------------------------------------------------ #
    #  Step 5 — Technical factors
    # ------------------------------------------------------------------ #
    def add_technical(self, df):
        print("\nStep 5: Technical factors...")

        w_lag  = Window.partitionBy("ticker") \
                       .orderBy("date")
        w_20   = Window.partitionBy("ticker") \
                       .orderBy("date") \
                       .rowsBetween(-19, 0)
        w_50   = Window.partitionBy("ticker") \
                       .orderBy("date") \
                       .rowsBetween(-49, 0)
        w_200  = Window.partitionBy("ticker") \
                       .orderBy("date") \
                       .rowsBetween(-199, 0)

        # VWAP ratio (already in silver)
        if "vwap" in df.columns:
            df = df.withColumn(
                "vwap_ratio",
                F.col("close") /
                (F.col("vwap") + F.lit(1e-8)) - 1
            )

        # Moving average ratios
        df = df.withColumn(
            "ma_20d",
            F.mean("close").over(w_20)
        ).withColumn(
            "price_to_ma20",
            F.col("close") /
            (F.col("ma_20d") + F.lit(1e-8)) - 1
        )

        df = df.withColumn(
            "ma_50d",
            F.mean("close").over(w_50)
        ).withColumn(
            "price_to_ma50",
            F.col("close") /
            (F.col("ma_50d") + F.lit(1e-8)) - 1
        )

        df = df.withColumn(
            "ma_200d",
            F.mean("close").over(w_200)
        ).withColumn(
            "price_to_ma200",
            F.col("close") /
            (F.col("ma_200d") + F.lit(1e-8)) - 1
        )

        # MA crossover signals
        df = df.withColumn(
            "ma_cross_20_50",
            F.col("ma_20d") - F.col("ma_50d")
        ).withColumn(
            "ma_cross_50_200",
            F.col("ma_50d") - F.col("ma_200d")
        )

        # Daily range (already in silver)
        if "daily_range" not in df.columns:
            df = df.withColumn(
                "daily_range",
                (F.col("high") - F.col("low")) /
                F.col("close")
            )

        # Gap (open vs prior close)
        df = df.withColumn(
            "gap",
            F.col("open") /
            (F.lag("close", 1).over(w_lag) +
             F.lit(1e-8)) - 1
        )

        # RSI (14-day simplified)
        df = df.withColumn(
            "gain",
            F.when(
                F.col("return_1d") > 0,
                F.col("return_1d")
            ).otherwise(F.lit(0.0))
        ).withColumn(
            "loss",
            F.when(
                F.col("return_1d") < 0,
                -F.col("return_1d")
            ).otherwise(F.lit(0.0))
        )

        w_14 = Window.partitionBy("ticker") \
                     .orderBy("date") \
                     .rowsBetween(-13, 0)
        df = df.withColumn(
            "avg_gain_14d",
            F.mean("gain").over(w_14)
        ).withColumn(
            "avg_loss_14d",
            F.mean("loss").over(w_14)
        ).withColumn(
            "rsi_14d",
            F.lit(100.0) - (
                F.lit(100.0) /
                (F.lit(1.0) + F.col("avg_gain_14d") /
                 (F.col("avg_loss_14d") + F.lit(1e-8)))
            )
        ).drop("gain","loss","avg_gain_14d","avg_loss_14d")

        print("  Technical factors added ✓")
        return df

    # ------------------------------------------------------------------ #
    #  Step 6 — Quality factors
    # ------------------------------------------------------------------ #
    def add_quality(self, df):
        print("\nStep 6: Quality factors...")

        w_21 = Window.partitionBy("ticker") \
                     .orderBy("date") \
                     .rowsBetween(-20, 0)
        w_63 = Window.partitionBy("ticker") \
                     .orderBy("date") \
                     .rowsBetween(-62, 0)

        # Sharpe (21d)
        df = df.withColumn(
            "sharpe_21d",
            F.mean("return_1d").over(w_21) *
            F.sqrt(F.lit(252.0)) /
            (F.stddev("return_1d").over(w_21) *
             F.sqrt(F.lit(252.0)) + F.lit(1e-8))
        )

        # Sharpe (63d)
        df = df.withColumn(
            "sharpe_63d",
            F.mean("return_1d").over(w_63) *
            F.sqrt(F.lit(252.0)) /
            (F.stddev("return_1d").over(w_63) *
             F.sqrt(F.lit(252.0)) + F.lit(1e-8))
        )

        # Sortino (21d)
        df = df.withColumn(
            "neg_ret_sq",
            F.when(
                F.col("return_1d") < 0,
                F.col("return_1d") * F.col("return_1d")
            ).otherwise(F.lit(0.0))
        )
        df = df.withColumn(
            "sortino_21d",
            F.mean("return_1d").over(w_21) *
            F.sqrt(F.lit(252.0)) /
            (F.sqrt(F.mean("neg_ret_sq").over(w_21)) *
             F.sqrt(F.lit(252.0)) + F.lit(1e-8))
        ).drop("neg_ret_sq")

        # Calmar (21d annualized return / max drawdown)
        df = df.withColumn(
            "rolling_min_21d",
            F.min("close").over(w_21)
        ).withColumn(
            "rolling_max_21d",
            F.max("close").over(w_21)
        ).withColumn(
            "max_dd_21d",
            (F.col("rolling_min_21d") -
             F.col("rolling_max_21d")) /
            (F.col("rolling_max_21d") + F.lit(1e-8))
        ).drop("rolling_min_21d","rolling_max_21d")

        print("  Quality factors added ✓")
        return df

    # ------------------------------------------------------------------ #
    #  Step 7 — Cross-sectional features
    # ------------------------------------------------------------------ #
    def add_cross_sectional(self, df):
        print("\nStep 7: Cross-sectional features...")

        w_date = Window.partitionBy("date")

        # Key factors to cross-sectionally normalize
        cs_factors = [
            "mom_1d","mom_5d","mom_21d",
            "mom_63d","mom_252d",
            "rev_1d","rev_5d",
            "vol_21d","vol_63d",
            "dollar_volume","volume_ratio",
            "sharpe_21d","sharpe_63d",
            "rsi_14d","price_to_ma20",
            "price_to_ma50","amihud_21d",
        ]

        for factor in cs_factors:
            if factor not in df.columns:
                continue

            # CS rank
            df = df.withColumn(
                f"{factor}_rank",
                F.percent_rank().over(
                    w_date.orderBy(factor)
                )
            )

            # CS z-score
            df = df.withColumn(
                f"{factor}_zscore",
                (F.col(factor) -
                 F.mean(factor).over(w_date)) /
                (F.stddev(factor).over(w_date) +
                 F.lit(1e-8))
            )

        print("  Cross-sectional features added ✓")
        return df

    # ------------------------------------------------------------------ #
    #  Step 8 — Forward returns (labels)
    # ------------------------------------------------------------------ #
    def add_forward_returns(self, df):
        print("\nStep 8: Forward return labels...")

        w = Window.partitionBy("ticker").orderBy("date")

        for h in [1, 5, 10, 21, 63]:
            df = df.withColumn(
                f"fwd_return_{h}d",
                F.avg("return_1d").over(
                    w.rowsBetween(1, h)
                )
            )

        # Forward vol
        df = df.withColumn(
            "fwd_vol_21d",
            F.stddev("return_1d").over(
                w.rowsBetween(1, 21)
            ) * F.sqrt(F.lit(252.0))
        )

        # Forward Sharpe
        df = df.withColumn(
            "fwd_sharpe_21d",
            F.mean("return_1d").over(
                w.rowsBetween(1, 21)
            ) * F.sqrt(F.lit(252.0)) /
            (F.stddev("return_1d").over(
                w.rowsBetween(1, 21)
            ) * F.sqrt(F.lit(252.0)) + F.lit(1e-8))
        )

        print("  Forward returns added ✓")
        return df

    # ------------------------------------------------------------------ #
    #  Step 9 — Add regime labels
    # ------------------------------------------------------------------ #
    def add_regime_labels(self, df):
        print("\nStep 9: Joining regime labels...")

        try:
            regimes = self.spark.read.format("delta").load(
                f"{self.eda_path}/regime_analysis"
                f"/regime_labels"
            ).select(
                F.to_date("date").alias("date"),
                "regime_label",
                "prob_bull",
                "prob_bear",
                "prob_highvol",
                "hmm_state"
            ).dropDuplicates(["date"])

            df = df.join(
                regimes, on="date", how="left"
            ).withColumn(
                "regime_label",
                F.coalesce(
                    F.col("regime_label"),
                    F.lit("Unknown")
                )
            )

            regime_count = df.filter(
                F.col("regime_label") != "Unknown"
            ).count()
            print(f"  Regime rows joined : "
                  f"{regime_count:,}")
        except Exception as e:
            print(f"  Regime join skipped: {e}")
            df = df.withColumn(
                "regime_label", F.lit("Unknown")
            ).withColumn(
                "prob_bull",    F.lit(None).cast("double")
            ).withColumn(
                "prob_bear",    F.lit(None).cast("double")
            ).withColumn(
                "prob_highvol", F.lit(None).cast("double")
            ).withColumn(
                "hmm_state",    F.lit(None).cast("integer")
            )

        return df

    # ------------------------------------------------------------------ #
    #  Step 10 — Clean and finalize
    # ------------------------------------------------------------------ #
    def clean(self, df):
        print("\nStep 10: Cleaning...")

        # Drop intermediate columns
        drop_cols = [
            "ma_20d","ma_50d","ma_200d",
        ]
        for c in drop_cols:
            if c in df.columns:
                df = df.drop(c)

        # Winsorize z-scores at ±5
        zscore_cols = [
            c for c in df.columns
            if c.endswith("_zscore")
        ]
        for col in zscore_cols:
            df = df.withColumn(
                col,
                F.when(F.col(col) > 5,  5.0)
                .when(F.col(col) < -5, -5.0)
                .otherwise(F.col(col))
            )

        # Add metadata
        df = df.withColumn(
            "gold_created_at",
            F.lit(
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )
        )

        before = df.count()
        # Remove rows with null close
        df = df.filter(F.col("close").isNotNull())
        after = df.count()
        print(f"  Rows before clean : {before:,}")
        print(f"  Rows after clean  : {after:,}")
        print("  Cleaning done ✓")
        return df

    # ------------------------------------------------------------------ #
    #  Write
    # ------------------------------------------------------------------ #
    def write(self, df) -> None:
        print(f"\nWriting Gold Price Factors...")
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
            "spark.databricks.delta.retentionDurationCheck"
            ".enabled", "false"
        )
        self.spark.sql(
            f"VACUUM delta.`{self.gold_path}` "
            f"RETAIN 168 HOURS"
        )

        details = self.spark.sql(
            f"DESCRIBE DETAIL delta.`{self.gold_path}`"
        ).select(
            "numFiles","sizeInBytes"
        ).collect()[0]
        print(f"  Files : {details['numFiles']}")
        print(f"  Size  : "
              f"{details['sizeInBytes']/1e6:.1f} MB")
        print("  Write complete ✓")

    # ------------------------------------------------------------------ #
    #  Validate
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        print("\n" + "="*55)
        print("VALIDATION — Gold Price Factors")
        print("="*55)

        df     = self.spark.read.format("delta").load(
            self.gold_path
        )
        total  = df.count()
        tickers = df.select("ticker").distinct().count()

        print(f"\n  Total rows      : {total:,}")
        print(f"  Unique tickers  : {tickers:,}")
        print(f"  Total columns   : {len(df.columns):,}")

        date_range = df.agg(
            F.min("date").alias("min"),
            F.max("date").alias("max")
        ).collect()[0]
        print(f"  Date range      : "
              f"{date_range['min']} → "
              f"{date_range['max']}")

        print(f"\n  Regime distribution:")
        df.groupBy("regime_label").count() \
          .orderBy("count", ascending=False).show()

        print(f"\n  Sample (AAPL latest):")
        key_cols = [
            "date","ticker",
            "mom_5d","mom_21d","mom_252d",
            "vol_21d","sharpe_21d",
            "rsi_14d","price_to_ma20",
            "regime_label","fwd_return_21d"
        ]
        available = [
            c for c in key_cols if c in df.columns
        ]
        df.filter(F.col("ticker") == "AAPL") \
          .orderBy(F.col("date").desc()) \
          .select(*available) \
          .show(5)

        print(f"\n  Null check (key factors):")
        for col in [
            "mom_21d","vol_21d","sharpe_21d",
            "rsi_14d","fwd_return_21d"
        ]:
            if col in df.columns:
                n = df.filter(
                    F.col(col).isNull()
                ).count()
                pct = n/total*100
                print(f"    {col:20}: "
                      f"{n:,} nulls ({pct:.1f}%)")

        print(f"\n  Factor summary stats:")
        factor_cols = [
            "mom_21d","mom_252d","vol_21d",
            "sharpe_21d","rsi_14d","volume_ratio"
        ]
        stats_cols = [
            c for c in factor_cols
            if c in df.columns
        ]
        df.select(*stats_cols).describe().show()

        assert total > 0, "FAIL — empty table"
        print(f"\nValidation PASSED ✓")
        return df

    # ------------------------------------------------------------------ #
    #  Run
    # ------------------------------------------------------------------ #
    def run(self):
        print("="*55)
        print("Gold 01 — Price Factors Pipeline")
        print("="*55)
        start = datetime.now()

        df = self.load()
        df = self.add_momentum(df)
        df = self.add_volatility(df)
        df = self.add_volume(df)
        df = self.add_technical(df)
        df = self.add_quality(df)
        df = self.add_cross_sectional(df)
        df = self.add_forward_returns(df)
        df = self.add_regime_labels(df)
        df = self.clean(df)

        self.write(df)
        self.validate()

        elapsed = (
            datetime.now() - start
        ).seconds / 60
        print(f"\nTotal time : {elapsed:.1f} minutes")
        print("Gold 01 — Price Factors COMPLETE ✓")
        return df

# COMMAND ----------

class GoldPriceFactorCharts:
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
        "Unknown": "#9E9E9E"
    }

    def chart_factor_coverage(self,
                               df) -> None:
        """Chart 1 — Factor null coverage."""
        factor_cols = [
            "mom_1d","mom_5d","mom_21d",
            "mom_63d","mom_252d",
            "vol_21d","vol_63d",
            "sharpe_21d","sharpe_63d","sortino_21d",
            "rsi_14d","price_to_ma20","price_to_ma50",
            "vwap_ratio","volume_ratio","amihud_21d",
            "price_to_52w_high","mom_accel",
        ]
        available = [
            c for c in factor_cols if c in df.columns
        ]
        total     = df.count()

        null_pcts = []
        for col in available:
            n = df.filter(F.col(col).isNull()).count()
            null_pcts.append({
                "factor"   : col,
                "null_pct" : n/total*100,
                "coverage" : (1 - n/total)*100
            })

        cov_df = pd.DataFrame(null_pcts).sort_values(
            "coverage", ascending=True
        )

        colors = [
            self.COLORS["success"] if c >= 90
            else self.COLORS["warning"] if c >= 70
            else self.COLORS["secondary"]
            for c in cov_df["coverage"]
        ]

        fig = go.Figure(go.Bar(
            x=cov_df["coverage"],
            y=cov_df["factor"],
            orientation="h",
            marker_color=colors,
            text=cov_df["coverage"].apply(
                lambda x: f"{x:.1f}%"
            ),
            textposition="outside"
        ))

        fig.add_vline(
            x=90, line_dash="dash",
            line_color="green", opacity=0.6,
            annotation_text="90% threshold"
        )

        fig.update_layout(
            title="<b>Gold 01 — Factor Data Coverage</b>",
            template=self.TEMPLATE,
            height=600,
            xaxis_title="Coverage (%)",
            yaxis_title="Factor"
        )
        fig.show()

    def chart_factor_distributions(self,
                                    pdf: pd.DataFrame
                                    ) -> None:
        """Chart 2 — Factor distributions."""
        factor_groups = {
            "Momentum" : [
                "mom_5d","mom_21d","mom_252d"
            ],
            "Volatility": [
                "vol_21d","vol_63d","vol_252d"
            ],
            "Quality"  : [
                "sharpe_21d","sortino_21d","rsi_14d"
            ],
            "Technical": [
                "price_to_ma20","vwap_ratio",
                "price_to_52w_high"
            ],
        }

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(factor_groups.keys())
        )

        positions = [(1,1),(1,2),(2,1),(2,2)]
        colors    = list(px.colors.qualitative.Plotly)

        for idx, (group, factors) in enumerate(
            factor_groups.items()
        ):
            row, col = positions[idx]
            for i, factor in enumerate(factors):
                if factor not in pdf.columns:
                    continue
                vals = pdf[factor].dropna()
                vals = vals.clip(
                    vals.quantile(0.01),
                    vals.quantile(0.99)
                )
                fig.add_trace(go.Histogram(
                    x=vals,
                    name=factor,
                    nbinsx=50,
                    opacity=0.6,
                    marker_color=colors[
                        i % len(colors)
                    ],
                    showlegend=(idx == 0)
                ), row=row, col=col)

        fig.update_layout(
            title="<b>Gold 01 — Factor Distributions</b>",
            template=self.TEMPLATE,
            height=700,
            barmode="overlay"
        )
        fig.show()

    def chart_factor_correlation(self,
                                  pdf: pd.DataFrame
                                  ) -> None:
        """Chart 3 — Factor correlation heatmap."""
        factor_cols = [
            "mom_5d","mom_21d","mom_252d",
            "vol_21d","sharpe_21d","rsi_14d",
            "price_to_ma20","volume_ratio",
            "amihud_21d","vwap_ratio"
        ]
        available = [
            c for c in factor_cols if c in pdf.columns
        ]

        sample = pdf[available].dropna().sample(
            min(10000, len(pdf)), random_state=42
        )
        corr   = sample.corr()

        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdYlGn",
            zmid=0, zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=9),
            colorbar=dict(title="Corr")
        ))

        fig.update_layout(
            title="<b>Gold 01 — Factor Correlation Matrix</b>",
            template=self.TEMPLATE,
            height=600,
            xaxis=dict(tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9))
        )
        fig.show()

    def chart_regime_factor_profile(self,
                                     pdf: pd.DataFrame
                                     ) -> None:
        """Chart 4 — Factor means by regime."""
        if "regime_label" not in pdf.columns:
            return

        factors = [
            "mom_21d","vol_21d","sharpe_21d",
            "rsi_14d","volume_ratio","amihud_21d"
        ]
        available = [
            f for f in factors if f in pdf.columns
        ]

        regime_means = pdf.groupby(
            "regime_label"
        )[available].mean().reset_index()

        fig = go.Figure()
        colors = {
            "Bull"   : self.COLORS["success"],
            "Bear"   : self.COLORS["secondary"],
            "HighVol": self.COLORS["warning"],
            "Unknown": "#9E9E9E"
        }

        for _, row in regime_means.iterrows():
            regime = row["regime_label"]
            vals   = [row.get(f, 0) for f in available]
            fig.add_trace(go.Bar(
                x=available,
                y=vals,
                name=regime,
                marker_color=colors.get(
                    regime, "#9E9E9E"
                ),
                opacity=0.8
            ))

        fig.update_layout(
            title="<b>Gold 01 — Factor Means by Regime</b>",
            template=self.TEMPLATE,
            height=500,
            barmode="group",
            xaxis_title="Factor",
            yaxis_title="Mean Value"
        )
        fig.show()

    def chart_momentum_decay(self,
                              pdf: pd.DataFrame
                              ) -> None:
        """Chart 5 — Momentum factor IC decay."""
        mom_cols = [
            ("mom_1d",  1),
            ("mom_5d",  5),
            ("mom_21d", 21),
            ("mom_63d", 63),
            ("mom_252d",252),
        ]
        fwd_col = "fwd_return_21d"

        if fwd_col not in pdf.columns:
            return

        ic_vals = []
        for col, h in mom_cols:
            if col not in pdf.columns:
                continue
            valid = pdf[[col, fwd_col]].dropna()
            if len(valid) < 100:
                continue
            ic = float(np.corrcoef(
                stats.rankdata(valid[col]),
                stats.rankdata(valid[fwd_col])
            )[0, 1])
            ic_vals.append({
                "horizon": h,
                "factor" : col,
                "ic"     : ic
            })

        if not ic_vals:
            return

        ic_df = pd.DataFrame(ic_vals)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ic_df["horizon"],
            y=ic_df["ic"],
            mode="lines+markers",
            line=dict(
                color=self.COLORS["primary"], width=2.5
            ),
            marker=dict(size=10),
            text=ic_df["factor"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Horizon: %{x}d<br>"
                "IC: %{y:.4f}<extra></extra>"
            )
        ))

        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3
        )
        fig.add_hline(
            y=0.02, line_dash="dot",
            line_color="green", opacity=0.5,
            annotation_text="IC=0.02"
        )
        fig.add_hline(
            y=-0.02, line_dash="dot",
            line_color="green", opacity=0.5
        )

        fig.update_layout(
            title="<b>Gold 01 — Momentum IC Decay "
                  "(vs 21d Forward Return)</b>",
            template=self.TEMPLATE,
            height=450,
            xaxis_title="Lookback Horizon (days)",
            yaxis_title="IC (Spearman)"
        )
        fig.show()

    def chart_factor_ic_vs_fwd(self,
                                pdf: pd.DataFrame
                                ) -> None:
        """Chart 6 — All factor IC vs all horizons."""
        factor_cols = [
            "mom_5d","mom_21d","mom_252d",
            "vol_21d","sharpe_21d","rsi_14d",
            "vwap_ratio","volume_ratio","rev_5d",
        ]
        fwd_cols = {
            "1d" : "fwd_return_1d",
            "5d" : "fwd_return_5d",
            "21d": "fwd_return_21d",
        }

        rows = []
        for factor in factor_cols:
            if factor not in pdf.columns:
                continue
            for h_label, fwd_col in fwd_cols.items():
                if fwd_col not in pdf.columns:
                    continue
                valid = pdf[
                    [factor, fwd_col]
                ].dropna()
                if len(valid) < 100:
                    continue
                ic = float(np.corrcoef(
                    stats.rankdata(valid[factor]),
                    stats.rankdata(valid[fwd_col])
                )[0, 1])
                rows.append({
                    "factor"  : factor,
                    "horizon" : h_label,
                    "ic"      : ic,
                    "ic_abs"  : abs(ic),
                })

        if not rows:
            return

        ic_df = pd.DataFrame(rows)
        pivot = ic_df.pivot_table(
            index="factor",
            columns="horizon",
            values="ic"
        ).reindex(
            ic_df.groupby("factor")["ic_abs"].mean()
                 .sort_values(ascending=False).index
        )

        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="RdYlGn",
            zmid=0,
            text=np.round(pivot.values, 3),
            texttemplate="%{text}",
            textfont=dict(size=12),
            colorbar=dict(title="IC")
        ))

        fig.update_layout(
            title="<b>Gold 01 — Factor IC Heatmap "
                  "(Factor × Forward Horizon)</b>",
            template=self.TEMPLATE,
            height=500,
            xaxis_title="Forward Return Horizon",
            yaxis_title="Factor"
        )
        fig.show()

    def chart_sharpe_by_regime(self,
                                pdf: pd.DataFrame
                                ) -> None:
        """Chart 7 — Sharpe distribution by regime."""
        if "regime_label" not in pdf.columns or \
           "sharpe_21d" not in pdf.columns:
            return

        fig = go.Figure()
        for regime, color in {
            "Bull"   : self.COLORS["success"],
            "Bear"   : self.COLORS["secondary"],
            "HighVol": self.COLORS["warning"],
        }.items():
            vals = pdf[
                pdf["regime_label"] == regime
            ]["sharpe_21d"].dropna()

            if len(vals) == 0:
                continue

            fig.add_trace(go.Violin(
                y=vals.clip(-5, 5),
                name=regime,
                fillcolor=color,
                line_color="white",
                opacity=0.7,
                box_visible=True,
                meanline_visible=True
            ))

        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3
        )

        fig.update_layout(
            title="<b>Gold 01 — 21d Sharpe "
                  "Distribution by Regime</b>",
            template=self.TEMPLATE,
            height=500,
            yaxis_title="21d Rolling Sharpe"
        )
        fig.show()

    def run_all(self, spark, gold_path) -> None:
        print("\n" + "="*55)
        print("Generating Gold 01 Charts...")
        print("="*55)

        df_spark = spark.read.format("delta").load(
            gold_path
        )

        print("\n[1/7] Factor Coverage...")
        self.chart_factor_coverage(df_spark)

        # Sample to pandas for remaining charts
        print("  Sampling to pandas...")
        pdf = df_spark.sample(
            fraction=0.1, seed=42
        ).toPandas()
        print(f"  Sample size : {len(pdf):,}")

        print("[2/7] Factor Distributions...")
        self.chart_factor_distributions(pdf)

        print("[3/7] Factor Correlation Matrix...")
        self.chart_factor_correlation(pdf)

        print("[4/7] Regime Factor Profile...")
        self.chart_regime_factor_profile(pdf)

        print("[5/7] Momentum IC Decay...")
        self.chart_momentum_decay(pdf)

        print("[6/7] Factor IC Heatmap...")
        self.chart_factor_ic_vs_fwd(pdf)

        print("[7/7] Sharpe by Regime...")
        self.chart_sharpe_by_regime(pdf)

        print("\nAll 7 charts ✓")

# COMMAND ----------

pipeline = GoldPriceFactors(
    spark       = spark,
    silver_path = SILVER_PATH,
    eda_path    = EDA_PATH,
    gold_path   = GOLD_PATH
)

df = pipeline.run()

charts = GoldPriceFactorCharts()
charts.run_all(
    spark     = spark,
    gold_path = f"{GOLD_PATH}/price_factors"
)

print("\nGold 01 COMPLETE ✓")

# COMMAND ----------

df = spark.read.format("delta").load(
    f"{GOLD_PATH}/price_factors"
)

print("="*55)
print("Gold 01 — Price Factors Summary")
print("="*55)
print(f"Total rows     : {df.count():,}")
print(f"Total columns  : {len(df.columns):,}")
print(f"\nAll columns:")
for i, c in enumerate(sorted(df.columns)):
    print(f"  {i+1:3}. {c}")

print(f"\nRegime distribution:")
df.groupBy("regime_label").count() \
  .orderBy("count", ascending=False).show()

print(f"\nLatest AAPL factors:")
df.filter(F.col("ticker") == "AAPL") \
  .orderBy(F.desc("date")) \
  .select(
    "date","mom_21d","mom_252d",
    "vol_21d","sharpe_21d","rsi_14d",
    "price_to_ma20","regime_label",
    "fwd_return_21d"
  ).show(5)