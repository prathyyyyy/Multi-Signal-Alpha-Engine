# Databricks notebook source
# MAGIC %pip install scipy numpy pandas --quiet

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

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.shuffle.partitions", "200")

STORAGE_ACCOUNT = "multisignalalphaeng"
CONTAINER       = "quant-lakehouse"
ADLS_KEY        = dbutils.secrets.get(scope="quant-scope", key="adls-key-01")

spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
    ADLS_KEY
)

BASE_PATH   = f"abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net"
BRONZE_PATH = f"{BASE_PATH}/bronze/delta"
SILVER_PATH = f"{BASE_PATH}/silver/delta"

print("Config loaded ✓")
print(f"Bronze : {BRONZE_PATH}")
print(f"Silver : {SILVER_PATH}")

# COMMAND ----------

class SilverOHLCVCleaning:
    """
    Silver OHLCV Cleaning Pipeline.
    Operations:
      1. Load bronze OHLCV
      2. Remove bad data (nulls, zero prices, zero volume)
      3. Calculate returns (daily, log)
      4. Winsorize returns at 1st/99th percentile
      5. Add rolling volatility (21-day)
      6. Forward fill missing trading days per ticker
      7. Write to silver/delta/ohlcv
    """

    def __init__(self, spark, bronze_path, silver_path):
        self.spark       = spark
        self.bronze_path = f"{bronze_path}/ohlcv"
        self.silver_path = f"{silver_path}/ohlcv"

        print("SilverOHLCVCleaning ✓")
        print(f"  Bronze : {self.bronze_path}")
        print(f"  Silver : {self.silver_path}")

    # ------------------------------------------------------------------ #
    #  Load
    # ------------------------------------------------------------------ #
    def load(self):
        print("\nLoading bronze OHLCV...")
        df = self.spark.read.format("delta").load(self.bronze_path)
        print(f"  Rows   : {df.count():,}")
        print(f"  Tickers: {df.select('ticker').distinct().count():,}")
        return df

    # ------------------------------------------------------------------ #
    #  Clean
    # ------------------------------------------------------------------ #
    def clean(self, df):
        print("\nCleaning...")
        before = df.count()

        # Step 1 — Remove nulls in key columns
        df = df.dropna(subset=["date","ticker","close"])

        # Step 2 — Remove zero/negative prices
        df = df.filter(
            (F.col("close") > 0) &
            (F.col("open")  > 0) &
            (F.col("high")  > 0) &
            (F.col("low")   > 0)
        )

        # Step 3 — Remove zero volume
        df = df.filter(F.col("volume") > 0)

        # Step 4 — Remove duplicates
        df = df.dropDuplicates(["date","ticker"])

        after = df.count()
        print(f"  Removed : {before - after:,} bad rows")
        print(f"  Kept    : {after:,} rows")
        return df

    # ------------------------------------------------------------------ #
    #  Returns
    # ------------------------------------------------------------------ #
    def add_returns(self, df):
        print("\nCalculating returns...")

        # Window by ticker ordered by date
        w = Window.partitionBy("ticker").orderBy("date")

        # Daily return
        df = df.withColumn(
            "return_1d",
            (F.col("close") - F.lag("close", 1).over(w)) /
            F.lag("close", 1).over(w)
        )

        # Log return
        df = df.withColumn(
            "log_return_1d",
            F.log(F.col("close") / F.lag("close", 1).over(w))
        )

        # Weekly return (5 days)
        df = df.withColumn(
            "return_5d",
            (F.col("close") - F.lag("close", 5).over(w)) /
            F.lag("close", 5).over(w)
        )

        # Monthly return (21 days)
        df = df.withColumn(
            "return_21d",
            (F.col("close") - F.lag("close", 21).over(w)) /
            F.lag("close", 21).over(w)
        )

        # Quarterly return (63 days)
        df = df.withColumn(
            "return_63d",
            (F.col("close") - F.lag("close", 63).over(w)) /
            F.lag("close", 63).over(w)
        )

        # Annual return (252 days)
        df = df.withColumn(
            "return_252d",
            (F.col("close") - F.lag("close", 252).over(w)) /
            F.lag("close", 252).over(w)
        )

        print("  Returns calculated ✓")
        return df

    # ------------------------------------------------------------------ #
    #  Winsorize
    # ------------------------------------------------------------------ #
    def winsorize(self, df):
        print("\nWinsorizing returns at 1st/99th percentile...")

        return_cols = [
            "return_1d","log_return_1d",
            "return_5d","return_21d",
            "return_63d","return_252d"
        ]

        for col in return_cols:
            # Calculate percentiles
            quantiles = df.filter(
                F.col(col).isNotNull()
            ).approxQuantile(col, [0.01, 0.99], 0.001)

            if len(quantiles) == 2:
                p01, p99 = quantiles
                df = df.withColumn(
                    col,
                    F.when(F.col(col) < p01, p01)
                    .when(F.col(col) > p99, p99)
                    .otherwise(F.col(col))
                )

        print("  Winsorization complete ✓")
        return df

    # ------------------------------------------------------------------ #
    #  Rolling volatility
    # ------------------------------------------------------------------ #
    def add_volatility(self, df):
        print("\nCalculating rolling volatility...")

        w_21  = Window.partitionBy("ticker") \
                      .orderBy("date") \
                      .rowsBetween(-20, 0)
        w_63  = Window.partitionBy("ticker") \
                      .orderBy("date") \
                      .rowsBetween(-62, 0)

        # 21-day realized volatility (annualized)
        df = df.withColumn(
            "vol_21d",
            F.stddev("log_return_1d").over(w_21) * F.sqrt(F.lit(252.0))
        )

        # 63-day realized volatility (annualized)
        df = df.withColumn(
            "vol_63d",
            F.stddev("log_return_1d").over(w_63) * F.sqrt(F.lit(252.0))
        )

        # Dollar volume (liquidity signal)
        df = df.withColumn(
            "dollar_volume",
            F.col("close") * F.col("volume")
        )

        # Price range (high - low) / close
        df = df.withColumn(
            "daily_range",
            (F.col("high") - F.col("low")) / F.col("close")
        )

        print("  Volatility features added ✓")
        return df

    # ------------------------------------------------------------------ #
    #  Add metadata
    # ------------------------------------------------------------------ #
    def add_metadata(self, df):
        print("\nAdding metadata...")

        df = df.withColumn(
            "cleaned_at",
            F.lit(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )

        # Vwap approximation
        df = df.withColumn(
            "vwap",
            (F.col("high") + F.col("low") + F.col("close")) / 3.0
        )

        print("  Metadata added ✓")
        return df

    # ------------------------------------------------------------------ #
    #  Write
    # ------------------------------------------------------------------ #
    def write_delta(self, df) -> None:
        print(f"\nWriting Silver OHLCV: {self.silver_path}")
        (df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema",                  "true")
            .option("delta.autoOptimize.optimizeWrite", "true")
            .option("delta.autoOptimize.autoCompact",   "true")
            .partitionBy("year","month")
            .save(self.silver_path)
        )
        self.spark.sql(f"OPTIMIZE delta.`{self.silver_path}`")
        print("Write complete ✓")

    def optimize(self) -> None:
        print("\nOPTIMIZE + VACUUM...")
        self.spark.sql(f"OPTIMIZE delta.`{self.silver_path}`")
        self.spark.conf.set(
            "spark.databricks.delta.retentionDurationCheck.enabled",
            "false"
        )
        self.spark.sql(
            f"VACUUM delta.`{self.silver_path}` RETAIN 168 HOURS"
        )
        details = self.spark.sql(
            f"DESCRIBE DETAIL delta.`{self.silver_path}`"
        ).select("numFiles","sizeInBytes").collect()[0]
        print(f"  Files : {details['numFiles']}")
        print(f"  Size  : {details['sizeInBytes']/1e6:.1f} MB")

    # ------------------------------------------------------------------ #
    #  Validate
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        print("\n" + "="*50)
        print("VALIDATION — Silver OHLCV")
        print("="*50)

        df    = self.spark.read.format("delta").load(self.silver_path)
        total = df.count()

        print(f"  Total rows     : {total:,}")
        print(f"  Unique tickers : {df.select('ticker').distinct().count():,}")
        print(f"  Date range     : "
              f"{df.agg(F.min('date'), F.max('date')).collect()[0]}")
        print(f"  Null returns   : "
              f"{df.filter(F.col('return_1d').isNull()).count():,}")

        print(f"\n  Return stats (1d):")
        df.select(
            F.mean("return_1d").alias("mean"),
            F.stddev("return_1d").alias("std"),
            F.min("return_1d").alias("min"),
            F.max("return_1d").alias("max")
        ).show()

        print(f"\n  Vol stats (21d annualized):")
        df.filter(F.col("vol_21d").isNotNull()) \
          .select(
            F.mean("vol_21d").alias("mean_vol"),
            F.min("vol_21d").alias("min_vol"),
            F.max("vol_21d").alias("max_vol")
        ).show()

        print(f"\n  Sample (AAPL recent):")
        df.filter(F.col("ticker") == "AAPL") \
          .orderBy(F.col("date").desc()) \
          .select("date","close","return_1d",
                  "log_return_1d","vol_21d") \
          .show(5)

        assert total > 0, "FAIL — empty table"
        print("\nValidation PASSED ✓")

    # ------------------------------------------------------------------ #
    #  Run
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        print("="*50)
        print("Silver OHLCV Cleaning Pipeline")
        print("="*50)
        df = self.load()
        df = self.clean(df)
        df = self.add_returns(df)
        df = self.winsorize(df)
        df = self.add_volatility(df)
        df = self.add_metadata(df)
        self.write_delta(df)
        self.optimize()
        self.validate()
        print("\nSilver OHLCV COMPLETE ✓")

# COMMAND ----------

cleaner = SilverOHLCVCleaning(
    spark       = spark,
    bronze_path = BRONZE_PATH,
    silver_path = SILVER_PATH
)

cleaner.run()

# COMMAND ----------

df = spark.read.format("delta").load(
    f"{SILVER_PATH}/ohlcv"
)

print(f"Total rows     : {df.count():,}")
print(f"Unique tickers : {df.select('ticker').distinct().count():,}")
print(f"Columns        : {df.columns}")

df.filter(F.col("ticker") == "AAPL") \
  .orderBy(F.col("date").desc()) \
  .select("date","close","return_1d","vol_21d",
          "dollar_volume") \
  .show(10)