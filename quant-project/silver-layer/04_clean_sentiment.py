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

# COMMAND ----------

class SilverSentimentCleaning:
    """
    Silver Sentiment Cleaning Pipeline.
    Bronze schema: date, ticker, headline, summary,
                   source, url, sentiment_label,
                   sentiment_score, ingested_at
    Operations:
      1. Load bronze sentiment
      2. Remove duplicates and bad scores
      3. Aggregate to daily level per ticker
      4. Add rolling features (3d, 7d, 14d)
      5. Add cross-sectional rank and z-score
      6. Write to silver/delta/sentiment
    """

    def __init__(self, spark, bronze_path, silver_path):
        self.spark       = spark
        self.bronze_path = f"{bronze_path}/sentiment"
        self.silver_path = f"{silver_path}/sentiment"

        print("SilverSentimentCleaning ✓")
        print(f"  Bronze : {self.bronze_path}")
        print(f"  Silver : {self.silver_path}")

    # ------------------------------------------------------------------ #
    #  Load
    # ------------------------------------------------------------------ #
    def load(self):
        print("\nLoading bronze sentiment...")
        df = self.spark.read.format("delta").load(self.bronze_path)
        print(f"  Rows    : {df.count():,}")
        print(f"  Tickers : {df.select('ticker').distinct().count():,}")
        print(f"  Columns : {df.columns}")
        print(f"\n  Label distribution:")
        df.groupBy("sentiment_label").count() \
          .orderBy("count", ascending=False).show()
        return df

    # ------------------------------------------------------------------ #
    #  Clean
    # ------------------------------------------------------------------ #
    def clean(self, df):
        print("\nCleaning sentiment data...")
        before = df.count()

        # Remove nulls in key columns
        df = df.dropna(subset=[
            "ticker","date","sentiment_score","sentiment_label"
        ])

        # Remove invalid scores (must be between -1 and 1)
        df = df.filter(
            (F.col("sentiment_score") >= -1.0) &
            (F.col("sentiment_score") <=  1.0)
        )

        # Remove empty headlines
        df = df.filter(
            F.col("headline").isNotNull() &
            (F.length(F.col("headline")) > 5)
        )

        # Remove duplicates on ticker + date + headline
        df = df.dropDuplicates(["ticker","date","headline"])

        # Normalize date to date type
        df = df.withColumn(
            "date", F.to_date(F.col("date"))
        )

        after = df.count()
        print(f"  Removed : {before - after:,} bad rows")
        print(f"  Kept    : {after:,} rows")
        return df

    # ------------------------------------------------------------------ #
    #  Daily aggregation
    # ------------------------------------------------------------------ #
    def aggregate_daily(self, df):
        print("\nAggregating to daily level per ticker...")

        daily = df.groupBy("ticker","date").agg(
            # Mean sentiment score
            F.mean("sentiment_score").alias("sentiment_mean"),

            # News volume
            F.count("*").alias("news_count"),

            # Label counts
            F.sum(
                F.when(
                    F.col("sentiment_label") == "positive", 1
                ).otherwise(0)
            ).alias("positive_count"),
            F.sum(
                F.when(
                    F.col("sentiment_label") == "negative", 1
                ).otherwise(0)
            ).alias("negative_count"),
            F.sum(
                F.when(
                    F.col("sentiment_label") == "neutral", 1
                ).otherwise(0)
            ).alias("neutral_count"),

            # Sentiment volatility within a day
            F.stddev("sentiment_score").alias("sentiment_std"),

            # Max/min sentiment of the day
            F.max("sentiment_score").alias("sentiment_max"),
            F.min("sentiment_score").alias("sentiment_min"),

            # Source diversity
            F.countDistinct("source").alias("source_count"),
        )

        # Add year/month partition columns
        daily = daily \
            .withColumn("year",  F.year(F.col("date"))) \
            .withColumn("month", F.month(F.col("date")))

        # Use mean as main sentiment signal
        # (no confidence column in this dataset)
        daily = daily.withColumn(
            "sentiment_weighted",
            F.col("sentiment_mean")
        )

        # Bullish/bearish/neutral ratios
        daily = daily \
            .withColumn(
                "bullish_ratio",
                F.col("positive_count") / F.col("news_count")
            ) \
            .withColumn(
                "bearish_ratio",
                F.col("negative_count") / F.col("news_count")
            ) \
            .withColumn(
                "neutral_ratio",
                F.col("neutral_count") / F.col("news_count")
            )

        # Sentiment range
        daily = daily.withColumn(
            "sentiment_range",
            F.col("sentiment_max") - F.col("sentiment_min")
        )

        # Fill null std (single headline days)
        daily = daily.withColumn(
            "sentiment_std",
            F.coalesce(F.col("sentiment_std"), F.lit(0.0))
        )

        print(f"  Daily rows : {daily.count():,}")
        print(f"  Tickers    : {daily.select('ticker').distinct().count():,}")
        return daily

    # ------------------------------------------------------------------ #
    #  Rolling features
    # ------------------------------------------------------------------ #
    def add_rolling_features(self, df):
        print("\nAdding rolling sentiment features...")

        w3  = Window.partitionBy("ticker") \
                    .orderBy("date") \
                    .rowsBetween(-2, 0)
        w7  = Window.partitionBy("ticker") \
                    .orderBy("date") \
                    .rowsBetween(-6, 0)
        w14 = Window.partitionBy("ticker") \
                    .orderBy("date") \
                    .rowsBetween(-13, 0)
        w_lag = Window.partitionBy("ticker").orderBy("date")

        # Rolling mean sentiment
        df = df \
            .withColumn(
                "sentiment_3d",
                F.mean("sentiment_weighted").over(w3)
            ) \
            .withColumn(
                "sentiment_7d",
                F.mean("sentiment_weighted").over(w7)
            ) \
            .withColumn(
                "sentiment_14d",
                F.mean("sentiment_weighted").over(w14)
            )

        # Rolling news volume
        df = df \
            .withColumn(
                "news_count_3d",
                F.sum("news_count").over(w3)
            ) \
            .withColumn(
                "news_count_7d",
                F.sum("news_count").over(w7)
            )

        # Sentiment momentum (change from previous period)
        df = df \
            .withColumn(
                "sentiment_momentum_1d",
                F.col("sentiment_weighted") -
                F.lag("sentiment_weighted", 1).over(w_lag)
            ) \
            .withColumn(
                "sentiment_momentum_3d",
                F.col("sentiment_weighted") -
                F.lag("sentiment_weighted", 3).over(w_lag)
            )

        # Sentiment trend (short vs long window)
        df = df.withColumn(
            "sentiment_trend",
            F.col("sentiment_3d") - F.col("sentiment_7d")
        )

        # Rolling bullish ratio
        df = df.withColumn(
            "bullish_ratio_7d",
            F.mean("bullish_ratio").over(w7)
        )

        print("  Rolling features added ✓")
        return df

    # ------------------------------------------------------------------ #
    #  Cross-sectional features (IC preparation)
    # ------------------------------------------------------------------ #
    def add_ic_features(self, df):
        print("\nAdding cross-sectional IC features...")

        w_date = Window.partitionBy("date")

        # Cross-sectional rank (for IC calculation in Gold)
        df = df.withColumn(
            "sentiment_rank",
            F.percent_rank().over(
                w_date.orderBy("sentiment_weighted")
            )
        )

        # Cross-sectional z-score
        df = df.withColumn(
            "sentiment_zscore",
            (F.col("sentiment_weighted") -
             F.mean("sentiment_weighted").over(w_date)) /
            (F.stddev("sentiment_weighted").over(w_date) +
             F.lit(1e-8))
        )

        # Cross-sectional news volume rank
        df = df.withColumn(
            "news_volume_rank",
            F.percent_rank().over(
                w_date.orderBy("news_count")
            )
        )

        print("  IC features added ✓")
        return df

    # ------------------------------------------------------------------ #
    #  Add metadata
    # ------------------------------------------------------------------ #
    def add_metadata(self, df):
        df = df.withColumn(
            "cleaned_at",
            F.lit(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        return df

    # ------------------------------------------------------------------ #
    #  Write
    # ------------------------------------------------------------------ #
    def write_delta(self, df) -> None:
        print(f"\nWriting Silver Sentiment: {self.silver_path}")
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
        print("VALIDATION — Silver Sentiment")
        print("="*50)

        df    = self.spark.read.format("delta").load(self.silver_path)
        total = df.count()

        print(f"  Total rows  : {total:,}")
        print(f"  Tickers     : {df.select('ticker').distinct().count():,}")
        print(f"  Date range  : "
              f"{df.agg(F.min('date'), F.max('date')).collect()[0]}")
        print(f"  Columns     : {len(df.columns)}")

        print(f"\n  Sentiment stats:")
        df.select(
            F.mean("sentiment_weighted").alias("avg_sentiment"),
            F.stddev("sentiment_weighted").alias("std_sentiment"),
            F.mean("news_count").alias("avg_daily_news"),
            F.mean("bullish_ratio").alias("avg_bullish_ratio"),
            F.mean("bearish_ratio").alias("avg_bearish_ratio")
        ).show()

        print(f"\n  Sample (AAPL):")
        df.filter(F.col("ticker") == "AAPL") \
          .orderBy(F.col("date").desc()) \
          .select(
            "date","sentiment_weighted","sentiment_3d",
            "news_count","bullish_ratio","sentiment_rank",
            "sentiment_zscore"
        ).show(5)

        print(f"\n  Top bullish tickers (latest date):")
        latest = df.agg(F.max("date")).collect()[0][0]
        df.filter(F.col("date") == latest) \
          .orderBy(F.col("sentiment_weighted").desc()) \
          .select(
            "ticker","sentiment_weighted",
            "news_count","bullish_ratio"
        ).show(10)

        assert total > 0, "FAIL — empty table"
        print("\nValidation PASSED ✓")

    # ------------------------------------------------------------------ #
    #  Run
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        print("="*50)
        print("Silver Sentiment Cleaning Pipeline")
        print("="*50)
        df = self.load()
        df = self.clean(df)
        df = self.aggregate_daily(df)
        df = self.add_rolling_features(df)
        df = self.add_ic_features(df)
        df = self.add_metadata(df)
        self.write_delta(df)
        self.optimize()
        self.validate()
        print("\nSilver Sentiment COMPLETE ✓")

# COMMAND ----------

cleaner = SilverSentimentCleaning(
    spark       = spark,
    bronze_path = BRONZE_PATH,
    silver_path = SILVER_PATH
)

cleaner.run()

# COMMAND ----------

df = spark.read.format("delta").load(f"{SILVER_PATH}/sentiment")

print(f"Total rows  : {df.count():,}")
print(f"Tickers     : {df.select('ticker').distinct().count():,}")
print(f"Columns     : {df.columns}")

df.orderBy(F.col("date").desc()) \
  .select(
    "date","ticker","sentiment_weighted",
    "sentiment_3d","news_count",
    "bullish_ratio","sentiment_rank"
  ).show(10)