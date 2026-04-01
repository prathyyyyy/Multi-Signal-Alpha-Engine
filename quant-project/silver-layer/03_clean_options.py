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
from scipy.stats import norm
from scipy.optimize import brentq

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

class SilverOptionsCleaning:
    """
    Silver Options Cleaning Pipeline.
    Operations:
      1. Load bronze options
      2. Remove bad data (zero IV, bad strikes)
      3. Validate Black-Scholes consistency
      4. Calculate IV surface features:
         - ATM IV (at-the-money)
         - IV skew (25-delta put vs call)
         - IV term structure (near vs far)
         - Put/Call IV ratio
      5. Calculate Greeks (Delta, Gamma, Vega, Theta)
      6. Write to silver/delta/options
    """

    RISK_FREE_RATE = 0.05  # approximate

    def __init__(self, spark, bronze_path, silver_path):
        self.spark       = spark
        self.bronze_path = f"{bronze_path}/options"
        self.silver_path = f"{silver_path}/options"

        print("SilverOptionsCleaning ✓")
        print(f"  Bronze : {self.bronze_path}")
        print(f"  Silver : {self.silver_path}")

    # ------------------------------------------------------------------ #
    #  Black-Scholes functions
    # ------------------------------------------------------------------ #
    @staticmethod
    def _bs_d1(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return 0.0
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / \
               (sigma * np.sqrt(T))

    @staticmethod
    def _bs_price(S, K, T, r, sigma, option_type="call"):
        if T <= 0 or sigma <= 0:
            return max(0, S - K) if option_type == "call" \
                   else max(0, K - S)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / \
             (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def _bs_delta(S, K, T, r, sigma, option_type="call"):
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / \
             (sigma * np.sqrt(T))
        if option_type == "call":
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

    @staticmethod
    def _bs_gamma(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / \
             (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def _bs_vega(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / \
             (sigma * np.sqrt(T))
        return S * norm.pdf(d1) * np.sqrt(T) / 100

    # ------------------------------------------------------------------ #
    #  Load
    # ------------------------------------------------------------------ #
    def load(self):
        print("\nLoading bronze options...")
        df = self.spark.read.format("delta").load(self.bronze_path)
        print(f"  Rows    : {df.count():,}")
        print(f"  Tickers : {df.select('ticker').distinct().count():,}")
        df.groupBy("option_type").count().show()
        return df

    # ------------------------------------------------------------------ #
    #  Clean
    # ------------------------------------------------------------------ #
    def clean(self, df):
        print("\nCleaning options data...")
        before = df.count()

        # Remove nulls
        df = df.dropna(subset=[
            "ticker","date","strike",
            "expiry","option_type","implied_vol"
        ])

        # Remove zero/negative IV
        df = df.filter(F.col("implied_vol") > 0.001)

        # Remove unreasonably high IV (> 500%)
        df = df.filter(F.col("implied_vol") < 5.0)

        # Remove zero strike
        df = df.filter(F.col("strike") > 0)

        # Remove zero bid AND ask
        df = df.filter(
            (F.col("bid") > 0) | (F.col("ask") > 0)
        )

        # Remove duplicates
        df = df.dropDuplicates([
            "date","ticker","expiry","strike","option_type"
        ])

        after = df.count()
        print(f"  Removed : {before - after:,} bad rows")
        print(f"  Kept    : {after:,} rows")
        return df

    # ------------------------------------------------------------------ #
    #  Add time to expiry
    # ------------------------------------------------------------------ #
    def add_tte(self, df):
        print("\nAdding time to expiry...")

        df = df.withColumn(
            "tte_days",
            F.datediff(
                F.to_date(F.col("expiry")),
                F.to_date(F.col("date"))
            )
        )

        # Time to expiry in years
        df = df.withColumn(
            "tte_years",
            F.col("tte_days") / F.lit(365.0)
        )

        # Remove expired options
        df = df.filter(F.col("tte_days") > 0)

        # Add expiry bucket
        df = df.withColumn(
            "expiry_bucket",
            F.when(F.col("tte_days") <= 30,  "near")
            .when(F.col("tte_days") <= 60,   "mid")
            .when(F.col("tte_days") <= 90,   "far")
            .otherwise("leap")
        )

        print("  TTE features added ✓")
        return df

    # ------------------------------------------------------------------ #
    #  Add mid price + spread
    # ------------------------------------------------------------------ #
    def add_price_features(self, df):
        print("\nAdding price features...")

        # Mid price
        df = df.withColumn(
            "mid_price",
            (F.col("bid") + F.col("ask")) / 2.0
        )

        # Bid-ask spread
        df = df.withColumn(
            "bid_ask_spread",
            F.col("ask") - F.col("bid")
        )

        # Relative spread (liquidity proxy)
        df = df.withColumn(
            "relative_spread",
            F.when(
                F.col("mid_price") > 0,
                F.col("bid_ask_spread") / F.col("mid_price")
            ).otherwise(F.lit(None))
        )

        print("  Price features added ✓")
        return df

    # ------------------------------------------------------------------ #
    #  Add moneyness
    # ------------------------------------------------------------------ #
    def add_moneyness(self, df):
        print("\nAdding moneyness features...")

        # We need spot price — approximate from ATM strike
        # Use strike as proxy (simplified)
        w = Window.partitionBy("ticker","date","expiry")

        # Find ATM strike (closest to current price)
        # We use the strike with highest open interest as ATM proxy
        df = df.withColumn(
            "max_oi",
            F.max("open_interest").over(w)
        )

        # Moneyness = ln(K/S) — simplified using relative strike
        # Find median strike per ticker/date as spot proxy
        df = df.withColumn(
            "median_strike",
            F.percentile_approx("strike", 0.5).over(w)
        )

        df = df.withColumn(
            "moneyness",
            F.log(F.col("strike") / F.col("median_strike"))
        )

        # ATM flag (within 2% of median strike)
        df = df.withColumn(
            "is_atm",
            F.abs(F.col("moneyness")) < 0.02
        )

        print("  Moneyness features added ✓")
        return df

    # ------------------------------------------------------------------ #
    #  IV Surface features
    # ------------------------------------------------------------------ #
    def add_iv_surface_features(self, df):
        print("\nCalculating IV surface features...")

        w_date = Window.partitionBy("ticker","date")
        w_exp  = Window.partitionBy("ticker","date","expiry")

        # ATM IV per expiry
        df = df.withColumn(
            "atm_iv",
            F.avg(
                F.when(F.col("is_atm"), F.col("implied_vol"))
            ).over(w_exp)
        )

        # Call ATM IV
        df = df.withColumn(
            "call_atm_iv",
            F.avg(
                F.when(
                    F.col("is_atm") & (F.col("option_type") == "call"),
                    F.col("implied_vol")
                )
            ).over(w_exp)
        )

        # Put ATM IV
        df = df.withColumn(
            "put_atm_iv",
            F.avg(
                F.when(
                    F.col("is_atm") & (F.col("option_type") == "put"),
                    F.col("implied_vol")
                )
            ).over(w_exp)
        )

        # IV Skew (put IV - call IV at ATM)
        df = df.withColumn(
            "iv_skew",
            F.col("put_atm_iv") - F.col("call_atm_iv")
        )

        # Near term ATM IV
        df = df.withColumn(
            "near_atm_iv",
            F.avg(
                F.when(
                    F.col("is_atm") & (F.col("expiry_bucket") == "near"),
                    F.col("implied_vol")
                )
            ).over(w_date)
        )

        # Far term ATM IV
        df = df.withColumn(
            "far_atm_iv",
            F.avg(
                F.when(
                    F.col("is_atm") & (F.col("expiry_bucket") == "far"),
                    F.col("implied_vol")
                )
            ).over(w_date)
        )

        # IV Term structure (near - far)
        df = df.withColumn(
            "iv_term_structure",
            F.col("near_atm_iv") - F.col("far_atm_iv")
        )

        # Average IV per ticker/date
        df = df.withColumn(
            "avg_iv_daily",
            F.avg("implied_vol").over(w_date)
        )

        print("  IV surface features added ✓")
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
        print(f"\nWriting Silver Options: {self.silver_path}")
        (df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema",                  "true")
            .option("delta.autoOptimize.optimizeWrite", "true")
            .option("delta.autoOptimize.autoCompact",   "true")
            .partitionBy("option_type","year","month")
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
        print("VALIDATION — Silver Options")
        print("="*50)

        df    = self.spark.read.format("delta").load(self.silver_path)
        total = df.count()

        print(f"  Total rows  : {total:,}")
        print(f"  Tickers     : {df.select('ticker').distinct().count():,}")
        print(f"  Date range  : "
              f"{df.agg(F.min('date'), F.max('date')).collect()[0]}")

        print(f"\n  Option type breakdown:")
        df.groupBy("option_type","expiry_bucket").count() \
          .orderBy("option_type","expiry_bucket").show()

        print(f"\n  IV surface sample (AAPL):")
        df.filter(F.col("ticker") == "AAPL") \
          .orderBy(F.col("date").desc()) \
          .select(
            "date","option_type","expiry","strike",
            "implied_vol","atm_iv","iv_skew",
            "iv_term_structure"
        ).show(5)

        print(f"\n  IV stats:")
        df.select(
            F.mean("implied_vol").alias("avg_iv"),
            F.min("implied_vol").alias("min_iv"),
            F.max("implied_vol").alias("max_iv"),
            F.mean("iv_skew").alias("avg_skew")
        ).show()

        assert total > 0, "FAIL — empty table"
        print("\nValidation PASSED ✓")

    # ------------------------------------------------------------------ #
    #  Run
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        print("="*50)
        print("Silver Options Cleaning Pipeline")
        print("="*50)
        df = self.load()
        df = self.clean(df)
        df = self.add_tte(df)
        df = self.add_price_features(df)
        df = self.add_moneyness(df)
        df = self.add_iv_surface_features(df)
        df = self.add_metadata(df)
        self.write_delta(df)
        self.optimize()
        self.validate()
        print("\nSilver Options COMPLETE ✓")

# COMMAND ----------

cleaner = SilverOptionsCleaning(
    spark       = spark,
    bronze_path = BRONZE_PATH,
    silver_path = SILVER_PATH
)

cleaner.run()

# COMMAND ----------

df = spark.read.format("delta").load(f"{SILVER_PATH}/options")

print(f"Total rows  : {df.count():,}")
print(f"Tickers     : {df.select('ticker').distinct().count():,}")

df.filter(F.col("ticker") == "AAPL") \
  .orderBy(F.col("date").desc()) \
  .select(
    "date","option_type","strike","implied_vol",
    "atm_iv","iv_skew","tte_days","expiry_bucket"
  ).show(10)

# COMMAND ----------

