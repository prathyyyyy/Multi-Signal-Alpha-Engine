# Databricks notebook source
# MAGIC %pip install pykalman scipy numpy pandas --quiet

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
from pykalman import KalmanFilter

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

class SilverMacroCleaning:
    """
    Silver Macro Cleaning Pipeline.
    Operations:
      1. Load bronze macro (FRED series)
      2. Forward fill missing dates (weekends/holidays)
      3. Align all series to daily frequency
      4. Apply Kalman filter to smooth noisy series
      5. Calculate derived features:
         - Yield curve spread (10Y - 2Y)
         - Real rate (DGS10 - CPI YoY)
         - Credit spread changes
         - VIX regime labels
      6. Write to silver/delta/macro
    """

    # Macro series and their types
    SERIES_META = {
        "DGS10"        : {"type": "rate",   "desc": "10Y Treasury"},
        "DGS2"         : {"type": "rate",   "desc": "2Y Treasury"},
        "T10Y2Y"       : {"type": "spread", "desc": "Yield Curve"},
        "VIXCLS"       : {"type": "index",  "desc": "VIX"},
        "DCOILWTICO"   : {"type": "price",  "desc": "WTI Crude"},
        "CPIAUCSL"     : {"type": "level",  "desc": "CPI"},
        "UNRATE"       : {"type": "rate",   "desc": "Unemployment"},
        "FEDFUNDS"     : {"type": "rate",   "desc": "Fed Funds"},
        "BAMLH0A0HYM2" : {"type": "spread", "desc": "HY Spread"},
        "DTWEXBGS"     : {"type": "index",  "desc": "USD Index"},
    }

    def __init__(self, spark, bronze_path, silver_path):
        self.spark       = spark
        self.bronze_path = f"{bronze_path}/macro"
        self.silver_path = f"{silver_path}/macro"

        print("SilverMacroCleaning ✓")
        print(f"  Bronze : {self.bronze_path}")
        print(f"  Silver : {self.silver_path}")

    # ------------------------------------------------------------------ #
    #  Load
    # ------------------------------------------------------------------ #
    def load(self):
        print("\nLoading bronze macro...")
        df = self.spark.read.format("delta").load(self.bronze_path)
        print(f"  Rows   : {df.count():,}")
        print(f"  Series : {df.select('series_id').distinct().count()}")
        df.groupBy("series_id").count().orderBy("series_id").show()
        return df

    # ------------------------------------------------------------------ #
    #  Pivot to wide format
    # ------------------------------------------------------------------ #
    def pivot_wide(self, df) -> pd.DataFrame:
        """Convert long format → wide format (one column per series)."""
        print("\nPivoting to wide format...")

        # Convert to pandas for easier manipulation
        pdf = df.select(
            "date","series_id","value"
        ).toPandas()

        pdf["date"]  = pd.to_datetime(pdf["date"])
        pdf["value"] = pd.to_numeric(pdf["value"], errors="coerce")

        # Pivot
        wide = pdf.pivot_table(
            index   = "date",
            columns = "series_id",
            values  = "value",
            aggfunc = "mean"
        )

        wide = wide.sort_index()
        print(f"  Wide shape : {wide.shape}")
        print(f"  Date range : {wide.index.min()} → {wide.index.max()}")
        return wide

    # ------------------------------------------------------------------ #
    #  Forward fill to daily frequency
    # ------------------------------------------------------------------ #
    def forward_fill(self, wide: pd.DataFrame) -> pd.DataFrame:
        """Resample to daily and forward fill."""
        print("\nForward filling to daily frequency...")

        # Create full daily date range
        full_idx = pd.date_range(
            start = wide.index.min(),
            end   = wide.index.max(),
            freq  = "D"
        )

        wide = wide.reindex(full_idx)

        # Forward fill — max 7 days to avoid stale data
        wide = wide.fillna(method="ffill", limit=7)

        print(f"  Rows after ffill : {len(wide):,}")
        print(f"  Null pct         : "
              f"{wide.isnull().sum().sum() / wide.size * 100:.1f}%")
        return wide

    # ------------------------------------------------------------------ #
    #  Kalman filter smoothing
    # ------------------------------------------------------------------ #
    def kalman_smooth(self, series: pd.Series) -> pd.Series:
        """Apply Kalman filter to smooth a single series."""
        try:
            values = series.dropna().values.reshape(-1, 1)
            if len(values) < 10:
                return series

            kf = KalmanFilter(
                transition_matrices    = [[1]],
                observation_matrices   = [[1]],
                initial_state_mean     = values[0],
                initial_state_covariance = [[1]],
                observation_covariance   = [[1]],
                transition_covariance    = [[0.1]]
            )
            smoothed, _ = kf.smooth(values)
            result = series.copy()
            result[series.notna()] = smoothed.flatten()
            return result
        except Exception:
            return series

    def apply_kalman(self, wide: pd.DataFrame) -> pd.DataFrame:
        """Apply Kalman smoothing to rate/spread series."""
        print("\nApplying Kalman filter smoothing...")

        # Apply to rate and spread series only
        smooth_series = [
            "DGS10","DGS2","T10Y2Y",
            "FEDFUNDS","BAMLH0A0HYM2"
        ]

        for col in smooth_series:
            if col in wide.columns:
                wide[f"{col}_kalman"] = self.kalman_smooth(wide[col])
                print(f"  ✓ {col} smoothed")

        print("  Kalman smoothing complete ✓")
        return wide

    # ------------------------------------------------------------------ #
    #  Derived features
    # ------------------------------------------------------------------ #
    def add_derived_features(self, wide: pd.DataFrame) -> pd.DataFrame:
        """Calculate macro derived features."""
        print("\nCalculating derived features...")

        # Yield curve spread (if not already present)
        if "DGS10" in wide.columns and "DGS2" in wide.columns:
            wide["yield_spread_10y2y"] = wide["DGS10"] - wide["DGS2"]
            print("  ✓ yield_spread_10y2y")

        # CPI YoY change
        if "CPIAUCSL" in wide.columns:
            wide["cpi_yoy"] = wide["CPIAUCSL"].pct_change(252) * 100
            print("  ✓ cpi_yoy")

        # Real rate (10Y - CPI YoY)
        if "DGS10" in wide.columns and "cpi_yoy" in wide.columns:
            wide["real_rate_10y"] = wide["DGS10"] - wide["cpi_yoy"]
            print("  ✓ real_rate_10y")

        # VIX regime (low/medium/high/extreme)
        if "VIXCLS" in wide.columns:
            wide["vix_regime"] = pd.cut(
                wide["VIXCLS"],
                bins   = [0, 15, 20, 30, 100],
                labels = [1, 2, 3, 4]   # 1=low, 4=extreme
            ).astype(float)
            print("  ✓ vix_regime")

        # VIX 1-day change
        if "VIXCLS" in wide.columns:
            wide["vix_change_1d"] = wide["VIXCLS"].pct_change(1)
            print("  ✓ vix_change_1d")

        # HY spread change
        if "BAMLH0A0HYM2" in wide.columns:
            wide["hy_spread_change"] = wide["BAMLH0A0HYM2"].diff(1)
            print("  ✓ hy_spread_change")

        # Oil price change
        if "DCOILWTICO" in wide.columns:
            wide["oil_change_1d"] = wide["DCOILWTICO"].pct_change(1)
            print("  ✓ oil_change_1d")

        # USD change
        if "DTWEXBGS" in wide.columns:
            wide["usd_change_1d"] = wide["DTWEXBGS"].pct_change(1)
            print("  ✓ usd_change_1d")

        # Fed rate change
        if "FEDFUNDS" in wide.columns:
            wide["fedfunds_change"] = wide["FEDFUNDS"].diff(1)
            print("  ✓ fedfunds_change")

        print("  Derived features complete ✓")
        return wide

    # ------------------------------------------------------------------ #
    #  Convert back to Spark
    # ------------------------------------------------------------------ #
    def to_spark(self, wide: pd.DataFrame):
        """Convert wide pandas DataFrame back to Spark."""
        print("\nConverting to Spark...")

        # Reset index
        wide = wide.reset_index()
        wide = wide.rename(columns={"index": "date"})

        # Add partition columns
        wide["date"]  = pd.to_datetime(wide["date"]).dt.date
        wide["year"]  = pd.to_datetime(wide["date"]).dt.year
        wide["month"] = pd.to_datetime(wide["date"]).dt.month
        wide["cleaned_at"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Convert all numeric columns to float
        for col in wide.columns:
            if col not in ["date","year","month","cleaned_at"]:
                wide[col] = pd.to_numeric(
                    wide[col], errors="coerce"
                ).astype(float)

        sdf = self.spark.createDataFrame(wide)
        print(f"  Columns : {len(sdf.columns)}")
        print(f"  Rows    : {sdf.count():,}")
        return sdf

    # ------------------------------------------------------------------ #
    #  Write
    # ------------------------------------------------------------------ #
    def write_delta(self, sdf) -> None:
        print(f"\nWriting Silver Macro: {self.silver_path}")
        (sdf.write
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
        print("VALIDATION — Silver Macro")
        print("="*50)

        df    = self.spark.read.format("delta").load(self.silver_path)
        total = df.count()

        print(f"  Total rows : {total:,}")
        print(f"  Columns    : {len(df.columns)}")
        print(f"  Date range : "
              f"{df.agg(F.min('date'), F.max('date')).collect()[0]}")

        print(f"\n  Latest macro values:")
        df.orderBy(F.col("date").desc()) \
          .select(
            "date","VIXCLS","DGS10","DGS2",
            "yield_spread_10y2y","FEDFUNDS",
            "BAMLH0A0HYM2","vix_regime"
        ).show(5)

        print(f"\n  Null counts per key column:")
        for col in ["VIXCLS","DGS10","DGS2","FEDFUNDS",
                    "yield_spread_10y2y","vix_regime"]:
            if col in df.columns:
                null_cnt = df.filter(F.col(col).isNull()).count()
                print(f"    {col}: {null_cnt:,} nulls")

        assert total > 0, "FAIL — empty table"
        print("\nValidation PASSED ✓")

    # ------------------------------------------------------------------ #
    #  Run
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        print("="*50)
        print("Silver Macro Cleaning Pipeline")
        print("="*50)
        df   = self.load()
        wide = self.pivot_wide(df)
        wide = self.forward_fill(wide)
        wide = self.apply_kalman(wide)
        wide = self.add_derived_features(wide)
        sdf  = self.to_spark(wide)
        self.write_delta(sdf)
        self.optimize()
        self.validate()
        print("\nSilver Macro COMPLETE ✓")

# COMMAND ----------

cleaner = SilverMacroCleaning(
    spark       = spark,
    bronze_path = BRONZE_PATH,
    silver_path = SILVER_PATH
)

cleaner.run()

# COMMAND ----------

df = spark.read.format("delta").load(f"{SILVER_PATH}/macro")

print(f"Total rows : {df.count():,}")
print(f"Columns    : {df.columns}")

df.orderBy(F.col("date").desc()).show(5)