# Databricks notebook source
# MAGIC %pip install requests pandas --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime, date, timedelta
import pandas as pd
import requests
import time
import os
import io

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.shuffle.partitions", "8")
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

STORAGE_ACCOUNT = "multisignalalphaeng"
CONTAINER       = "quant-lakehouse"
ADLS_KEY        = dbutils.secrets.get(scope="quant-scope", key="adls-key-01")

spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
    ADLS_KEY
)

BASE_PATH = f"abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net"
print(f"Config loaded ✓")
print(f"Base path: {BASE_PATH}")

# COMMAND ----------

class BronzeCBOEImpliedVolatility:
    """
    Bronze CBOE Implied Volatility ingestion.
    Sources:
      - VIX daily history       (CBOE free CSV)
      - VIX9D daily history     (9-day VIX)
      - VIX3M daily history     (3-month VIX)
      - VIX6M daily history     (6-month VIX)
      - VVIX daily history      (VIX of VIX)
      - SKEW daily history      (CBOE Skew Index)
      - PUT/CALL ratio history  (CBOE equity P/C ratio)
    All free, no API key required.
    Expected size : ~50-100MB Delta
    Expected time : ~5-10 minutes
    """

    # CBOE free data URLs
    CBOE_URLS = {
        "VIX"   : "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
        "VIX9D" : "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv",
        "VIX3M" : "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv",
        "VIX6M" : "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX6M_History.csv",
        "VVIX"  : "https://cdn.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv",
        "SKEW"  : "https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv",
    }

    # CBOE Put/Call ratio
    PC_URL = "https://cdn.cboe.com/data/us/options/market_statistics/daily_historical/equity_pc_ratio.csv"

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Accept"    : "text/html,application/xhtml+xml,*/*",
    }

    def __init__(self, spark, base_path):
        self.spark     = spark
        self.base_path = base_path
        self.path      = f"{base_path}/bronze/delta/cboe_iv"
        self.failed    = []

        print(f"BronzeCBOEImpliedVolatility ✓")
        print(f"  Sources : VIX, VIX9D, VIX3M, VIX6M, VVIX, SKEW, P/C")
        print(f"  Output  : {self.path}")
        print(f"  Est time: ~5-10 minutes")

    # ------------------------------------------------------------------ #
    #  Fetch VIX-style indices
    # ------------------------------------------------------------------ #
    def _fetch_cboe_index(self, name: str, url: str) -> pd.DataFrame:
        """Fetch a single CBOE index CSV."""
        try:
            print(f"  Fetching {name}...")
            r = requests.get(url, headers=self.HEADERS, timeout=30)

            if r.status_code != 200:
                print(f"  ✗ {name} failed: HTTP {r.status_code}")
                self.failed.append(name)
                return pd.DataFrame()

            # Parse CSV — CBOE CSVs sometimes have header rows to skip
            content = r.text
            lines   = content.strip().split("\n")

            # Find the actual header row
            header_idx = 0
            for i, line in enumerate(lines):
                if "DATE" in line.upper() or "date" in line.lower():
                    header_idx = i
                    break

            df = pd.read_csv(
                io.StringIO("\n".join(lines[header_idx:])),
                on_bad_lines="skip"
            )

            if df.empty:
                self.failed.append(name)
                return pd.DataFrame()

            # Normalize columns
            df.columns = [c.strip().upper() for c in df.columns]

            # Find date column
            date_col = None
            for col in df.columns:
                if "DATE" in col:
                    date_col = col
                    break

            if not date_col:
                print(f"  ✗ {name}: no date column found")
                self.failed.append(name)
                return pd.DataFrame()

            df = df.rename(columns={date_col: "date"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df["date"]  = df["date"].dt.date
            df["year"]  = pd.to_datetime(df["date"]).dt.year
            df["month"] = pd.to_datetime(df["date"]).dt.month
            df["index_name"] = name

            # Normalize OHLC columns
            col_map = {}
            for col in df.columns:
                if col in ["date","year","month","index_name"]:
                    continue
                col_lower = col.strip().lower()
                if "open"  in col_lower: col_map[col] = "open"
                elif "high"  in col_lower: col_map[col] = "high"
                elif "low"   in col_lower: col_map[col] = "low"
                elif "close" in col_lower: col_map[col] = "close"

            df = df.rename(columns=col_map)

            # Keep only standard columns
            keep = ["date","year","month","index_name",
                    "open","high","low","close"]
            df = df[[c for c in keep if c in df.columns]]

            # Convert to float
            for col in ["open","high","low","close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(
                        df[col], errors="coerce"
                    ).fillna(0.0).astype(float)

            df = df.dropna(subset=["close"])
            df = df[df["close"] > 0]

            print(f"  ✓ {name}: {len(df):,} rows "
                  f"({df['date'].min()} → {df['date'].max()})")
            return df

        except Exception as e:
            print(f"  ✗ {name} error: {e}")
            self.failed.append(name)
            return pd.DataFrame()

    # ------------------------------------------------------------------ #
    #  Fetch Put/Call ratio
    # ------------------------------------------------------------------ #
    def _fetch_put_call_ratio(self) -> pd.DataFrame:
        """Fetch CBOE equity put/call ratio."""
        try:
            print(f"  Fetching PUT/CALL ratio...")
            r = requests.get(
                self.PC_URL, headers=self.HEADERS, timeout=30
            )

            if r.status_code != 200:
                # Try alternate URL
                alt_url = (
                    "https://www.cboe.com/trading/data/EquityPC.csv"
                )
                r = requests.get(
                    alt_url, headers=self.HEADERS, timeout=30
                )

            if r.status_code != 200:
                print(f"  ✗ PUT/CALL: HTTP {r.status_code}")
                self.failed.append("PUT_CALL")
                return pd.DataFrame()

            content = r.text
            lines   = content.strip().split("\n")

            header_idx = 0
            for i, line in enumerate(lines):
                if "DATE" in line.upper():
                    header_idx = i
                    break

            df = pd.read_csv(
                io.StringIO("\n".join(lines[header_idx:])),
                on_bad_lines="skip"
            )

            if df.empty:
                self.failed.append("PUT_CALL")
                return pd.DataFrame()

            df.columns = [c.strip().upper() for c in df.columns]

            # Find date and ratio columns
            date_col  = None
            ratio_col = None
            for col in df.columns:
                if "DATE" in col:
                    date_col = col
                if "RATIO" in col or "P/C" in col or "PUT" in col:
                    ratio_col = col

            if not date_col:
                self.failed.append("PUT_CALL")
                return pd.DataFrame()

            df = df.rename(columns={date_col: "date"})
            if ratio_col:
                df = df.rename(columns={ratio_col: "close"})

            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df["date"]       = df["date"].dt.date
            df["year"]       = pd.to_datetime(df["date"]).dt.year
            df["month"]      = pd.to_datetime(df["date"]).dt.month
            df["index_name"] = "PUT_CALL_RATIO"

            if "close" not in df.columns:
                # Take first numeric column as close
                for col in df.columns:
                    if col not in ["date","year","month","index_name"]:
                        try:
                            df["close"] = pd.to_numeric(
                                df[col], errors="coerce"
                            )
                            break
                        except Exception:
                            pass

            keep = ["date","year","month","index_name","close"]
            df   = df[[c for c in keep if c in df.columns]]

            if "open"  not in df.columns: df["open"]  = df["close"]
            if "high"  not in df.columns: df["high"]  = df["close"]
            if "low"   not in df.columns: df["low"]   = df["close"]

            df["close"] = pd.to_numeric(
                df["close"], errors="coerce"
            ).fillna(0.0)
            df = df.dropna(subset=["close"])
            df = df[df["close"] > 0]

            print(f"  ✓ PUT_CALL_RATIO: {len(df):,} rows "
                  f"({df['date'].min()} → {df['date'].max()})")
            return df

        except Exception as e:
            print(f"  ✗ PUT/CALL error: {e}")
            self.failed.append("PUT_CALL")
            return pd.DataFrame()

    # ------------------------------------------------------------------ #
    #  Fetch all
    # ------------------------------------------------------------------ #
    def fetch_all(self) -> pd.DataFrame:
        print(f"\nFetching CBOE data...")
        all_frames = []

        # Fetch VIX indices
        for name, url in self.CBOE_URLS.items():
            df = self._fetch_cboe_index(name, url)
            if not df.empty:
                all_frames.append(df)
            time.sleep(0.5)  # polite delay

        # Fetch Put/Call ratio
        df_pc = self._fetch_put_call_ratio()
        if not df_pc.empty:
            all_frames.append(df_pc)

        if not all_frames:
            raise ValueError("No CBOE data fetched")

        combined = pd.concat(all_frames, ignore_index=True)

        print(f"\nFetch complete:")
        print(f"  Total rows   : {len(combined):,}")
        print(f"  Indices      : {combined['index_name'].nunique()}")
        print(f"  Index list   : {combined['index_name'].unique().tolist()}")
        print(f"  Failed       : {self.failed}")
        return combined

    # ------------------------------------------------------------------ #
    #  Spark
    # ------------------------------------------------------------------ #
    def _to_spark(self, pdf: pd.DataFrame):
        pdf = pdf.copy()
        pdf["date"]       = pd.to_datetime(pdf["date"]).dt.date
        pdf["year"]       = pdf["year"].astype(int)
        pdf["month"]      = pdf["month"].astype(int)
        pdf["index_name"] = pdf["index_name"].astype(str)
        for col in ["open","high","low","close"]:
            if col in pdf.columns:
                pdf[col] = pd.to_numeric(
                    pdf[col], errors="coerce"
                ).fillna(0.0).astype(float)
            else:
                pdf[col] = 0.0

        schema = StructType([
            StructField("date",       DateType(),    False),
            StructField("year",       IntegerType(), False),
            StructField("month",      IntegerType(), False),
            StructField("index_name", StringType(),  False),
            StructField("open",       DoubleType(),  True),
            StructField("high",       DoubleType(),  True),
            StructField("low",        DoubleType(),  True),
            StructField("close",      DoubleType(),  True),
        ])
        return self.spark.createDataFrame(pdf, schema=schema)

    # ------------------------------------------------------------------ #
    #  Write + Optimize
    # ------------------------------------------------------------------ #
    def write_delta(self, sdf) -> None:
        print(f"\nWriting Delta: {self.path}")
        (sdf.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema",                  "true")
            .option("delta.autoOptimize.optimizeWrite", "true")
            .option("delta.autoOptimize.autoCompact",   "true")
            .partitionBy("index_name", "year")
            .save(self.path)
        )
        self.spark.sql(f"OPTIMIZE delta.`{self.path}`")
        print("Write complete ✓")

    def optimize(self) -> None:
        print("\nOPTIMIZE + VACUUM...")
        self.spark.sql(f"OPTIMIZE delta.`{self.path}`")
        self.spark.conf.set(
            "spark.databricks.delta.retentionDurationCheck.enabled",
            "false"
        )
        self.spark.sql(
            f"VACUUM delta.`{self.path}` RETAIN 168 HOURS"
        )
        details = self.spark.sql(
            f"DESCRIBE DETAIL delta.`{self.path}`"
        ).select("numFiles","sizeInBytes").collect()[0]
        print(f"  Files : {details['numFiles']}")
        print(f"  Size  : {details['sizeInBytes']/1e6:.1f} MB")

    # ------------------------------------------------------------------ #
    #  Validate
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        print("\n" + "="*45)
        print("VALIDATION — Bronze CBOE IV")
        print("="*45)
        df    = self.spark.read.format("delta").load(self.path)
        total = df.count()

        print(f"  Total rows  : {total:,}")
        print(f"  Indices     : {df.select('index_name').distinct().count()}")
        print(f"\n  Rows per index:")
        df.groupBy("index_name").agg(
            F.count("*").alias("rows"),
            F.min("date").alias("from"),
            F.max("date").alias("to"),
            F.avg("close").alias("avg_close")
        ).orderBy("index_name").show()

        print(f"\n  Latest VIX values:")
        df.filter(F.col("index_name") == "VIX") \
          .orderBy(F.col("date").desc()) \
          .select("date","open","high","low","close") \
          .show(5)

        assert total > 0, "FAIL — empty table"
        print("\nValidation PASSED ✓")

    # ------------------------------------------------------------------ #
    #  Run
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        print("="*45)
        print("Bronze CBOE IV Pipeline")
        print("="*45)
        pdf = self.fetch_all()
        sdf = self._to_spark(pdf)
        self.write_delta(sdf)
        self.optimize()
        self.validate()
        print("\nBronze CBOE IV COMPLETE ✓")

# COMMAND ----------

ingestion = BronzeCBOEImpliedVolatility(
    spark     = spark,
    base_path = BASE_PATH
)

ingestion.run()

# COMMAND ----------

df = spark.read.format("delta").load(
    f"{BASE_PATH}/bronze/delta/cboe_iv"
)

print(f"Total rows : {df.count():,}")
df.groupBy("index_name").agg(
    F.count("*").alias("rows"),
    F.min("date").alias("from"),
    F.max("date").alias("to"),
    F.avg("close").cast("decimal(10,2)").alias("avg_close")
).orderBy("index_name").show()

# Latest values
df.orderBy(F.col("date").desc()) \
  .select("date","index_name","close") \
  .show(10)