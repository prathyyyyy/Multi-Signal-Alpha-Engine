# Databricks notebook source
# MAGIC %pip install fredapi --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime
import pandas as pd
from fredapi import Fred

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.shuffle.partitions", "8")

# COMMAND ----------

STORAGE_ACCOUNT = "multisignalalphaeng"
CONTAINER       = "quant-lakehouse"
ADLS_KEY        = dbutils.secrets.get(scope="quant-scope", key="adls-key-01")
FRED_KEY        = dbutils.secrets.get(scope="quant-scope", key="fred-api")

spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
    ADLS_KEY
)

BASE_PATH  = f"abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net"
MACRO_PATH = f"{BASE_PATH}/bronze/delta/macro"

print("Config loaded ✓")

# COMMAND ----------

class BronzeMacroIngestion:
    """
    Production-grade Bronze macro data ingestion.
    Pulls FRED series → Spark DataFrame → Delta Lake
    partitioned by series_id.

    Design principles:
    - Raw data only — no transformations at Bronze
    - DataFrame writer only — Unity Catalog compatible
    - Full audit trail per row
    - Assertions in validation catch silent failures
    - Post-write Delta optimizations
    """

    FRED_SERIES = {
        "DGS10":        "10Y Treasury Yield",
        "DGS2":         "2Y Treasury Yield",
        "T10Y2Y":       "Yield Curve Spread 10Y-2Y",
        "VIXCLS":       "VIX Close",
        "DCOILWTICO":   "WTI Crude Oil Price",
        "CPIAUCSL":     "CPI All Items",
        "UNRATE":       "Unemployment Rate",
        "FEDFUNDS":     "Federal Funds Rate",
        "BAMLH0A0HYM2": "HY Credit Spread",
        "DTWEXBGS":     "USD Broad Index",
    }

    def __init__(self, spark, fred_key, base_path,
                 start_date="1990-01-01",
                 end_date="2026-01-31"):
        self.spark      = spark
        self.fred       = Fred(api_key=fred_key)
        self.base_path  = base_path
        self.start_date = start_date
        self.end_date   = end_date
        self.macro_path = f"{base_path}/bronze/delta/macro"
        self.errors     = []

        print("BronzeMacroIngestion initialized ✓")
        print(f"  Series:     {len(self.FRED_SERIES)}")
        print(f"  Date range: {start_date} → {end_date}")
        print(f"  Output:     {self.macro_path}")

    def _fetch_series(self, series_id: str,
                      series_name: str) -> "pd.DataFrame":
        """Fetch single FRED series → clean pandas DataFrame."""
        try:
            series    = self.fred.get_series(
                series_id,
                observation_start=self.start_date,
                observation_end=self.end_date,
            )
            info      = self.fred.get_series_info(series_id)
            frequency = info.get("frequency_short", "D")

            df                = series.reset_index()
            df.columns        = ["date", "value"]
            df["series_id"]   = series_id
            df["series_name"] = series_name
            df["frequency"]   = frequency
            df["source"]      = "FRED"
            df["ingested_at"] = datetime.now().strftime("%Y-%m-%d")
            df = df.dropna(subset=["value"])
            df["date"] = pd.to_datetime(df["date"]).dt.date

            print(f"  ✓ {series_id:20}  {len(df):>6,} obs | {frequency}")
            return df

        except Exception as e:
            self.errors.append(series_id)
            print(f"  ✗ {series_id:20}  FAILED — {e}")
            return pd.DataFrame()

    def fetch_all(self) -> "pd.DataFrame":
        """Fetch all FRED series → combined DataFrame."""
        print(f"\nFetching {len(self.FRED_SERIES)} FRED series...")
        frames = []
        for series_id, series_name in self.FRED_SERIES.items():
            df = self._fetch_series(series_id, series_name)
            if not df.empty:
                frames.append(df)

        if not frames:
            raise ValueError("No macro data fetched — check FRED API key")

        combined = pd.concat(frames, ignore_index=True)
        combined = combined[[
            "date", "series_id", "series_name",
            "value", "frequency", "source", "ingested_at"
        ]]

        print(f"\nFetch complete:")
        print(f"  Total rows:  {len(combined):,}")
        print(f"  Series ok:   {combined['series_id'].nunique()}")
        print(f"  Failed:      {len(self.errors)} — {self.errors}")
        return combined

    def _to_spark(self, pdf: "pd.DataFrame"):
        """Convert pandas → typed Spark DataFrame."""
        schema = StructType([
            StructField("date",        DateType(),   False),
            StructField("series_id",   StringType(), False),
            StructField("series_name", StringType(), True),
            StructField("value",       DoubleType(), True),
            StructField("frequency",   StringType(), True),
            StructField("source",      StringType(), True),
            StructField("ingested_at", StringType(), True),
        ])
        sdf = self.spark.createDataFrame(pdf, schema=schema)
        sdf = (sdf
            .withColumn("year",  F.year("date"))
            .withColumn("month", F.month("date"))
        )
        return sdf

    def write_delta(self, sdf) -> None:
        """
        Write to Delta using DataFrame writer only.
        No SQL DDL — fully Unity Catalog compatible.
        Partitioned by series_id for query performance.
        """
        print(f"\nWriting Delta: {self.macro_path}")

        (sdf.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema",                 "true")
            .option("delta.enableChangeDataFeed",      "true")
            .option("delta.autoOptimize.optimizeWrite","true")
            .option("delta.autoOptimize.autoCompact",  "true")
            .partitionBy("series_id")
            .save(self.macro_path)
        )
        print("Delta write complete ✓")

        print("Running OPTIMIZE...")
        self.spark.sql(f"OPTIMIZE delta.`{self.macro_path}`")
        print("OPTIMIZE complete ✓")

    def optimize(self) -> None:
        """Apply Delta optimizations after write."""
        path = self.macro_path
        print(f"\nApplying Delta optimizations...")

        # OPTIMIZE
        self.spark.sql(f"OPTIMIZE delta.`{path}`")
        print("  OPTIMIZE complete ✓")

        # VACUUM old versions
        self.spark.conf.set(
            "spark.databricks.delta.retentionDurationCheck.enabled",
            "false"
        )
        self.spark.sql(f"VACUUM delta.`{path}` RETAIN 168 HOURS")
        print("  VACUUM complete ✓")

        # Size check
        details = self.spark.sql(f"""
            DESCRIBE DETAIL delta.`{path}`
        """).select("numFiles", "sizeInBytes").collect()[0]

        print(f"\n  Post-optimization:")
        print(f"  Files: {details['numFiles']}")
        print(f"  Size:  {details['sizeInBytes']/1e6:.1f} MB")

    def validate(self) -> None:
        """Validate written Delta table."""
        print("\n" + "=" * 45)
        print("VALIDATION — Bronze Macro")
        print("=" * 45)
        df         = self.spark.read.format("delta").load(self.macro_path)
        row_count  = df.count()
        series_cnt = df.select("series_id").distinct().count()
        date_stats = df.agg(
            F.min("date").alias("min_date"),
            F.max("date").alias("max_date")
        ).collect()[0]
        null_count = df.filter(F.col("value").isNull()).count()
        details    = self.spark.sql(f"""
            DESCRIBE DETAIL delta.`{self.macro_path}`
        """).select("numFiles", "sizeInBytes").collect()[0]

        print(f"  Rows:               {row_count:,}")
        print(f"  Series:             {series_cnt}")
        print(f"  Date range:         {date_stats['min_date']} → {date_stats['max_date']}")
        print(f"  Null values:        {null_count}")
        print(f"  Delta files:        {details['numFiles']}")
        print(f"  Size:               {details['sizeInBytes']/1e6:.1f} MB")

        print(f"\nSample (T10Y2Y — yield curve):")
        df.filter(F.col("series_id") == "T10Y2Y") \
          .orderBy(F.col("date").desc()) \
          .show(5)

        assert row_count  > 0,  "FAIL — empty table"
        assert null_count == 0, "FAIL — nulls in value column"
        assert series_cnt >= 8, "FAIL — missing series"
        print("\nValidation PASSED ✓")

    def run(self) -> None:
        """Full pipeline — fetch → convert → write → optimize → validate."""
        print("Starting Bronze Macro Ingestion Pipeline")
        print("=" * 45)
        pdf = self.fetch_all()
        sdf = self._to_spark(pdf)
        self.write_delta(sdf)
        self.optimize()
        self.validate()
        print("\nBronze Macro Pipeline COMPLETE ✓")

# COMMAND ----------

ingestion = BronzeMacroIngestion(
    spark      = spark,
    fred_key   = FRED_KEY,
    base_path  = BASE_PATH,
    start_date = "1990-01-01",
    end_date   = "2026-01-31"
)

ingestion.run()

# COMMAND ----------

# Check current Bronze file sizes
for table, path in [
    ("OHLCV",  f"{BASE_PATH}/bronze/delta/ohlcv"),
    ("Macro",  f"{BASE_PATH}/bronze/delta/macro"),
]:
    details = spark.sql(f"""
        DESCRIBE DETAIL delta.`{path}`
    """).select(
        "numFiles",
        "sizeInBytes", 
        "clusteringColumns"
    ).collect()[0]
    
    print(f"{table}:")
    print(f"  Files:       {details['numFiles']}")
    print(f"  Size:        {details['sizeInBytes']/1e6:.1f} MB")
    print(f"  Clustering:  {details['clusteringColumns']}")
    print()