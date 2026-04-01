# Databricks notebook source
# MAGIC %pip install yfinance --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime, date, timedelta
import pandas as pd
import yfinance as yf
import time

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.shuffle.partitions", "8")

STORAGE_ACCOUNT = "multisignalalphaeng"
CONTAINER = "quant-lakehouse"
ADLS_KEY = dbutils.secrets.get(scope="quant-scope", key="adls-key-01")

spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net", 
    ADLS_KEY
)

BASE_PATH = f"abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net"
print("Config loaded ✓")

# COMMAND ----------

class BronzeIntraday1HrIngestion:
    """
    Ingests 1-hour bars from yfinance.
    - S&P 500 (509 tickers) + major ETFs
    - period = 2y (max allowed for 1h interval)
    - Realistic output: ~60-80MB Delta compressed
    """

    SP500_TICKERS = [
        "A","AAL","AAPL","ABBV","ABNB","ABT","ACGL","ACN","ADBE","ADI","ADM","ADP","ADSK",
        "AEE","AEP","AES","AFL","AIG","AIZ","AJG","AKAM","ALB","ALGN","ALK","ALL","ALLE",
        "AMAT","AMCR","AMD","AME","AMGN","AMP","AMZN","ANET","ANSS","AON","AOS","APA","APD",
        "APH","ARE","ATVI","AVB","AVGO","AVY","AWK","AXON","AXP","AZO","BA","BAC","BALL",
        "BAX","BBWI","BBY","BDX","BEN","BG","BIIB","BIO","BK","BKNG","BKR","BLDR","BLK",
        "BMY","BR","BRK-B","BRO","BSX","BWA","BXP","C","CAG","CAH","CARR","CAT","CB","CBOE",
        "CBRE","CCI","CCL","CDAY","CDNS","CDW","CE","CF","CFG","CHD","CHRW","CHTR","CI",
        "CINF","CL","CLX","CMA","CMCSA","CME","CMG","CMI","CMS","CNC","CNP","COF","COO",
        "COP","COR","COST","CPAY","CPB","CPRT","CPT","CRL","CRM","CRWD","CSCO","CSGP","CSX",
        "CTAS","CTSH","CTVA","CVS","CVX","CZR","D","DAL","DD","DE","DECK","DELL","DFS","DG",
        "DGX","DHI","DHR","DIS","DLR","DLTR","DOC","DOV","DOW","DPZ","DRI","DTE","DUK","DVA",
        "DVN","DXCM","EA","EBAY","ECL","ED","EFX","EG","EIX","EL","ELV","EMN","EMR","ENPH",
        "EOG","EPAM","EQIX","EQR","EQT","ERIE","ES","ESS","ETN","ETR","ETSY","EVRG","EW",
        "EXC","EXPD","EXPE","EXR","F","FANG","FAST","FCX","FDS","FDX","FE","FFIV","FICO",
        "FIS","FISV","FITB","FMC","FOX","FOXA","FRT","FTNT","FTV","GD","GE","GEHC","GEN",
        "GEV","GILD","GIS","GL","GLW","GM","GNRC","GOOG","GOOGL","GPC","GPN","GRMN","GS",
        "GWW","H","HAL","HAS","HBAN","HCA","HD","HES","HIG","HII","HLT","HOLX","HON","HPE",
        "HPQ","HRL","HSIC","HST","HSY","HUBB","HUM","HWM","IBM","ICE","IDXX","IEX","IFF",
        "INTC","INTU","INVH","INCY","IP","IPG","IQV","IR","IRM","ISRG","IT","ITW","IVZ",
        "J","JBHT","JBL","JCI","JKHY","JNJ","JNPR","JPM","K","KDP","KEY","KEYS","KHC","KIM",
        "KKR","KLAC","KMB","KMI","KMX","KO","KR","KVUE","L","LDOS","LEN","LH","LHX","LIN",
        "LKQ","LLY","LMT","LNT","LOW","LRCX","LULU","LYB","LYV","MA","MAA","MAR","MAS",
        "MCD","MCHP","MCK","MCO","MDLZ","MDT","MET","META","MGM","MHK","MKC","MKTX","MLM",
        "MMC","MMM","MNST","MO","MOH","MOS","MPC","MPWR","MRK","MRNA","MRO","MS","MSCI",
        "MSFT","MSI","MTB","MTCH","MTD","MU","NCLH","NDAQ","NDSN","NEE","NEM","NFLX","NI",
        "NKE","NOC","NOW","NRG","NSC","NTAP","NTRS","NUE","NVDA","NVR","NWL","NWS","NWSA",
        "NXPI","O","ODFL","OKE","OMC","ONTO","ORCL","ORLY","OTIS","OXY","PANW","PARA","PAYC",
        "PAYX","PCAR","PCG","PEG","PEP","PFE","PFG","PG","PGR","PH","PHM","PKG","PLD","PM",
        "PNC","PNR","PNW","PODD","POOL","PPG","PPL","PRU","PSA","PSX","PTC","PWR","QCOM",
        "RCL","REG","REGN","RF","RJF","RL","RMD","ROK","ROP","ROST","RSG","RTX","RVTY",
        "S","SJM","SLB","SMCI","SNA","SNPS","SO","SOLV","SPG","SPGI","SRE","STE","STLD",
        "STT","STX","STZ","SW","SWK","SWKS","SYF","SYK","SYY","T","TAP","TDG","TDY","TECH",
        "TEL","TER","TFC","TFX","TGT","TJX","TKO","TMO","TMUS","TPR","TRGP","TRMB","TROW",
        "TRV","TSCO","TSLA","TSN","TT","TTWO","TXN","TXT","TYL","UBER","UDR","UHS","ULTA",
        "UNH","UNP","UPS","URI","USB","V","VFC","VICI","VLO","VLTO","VMC","VNO","VRSK",
        "VRSN","VRTX","VST","VTR","VTRS","VZ","WAB","WAT","WBA","WBD","WDC","WEC","WELL",
        "WFC","WM","WMB","WMT","WRB","WST","WTW","WY","WYNN","XEL","XOM","XYL","YUM",
        "ZBH","ZBRA","ZTS"
    ]

    # Major ETFs — adds breadth and useful benchmark signals for Gold features
    ETF_TICKERS = [
        "SPY","QQQ","IWM","DIA","VTI",          # broad market
        "XLK","XLF","XLE","XLV","XLI",          # sector ETFs
        "XLP","XLY","XLU","XLB","XLRE",         # remaining sectors
        "GLD","SLV","TLT","HYG","LQD",          # commodities + bonds
        "VXX","UVXY",                            # vol ETFs
        "EEM","EFA","FXI",                       # international
    ]

    def __init__(self, spark, base_path: str, include_etfs: bool = True):
        self.spark      = spark
        self.base_path  = base_path
        self.path       = f"{base_path}/bronze/delta/intraday_1hr"
        self.failed     = []

        self.tickers = self.SP500_TICKERS.copy()
        if include_etfs:
            self.tickers += self.ETF_TICKERS

        # Deduplicate
        self.tickers = list(dict.fromkeys(self.tickers))

        print(f"BronzeIntraday1HrIngestion initialized ✓")
        print(f"  S&P 500 tickers : {len(self.SP500_TICKERS)}")
        print(f"  ETFs added      : {len(self.ETF_TICKERS) if include_etfs else 0}")
        print(f"  Total tickers   : {len(self.tickers)}")
        print(f"  Interval        : 1h | Period: 2y (~504 trading days)")
        print(f"  Expected rows   : ~{len(self.tickers) * 504 * 7:,} (est.)")

    # ------------------------------------------------------------------ #
    #  Fetch
    # ------------------------------------------------------------------ #
    def _fetch_ticker_1h(self, ticker: str) -> pd.DataFrame:
        """Fetch 1-hour bars for one ticker. Max = 2y for 1h interval."""
        try:
            t  = yf.Ticker(ticker)
            df = t.history(
                period     = "2y",
                interval   = "1h",
                auto_adjust= True,
                prepost    = False   # regular market hours only
            )

            if df is None or df.empty:
                self.failed.append((ticker, "empty response"))
                return pd.DataFrame()

            df = df.reset_index()

            # Identify datetime column — yfinance returns 'Datetime' for intraday
            if "Datetime" in df.columns:
                df["timestamp"] = pd.to_datetime(df["Datetime"], utc=True).dt.tz_localize(None)
            elif "Date" in df.columns:
                df["timestamp"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
            else:
                self.failed.append((ticker, "no datetime column"))
                return pd.DataFrame()

            df["ticker"] = ticker
            df.columns   = [c.lower() for c in df.columns]

            # Partition columns
            df["date"]  = df["timestamp"].dt.date
            df["year"]  = df["timestamp"].dt.year
            df["month"] = df["timestamp"].dt.month

            keep = ["timestamp","ticker","open","high","low","close","volume","date","year","month"]
            df   = df[[c for c in keep if c in df.columns]]
            df   = df.dropna(subset=["close"])

            return df

        except Exception as e:
            self.failed.append((ticker, str(e)))
            return pd.DataFrame()

    def fetch_all(self) -> pd.DataFrame:
        print(f"\nFetching 1h bars for {len(self.tickers)} tickers...")
        print("Estimated time: ~10-15 minutes")

        all_frames = []

        for i, ticker in enumerate(self.tickers):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(self.tickers)}")

            df = self._fetch_ticker_1h(ticker)
            if not df.empty:
                all_frames.append(df)

            # Rate limit — polite delay every 10 tickers
            if i % 10 == 9:
                time.sleep(0.3)

        if not all_frames:
            raise ValueError("No hourly data fetched. Check yfinance availability.")

        combined = pd.concat(all_frames, ignore_index=True)

        print(f"\nFetch complete:")
        print(f"  Total rows      : {len(combined):,}")
        print(f"  Unique tickers  : {combined['ticker'].nunique()}")
        print(f"  Date range      : {combined['date'].min()} → {combined['date'].max()}")
        print(f"  Failed tickers  : {len(self.failed)}")
        if self.failed:
            print(f"  Failed sample   : {[t for t,_ in self.failed[:10]]}")

        return combined

    # ------------------------------------------------------------------ #
    #  Spark
    # ------------------------------------------------------------------ #
    def _to_spark(self, pdf: pd.DataFrame):
        schema = StructType([
            StructField("timestamp", TimestampType(), False),
            StructField("ticker",    StringType(),    False),
            StructField("open",      DoubleType(),    True),
            StructField("high",      DoubleType(),    True),
            StructField("low",       DoubleType(),    True),
            StructField("close",     DoubleType(),    True),
            StructField("volume",    LongType(),      True),
            StructField("date",      DateType(),      False),
            StructField("year",      IntegerType(),   False),
            StructField("month",     IntegerType(),   False),
        ])
        return self.spark.createDataFrame(pdf, schema=schema)

    # ------------------------------------------------------------------ #
    #  Write
    # ------------------------------------------------------------------ #
    def write_delta(self, sdf) -> None:
        print(f"\nWriting to Delta: {self.path}")

        (sdf.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .option("delta.autoOptimize.optimizeWrite", "true")
            .option("delta.autoOptimize.autoCompact",   "true")
            .partitionBy("year", "month")       # same pattern as ohlcv
            .save(self.path)
        )

        self.spark.sql(f"OPTIMIZE delta.`{self.path}`")
        print("Delta write complete ✓")

    # ------------------------------------------------------------------ #
    #  Optimize
    # ------------------------------------------------------------------ #
    def optimize(self) -> None:
        print("\nRunning OPTIMIZE + VACUUM...")
        self.spark.sql(f"OPTIMIZE delta.`{self.path}`")
        self.spark.conf.set(
            "spark.databricks.delta.retentionDurationCheck.enabled", "false"
        )
        self.spark.sql(f"VACUUM delta.`{self.path}` RETAIN 168 HOURS")

        details = self.spark.sql(
            f"DESCRIBE DETAIL delta.`{self.path}`"
        ).select("numFiles", "sizeInBytes").collect()[0]
        print(f"  Files : {details['numFiles']}")
        print(f"  Size  : {details['sizeInBytes']/1e6:.1f} MB")

    # ------------------------------------------------------------------ #
    #  Validate
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        print("\n" + "=" * 50)
        print("VALIDATION — Bronze Intraday 1-Hr")
        print("=" * 50)

        df = self.spark.read.format("delta").load(self.path)

        row_count   = df.count()
        ticker_cnt  = df.select("ticker").distinct().count()
        date_range  = df.agg(F.min("date"), F.max("date")).collect()[0]
        null_close  = df.filter(F.col("close").isNull()).count()

        print(f"  Rows            : {row_count:,}")
        print(f"  Unique tickers  : {ticker_cnt}")
        print(f"  Date range      : {date_range[0]} → {date_range[1]}")
        print(f"  Null closes     : {null_close}")

        # Bars per ticker sanity check
        print("\nBars per ticker (sample):")
        df.groupBy("ticker").count() \
          .filter(F.col("ticker").isin("AAPL","MSFT","NVDA","SPY","QQQ")) \
          .orderBy("ticker") \
          .show()

        # Volume sanity
        print("Sample rows (AAPL):")
        df.filter(F.col("ticker") == "AAPL") \
          .orderBy(F.col("timestamp").desc()) \
          .select("timestamp","open","high","low","close","volume") \
          .show(5)

        assert row_count > 0,   "FAIL — empty table"
        assert null_close == 0, "FAIL — null closes found"
        print("\nValidation PASSED ✓")

    # ------------------------------------------------------------------ #
    #  Run
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        print("=" * 50)
        print("Bronze Intraday 1-Hr Pipeline")
        print("=" * 50)

        pdf = self.fetch_all()
        sdf = self._to_spark(pdf)
        self.write_delta(sdf)
        self.optimize()
        self.validate()

        print("\nBronze 06 — Intraday 1-Hr COMPLETE ✓")

# COMMAND ----------

ingestion = BronzeIntraday1HrIngestion(
    spark       = spark,
    base_path   = BASE_PATH,
    include_etfs= True     
)

ingestion.run()

# COMMAND ----------

df = spark.read.format("delta").load(f"{BASE_PATH}/bronze/delta/intraday_1hr")

print("Bronze 06 Summary")
print("=" * 40)
print(f"Total rows     : {df.count():,}")
print(f"Total tickers  : {df.select('ticker').distinct().count()}")
print(f"Date range     : {df.agg(F.min('date'), F.max('date')).collect()[0]}")
print(f"Years covered  : {df.select('year').distinct().orderBy('year').collect()}")

# Rows per year
df.groupBy("year").count().orderBy("year").show()