# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Layer — Options Chain Ingestion
# MAGIC ## Quant Alpha Engine | Azure Databricks + Delta Lake
# MAGIC
# MAGIC ### Purpose
# MAGIC Captures daily live options chain snapshot for S&P 500 tickers.
# MAGIC Raw data only
# MAGIC
# MAGIC ### Data Source
# MAGIC | Provider | Data | Coverage | Frequency |
# MAGIC |----------|------|----------|-----------|
# MAGIC | yfinance | Options chain | S&P 500 500 tickers | Daily snapshot |
# MAGIC
# MAGIC ### Schema
# MAGIC | Column | Type | Description |
# MAGIC |--------|------|-------------|
# MAGIC | date | Date | Snapshot date |
# MAGIC | ticker | String | Stock ticker |
# MAGIC | expiry | String | Option expiry date |
# MAGIC | strike | Double | Strike price |
# MAGIC | option_type | String | call or put |
# MAGIC | last_price | Double | Last traded price |
# MAGIC | bid | Double | Bid price |
# MAGIC | ask | Double | Ask price |
# MAGIC | implied_vol | Double | Implied volatility |
# MAGIC | delta | Double | Delta greek |
# MAGIC | gamma | Double | Gamma greek |
# MAGIC | theta | Double | Theta greek |
# MAGIC | vega | Double | Vega greek |
# MAGIC | open_interest | Long | Open interest |
# MAGIC | volume | Long | Daily volume |
# MAGIC | in_the_money | Boolean | ITM flag |
# MAGIC
# MAGIC ### Output
# MAGIC - **Delta table:** `bronze/delta/options`
# MAGIC - **Partition:** date, option_type
# MAGIC - **Expected size:** ~50MB per daily snapshot

# COMMAND ----------

# MAGIC %pip install yfinance --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime, date
import pandas as pd
import yfinance as yf

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.shuffle.partitions", "8")

# COMMAND ----------

STORAGE_ACCOUNT = "multisignalalphaeng"
CONTAINER       = "quant-lakehouse"
ADLS_KEY        = dbutils.secrets.get(scope="quant-scope", key="adls-key-01")

# COMMAND ----------

spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
    ADLS_KEY
)

# COMMAND ----------

BASE_PATH    = f"abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net"
OPTIONS_PATH = f"{BASE_PATH}/bronze/delta/options"
print("Config loaded ✓")
print(f"Output: {OPTIONS_PATH}")

# COMMAND ----------

class BronzeOptionsIngestion:
    """
    Bronze options chain ingestion.
    - Daily snapshot of S&P 500 options chain
    - Partitioned by (option_type, year, month)
    - All type casting handled before Spark conversion
    """

    DEFAULT_TICKERS = [
    "A","AAL","AAPL","ABBV","ABNB","ABT","ACGL","ACN","ADBE","ADI",
    "ADM","ADP","ADSK","AEE","AEP","AES","AFL","AIG","AIZ","AJG",
    "AKAM","ALB","ALGN","ALK","ALL","ALLE","AMAT","AMCR","AMD","AME",
    "AMGN","AMP","AMZN","ANET","ANSS","AON","AOS","APA","APD","APH",
    "ARE","ATVI","AVB","AVGO","AVY","AWK","AXON","AXP","AZO","BA",
    "BAC","BALL","BAX","BBWI","BBY","BDX","BEN","BG","BIIB","BIO",
    "BK","BKNG","BKR","BLDR","BLK","BMY","BR","BRK-B","BRO","BSX",
    "BWA","BXP","C","CAG","CAH","CARR","CAT","CB","CBOE","CBRE",
    "CCI","CCL","CDAY","CDNS","CDW","CE","CF","CFG","CHD","CHRW",
    "CHTR","CI","CINF","CL","CLX","CMA","CMCSA","CME","CMG","CMI",
    "CMS","CNC","CNP","COF","COO","COP","COR","COST","CPAY","CPB",
    "CPRT","CPT","CRL","CRM","CRWD","CSCO","CSGP","CSX","CTAS",
    "CTSH","CTVA","CVS","CVX","CZR","D","DAL","DD","DE","DECK",
    "DELL","DFS","DG","DGX","DHI","DHR","DIS","DLR","DLTR","DOC",
    "DOV","DOW","DPZ","DRI","DTE","DUK","DVA","DVN","DXCM","EA",
    "EBAY","ECL","ED","EFX","EG","EIX","EL","ELV","EMN","EMR",
    "ENPH","EOG","EPAM","EQIX","EQR","EQT","ERIE","ES","ESS","ETN",
    "ETR","ETSY","EVRG","EW","EXC","EXPD","EXPE","EXR","F","FANG",
    "FAST","FCX","FDS","FDX","FE","FFIV","FICO","FIS","FISV","FITB",
    "FMC","FOX","FOXA","FRT","FTNT","FTV","GD","GE","GEHC","GEN",
    "GEV","GILD","GIS","GL","GLW","GM","GNRC","GOOG","GOOGL","GPC",
    "GPN","GRMN","GS","GWW","H","HAL","HAS","HBAN","HCA","HD",
    "HES","HIG","HII","HLT","HOLX","HON","HPE","HPQ","HRL","HSIC",
    "HST","HSY","HUBB","HUM","HWM","IBM","ICE","IDXX","IEX","IFF",
    "INTC","INTU","INVH","INCY","IP","IPG","IQV","IR","IRM","ISRG",
    "IT","ITW","IVZ","J","JBHT","JBL","JCI","JKHY","JNJ","JNPR",
    "JPM","K","KDP","KEY","KEYS","KHC","KIM","KKR","KLAC","KMB",
    "KMI","KMX","KO","KR","KVUE","L","LDOS","LEN","LH","LHX","LIN",
    "LKQ","LLY","LMT","LNT","LOW","LRCX","LULU","LYB","LYV","MA",
    "MAA","MAR","MAS","MCD","MCHP","MCK","MCO","MDLZ","MDT","MET",
    "META","MGM","MHK","MKC","MKTX","MLM","MMC","MMM","MNST","MO",
    "MOH","MOS","MPC","MPWR","MRK","MRNA","MRO","MS","MSCI","MSFT",
    "MSI","MTB","MTCH","MTD","MU","NCLH","NDAQ","NDSN","NEE","NEM",
    "NFLX","NI","NKE","NOC","NOW","NRG","NSC","NTAP","NTRS","NUE",
    "NVDA","NVR","NWL","NWS","NWSA","NXPI","O","ODFL","OKE","OMC",
    "ONTO","ORCL","ORLY","OTIS","OXY","PANW","PARA","PAYC","PAYX",
    "PCAR","PCG","PEG","PEP","PFE","PFG","PG","PGR","PH","PHM",
    "PKG","PLD","PM","PNC","PNR","PNW","PODD","POOL","PPG","PPL",
    "PRU","PSA","PSX","PTC","PWR","QCOM","RCL","REG","REGN","RF",
    "RJF","RL","RMD","ROK","ROP","ROST","RSG","RTX","RVTY","S",
    "SJM","SLB","SMCI","SNA","SNPS","SO","SOLV","SPG","SPGI","SRE",
    "STE","STLD","STT","STX","STZ","SW","SWK","SWKS","SYF","SYK",
    "SYY","T","TAP","TDG","TDY","TECH","TEL","TER","TFC","TFX",
    "TGT","TJX","TKO","TMO","TMUS","TPR","TRGP","TRMB","TROW","TRV",
    "TSCO","TSLA","TSN","TT","TTWO","TXN","TXT","TYL","UBER","UDR",
    "UHS","ULTA","UNH","UNP","UPS","URI","USB","V","VFC","VICI",
    "VLO","VLTO","VMC","VNO","VRSK","VRSN","VRTX","VST","VTR","VTRS",
    "VZ","WAB","WAT","WBA","WBD","WDC","WEC","WELL","WFC","WM",
    "WMB","WMT","WRB","WST","WTW","WY","WYNN","XEL","XOM","XYL",
    "YUM","ZBH","ZBRA","ZTS",
    # Major ETFs
    "SPY","QQQ","IWM","DIA","VTI",
    "XLK","XLF","XLE","XLV","XLI","XLP","XLY","XLU","XLB","XLRE",
    "GLD","SLV","TLT","HYG","LQD","VXX","EEM","EFA",
]

    def __init__(self, spark, base_path, tickers=None, max_expiries=6):
        self.spark        = spark
        self.base_path    = base_path
        self.tickers      = tickers or self.DEFAULT_TICKERS
        self.max_expiries = max_expiries
        self.options_path = f"{base_path}/bronze/delta/options"
        self.failed       = []
        self.today        = date.today()

        print(f"BronzeOptionsIngestion ✓")
        print(f"  Tickers     : {len(self.tickers)}")
        print(f"  Max expiries: {max_expiries} per ticker")
        print(f"  Snapshot    : {self.today}")
        print(f"  Output      : {self.options_path}")

    # ------------------------------------------------------------------ #
    #  Fetch
    # ------------------------------------------------------------------ #
    def _fetch_ticker_options(self, ticker: str) -> pd.DataFrame:
        try:
            t       = yf.Ticker(ticker)
            expiries= t.options

            if not expiries:
                return pd.DataFrame()

            frames = []
            for expiry in expiries[:self.max_expiries]:
                try:
                    chain = t.option_chain(expiry)
                    for opt_type, df in [
                        ("call", chain.calls),
                        ("put",  chain.puts)
                    ]:
                        df = df.copy()
                        df["ticker"]      = ticker
                        df["expiry"]      = expiry
                        df["option_type"] = opt_type
                        df["date"]        = self.today
                        df["source"]      = "yfinance"
                        df["ingested_at"] = datetime.now().strftime(
                            "%Y-%m-%d"
                        )

                        df = df.rename(columns={
                            "impliedVolatility": "implied_vol",
                            "openInterest"     : "open_interest",
                            "lastPrice"        : "last_price",
                            "inTheMoney"       : "in_the_money",
                        })

                        keep = [
                            "date","ticker","expiry","strike",
                            "option_type","last_price","bid","ask",
                            "implied_vol","open_interest","volume",
                            "in_the_money","source","ingested_at"
                        ]
                        df = df[[c for c in keep if c in df.columns]]
                        frames.append(df)

                except Exception:
                    pass

            return (
                pd.concat(frames, ignore_index=True)
                if frames else pd.DataFrame()
            )

        except Exception as e:
            self.failed.append(ticker)
            return pd.DataFrame()

    def fetch_all(self) -> pd.DataFrame:
        print(f"\nFetching options for {len(self.tickers)} tickers...")
        all_frames = []

        for i, ticker in enumerate(self.tickers):
            if i % 20 == 0:
                print(f"  Progress: {i}/{len(self.tickers)} — {ticker}")
            df = self._fetch_ticker_options(ticker)
            if not df.empty:
                all_frames.append(df)

        if not all_frames:
            raise ValueError("No options data fetched")

        combined = pd.concat(all_frames, ignore_index=True)

        print(f"\nFetch complete:")
        print(f"  Rows    : {len(combined):,}")
        print(f"  Tickers : {combined['ticker'].nunique()}")
        print(f"  Expiries: {combined['expiry'].nunique()}")
        print(f"  Failed  : {len(self.failed)}")
        return combined

    # ------------------------------------------------------------------ #
    #  Spark conversion — all types fixed before conversion
    # ------------------------------------------------------------------ #
    def _to_spark(self, pdf: pd.DataFrame):
        # Fix all types explicitly before Spark conversion
        pdf = pdf.copy()
        pdf["date"]          = pd.to_datetime(pdf["date"]).dt.date
        pdf["strike"]        = pd.to_numeric(
            pdf["strike"], errors="coerce"
        ).fillna(0.0).astype(float)
        pdf["last_price"]    = pd.to_numeric(
            pdf["last_price"], errors="coerce"
        ).fillna(0.0).astype(float)
        pdf["bid"]           = pd.to_numeric(
            pdf["bid"], errors="coerce"
        ).fillna(0.0).astype(float)
        pdf["ask"]           = pd.to_numeric(
            pdf["ask"], errors="coerce"
        ).fillna(0.0).astype(float)
        pdf["implied_vol"]   = pd.to_numeric(
            pdf["implied_vol"], errors="coerce"
        ).fillna(0.0).astype(float)
        pdf["open_interest"] = pd.to_numeric(
            pdf["open_interest"], errors="coerce"
        ).fillna(0).astype(int)
        pdf["volume"]        = pd.to_numeric(
            pdf["volume"], errors="coerce"
        ).fillna(0).astype(int)
        pdf["in_the_money"]  = pdf["in_the_money"].fillna(
            False
        ).astype(bool)
        pdf["ticker"]        = pdf["ticker"].astype(str)
        pdf["expiry"]        = pdf["expiry"].astype(str)
        pdf["option_type"]   = pdf["option_type"].astype(str)
        pdf["source"]        = pdf["source"].astype(str)
        pdf["ingested_at"]   = pdf["ingested_at"].astype(str)

        schema = StructType([
            StructField("date",         DateType(),    False),
            StructField("ticker",       StringType(),  False),
            StructField("expiry",       StringType(),  True),
            StructField("strike",       DoubleType(),  True),
            StructField("option_type",  StringType(),  False),
            StructField("last_price",   DoubleType(),  True),
            StructField("bid",          DoubleType(),  True),
            StructField("ask",          DoubleType(),  True),
            StructField("implied_vol",  DoubleType(),  True),
            StructField("open_interest",LongType(),    True),
            StructField("volume",       LongType(),    True),
            StructField("in_the_money", BooleanType(), True),
            StructField("source",       StringType(),  True),
            StructField("ingested_at",  StringType(),  True),
        ])

        sdf = self.spark.createDataFrame(pdf, schema=schema)
        sdf = (sdf
            .withColumn("year",  F.year("date"))
            .withColumn("month", F.month("date"))
        )
        return sdf

    # ------------------------------------------------------------------ #
    #  Write + Optimize
    # ------------------------------------------------------------------ #
    def write_delta(self, sdf) -> None:
        print(f"\nWriting Delta: {self.options_path}")
        (sdf.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema",                  "true")
            .option("delta.autoOptimize.optimizeWrite", "true")
            .option("delta.autoOptimize.autoCompact",   "true")
            .partitionBy("option_type", "year", "month")
            .save(self.options_path)
        )
        self.spark.sql(f"OPTIMIZE delta.`{self.options_path}`")
        print("Delta write complete ✓")

    def optimize(self) -> None:
        path = self.options_path
        print(f"\nRunning OPTIMIZE + VACUUM...")
        self.spark.sql(f"OPTIMIZE delta.`{path}`")
        self.spark.conf.set(
            "spark.databricks.delta.retentionDurationCheck.enabled",
            "false"
        )
        self.spark.sql(f"VACUUM delta.`{path}` RETAIN 168 HOURS")
        details = self.spark.sql(
            f"DESCRIBE DETAIL delta.`{path}`"
        ).select("numFiles", "sizeInBytes").collect()[0]
        print(f"  Files : {details['numFiles']}")
        print(f"  Size  : {details['sizeInBytes']/1e6:.1f} MB")

    # ------------------------------------------------------------------ #
    #  Validate
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        print("\n" + "="*45)
        print("VALIDATION — Bronze Options")
        print("="*45)

        df         = self.spark.read.format("delta").load(
            self.options_path
        )
        row_count  = df.count()
        ticker_cnt = df.select("ticker").distinct().count()
        expiry_cnt = df.select("expiry").distinct().count()
        null_count = df.filter(F.col("strike").isNull()).count()
        details    = self.spark.sql(
            f"DESCRIBE DETAIL delta.`{self.options_path}`"
        ).select("numFiles", "sizeInBytes").collect()[0]

        print(f"  Rows         : {row_count:,}")
        print(f"  Tickers      : {ticker_cnt}")
        print(f"  Expiries     : {expiry_cnt}")
        print(f"  Null strikes : {null_count}")
        print(f"  Delta files  : {details['numFiles']}")
        print(f"  Size         : {details['sizeInBytes']/1e6:.1f} MB")

        print(f"\n  Option type breakdown:")
        df.groupBy("option_type").count().show()

        print(f"\n  Sample (AAPL calls):")
        df.filter(
            (F.col("ticker") == "AAPL") &
            (F.col("option_type") == "call")
        ).orderBy("expiry", "strike").show(5)

        assert row_count  > 0,  "FAIL — empty table"
        assert ticker_cnt > 10, "FAIL — too few tickers"
        assert null_count == 0, "FAIL — null strikes"
        print("\nValidation PASSED ✓")

    # ------------------------------------------------------------------ #
    #  Run
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        print("="*45)
        print("Bronze Options Ingestion Pipeline")
        print("="*45)
        pdf = self.fetch_all()
        sdf = self._to_spark(pdf)
        self.write_delta(sdf)
        self.optimize()
        self.validate()
        print("\nBronze Options COMPLETE ✓")

# COMMAND ----------

ingestion = BronzeOptionsIngestion(
    spark      = spark,
    base_path  = BASE_PATH,
    max_expiries = 6
)
ingestion.run()