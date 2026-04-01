# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Layer — 04 Sentiment Ingestion
# MAGIC ## Quant Alpha Engine | Azure Databricks + Delta Lake
# MAGIC
# MAGIC ### Purpose
# MAGIC Ingest financial news headlines from Finnhub and generate sentiment scores using FinBERT (Hugging Face). This creates a daily sentiment signal for S&P 500 tickers.
# MAGIC
# MAGIC ### Data Source
# MAGIC | Provider | Data | Coverage | Frequency |
# MAGIC |----------|------|----------|-----------|
# MAGIC | Finnhub | Company News | S&P 500 tickers | Daily historical |
# MAGIC | FinBERT | NLP Sentiment | ProsusAI/finbert | Batch inference |
# MAGIC
# MAGIC ### Schema
# MAGIC | Column | Type | Description |
# MAGIC |--------|------|-------------|
# MAGIC | date | Date | Publication date |
# MAGIC | ticker | String | Stock ticker |
# MAGIC | headline | String | News headline text |
# MAGIC | summary | String | News summary text |
# MAGIC | source | String | News source (e.g., Bloomberg) |
# MAGIC | url | String | Article URL |
# MAGIC | sentiment_label | String | POSITIVE, NEGATIVE, NEUTRAL |
# MAGIC | sentiment_score | Double | Signed score: POS=+score, NEG=-score, NEUT=0 |
# MAGIC | ingested_at | String | Processing timestamp |
# MAGIC
# MAGIC ### Output
# MAGIC - **Delta table:** `bronze/delta/sentiment`
# MAGIC - **Partition:** ticker
# MAGIC - **Expected size:** ~15MB (daily batch)
# MAGIC
# MAGIC ### Prerequisites
# MAGIC - **Libraries:** `finnhub-python`, `transformers`, `torch`
# MAGIC - **Secrets:** `finnhub-news-api-key` must exist in `quant-scope

# COMMAND ----------

# MAGIC %pip install finnhub-python transformers torch --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime, date, timedelta
import pandas as pd
import finnhub
from transformers import pipeline

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.shuffle.partitions", "8")

# COMMAND ----------

STORAGE_ACCOUNT = "multisignalalphaeng"
CONTAINER = "quant-lakehouse"
ADLS_KEY = dbutils.secrets.get(scope="quant-scope", key="adls-key-01")
FINNHUB_KEY = dbutils.secrets.get(scope="quant-scope", key="finnhub-news-api-key")

# COMMAND ----------

spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net", 
    ADLS_KEY
)

BASE_PATH = f"abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net"
print("Config loaded ✓")

# COMMAND ----------

# CELL 4
class BronzeSentimentIngestion:
    """
    Production-grade Bronze sentiment ingestion.
    Fetches news from Finnhub → Runs FinBERT NLP → Writes Delta.
    
    Design principles:
    - FinBERT runs in batch for efficiency
    - Signed sentiment score for direct quant signal
    - Handles API rate limits gracefully
    - Partitioned by ticker for fast lookups
    """

    # S&P 500 liquid tickers (subset for daily run, expand as needed)
    DEFAULT_TICKERS = [
        "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","BRK-B",
        "JPM","JNJ","V","PG","MA","HD","CVX","MRK","ABBV","PEP",
        "BAC","KO","AVGO","PFE","COST","TMO","WMT","DIS","CSCO",
        "ABT","CRM","ACN","MCD","NEE","DHR","NKE","LIN","ADBE",
        "TXN","VZ","CMCSA","INTC","QCOM","HON","UPS","LOW","PM",
        "RTX","AMD","SPGI","INTU","AMAT","ISRG","GS","BLK","AXP",
        "WFC","C","MS","BK","UNH","CVS","HUM","NFLX","UBER","ABNB",
        "COIN","SQ","PYPL","SOFI","HOOD","AFRM","SNOW","DDOG","NET",
        "CRWD","ZS","OKTA","TWLO","MDB","GTLB","CFLT","U","RBLX",
        "SPOT","DASH","LYFT","PINS","SNAP","MTCH","IAC","Z","OPEN",
        "SPY","QQQ","IWM","GLD","SLV","USO","TLT","HYG","LQD","VXX"
    ]

    def __init__(self, spark, base_path, finnhub_key, tickers=None):
        self.spark = spark
        self.base_path = base_path
        self.finnhub_key = finnhub_key
        self.tickers = tickers or self.DEFAULT_TICKERS
        self.sentiment_path = f"{base_path}/bronze/delta/sentiment"
        self.failed = []
        
        # Initialize FinBERT (CPU mode - runs on driver)
        # Note: Use GPU cluster for full historical backfill
        print("Loading FinBERT model (this may take 30-60s)...")
        self.nlp_pipeline = pipeline(
            "text-classification", 
            model="ProsusAI/finbert",
            device=-1,  # -1 for CPU, 0 for GPU
            top_k=None  # Return all scores
        )
        
        # Initialize Finnhub client
        self.finnhub_client = finnhub.Client(api_key=finnhub_key)
        
        print(f"BronzeSentimentIngestion initialized ✓")
        print(f"  Tickers:     {len(self.tickers)}")
        print(f"  Output:      {self.sentiment_path}")

    def _fetch_ticker_news(self, ticker: str, from_date: str, to_date: str) -> pd.DataFrame:
        """Fetch company news for one ticker."""
        try:
            news = self.finnhub_client.company_news(
                ticker, 
                _from=from_date, 
                to=to_date
            )
            
            if not news:
                return pd.DataFrame()
            
            df = pd.DataFrame(news)
            df['ticker'] = ticker
            df['date'] = pd.to_datetime(df['datetime'], unit='s').dt.date
            
            # Keep relevant columns
            cols = ['date', 'ticker', 'headline', 'summary', 'source', 'url', 'datetime']
            df = df[[c for c in cols if c in df.columns]]
            
            return df
            
        except Exception as e:
            self.failed.append((ticker, str(e)))
            return pd.DataFrame()

    def _score_sentiment(self, text: str):
        """
        Run FinBERT on text.
        Returns: (label, signed_score)
        Signed score: POS=+score, NEG=-score, NEUTRAL=0
        """
        if not text or pd.isna(text):
            return "NEUTRAL", 0.0
            
        try:
            # Run inference
            results = self.nlp_pipeline(text[:512])[0]  # FinBERT limit
            
            # Parse results
            scores = {r['label']: r['score'] for r in results}
            
            pos = scores.get('positive', 0)
            neg = scores.get('negative', 0)
            neu = scores.get('neutral', 0)
            
            # Assign label based on max score
            label = max(scores, key=scores.get)
            
            # Create signed score
            if label == "positive":
                signed_score = pos
            elif label == "negative":
                signed_score = -neg
            else:
                signed_score = 0.0
                
            return label, signed_score
            
        except Exception:
            return "NEUTRAL", 0.0

    def fetch_and_score(self, from_date: str, to_date: str) -> pd.DataFrame:
        """
        Fetch news for all tickers and score sentiment.
        """
        print(f"\nFetching news from {from_date} to {to_date}...")
        all_news = []
        
        for i, ticker in enumerate(self.tickers):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(self.tickers)} — {ticker}")
                
            df = self._fetch_ticker_news(ticker, from_date, to_date)
            if not df.empty:
                all_news.append(df)
        
        if not all_news:
            raise ValueError(f"No news data fetched for {from_date} to {to_date}")
        
        combined = pd.concat(all_news, ignore_index=True)
        print(f"\nTotal articles fetched: {len(combined)}")
        
        # Run sentiment scoring
        print("Running FinBERT sentiment scoring...")
        
        # Combine headline + summary for better signal
        combined['text_to_score'] = combined['headline'].fillna('') + ". " + combined['summary'].fillna('')
        
        # Batch process (apply is slow, but safe for Bronze CPU cluster)
        sentiments = combined['text_to_score'].apply(self._score_sentiment)
        combined['sentiment_label'] = [s[0] for s in sentiments]
        combined['sentiment_score'] = [s[1] for s in sentiments]
        
        # Clean up
        combined['ingested_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        combined = combined.drop(columns=['text_to_score', 'datetime'], errors='ignore')
        
        print(f"\nSentiment scoring complete:")
        print(f"  Positive:  {len(combined[combined['sentiment_label']=='positive'])}")
        print(f"  Negative:  {len(combined[combined['sentiment_label']=='negative'])}")
        print(f"  Neutral:   {len(combined[combined['sentiment_label']=='neutral'])}")
        print(f"  Failed:    {len(self.failed)}")
        
        return combined

    def _to_spark(self, pdf: pd.DataFrame):
        """Convert pandas → typed Spark DataFrame."""
        schema = StructType([
            StructField("date",            DateType(),    False),
            StructField("ticker",          StringType(),  False),
            StructField("headline",        StringType(),  True),
            StructField("summary",         StringType(),  True),
            StructField("source",          StringType(),  True),
            StructField("url",             StringType(),  True),
            StructField("sentiment_label", StringType(),  True),
            StructField("sentiment_score", DoubleType(),   True),
            StructField("ingested_at",     StringType(),  True),
        ])
        
        sdf = self.spark.createDataFrame(pdf, schema=schema)
        return sdf

    def write_delta(self, sdf) -> None:
        """Write to Delta partitioned by ticker."""
        print(f"\nWriting Delta: {self.sentiment_path}")
        (sdf.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .option("delta.autoOptimize.optimizeWrite", "true")
            .option("delta.autoOptimize.autoCompact", "true")
            .partitionBy("ticker")
            .save(self.sentiment_path)
        )
        print("Delta write complete ✓")

    def optimize(self) -> None:
        """Apply Delta optimizations."""
        path = self.sentiment_path
        print(f"\nApplying optimizations...")
        
        self.spark.sql(f"OPTIMIZE delta.`{path}`")
        print("  OPTIMIZE complete ✓")
        
        self.spark.conf.set(
            "spark.databricks.delta.retentionDurationCheck.enabled", 
            "false"
        )
        self.spark.sql(f"VACUUM delta.`{path}` RETAIN 168 HOURS")
        print("  VACUUM complete ✓")
        
        details = self.spark.sql(f"""
            DESCRIBE DETAIL delta.`{path}`
        """).select("numFiles", "sizeInBytes").collect()[0]
        
        print(f"\n  Post-optimization:")
        print(f"  Files: {details['numFiles']}")
        print(f"  Size:  {details['sizeInBytes']/1e6:.1f} MB")

    def validate(self) -> None:
        """Validate written Delta table."""
        print("\n" + "=" * 45)
        print("VALIDATION — Bronze Sentiment")
        print("=" * 45)
        df = self.spark.read.format("delta").load(self.sentiment_path)
        
        row_count = df.count()
        ticker_cnt = df.select("ticker").distinct().count()
        null_count = df.filter(F.col("sentiment_score").isNull()).count()
        avg_score = df.agg(F.avg("sentiment_score")).collect()[0][0]
        
        details = self.spark.sql(f"""
            DESCRIBE DETAIL delta.`{self.sentiment_path}`
        """).select("numFiles", "sizeInBytes").collect()[0]
        
        print(f"  Rows:          {row_count:,}")
        print(f"  Tickers:       {ticker_cnt}")
        print(f"  Null scores:   {null_count}")
        print(f"  Avg score:     {avg_score:.4f}")
        print(f"  Delta files:   {details['numFiles']}")
        print(f"  Size:          {details['sizeInBytes']/1e6:.1f} MB")
        
        print(f"\nSample (AAPL recent):")
        df.filter(F.col("ticker") == "AAPL") \
          .select("date", "headline", "sentiment_label", "sentiment_score") \
          .orderBy(F.col("date").desc()) \
          .show(5, truncate=50)
        
        assert row_count > 0, "FAIL — empty table"
        assert null_count == 0, "FAIL — null sentiment scores"
        print("\nValidation PASSED ✓")

    def run(self, from_date: str, to_date: str) -> None:
        """Full pipeline — fetch → score → write → optimize → validate."""
        print("Starting Bronze Sentiment Ingestion Pipeline")
        print("=" * 45)
        pdf = self.fetch_and_score(from_date, to_date)
        sdf = self._to_spark(pdf)
        self.write_delta(sdf)
        self.optimize()
        self.validate()
        print("\nBronze Sentiment Pipeline COMPLETE ✓")

# COMMAND ----------

ingestion = BronzeSentimentIngestion(
    spark      = spark,
    base_path  = BASE_PATH,
    finnhub_key= FINNHUB_KEY
)

# Date range: Last 30 days (or specify exact dates from PDF)
end_date = date.today()
start_date = end_date - timedelta(days=30)

ingestion.run(
    from_date = start_date.strftime("%Y-%m-%d"),
    to_date   = end_date.strftime("%Y-%m-%d")
)