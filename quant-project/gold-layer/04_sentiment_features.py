# Databricks notebook source
# MAGIC %pip install scipy statsmodels==0.14.5 plotly pandas numpy --quiet

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

class GoldSentimentFeatures:
    """
    Gold 04 — Sentiment Features.

    Data situation:
      Silver sentiment : 1,251 rows, 59 tickers,
                         2026-02-19 → 2026-03-21
      OHLCV max date   : 2026-01-30

    Strategy:
      1. Load silver sentiment (ticker-level)
      2. Aggregate all features per ticker
      3. Join to Gold price factors on ticker
      4. Add sentiment IC features
      5. Add regime-conditional sentiment
      6. Save as Gold sentiment features

    Features built:
      Base         : sentiment_weighted, bullish_ratio,
                     bearish_ratio, news_count
      Rolling      : 3d/7d/14d rolling averages
      Momentum     : sentiment_momentum_1d/3d
      CS           : sentiment_rank, sentiment_zscore
      Regime       : sentiment × regime interaction
      Signal       : sentiment_strength, conviction
    """

    def __init__(self, spark, silver_path,
                 eda_path, gold_path):
        self.spark       = spark
        self.silver_path = silver_path
        self.eda_path    = eda_path
        self.gold_path   = f"{gold_path}/sentiment_features"
        self.price_path  = f"{gold_path}/price_factors"
        print("GoldSentimentFeatures ✓")
        print(f"  Output : {self.gold_path}")

    # ------------------------------------------------------------------ #
    #  Step 1 — Load silver sentiment
    # ------------------------------------------------------------------ #
    def load_sentiment(self):
        print("\nStep 1: Loading silver sentiment...")
        start = datetime.now()

        sent = self.spark.read.format("delta").load(
            f"{self.silver_path}/sentiment"
        ).withColumn("date", F.to_date(F.col("date")))

        total   = sent.count()
        tickers = sent.select("ticker").distinct().count()
        elapsed = (datetime.now() - start).seconds

        dr = sent.agg(
            F.min("date").alias("min"),
            F.max("date").alias("max")
        ).collect()[0]

        print(f"  Rows    : {total:,}")
        print(f"  Tickers : {tickers:,}")
        print(f"  Dates   : {dr['min']} → {dr['max']}")
        print(f"  Elapsed : {elapsed}s")
        print(f"  Columns : {sent.columns}")
        return sent

    # ------------------------------------------------------------------ #
    #  Step 2 — Aggregate per ticker (cross-sectional)
    # ------------------------------------------------------------------ #
    def aggregate_per_ticker(self, sent):
        print("\nStep 2: Aggregating per ticker...")

        # Available columns check
        avail = sent.columns

        agg_exprs = [
            F.mean("sentiment_weighted").alias(
                "sentiment_weighted"
            ),
            F.mean("sentiment_mean").alias(
                "sentiment_mean"
            ),
            F.mean("bullish_ratio").alias(
                "bullish_ratio"
            ),
            F.mean("bearish_ratio").alias(
                "bearish_ratio"
            ),
            F.sum("news_count").alias("total_news"),
            F.mean("news_count").alias("avg_daily_news"),
            F.count("*").alias("n_sentiment_days"),
            F.max("date").alias("last_sent_date"),
            F.min("date").alias("first_sent_date"),
            F.stddev("sentiment_weighted").alias(
                "sentiment_volatility"
            ),
            F.max("sentiment_weighted").alias(
                "sentiment_max"
            ),
            F.min("sentiment_weighted").alias(
                "sentiment_min"
            ),
        ]

        # Optional columns
        optional = {
            "sentiment_3d"         : "sentiment_3d_avg",
            "sentiment_7d"         : "sentiment_7d_avg",
            "sentiment_14d"        : "sentiment_14d_avg",
            "sentiment_rank"       : "sentiment_rank_avg",
            "sentiment_zscore"     : "sentiment_zscore_avg",
            "sentiment_momentum_1d": "sentiment_mom_1d_avg",
            "sentiment_momentum_3d": "sentiment_mom_3d_avg",
            "sentiment_trend"      : "sentiment_trend_avg",
            "bullish_ratio_7d"     : "bullish_ratio_7d",
            "neutral_ratio"        : "neutral_ratio_avg",
            "news_volume_rank"     : "news_volume_rank_avg",
        }
        for col, alias in optional.items():
            if col in avail:
                agg_exprs.append(
                    F.mean(col).alias(alias)
                )

        ticker_agg = sent.groupBy("ticker").agg(*agg_exprs)

        # Derived features
        ticker_agg = ticker_agg.withColumn(
            "sentiment_range",
            F.col("sentiment_max") - F.col("sentiment_min")
        ).withColumn(
            "bull_bear_ratio",
            F.col("bullish_ratio") /
            (F.col("bearish_ratio") + F.lit(1e-8))
        ).withColumn(
            "sentiment_conviction",
            F.abs(F.col("sentiment_weighted")) *
            F.col("avg_daily_news")
        ).withColumn(
            "is_bullish",
            F.when(
                F.col("sentiment_weighted") > 0.1,
                F.lit(1)
            ).otherwise(F.lit(0))
        ).withColumn(
            "is_bearish",
            F.when(
                F.col("sentiment_weighted") < -0.1,
                F.lit(1)
            ).otherwise(F.lit(0))
        ).withColumn(
            "news_coverage_days",
            F.col("n_sentiment_days")
        )

        count = ticker_agg.count()
        print(f"  Ticker aggregates : {count:,}")
        return ticker_agg

    # ------------------------------------------------------------------ #
    #  Step 3 — Add cross-sectional features
    # ------------------------------------------------------------------ #
    def add_cross_sectional(self, ticker_agg):
        print("\nStep 3: Cross-sectional features...")

        # Single partition — all tickers are one group
        w = Window.partitionBy(F.lit(1))

        cs_factors = [
            "sentiment_weighted",
            "bullish_ratio",
            "bearish_ratio",
            "sentiment_volatility",
            "sentiment_conviction",
            "bull_bear_ratio",
            "avg_daily_news",
            "total_news",
        ]

        for factor in cs_factors:
            if factor not in ticker_agg.columns:
                continue
            ticker_agg = ticker_agg.withColumn(
                f"{factor}_rank",
                F.percent_rank().over(
                    w.orderBy(factor)
                )
            ).withColumn(
                f"{factor}_zscore",
                (F.col(factor) -
                 F.mean(factor).over(w)) /
                (F.stddev(factor).over(w) +
                 F.lit(1e-8))
            )

        print("  CS features added ✓")
        return ticker_agg

    # ------------------------------------------------------------------ #
    #  Step 4 — Join to Gold price factors
    # ------------------------------------------------------------------ #
    def join_to_price_factors(self, ticker_agg):
        print("\nStep 4: Joining to Gold price factors...")

        try:
            # Get latest row per ticker from price factors
            price_factors = self.spark.read.format(
                "delta"
            ).load(self.price_path)

            w_last = Window.partitionBy("ticker").orderBy(
                F.desc("date")
            )
            price_latest = price_factors.withColumn(
                "rn", F.row_number().over(w_last)
            ).filter(F.col("rn") == 1).drop("rn")

            # Select key columns from price factors
            price_cols = [
                "ticker","date",
                "mom_21d","mom_252d",
                "vol_21d","sharpe_21d",
                "rsi_14d","price_to_ma20",
                "regime_label","prob_bull",
                "prob_bear","prob_highvol",
                "position_size_weight",
                "fwd_return_21d",
            ]
            price_avail = [
                c for c in price_cols
                if c in price_latest.columns
            ]
            price_select = price_latest.select(
                *price_avail
            ).withColumnRenamed("date","ohlcv_date")

            # Join on ticker
            joined = ticker_agg.join(
                price_select,
                on="ticker",
                how="left"
            )

            joined_count = joined.count()
            print(f"  Joined rows : {joined_count:,}")

            # Rename date col
            if "ohlcv_date" in joined.columns:
                joined = joined.withColumnRenamed(
                    "ohlcv_date","price_date"
                )

            return joined

        except Exception as e:
            print(f"  Price factor join failed: {e}")
            print(f"  Continuing without price join")
            return ticker_agg

    # ------------------------------------------------------------------ #
    #  Step 5 — Regime-conditional sentiment
    # ------------------------------------------------------------------ #
    def add_regime_sentiment(self, df):
        print("\nStep 5: Regime-conditional features...")

        if "regime_label" not in df.columns:
            print("  No regime_label — skipping")
            return df

        # Sentiment × regime interaction
        df = df.withColumn(
            "sentiment_bull_regime",
            F.when(
                F.col("regime_label") == "Bull",
                F.col("sentiment_weighted")
            ).otherwise(F.lit(0.0))
        ).withColumn(
            "sentiment_bear_regime",
            F.when(
                F.col("regime_label") == "Bear",
                F.col("sentiment_weighted")
            ).otherwise(F.lit(0.0))
        ).withColumn(
            "sentiment_highvol_regime",
            F.when(
                F.col("regime_label") == "HighVol",
                F.col("sentiment_weighted")
            ).otherwise(F.lit(0.0))
        )

        # Regime-scaled position signal
        if "position_size_weight" in df.columns:
            df = df.withColumn(
                "sentiment_position_signal",
                F.col("sentiment_weighted") *
                F.col("position_size_weight")
            )

        # Sentiment alignment with regime
        df = df.withColumn(
            "regime_sentiment_aligned",
            F.when(
                (F.col("regime_label") == "Bull") &
                (F.col("sentiment_weighted") > 0),
                F.lit(1)
            ).when(
                (F.col("regime_label") == "Bear") &
                (F.col("sentiment_weighted") < 0),
                F.lit(1)
            ).otherwise(F.lit(0))
        )

        # Combined sentiment + momentum signal
        if "mom_21d" in df.columns:
            df = df.withColumn(
                "sentiment_momentum_combo",
                F.col("sentiment_weighted") +
                F.col("mom_21d")
            )

        print("  Regime sentiment features added ✓")
        return df

    # ------------------------------------------------------------------ #
    #  Step 6 — Add signal strength features
    # ------------------------------------------------------------------ #
    def add_signal_strength(self, df):
        print("\nStep 6: Signal strength features...")

        w = Window.partitionBy(F.lit(1))

        # Overall sentiment signal strength
        df = df.withColumn(
            "sentiment_signal_strength",
            F.abs(F.col("sentiment_weighted")) *
            F.col("bullish_ratio").cast("double")
        )

        # High conviction = strong sentiment + high news
        if "total_news" in df.columns:
            df = df.withColumn(
                "high_conviction_flag",
                F.when(
                    (F.abs(F.col("sentiment_weighted"))
                     > 0.3) &
                    (F.col("total_news") > 5),
                    F.lit(1)
                ).otherwise(F.lit(0))
            )

        # Sentiment quality score
        df = df.withColumn(
            "sentiment_quality",
            F.abs(F.col("sentiment_weighted")) *
            F.col("n_sentiment_days").cast("double")
        )

        # Contrarian signal (extreme sentiment reversal)
        df = df.withColumn(
            "contrarian_signal",
            F.when(
                F.col("sentiment_weighted") > 0.5,
                F.lit(-1)   # extremely bullish → fade
            ).when(
                F.col("sentiment_weighted") < -0.5,
                F.lit(1)    # extremely bearish → buy
            ).otherwise(F.lit(0))
        )

        # CS rank of signal strength
        df = df.withColumn(
            "sentiment_signal_rank",
            F.percent_rank().over(
                w.orderBy("sentiment_signal_strength")
            )
        )

        print("  Signal strength features added ✓")
        return df

    # ------------------------------------------------------------------ #
    #  Step 7 — Metadata
    # ------------------------------------------------------------------ #
    def add_metadata(self, df):
        df = df.withColumn(
            "gold_created_at",
            F.lit(
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )
        ).withColumn(
            "year",
            F.year(F.col("last_sent_date"))
        ).withColumn(
            "month",
            F.month(F.col("last_sent_date"))
        )
        return df

    # ------------------------------------------------------------------ #
    #  Write
    # ------------------------------------------------------------------ #
    def write(self, df) -> None:
        print(f"\nWriting Gold Sentiment Features...")
        print(f"  Path : {self.gold_path}")

        (df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema",                  "true")
            .option("delta.autoOptimize.optimizeWrite", "true")
            .option("delta.autoOptimize.autoCompact",   "true")
            .save(self.gold_path)
        )

        self.spark.sql(
            f"OPTIMIZE delta.`{self.gold_path}`"
        )
        self.spark.conf.set(
            "spark.databricks.delta.retentionDurationCheck.enabled",
            "false"
        )
        self.spark.sql(
            f"VACUUM delta.`{self.gold_path}` RETAIN 168 HOURS"
        )

        details = self.spark.sql(
            f"DESCRIBE DETAIL delta.`{self.gold_path}`"
        ).select("numFiles","sizeInBytes").collect()[0]
        print(f"  Files : {details['numFiles']}")
        print(f"  Size  : "
              f"{details['sizeInBytes']/1e6:.1f} MB")
        print("  Write complete ✓")

    # ------------------------------------------------------------------ #
    #  Validate
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        print("\n" + "="*55)
        print("VALIDATION — Gold Sentiment Features")
        print("="*55)

        df      = self.spark.read.format("delta").load(
            self.gold_path
        )
        total   = df.count()
        tickers = df.select("ticker").distinct().count()

        print(f"\n  Total rows  : {total:,}")
        print(f"  Tickers     : {tickers:,}")
        print(f"  Columns     : {len(df.columns):,}")

        print(f"\n  Sentiment stats:")
        df.select(
            F.mean("sentiment_weighted").alias(
                "mean_sentiment"
            ),
            F.mean("bullish_ratio").alias(
                "mean_bull_ratio"
            ),
            F.mean("avg_daily_news").alias(
                "mean_daily_news"
            ),
            F.sum("high_conviction_flag").alias(
                "high_conviction_count"
            )
        ).show()

        print(f"\n  Top 10 most bullish tickers:")
        key_cols = [
            "ticker","sentiment_weighted",
            "bullish_ratio","total_news",
            "sentiment_conviction",
            "high_conviction_flag"
        ]
        avail = [
            c for c in key_cols if c in df.columns
        ]
        df.orderBy(
            F.desc("sentiment_weighted")
        ).select(*avail).show(10)

        print(f"\n  Top 10 most bearish tickers:")
        df.orderBy(
            F.asc("sentiment_weighted")
        ).select(*avail).show(10)

        if "regime_label" in df.columns:
            print(f"\n  Regime-sentiment alignment:")
            df.groupBy("regime_label").agg(
                F.count("*").alias("n_tickers"),
                F.mean("sentiment_weighted").alias(
                    "avg_sentiment"
                ),
                F.sum("regime_sentiment_aligned").alias(
                    "aligned_count"
                )
            ).orderBy("avg_sentiment",
                      ascending=False).show()

        assert total > 0, "FAIL — empty table"
        print(f"\nValidation PASSED ✓")

    # ------------------------------------------------------------------ #
    #  Run
    # ------------------------------------------------------------------ #
    def run(self):
        print("="*55)
        print("Gold 04 — Sentiment Features Pipeline")
        print("="*55)
        start = datetime.now()

        sent       = self.load_sentiment()
        ticker_agg = self.aggregate_per_ticker(sent)
        ticker_agg = self.add_cross_sectional(ticker_agg)
        df         = self.join_to_price_factors(ticker_agg)
        df         = self.add_regime_sentiment(df)
        df         = self.add_signal_strength(df)
        df         = self.add_metadata(df)

        self.write(df)
        self.validate()

        elapsed = (
            datetime.now() - start
        ).seconds / 60
        print(f"\nTotal time : {elapsed:.1f} minutes")
        print("Gold 04 — Sentiment Features COMPLETE ✓")
        return df

# COMMAND ----------

class GoldSentimentCharts:
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
        "Unknown": "#9E9E9E",
    }

    def _load(self, spark, gold_path):
        return spark.read.format("delta") \
                    .load(gold_path).toPandas()

    def chart_sentiment_overview(self, spark,
                                  gold_path) -> None:
        """Chart 1 — Sentiment overview."""
        pdf = self._load(spark, gold_path)
        if len(pdf) == 0:
            return

        pdf = pdf.sort_values(
            "sentiment_weighted", ascending=False
        )

        fig = go.Figure(go.Bar(
            x=pdf["ticker"],
            y=pdf["sentiment_weighted"],
            marker=dict(
                color=pdf["sentiment_weighted"],
                colorscale="RdYlGn",
                cmid=0,
                colorbar=dict(title="Sentiment")
            ),
            text=pdf["sentiment_weighted"].round(3),
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Sentiment: %{y:.3f}<br>"
                "Bull Ratio: %{customdata[0]:.1%}<br>"
                "Total News: %{customdata[1]}"
                "<extra></extra>"
            ),
            customdata=pdf[[
                "bullish_ratio","total_news"
            ]].values
        ))

        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.4
        )
        fig.add_hline(
            y=0.1, line_dash="dot",
            line_color="green", opacity=0.5,
            annotation_text="Bullish threshold"
        )
        fig.add_hline(
            y=-0.1, line_dash="dot",
            line_color="red", opacity=0.5,
            annotation_text="Bearish threshold"
        )

        fig.update_layout(
            title="<b>Gold 04 — Sentiment Score "
                  "by Ticker (All 59)</b>",
            template=self.TEMPLATE,
            height=550,
            xaxis_title="Ticker",
            yaxis_title="Mean Sentiment Score"
        )
        fig.show()

    def chart_sentiment_distribution(self, spark,
                                      gold_path) -> None:
        """Chart 2 — Sentiment distributions."""
        pdf = self._load(spark, gold_path)
        if len(pdf) == 0:
            return

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Sentiment Score Distribution",
                "Bullish vs Bearish Ratio",
                "News Count Distribution",
                "Sentiment Conviction"
            ]
        )

        # Sentiment distribution
        sw = pdf["sentiment_weighted"].dropna()
        fig.add_trace(go.Histogram(
            x=sw, nbinsx=30,
            name="Sentiment",
            marker_color=self.COLORS["primary"],
            opacity=0.8, showlegend=False
        ), row=1, col=1)
        fig.add_vline(
            x=0, line_dash="dash",
            line_color="white", opacity=0.4,
            row=1, col=1
        )
        fig.add_vline(
            x=sw.mean(),
            line_color=self.COLORS["warning"],
            line_width=2,
            annotation_text=f"Mean={sw.mean():.3f}",
            row=1, col=1
        )

        # Bull vs bear
        if "bullish_ratio" in pdf.columns and \
           "bearish_ratio" in pdf.columns:
            pdf_sorted = pdf.sort_values(
                "bullish_ratio", ascending=False
            ).head(30)
            fig.add_trace(go.Bar(
                x=pdf_sorted["ticker"],
                y=pdf_sorted["bullish_ratio"],
                name="Bull",
                marker_color=self.COLORS["success"],
                opacity=0.8
            ), row=1, col=2)
            fig.add_trace(go.Bar(
                x=pdf_sorted["ticker"],
                y=-pdf_sorted["bearish_ratio"],
                name="Bear",
                marker_color=self.COLORS["secondary"],
                opacity=0.8
            ), row=1, col=2)

        # News count
        if "avg_daily_news" in pdf.columns:
            fig.add_trace(go.Histogram(
                x=pdf["avg_daily_news"].dropna(),
                nbinsx=20,
                name="News",
                marker_color=self.COLORS["teal"],
                opacity=0.8,
                showlegend=False
            ), row=2, col=1)

        # Sentiment conviction
        if "sentiment_conviction" in pdf.columns:
            top = pdf.nlargest(20, "sentiment_conviction")
            fig.add_trace(go.Bar(
                x=top["ticker"],
                y=top["sentiment_conviction"],
                marker_color=self.COLORS["warning"],
                showlegend=False
            ), row=2, col=2)

        fig.update_layout(
            title="<b>Gold 04 — Sentiment "
                  "Distribution Analysis</b>",
            template=self.TEMPLATE,
            height=700,
            barmode="relative"
        )
        fig.show()

    def chart_bull_bear_scatter(self, spark,
                                 gold_path) -> None:
        """Chart 3 — Bullish vs bearish scatter."""
        pdf = self._load(spark, gold_path)
        if len(pdf) == 0:
            return

        df = pdf[[
            "ticker","bullish_ratio","bearish_ratio",
            "sentiment_weighted","total_news"
        ]].dropna()

        if "regime_label" in pdf.columns:
            df = df.merge(
                pdf[["ticker","regime_label"]],
                on="ticker", how="left"
            )
            color_col = "regime_label"
            color_map = self.REGIME_COLORS
        else:
            df["color"] = df[
                "sentiment_weighted"
            ].apply(
                lambda x: "Bullish" if x > 0
                          else "Bearish"
            )
            color_col = "color"
            color_map = {
                "Bullish": self.COLORS["success"],
                "Bearish": self.COLORS["secondary"]
            }

        fig = px.scatter(
            df,
            x="bullish_ratio",
            y="bearish_ratio",
            color=color_col,
            color_discrete_map=color_map,
            text="ticker",
            size="total_news",
            size_max=20,
            template=self.TEMPLATE,
            title="<b>Gold 04 — Bullish vs Bearish "
                  "Ratio by Ticker</b>",
            labels={
                "bullish_ratio": "Bullish Ratio",
                "bearish_ratio": "Bearish Ratio",
            },
            hover_data={
                "ticker"            : True,
                "sentiment_weighted": ":.3f",
                "total_news"        : True,
            }
        )

        fig.update_traces(
            textposition="top center",
            textfont=dict(size=7)
        )

        # Diagonal line (bull = bear)
        fig.add_shape(
            type="line", x0=0, y0=0, x1=1, y1=1,
            line=dict(
                color="white", dash="dash", width=1
            ),
            opacity=0.3
        )

        fig.update_layout(height=600)
        fig.show()

    def chart_sentiment_vs_returns(self, spark,
                                    gold_path) -> None:
        """Chart 4 — Sentiment vs trailing returns."""
        pdf = self._load(spark, gold_path)
        if len(pdf) == 0:
            return

        ret_col = next(
            (c for c in [
                "mom_21d","mom_5d","fwd_return_21d"
            ] if c in pdf.columns),
            None
        )

        if ret_col is None:
            print("  [Skipped] No return column")
            return

        df = pdf[[
            "ticker","sentiment_weighted",
            ret_col
        ]].dropna()

        if len(df) < 5:
            return

        x    = df["sentiment_weighted"].values
        y    = df[ret_col].values
        coef = np.polyfit(x, y, 1)
        xl   = np.linspace(x.min(), x.max(), 100)
        yl   = np.polyval(coef, xl)

        ic = float(np.corrcoef(
            stats.rankdata(x),
            stats.rankdata(y)
        )[0, 1])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y * 100,
            mode="markers+text",
            text=df["ticker"],
            textfont=dict(size=8),
            textposition="top center",
            marker=dict(
                color=x,
                colorscale="RdYlGn",
                cmid=0,
                size=10,
                opacity=0.8,
                colorbar=dict(title="Sentiment")
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Sentiment: %{x:.3f}<br>"
                f"Return: %{{y:.2f}}%"
                "<extra></extra>"
            ),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=xl,
            y=yl * 100,
            mode="lines",
            name=f"OLS (IC={ic:.3f})",
            line=dict(
                color=self.COLORS["warning"],
                width=2, dash="dash"
            )
        ))

        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3
        )
        fig.add_vline(
            x=0, line_dash="dash",
            line_color="white", opacity=0.3
        )

        fig.update_layout(
            title=f"<b>Gold 04 — Sentiment vs "
                  f"{ret_col} (IC={ic:.3f})</b>",
            template=self.TEMPLATE,
            height=600,
            xaxis_title="Sentiment Score",
            yaxis_title=f"{ret_col} (%)"
        )
        fig.show()

    def chart_regime_sentiment(self, spark,
                                gold_path) -> None:
        """Chart 5 — Sentiment by regime."""
        pdf = self._load(spark, gold_path)
        if "regime_label" not in pdf.columns:
            print("  [Skipped] No regime_label")
            return

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Sentiment Distribution by Regime",
                "Regime-Sentiment Alignment"
            ]
        )

        # Violin by regime
        for regime, color in self.REGIME_COLORS.items():
            mask = pdf["regime_label"] == regime
            if not mask.any():
                continue
            fig.add_trace(go.Violin(
                y=pdf[mask]["sentiment_weighted"],
                name=regime,
                fillcolor=color,
                line_color="white",
                opacity=0.7,
                box_visible=True,
                meanline_visible=True
            ), row=1, col=1)

        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=1, col=1
        )

        # Alignment bar
        if "regime_sentiment_aligned" in pdf.columns:
            align = pdf.groupby("regime_label").agg(
                aligned_pct=(
                    "regime_sentiment_aligned","mean"
                ),
                n_tickers=("ticker","count")
            ).reset_index()

            colors = [
                self.REGIME_COLORS.get(r,"#9E9E9E")
                for r in align["regime_label"]
            ]
            fig.add_trace(go.Bar(
                x=align["regime_label"],
                y=align["aligned_pct"],
                marker_color=colors,
                text=align["aligned_pct"].apply(
                    lambda x: f"{x:.1%}"
                ),
                textposition="outside",
                showlegend=False
            ), row=1, col=2)
            fig.add_hline(
                y=0.5, line_dash="dash",
                line_color="white", opacity=0.3,
                row=1, col=2
            )

        fig.update_layout(
            title="<b>Gold 04 — Sentiment by Regime</b>",
            template=self.TEMPLATE,
            height=550
        )
        fig.update_yaxes(
            title_text="Sentiment Score", row=1, col=1
        )
        fig.update_yaxes(
            title_text="% Regime Aligned", row=1, col=2
        )
        fig.show()

    def chart_top_conviction(self, spark,
                              gold_path) -> None:
        """Chart 6 — Top conviction signals."""
        pdf = self._load(spark, gold_path)
        if len(pdf) == 0:
            return

        if "sentiment_conviction" not in pdf.columns:
            return

        top_bull  = pdf.nlargest(
            15,"sentiment_weighted"
        )
        top_bear  = pdf.nsmallest(
            15,"sentiment_weighted"
        )

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Top 15 Bullish Tickers",
                "Top 15 Bearish Tickers"
            ]
        )

        fig.add_trace(go.Bar(
            x=top_bull["ticker"],
            y=top_bull["sentiment_weighted"],
            marker_color=self.COLORS["success"],
            text=top_bull["sentiment_weighted"].round(3),
            textposition="outside",
            name="Bullish",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Sentiment: %{y:.3f}<br>"
                "Bull%: %{customdata[0]:.1%}<br>"
                "News: %{customdata[1]}"
                "<extra></extra>"
            ),
            customdata=top_bull[[
                "bullish_ratio","total_news"
            ]].values
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=top_bear["ticker"],
            y=top_bear["sentiment_weighted"],
            marker_color=self.COLORS["secondary"],
            text=top_bear["sentiment_weighted"].round(3),
            textposition="outside",
            name="Bearish",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Sentiment: %{y:.3f}<br>"
                "Bear%: %{customdata[0]:.1%}<br>"
                "News: %{customdata[1]}"
                "<extra></extra>"
            ),
            customdata=top_bear[[
                "bearish_ratio","total_news"
            ]].values
        ), row=1, col=2)

        for r, c in [(1,1),(1,2)]:
            fig.add_hline(
                y=0, line_dash="dash",
                line_color="white", opacity=0.3,
                row=r, col=c
            )

        fig.update_layout(
            title="<b>Gold 04 — Top Conviction "
                  "Signals</b>",
            template=self.TEMPLATE,
            height=500
        )
        fig.update_yaxes(
            title_text="Sentiment Score",
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="Sentiment Score",
            row=1, col=2
        )
        fig.show()

    def chart_sentiment_correlation(self, spark,
                                     gold_path) -> None:
        """Chart 7 — Sentiment feature correlation."""
        pdf = self._load(spark, gold_path)
        if len(pdf) == 0:
            return

        factor_cols = [
            "sentiment_weighted",
            "bullish_ratio",
            "bearish_ratio",
            "sentiment_volatility",
            "sentiment_conviction",
            "avg_daily_news",
            "sentiment_signal_strength",
        ]
        avail = [
            c for c in factor_cols if c in pdf.columns
        ]
        if len(avail) < 2:
            return

        corr = pdf[avail].dropna().corr()

        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdYlGn",
            zmid=0, zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorbar=dict(title="Corr")
        ))

        fig.update_layout(
            title="<b>Gold 04 — Sentiment Feature "
                  "Correlation</b>",
            template=self.TEMPLATE,
            height=550
        )
        fig.show()

    def run_all(self, spark, gold_path) -> None:
        print("\n" + "="*55)
        print("Generating Gold 04 Charts...")
        print("="*55)

        print("\n[1/7] Sentiment Overview...")
        self.chart_sentiment_overview(spark, gold_path)

        print("[2/7] Sentiment Distributions...")
        self.chart_sentiment_distribution(spark, gold_path)

        print("[3/7] Bull vs Bear Scatter...")
        self.chart_bull_bear_scatter(spark, gold_path)

        print("[4/7] Sentiment vs Returns...")
        self.chart_sentiment_vs_returns(spark, gold_path)

        print("[5/7] Regime Sentiment...")
        self.chart_regime_sentiment(spark, gold_path)

        print("[6/7] Top Conviction Signals...")
        self.chart_top_conviction(spark, gold_path)

        print("[7/7] Feature Correlation...")
        self.chart_sentiment_correlation(spark, gold_path)

        print("\nAll 7 charts ✓")

# COMMAND ----------

pipeline = GoldSentimentFeatures(
    spark       = spark,
    silver_path = SILVER_PATH,
    eda_path    = EDA_PATH,
    gold_path   = GOLD_PATH
)

df = pipeline.run()

charts = GoldSentimentCharts()
charts.run_all(
    spark     = spark,
    gold_path = f"{GOLD_PATH}/sentiment_features"
)

print("\nGold 04 COMPLETE ✓")

# COMMAND ----------

df = spark.read.format("delta").load(
    f"{GOLD_PATH}/sentiment_features"
)

print("="*55)
print("Gold 04 — Sentiment Features Summary")
print("="*55)
print(f"Total rows  : {df.count():,}")
print(f"Tickers     : "
      f"{df.select('ticker').distinct().count():,}")
print(f"Columns     : {len(df.columns):,}")

print(f"\nAll columns:")
for i, c in enumerate(sorted(df.columns)):
    print(f"  {i+1:3}. {c}")

print(f"\nTop 10 bullish tickers:")
df.orderBy(F.desc("sentiment_weighted")) \
  .select(
    "ticker","sentiment_weighted",
    "bullish_ratio","total_news",
    "high_conviction_flag",
    "sentiment_signal_strength"
  ).show(10)

print(f"\nTop 10 bearish tickers:")
df.orderBy(F.asc("sentiment_weighted")) \
  .select(
    "ticker","sentiment_weighted",
    "bearish_ratio","total_news",
    "high_conviction_flag"
  ).show(10)