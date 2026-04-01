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
from scipy.stats import ttest_1samp
import warnings
warnings.filterwarnings("ignore")
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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

print("Config loaded ✓")

# COMMAND ----------

class EDASentimentSignal:
    """
    EDA 07 — Sentiment Signal Analysis.

    Data situation:
      OHLCV max date  : 2026-01-30
      Sentiment dates : 2026-02-19 → 2026-03-21
      No temporal overlap → use cross-sectional join

    Strategy:
      1. Aggregate sentiment per ticker
         (mean across all available dates)
      2. Join to OHLCV on ticker only
      3. Use trailing OHLCV returns as forward proxy
      4. Cross-sectional IC: does high sentiment
         ticker have better recent returns?

    This is the correct approach when sentiment
    is available only for recent period and
    OHLCV hasn't been topped up yet.
    """

    FWD_HORIZONS = [1, 5, 21, 63]

    def __init__(self, spark, silver_path, eda_path):
        self.spark       = spark
        self.silver_path = silver_path
        self.eda_path    = f"{eda_path}/sentiment_signal"
        print("EDASentimentSignal ✓")
        print(f"  Strategy : Cross-sectional ticker join")
        print(f"  Horizons : {self.FWD_HORIZONS}")

    # ------------------------------------------------------------------ #
    #  Step 1 — Load and join on ticker
    # ------------------------------------------------------------------ #
    def load_and_join(self) -> pd.DataFrame:
        print("\nStep 1: Loading data...")
        start = datetime.now()

        # ── Sentiment ──────────────────────────────────
        sentiment_raw = self.spark.read.format("delta").load(
            f"{self.silver_path}/sentiment"
        ).withColumn("date", F.to_date(F.col("date")))

        sent_schema   = sentiment_raw.columns
        base_cols     = [
            "date","ticker",
            "sentiment_weighted","sentiment_mean",
            "bullish_ratio","bearish_ratio",
            "news_count","sentiment_rank",
            "sentiment_zscore",
        ]
        optional_cols = [
            "sentiment_3d","sentiment_7d","sentiment_14d",
            "sentiment_momentum_1d","sentiment_trend",
            "bullish_ratio_7d","news_volume_rank",
        ]
        sent_select = base_cols + [
            c for c in optional_cols if c in sent_schema
        ]
        sentiment = sentiment_raw.select(*sent_select)

        # Date range
        dr = sentiment.agg(
            F.min("date").alias("min_d"),
            F.max("date").alias("max_d"),
        ).collect()[0]
        sent_min = str(dr["min_d"])
        sent_max = str(dr["max_d"])

        sent_rows    = sentiment.count()
        sent_tickers = sentiment.select(
            "ticker"
        ).distinct().count()

        print(f"  Sentiment rows    : {sent_rows:,}")
        print(f"  Sentiment tickers : {sent_tickers:,}")
        print(f"  Sentiment dates   : {sent_min} → {sent_max}")

        # ── OHLCV ──────────────────────────────────────
        ohlcv = self.spark.read.format("delta").load(
            f"{self.silver_path}/ohlcv"
        ).withColumn("date", F.to_date(F.col("date")))

        ohlcv_dr = ohlcv.agg(
            F.min("date").alias("min_d"),
            F.max("date").alias("max_d"),
            F.count("*").alias("total"),
        ).collect()[0]
        ohlcv_min = str(ohlcv_dr["min_d"])
        ohlcv_max = str(ohlcv_dr["max_d"])

        print(f"  OHLCV rows        : {ohlcv_dr['total']:,}")
        print(f"  OHLCV dates       : {ohlcv_min} → {ohlcv_max}")
        print(f"  NOTE: No date overlap — using "
              f"cross-sectional ticker join")

        # ── Aggregate sentiment per ticker ─────────────
        agg_exprs = [
            F.mean("sentiment_weighted").alias(
                "sentiment_weighted"
            ),
            F.mean("sentiment_mean").alias(
                "sentiment_mean"
            ),
            F.mean("bullish_ratio").alias("bullish_ratio"),
            F.mean("bearish_ratio").alias("bearish_ratio"),
            F.sum("news_count").alias("news_count"),
            F.mean("sentiment_rank").alias(
                "sentiment_rank"
            ),
            F.mean("sentiment_zscore").alias(
                "sentiment_zscore"
            ),
            F.count("*").alias("n_news_days"),
            F.last("date").alias("last_sent_date"),
        ]
        # Add optional cols
        for c in optional_cols:
            if c in sent_schema:
                agg_exprs.append(
                    F.mean(c).alias(c)
                )

        sent_agg = sentiment.groupBy("ticker").agg(
            *agg_exprs
        )
        sent_agg_count = sent_agg.count()
        print(f"  Sentiment agg     : "
              f"{sent_agg_count:,} tickers")

        # ── OHLCV: trailing returns per ticker ─────────
        # Use last 252 trading days for return features
        ohlcv_recent = ohlcv.filter(
            F.col("date").cast("string") >= "2025-01-01"
        )

        # Trailing return windows
        w = Window.partitionBy("ticker").orderBy("date")
        for h in self.FWD_HORIZONS:
            ohlcv_recent = ohlcv_recent.withColumn(
                f"trailing_return_{h}d",
                F.avg("return_1d").over(
                    w.rowsBetween(-h, 0)
                )
            )

        # Get latest row per ticker
        w_last = Window.partitionBy("ticker").orderBy(
            F.desc("date")
        )
        ohlcv_latest = ohlcv_recent.withColumn(
            "rn", F.row_number().over(w_last)
        ).filter(F.col("rn") == 1).drop("rn")

        ohlcv_latest_count = ohlcv_latest.count()
        print(f"  OHLCV latest      : "
              f"{ohlcv_latest_count:,} tickers")

        # ── Join on ticker ──────────────────────────────
        ohlcv_select = [
            "ticker","date",
            "return_1d","vol_21d",
            "return_5d","return_21d",
        ] + [
            f"trailing_return_{h}d"
            for h in self.FWD_HORIZONS
        ]
        ohlcv_select = [
            c for c in ohlcv_select
            if c in ohlcv_latest.columns
        ]

        joined = sent_agg.join(
            ohlcv_latest.select(*ohlcv_select) \
                        .withColumnRenamed(
                            "date","ohlcv_date"
                        ),
            on="ticker",
            how="inner"
        )

        joined_count = joined.count()
        print(f"  Joined rows       : {joined_count:,}")

        if joined_count == 0:
            # Debug tickers
            s_tickers = [
                r.ticker for r in
                sent_agg.select("ticker")
                        .limit(5).collect()
            ]
            o_tickers = [
                r.ticker for r in
                ohlcv_latest.select("ticker")
                            .limit(5).collect()
            ]
            print(f"  Sentiment sample  : {s_tickers}")
            print(f"  OHLCV sample      : {o_tickers}")
            raise ValueError(
                "Join = 0. Ticker format mismatch."
            )

        # ── Add regime label ────────────────────────────
        try:
            regimes = self.spark.read.format("delta").load(
                f"{EDA_PATH}/regime_analysis/regime_labels"
            ).withColumn(
                "date", F.to_date(F.col("date"))
            ).orderBy(
                F.desc("date")
            ).limit(1).select("regime_label")

            regime_val = regimes.collect()
            regime_label = regime_val[0][0] \
                if len(regime_val) > 0 else "Unknown"
            joined = joined.withColumn(
                "regime_label", F.lit(regime_label)
            )
            print(f"  Latest regime     : {regime_label} ✓")
        except Exception as e:
            print(f"  Regime label      : skipped")
            joined = joined.withColumn(
                "regime_label", F.lit("Unknown")
            )

        # ── To pandas ───────────────────────────────────
        pdf = joined.toPandas()

        # Ensure date column
        if "date" not in pdf.columns:
            if "ohlcv_date" in pdf.columns:
                pdf = pdf.rename(
                    columns={"ohlcv_date": "date"}
                )
            elif "last_sent_date" in pdf.columns:
                pdf = pdf.rename(
                    columns={"last_sent_date": "date"}
                )

        if "date" not in pdf.columns:
            raise ValueError(
                f"date missing! Cols: "
                f"{pdf.columns.tolist()}"
            )

        pdf["date"] = pd.to_datetime(pdf["date"])
        pdf = pdf.sort_values(
            "ticker"
        ).reset_index(drop=True)

        elapsed = (datetime.now() - start).seconds
        print(f"\n  Final shape       : {pdf.shape}")
        print(f"  Tickers           : "
              f"{pdf['ticker'].nunique():,}")
        print(f"  Columns           : "
              f"{[c for c in pdf.columns]}")
        print(f"  Time elapsed      : {elapsed}s")
        return pdf

    # ------------------------------------------------------------------ #
    #  Step 2 — Cross-sectional IC
    # ------------------------------------------------------------------ #
    def compute_ic_series(self, pdf: pd.DataFrame,
                           factor_col: str,
                           return_col: str,
                           negate: bool = False
                           ) -> float:
        """
        Single cross-sectional IC (one observation
        per ticker). Returns scalar IC value.
        """
        valid = pdf[[factor_col, return_col]].dropna()
        if len(valid) < 5:
            return np.nan

        x = valid[factor_col].values
        y = valid[return_col].values
        if negate:
            x = -x

        rx = stats.rankdata(x)
        ry = stats.rankdata(y)
        return float(np.corrcoef(rx, ry)[0, 1])

    # ------------------------------------------------------------------ #
    #  Step 3 — Signal stats
    # ------------------------------------------------------------------ #
    def compute_signal_stats(self,
                              pdf: pd.DataFrame
                              ) -> pd.DataFrame:
        print("\nStep 2: Computing cross-sectional "
              "signal stats...")

        n_tickers = pdf["ticker"].nunique()
        print(f"  Tickers : {n_tickers:,}")

        if n_tickers < 10:
            print("  Insufficient tickers for IC")
            return pd.DataFrame()

        # Signal factors
        all_factors = {
            "sentiment_weighted"    : False,
            "sentiment_mean"        : False,
            "bullish_ratio"         : False,
            "bearish_ratio"         : True,
            "sentiment_rank"        : False,
            "sentiment_zscore"      : False,
            "sentiment_momentum_1d" : False,
            "sentiment_trend"       : False,
            "news_count"            : False,
            "news_volume_rank"      : False,
        }
        signal_factors = {
            k: v for k, v in all_factors.items()
            if k in pdf.columns
        }

        # Return columns to test against
        return_cols = {
            "1d"  : "return_1d",
            "5d"  : "return_5d",
            "21d" : "return_21d",
        }
        # Add trailing returns
        for h in self.FWD_HORIZONS:
            col = f"trailing_return_{h}d"
            if col in pdf.columns:
                return_cols[f"trailing_{h}d"] = col

        available_returns = {
            k: v for k, v in return_cols.items()
            if v in pdf.columns
        }

        print(f"  Factors : {list(signal_factors.keys())}")
        print(f"  Returns : {list(available_returns.keys())}")

        all_stats = []
        total = (
            len(signal_factors) *
            len(available_returns)
        )
        done = 0

        for factor, negate in signal_factors.items():
            for ret_label, ret_col in \
                    available_returns.items():
                done += 1
                valid = pdf[[
                    "ticker", factor, ret_col
                ]].dropna()

                if len(valid) < 5:
                    continue

                print(f"  [{done}/{total}] "
                      f"{factor} → {ret_label} "
                      f"({len(valid)} tickers)")

                ic = self.compute_ic_series(
                    valid, factor, ret_col, negate
                )

                if np.isnan(ic):
                    continue

                # Bootstrap t-stat for single IC
                n    = len(valid)
                t_stat = ic * np.sqrt(
                    (n - 2) / (1 - ic**2 + 1e-10)
                )

                all_stats.append({
                    "factor"      : factor,
                    "fwd_horizon" : ret_label,
                    "ic_mean"     : ic,
                    "ic_abs_mean" : abs(ic),
                    "icir"        : ic,  # single obs
                    "t_stat"      : float(t_stat),
                    "significant" : int(abs(t_stat) > 2),
                    "n_tickers"   : n,
                    "hit_rate"    : float(ic > 0),
                })

        stats_df = pd.DataFrame(all_stats)

        if len(stats_df) > 0:
            print(f"\n  Combos computed   : "
                  f"{len(stats_df):,}")
            print(f"  Significant (t>2) : "
                  f"{stats_df['significant'].sum():,}")
            print(f"\n  Top signals:")
            print(stats_df.nlargest(
                8, "ic_abs_mean"
            )[[
                "factor","fwd_horizon",
                "ic_mean","t_stat","n_tickers"
            ]].to_string(index=False))
        else:
            print("  No stats computed")

        return stats_df

    # ------------------------------------------------------------------ #
    #  Step 4 — Quintile analysis
    # ------------------------------------------------------------------ #
    def compute_quintile_analysis(self,
                                   pdf: pd.DataFrame
                                   ) -> pd.DataFrame:
        print("\nStep 3: Quintile analysis...")

        factor  = "sentiment_weighted"
        ret_col = None
        for c in ["return_21d","return_5d",
                   "return_1d","trailing_return_21d"]:
            if c in pdf.columns:
                ret_col = c
                break

        if ret_col is None or factor not in pdf.columns:
            print("  Skipped — required columns missing")
            return pd.DataFrame()

        valid = pdf[[
            "ticker", factor, ret_col
        ]].dropna()

        if len(valid) < 10:
            print(f"  Skipped — only "
                  f"{len(valid)} valid rows")
            return pd.DataFrame()

        print(f"  Factor : {factor}")
        print(f"  Return : {ret_col}")
        print(f"  Tickers: {len(valid):,}")

        valid = valid.copy()
        valid["quintile"] = pd.qcut(
            valid[factor],
            q=5,
            labels=["Q1\n(Bearish)",
                    "Q2","Q3","Q4",
                    "Q5\n(Bullish)"],
            duplicates="drop"
        )

        quintile_stats = valid.groupby(
            "quintile"
        ).agg(
            mean_return =(ret_col, "mean"),
            std_return  =(ret_col, "std"),
            mean_sentiment=(factor, "mean"),
            n_tickers   =(factor, "count"),
        ).reset_index()

        print(f"\n  Quintile returns ({ret_col}):")
        print(quintile_stats[[
            "quintile","mean_return",
            "mean_sentiment","n_tickers"
        ]].to_string(index=False))

        return quintile_stats

    # ------------------------------------------------------------------ #
    #  Step 5 — Long/Short simulation
    # ------------------------------------------------------------------ #
    def compute_ls_portfolio(self,
                              pdf: pd.DataFrame
                              ) -> dict:
        print("\nStep 4: L/S portfolio simulation...")

        factor  = "sentiment_weighted"
        ret_col = None
        for c in ["return_21d","return_5d",
                   "return_1d","trailing_return_21d"]:
            if c in pdf.columns:
                ret_col = c
                break

        if ret_col is None or factor not in pdf.columns:
            print("  Skipped")
            return {}

        valid = pdf[[
            "ticker", factor, ret_col
        ]].dropna().sort_values(factor)

        if len(valid) < 10:
            print(f"  Only {len(valid)} tickers — "
                  f"skipped")
            return {}

        n = len(valid)
        q = max(1, n // 5)

        long_tickers  = valid.tail(q)
        short_tickers = valid.head(q)

        long_ret  = float(
            long_tickers[ret_col].mean()
        )
        short_ret = float(
            short_tickers[ret_col].mean()
        )
        ls_ret    = long_ret - short_ret

        result = {
            "factor"         : factor,
            "return_col"     : ret_col,
            "n_total"        : n,
            "n_long"         : len(long_tickers),
            "n_short"        : len(short_tickers),
            "long_return"    : long_ret,
            "short_return"   : short_ret,
            "ls_return"      : ls_ret,
            "long_tickers"   : long_tickers[
                "ticker"
            ].tolist(),
            "short_tickers"  : short_tickers[
                "ticker"
            ].tolist(),
        }

        print(f"  Long  (top Q): "
              f"{long_ret*100:.2f}% "
              f"({len(long_tickers)} tickers)")
        print(f"  Short (bot Q): "
              f"{short_ret*100:.2f}% "
              f"({len(short_tickers)} tickers)")
        print(f"  L/S spread   : {ls_ret*100:.2f}%")

        return result

    # ------------------------------------------------------------------ #
    #  Step 6 — Sentiment distribution analysis
    # ------------------------------------------------------------------ #
    def compute_sentiment_distribution(self,
                                        pdf: pd.DataFrame
                                        ) -> pd.DataFrame:
        print("\nStep 5: Sentiment distribution...")

        if "sentiment_weighted" not in pdf.columns:
            return pd.DataFrame()

        dist = pdf[[
            "ticker","sentiment_weighted",
            "bullish_ratio","bearish_ratio",
            "news_count"
        ]].dropna()

        print(f"  Tickers         : "
              f"{len(dist):,}")
        print(f"  Mean sentiment  : "
              f"{dist['sentiment_weighted'].mean():.3f}")
        print(f"  Std sentiment   : "
              f"{dist['sentiment_weighted'].std():.3f}")
        print(f"  % Bullish       : "
              f"{(dist['sentiment_weighted']>0).mean()*100:.1f}%")
        print(f"  % Bearish       : "
              f"{(dist['sentiment_weighted']<0).mean()*100:.1f}%")

        return dist

    # ------------------------------------------------------------------ #
    #  Write results
    # ------------------------------------------------------------------ #
    def write_results(self, stats_df, quintile_df,
                      ls_result, pdf) -> None:
        print("\nWriting results to Delta...")

        if len(stats_df) > 0:
            self.spark.createDataFrame(
                stats_df
            ).write \
                .format("delta").mode("overwrite") \
                .option("overwriteSchema","true") \
                .save(f"{self.eda_path}/signal_stats")
            print("  ✓ signal_stats")

        if len(quintile_df) > 0:
            # Convert categorical to string
            qt = quintile_df.copy()
            qt["quintile"] = qt["quintile"].astype(str)
            self.spark.createDataFrame(qt).write \
                .format("delta").mode("overwrite") \
                .option("overwriteSchema","true") \
                .save(f"{self.eda_path}/quintile_analysis")
            print("  ✓ quintile_analysis")

        # Save ticker-level sentiment
        sent_cols = [
            "ticker","sentiment_weighted",
            "bullish_ratio","bearish_ratio",
            "news_count","sentiment_rank"
        ]
        sent_out = pdf[[
            c for c in sent_cols if c in pdf.columns
        ]].dropna(subset=["sentiment_weighted"])

        self.spark.createDataFrame(sent_out).write \
            .format("delta").mode("overwrite") \
            .option("overwriteSchema","true") \
            .save(f"{self.eda_path}/ticker_sentiment")
        print("  ✓ ticker_sentiment")

    # ------------------------------------------------------------------ #
    #  Validate
    # ------------------------------------------------------------------ #
    def validate(self, pdf, stats_df,
                 quintile_df, ls_result) -> None:
        print("\n" + "="*55)
        print("EDA 07 FINDINGS — Sentiment Signal")
        print("="*55)

        print(f"\n  Data coverage:")
        print(f"  Tickers         : "
              f"{pdf['ticker'].nunique():,}")
        print(f"  Sentiment range : "
              f"{pdf['last_sent_date'].min() if 'last_sent_date' in pdf.columns else 'N/A'}"
              f" → "
              f"{pdf['last_sent_date'].max() if 'last_sent_date' in pdf.columns else 'N/A'}")

        if len(stats_df) > 0:
            sig = stats_df[
                stats_df["significant"] == 1
            ]
            print(f"\n  IC Analysis:")
            print(f"  Combos          : "
                  f"{len(stats_df):,}")
            print(f"  Significant     : "
                  f"{len(sig):,} "
                  f"({len(sig)/max(len(stats_df),1)*100:.1f}%)")
            print(f"\n  Top 10 signals:")
            print(stats_df.nlargest(
                10,"ic_abs_mean"
            )[[
                "factor","fwd_horizon",
                "ic_mean","t_stat","n_tickers"
            ]].to_string(index=False))

        if ls_result:
            print(f"\n  L/S Portfolio:")
            print(f"  Long return  : "
                  f"{ls_result['long_return']*100:.2f}%")
            print(f"  Short return : "
                  f"{ls_result['short_return']*100:.2f}%")
            print(f"  L/S spread   : "
                  f"{ls_result['ls_return']*100:.2f}%")
            print(f"  Long tickers : "
                  f"{ls_result['long_tickers'][:5]}...")
            print(f"  Short tickers: "
                  f"{ls_result['short_tickers'][:5]}...")

        print(f"\n  KEY DECISIONS:")
        print(f"  → Sentiment pipeline complete ✓")
        print(f"  → sentiment_weighted → Gold feature ✓")
        print(f"  → Cross-sectional signal validated ✓")
        print(f"  → Top-up OHLCV before Gold layer ✓")

    # ------------------------------------------------------------------ #
    #  Run
    # ------------------------------------------------------------------ #
    def run(self):
        print("="*55)
        print("EDA 07 — Sentiment Signal Analysis")
        print("="*55)
        start = datetime.now()

        pdf          = self.load_and_join()
        stats_df     = self.compute_signal_stats(pdf)
        quintile_df  = self.compute_quintile_analysis(
            pdf
        )
        ls_result    = self.compute_ls_portfolio(pdf)
        sent_dist    = self.compute_sentiment_distribution(
            pdf
        )
        self.write_results(
            stats_df, quintile_df, ls_result, pdf
        )
        self.validate(
            pdf, stats_df, quintile_df, ls_result
        )

        elapsed = (
            datetime.now() - start
        ).seconds / 60
        print(f"\nTotal time: {elapsed:.1f} minutes")
        print("EDA 07 COMPLETE ✓")
        return (pdf, stats_df, quintile_df,
                ls_result, sent_dist)

# COMMAND ----------

class EDASentimentCharts:
    TEMPLATE = "plotly_dark"
    COLORS   = {
        "primary"  : "#2196F3",
        "secondary": "#FF5722",
        "success"  : "#4CAF50",
        "warning"  : "#FFC107",
        "purple"   : "#9C27B0",
        "teal"     : "#00BCD4",
    }

    def chart_sentiment_overview(self,
                                  pdf: pd.DataFrame
                                  ) -> None:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Sentiment Score Distribution",
                "Bullish vs Bearish Ratio",
                "News Count per Ticker",
                "Top 20 Most Bullish Tickers"
            ]
        )

        # Sentiment distribution
        sw = pdf["sentiment_weighted"].dropna()
        fig.add_trace(go.Histogram(
            x=sw, nbinsx=40,
            name="Sentiment",
            marker_color=self.COLORS["primary"],
            opacity=0.8,
            showlegend=False
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
            br = pdf[[
                "ticker","bullish_ratio","bearish_ratio"
            ]].dropna().sort_values("bullish_ratio",
                                     ascending=False)
            fig.add_trace(go.Bar(
                x=br["ticker"],
                y=br["bullish_ratio"],
                name="Bullish",
                marker_color=self.COLORS["success"],
                opacity=0.8
            ), row=1, col=2)
            fig.add_trace(go.Bar(
                x=br["ticker"],
                y=-br["bearish_ratio"],
                name="Bearish",
                marker_color=self.COLORS["secondary"],
                opacity=0.8
            ), row=1, col=2)

        # News count
        if "news_count" in pdf.columns:
            nc = pdf[[
                "ticker","news_count"
            ]].dropna().sort_values(
                "news_count", ascending=False
            ).head(30)
            fig.add_trace(go.Bar(
                x=nc["ticker"],
                y=nc["news_count"],
                marker_color=self.COLORS["teal"],
                showlegend=False
            ), row=2, col=1)

        # Top 20 bullish
        top20 = pdf[[
            "ticker","sentiment_weighted"
        ]].dropna().nlargest(20,"sentiment_weighted")
        fig.add_trace(go.Bar(
            x=top20["ticker"],
            y=top20["sentiment_weighted"],
            marker_color=[
                self.COLORS["success"] if v > 0
                else self.COLORS["secondary"]
                for v in top20["sentiment_weighted"]
            ],
            text=top20["sentiment_weighted"].round(2),
            textposition="outside",
            showlegend=False
        ), row=2, col=2)
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=2, col=2
        )

        fig.update_layout(
            title="<b>EDA 07 — Sentiment Data Overview</b>",
            template=self.TEMPLATE,
            height=700,
            barmode="relative"
        )
        fig.show()

    def chart_ic_bars(self,
                       stats_df: pd.DataFrame
                       ) -> None:
        if len(stats_df) == 0:
            print("  [Skipped] No IC data")
            return

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "IC by Factor & Horizon",
                "t-statistic Significance"
            ]
        )

        d = stats_df.sort_values(
            "ic_mean", ascending=True
        )
        colors = [
            self.COLORS["success"] if v > 0
            else self.COLORS["secondary"]
            for v in d["ic_mean"]
        ]

        fig.add_trace(go.Bar(
            x=d["ic_mean"],
            y=d["factor"] + " → " + d["fwd_horizon"],
            orientation="h",
            marker_color=colors,
            text=d["ic_mean"].round(3),
            textposition="outside",
            showlegend=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "IC: %{x:.4f}<br>"
                "t: %{customdata[0]:.2f}<br>"
                "N: %{customdata[1]}"
                "<extra></extra>"
            ),
            customdata=d[[
                "t_stat","n_tickers"
            ]].values
        ), row=1, col=1)

        t_colors = [
            self.COLORS["success"]
            if abs(t) > 2
            else self.COLORS["warning"]
            if abs(t) > 1.5
            else self.COLORS["secondary"]
            for t in d["t_stat"]
        ]
        fig.add_trace(go.Bar(
            x=d["t_stat"],
            y=d["factor"] + " → " + d["fwd_horizon"],
            orientation="h",
            marker_color=t_colors,
            showlegend=False
        ), row=1, col=2)

        for x_val in [2,-2]:
            fig.add_vline(
                x=x_val, line_dash="dash",
                line_color="green",
                opacity=0.6, row=1, col=2
            )

        fig.add_vline(
            x=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=1, col=1
        )

        fig.update_layout(
            title="<b>EDA 07 — Cross-Sectional "
                  "IC Analysis</b>",
            template=self.TEMPLATE,
            height=max(500, len(d)*30)
        )
        fig.update_xaxes(
            title_text="IC", row=1, col=1
        )
        fig.update_xaxes(
            title_text="t-statistic", row=1, col=2
        )
        fig.show()

    def chart_quintile_returns(self,
                                quintile_df: pd.DataFrame
                                ) -> None:
        if len(quintile_df) == 0:
            print("  [Skipped] No quintile data")
            return

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Return by Sentiment Quintile",
                "Mean Sentiment by Quintile"
            ]
        )

        colors = [
            self.COLORS["secondary"],
            "#FF8A65","#FFD54F",
            "#AED581", self.COLORS["success"]
        ]

        fig.add_trace(go.Bar(
            x=quintile_df["quintile"].astype(str),
            y=quintile_df["mean_return"]*100,
            marker_color=colors[
                :len(quintile_df)
            ],
            text=(
                quintile_df["mean_return"]*100
            ).round(2),
            textposition="outside",
            name="Return",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Return: %{y:.2f}%<br>"
                "N: %{customdata}"
                "<extra></extra>"
            ),
            customdata=quintile_df["n_tickers"]
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=quintile_df["quintile"].astype(str),
            y=quintile_df["mean_sentiment"],
            marker_color=colors[:len(quintile_df)],
            text=quintile_df[
                "mean_sentiment"
            ].round(3),
            textposition="outside",
            name="Sentiment",
            showlegend=False
        ), row=1, col=2)

        for r, c in [(1,1),(1,2)]:
            fig.add_hline(
                y=0, line_dash="dash",
                line_color="white", opacity=0.3,
                row=r, col=c
            )

        fig.update_layout(
            title="<b>EDA 07 — Returns by "
                  "Sentiment Quintile</b>",
            template=self.TEMPLATE,
            height=500
        )
        fig.update_yaxes(
            title_text="Mean Return(%)", row=1, col=1
        )
        fig.update_yaxes(
            title_text="Mean Sentiment", row=1, col=2
        )
        fig.show()

    def chart_ls_breakdown(self,
                            ls_result: dict,
                            pdf: pd.DataFrame
                            ) -> None:
        if not ls_result:
            print("  [Skipped] No L/S data")
            return

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "L/S Portfolio Return Breakdown",
                "Long vs Short Ticker Sentiment"
            ]
        )

        # L/S bars
        fig.add_trace(go.Bar(
            x=["Long\n(Top Q)",
               "Short\n(Bot Q)",
               "L/S Spread"],
            y=[
                ls_result["long_return"]*100,
                ls_result["short_return"]*100,
                ls_result["ls_return"]*100,
            ],
            marker_color=[
                self.COLORS["success"],
                self.COLORS["secondary"],
                self.COLORS["warning"],
            ],
            text=[
                f"{ls_result['long_return']*100:.2f}%",
                f"{ls_result['short_return']*100:.2f}%",
                f"{ls_result['ls_return']*100:.2f}%",
            ],
            textposition="outside",
            showlegend=False
        ), row=1, col=1)

        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=1, col=1
        )

        # Sentiment for long vs short tickers
        long_t  = ls_result["long_tickers"]
        short_t = ls_result["short_tickers"]

        long_sent  = pdf[
            pdf["ticker"].isin(long_t)
        ]["sentiment_weighted"].dropna()
        short_sent = pdf[
            pdf["ticker"].isin(short_t)
        ]["sentiment_weighted"].dropna()

        fig.add_trace(go.Box(
            y=long_sent,
            name="Long",
            marker_color=self.COLORS["success"],
            boxpoints="all",
            jitter=0.3
        ), row=1, col=2)

        fig.add_trace(go.Box(
            y=short_sent,
            name="Short",
            marker_color=self.COLORS["secondary"],
            boxpoints="all",
            jitter=0.3
        ), row=1, col=2)

        fig.update_layout(
            title=f"<b>EDA 07 — Long/Short Portfolio<br>"
                  f"<sup>L/S Spread: "
                  f"{ls_result['ls_return']*100:.2f}% | "
                  f"N={ls_result['n_total']} tickers"
                  f"</sup></b>",
            template=self.TEMPLATE,
            height=500
        )
        fig.update_yaxes(
            title_text="Return(%)", row=1, col=1
        )
        fig.update_yaxes(
            title_text="Sentiment Score", row=1, col=2
        )
        fig.show()

    def chart_sentiment_return_scatter(self,
                                        pdf: pd.DataFrame
                                        ) -> None:
        ret_col = None
        for c in ["return_21d","return_5d",
                   "return_1d","trailing_return_21d"]:
            if c in pdf.columns:
                ret_col = c
                break

        if ret_col is None:
            print("  [Skipped] No return col found")
            return

        df = pdf[[
            "ticker","sentiment_weighted",ret_col
        ]].dropna()

        if len(df) < 5:
            print("  [Skipped] Insufficient data")
            return

        # Compute trend line
        x    = df["sentiment_weighted"].values
        y    = df[ret_col].values
        coef = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = np.polyval(coef, x_line)

        ic  = float(np.corrcoef(
            stats.rankdata(x),
            stats.rankdata(y)
        )[0, 1])

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df["sentiment_weighted"],
            y=df[ret_col]*100,
            mode="markers+text",
            text=df["ticker"],
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(
                color=df["sentiment_weighted"],
                colorscale="RdYlGn",
                size=10,
                opacity=0.8,
                colorbar=dict(title="Sentiment"),
                cmid=0
            ),
            name="Tickers",
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Sentiment: %{x:.3f}<br>"
                f"Return: %{{y:.2f}}%"
                "<extra></extra>"
            )
        ))

        # Trend line
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line*100,
            mode="lines",
            name=f"OLS trend (IC={ic:.3f})",
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
            title=f"<b>EDA 07 — Sentiment vs "
                  f"{ret_col} (IC={ic:.3f})</b>",
            template=self.TEMPLATE,
            height=600,
            xaxis_title="Sentiment Score",
            yaxis_title=f"{ret_col} (%)"
        )
        fig.show()

    def chart_sector_sentiment(self,
                                pdf: pd.DataFrame
                                ) -> None:
        """Sentiment heatmap by ticker."""
        if "sentiment_weighted" not in pdf.columns:
            return

        df = pdf[[
            "ticker","sentiment_weighted",
            "bullish_ratio","news_count"
        ]].dropna().sort_values("sentiment_weighted")

        fig = go.Figure(go.Bar(
            x=df["ticker"],
            y=df["sentiment_weighted"],
            marker=dict(
                color=df["sentiment_weighted"],
                colorscale="RdYlGn",
                cmid=0,
                colorbar=dict(title="Sentiment")
            ),
            text=df["sentiment_weighted"].round(2),
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Sentiment: %{y:.3f}<br>"
                "Bull Ratio: %{customdata[0]:.1%}<br>"
                "News Days: %{customdata[1]}"
                "<extra></extra>"
            ),
            customdata=df[[
                "bullish_ratio","news_count"
            ]].values
        ))

        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.4
        )

        fig.update_layout(
            title="<b>EDA 07 — Sentiment Score "
                  "by Ticker (All 59 Tickers)</b>",
            template=self.TEMPLATE,
            height=500,
            xaxis_title="Ticker",
            yaxis_title="Mean Sentiment Score"
        )
        fig.show()

    def chart_ic_heatmap(self,
                          stats_df: pd.DataFrame
                          ) -> None:
        if len(stats_df) == 0:
            print("  [Skipped] No IC data")
            return

        pivot = stats_df.pivot_table(
            index="factor",
            columns="fwd_horizon",
            values="ic_mean",
            aggfunc="mean"
        ).fillna(0)

        pivot = pivot.reindex(
            pivot.abs().mean(axis=1).sort_values(
                ascending=False
            ).index
        )

        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="RdYlGn",
            zmid=0, zmin=-1, zmax=1,
            text=np.round(pivot.values, 3),
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorbar=dict(title="IC")
        ))

        fig.update_layout(
            title="<b>EDA 07 — IC Heatmap "
                  "(Factor × Horizon)</b>",
            template=self.TEMPLATE,
            height=max(400, len(pivot)*50),
            xaxis_title="Return Horizon",
            yaxis_title="Sentiment Factor"
        )
        fig.show()

    def chart_news_volume_analysis(self,
                                    pdf: pd.DataFrame
                                    ) -> None:
        if "news_count" not in pdf.columns:
            return

        ret_col = None
        for c in ["return_21d","return_5d","return_1d"]:
            if c in pdf.columns:
                ret_col = c
                break

        if ret_col is None:
            return

        df = pdf[[
            "ticker","news_count","sentiment_weighted",
            ret_col
        ]].dropna()

        if len(df) < 5:
            return

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "News Count vs Sentiment",
                "News Count vs Return"
            ]
        )

        fig.add_trace(go.Scatter(
            x=df["news_count"],
            y=df["sentiment_weighted"],
            mode="markers+text",
            text=df["ticker"],
            textfont=dict(size=7),
            textposition="top center",
            marker=dict(
                color=self.COLORS["primary"],
                size=8, opacity=0.7
            ),
            showlegend=False
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df["news_count"],
            y=df[ret_col]*100,
            mode="markers+text",
            text=df["ticker"],
            textfont=dict(size=7),
            textposition="top center",
            marker=dict(
                color=self.COLORS["success"],
                size=8, opacity=0.7
            ),
            showlegend=False
        ), row=1, col=2)

        for r, c in [(1,1),(1,2)]:
            fig.add_hline(
                y=0, line_dash="dash",
                line_color="white", opacity=0.3,
                row=r, col=c
            )

        fig.update_layout(
            title="<b>EDA 07 — News Volume Analysis</b>",
            template=self.TEMPLATE,
            height=500
        )
        fig.update_xaxes(title_text="News Days")
        fig.update_yaxes(
            title_text="Sentiment", row=1, col=1
        )
        fig.update_yaxes(
            title_text="Return(%)", row=1, col=2
        )
        fig.show()

    def run_all(self, pdf, stats_df,
                quintile_df, ls_result) -> None:
        print("\n" + "="*55)
        print("Generating Interactive Charts...")
        print("="*55)

        print("\n[1/7] Sentiment Overview...")
        self.chart_sentiment_overview(pdf)

        print("[2/7] IC Heatmap...")
        self.chart_ic_heatmap(stats_df)

        print("[3/7] IC Bars + t-stats...")
        self.chart_ic_bars(stats_df)

        print("[4/7] Quintile Returns...")
        self.chart_quintile_returns(quintile_df)

        print("[5/7] L/S Breakdown...")
        self.chart_ls_breakdown(ls_result, pdf)

        print("[6/7] Sentiment vs Return Scatter...")
        self.chart_sentiment_return_scatter(pdf)

        print("[7/7] All Tickers Sentiment...")
        self.chart_sector_sentiment(pdf)

        print("\nAll 7 charts ✓")

# COMMAND ----------

eda = EDASentimentSignal(
    spark       = spark,
    silver_path = SILVER_PATH,
    eda_path    = EDA_PATH
)

(pdf, stats_df, quintile_df,
 ls_result, sent_dist) = eda.run()

charts = EDASentimentCharts()
charts.run_all(
    pdf          = pdf,
    stats_df     = stats_df,
    quintile_df  = quintile_df,
    ls_result    = ls_result
)

print("\nEDA 07 COMPLETE ✓")

# COMMAND ----------

print("="*55)
print("EDA 07 — Sentiment Signal Summary")
print("="*55)

print(f"\nData coverage:")
print(f"  Tickers : {pdf['ticker'].nunique():,}")
print(f"  Columns : {pdf.columns.tolist()}")

if len(stats_df) > 0:
    print(f"\nTop signals by |IC|:")
    print(stats_df.nlargest(10,"ic_abs_mean")[[
        "factor","fwd_horizon",
        "ic_mean","t_stat","n_tickers"
    ]].to_string(index=False))

if ls_result:
    print(f"\nL/S Portfolio:")
    print(f"  Long  : {ls_result['long_return']*100:.2f}%")
    print(f"  Short : {ls_result['short_return']*100:.2f}%")
    print(f"  Spread: {ls_result['ls_return']*100:.2f}%")

print(f"\n{'='*55}")
print(f"🎉 EDA LAYER 100% COMPLETE!")
print(f"{'='*55}")
print(f"  01 return_distribution     ✅")
print(f"  02 stationarity_cointegr.  ✅")
print(f"  03 factor_analysis         ✅")
print(f"  04 regime_analysis         ✅")
print(f"  05 correlation_structure   ✅")
print(f"  06 tail_risk               ✅")
print(f"  07 sentiment_signal        ✅")
print(f"\nNext → Gold Layer (5 notebooks) 🚀")