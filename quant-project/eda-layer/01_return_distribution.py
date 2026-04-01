# Databricks notebook source
# MAGIC %pip install scipy statsmodels plotly pandas numpy --quiet

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
from scipy.stats import jarque_bera, shapiro
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

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
SILVER_PATH = f"{BASE_PATH}/silver/delta"
EDA_PATH    = f"{BASE_PATH}/eda/delta"

print("Config loaded ✓")

# COMMAND ----------

class EDAReturnDistribution:
    """
    EDA 01 — Return Distribution Analysis.
    Tests: JB, Shapiro-Wilk, Skewness, Kurtosis,
           Fat tails, QQ, Rolling moments
    Output: eda/delta/return_distribution
    """

    def __init__(self, spark, silver_path, eda_path):
        self.spark       = spark
        self.silver_path = f"{silver_path}/ohlcv"
        self.eda_path    = f"{eda_path}/return_distribution"
        print("EDAReturnDistribution ✓")

    def load(self):
        print("\nLoading silver OHLCV...")
        df = self.spark.read.format("delta").load(self.silver_path)
        print(f"  Rows    : {df.count():,}")
        print(f"  Tickers : {df.select('ticker').distinct().count():,}")
        return df

    def compute_ticker_stats(self, df) -> pd.DataFrame:
        print("\nComputing per-ticker stats...")
        returns_df = df.select(
            "ticker","date","return_1d","log_return_1d"
        ).dropna().toPandas()

        results = []
        tickers = returns_df["ticker"].unique()

        for i, ticker in enumerate(tickers):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(tickers)}")

            r = returns_df[
                returns_df["ticker"] == ticker
            ]["return_1d"].values

            if len(r) < 30:
                continue

            mean_r  = np.mean(r)
            std_r   = np.std(r)
            skew_r  = stats.skew(r)
            kurt_r  = stats.kurtosis(r)
            n       = len(r)

            jb_stat, jb_pval = jarque_bera(r)
            sample           = r[:5000] if len(r) > 5000 else r
            sw_stat, sw_pval = shapiro(sample)

            normal_tail    = 2 * stats.norm.cdf(-2)
            actual_tail    = np.mean(np.abs(r) > 2 * std_r)
            fat_tail_ratio = actual_tail / normal_tail \
                             if normal_tail > 0 else 1.0

            results.append({
                "ticker"          : ticker,
                "n_obs"           : n,
                "mean_return"     : mean_r,
                "std_return"      : std_r,
                "skewness"        : skew_r,
                "excess_kurtosis" : kurt_r,
                "min_return"      : np.min(r),
                "max_return"      : np.max(r),
                "p01"             : np.percentile(r, 1),
                "p05"             : np.percentile(r, 5),
                "p95"             : np.percentile(r, 95),
                "p99"             : np.percentile(r, 99),
                "jb_statistic"    : jb_stat,
                "jb_pvalue"       : jb_pval,
                "jb_reject_normal": int(jb_pval < 0.05),
                "sw_statistic"    : sw_stat,
                "sw_pvalue"       : sw_pval,
                "sw_reject_normal": int(sw_pval < 0.05),
                "fat_tail_ratio"  : fat_tail_ratio,
                "ann_return"      : mean_r * 252,
                "ann_vol"         : std_r * np.sqrt(252),
                "sharpe_ratio"    : (mean_r * 252) /
                                    (std_r * np.sqrt(252))
                                    if std_r > 0 else 0,
            })

        pdf = pd.DataFrame(results)
        print(f"  Tickers processed : {len(pdf):,}")
        print(f"  JB reject normal  : "
              f"{pdf['jb_reject_normal'].sum():,} "
              f"({pdf['jb_reject_normal'].mean()*100:.1f}%)")
        return pdf

    def compute_market_stats(self, df):
        print("\nComputing market-wide stats...")
        daily = df.groupBy("date").agg(
            F.mean("return_1d").alias("cs_mean"),
            F.stddev("return_1d").alias("cs_std"),
            F.count("return_1d").alias("cs_count"),
            F.percentile_approx("return_1d", 0.01).alias("cs_p01"),
            F.percentile_approx("return_1d", 0.05).alias("cs_p05"),
            F.percentile_approx("return_1d", 0.25).alias("cs_p25"),
            F.percentile_approx("return_1d", 0.75).alias("cs_p75"),
            F.percentile_approx("return_1d", 0.95).alias("cs_p95"),
            F.percentile_approx("return_1d", 0.99).alias("cs_p99"),
            F.sum(
                F.when(F.col("return_1d") > 0, 1).otherwise(0)
            ).alias("up_count"),
            F.sum(
                F.when(F.col("return_1d") < 0, 1).otherwise(0)
            ).alias("down_count"),
        ).withColumn("year",  F.year("date")) \
         .withColumn("month", F.month("date")) \
         .orderBy("date")
        print("  Market stats computed ✓")
        return daily

    def compute_rolling_stats(self, df):
        print("\nComputing rolling stats...")
        index_tickers = ["SPY","QQQ","AAPL","MSFT","NVDA"]
        w63  = Window.partitionBy("ticker") \
                     .orderBy("date").rowsBetween(-62, 0)
        w252 = Window.partitionBy("ticker") \
                     .orderBy("date").rowsBetween(-251, 0)

        rolling = df.filter(
            F.col("ticker").isin(index_tickers)
        ).withColumn(
            "rolling_skew_63d",
            F.skewness("return_1d").over(w63)
        ).withColumn(
            "rolling_kurt_63d",
            F.kurtosis("return_1d").over(w63)
        ).withColumn(
            "rolling_skew_252d",
            F.skewness("return_1d").over(w252)
        ).withColumn(
            "rolling_kurt_252d",
            F.kurtosis("return_1d").over(w252)
        ).select(
            "ticker","date","return_1d",
            "rolling_skew_63d","rolling_kurt_63d",
            "rolling_skew_252d","rolling_kurt_252d"
        ).dropna()

        print(f"  Rolling stats rows : {rolling.count():,}")
        return rolling

    def fit_distributions(self, df) -> pd.DataFrame:
        print("\nFitting distributions...")
        spy = df.filter(
            F.col("ticker") == "SPY"
        ).select("return_1d").dropna().toPandas()["return_1d"].values

        if len(spy) < 100:
            spy = df.filter(
                F.col("ticker") == "AAPL"
            ).select("return_1d").dropna().toPandas()["return_1d"].values

        distributions = {
            "normal" : stats.norm,
            "t"      : stats.t,
            "laplace": stats.laplace,
            "logistic": stats.logistic,
        }

        results = []
        for name, dist in distributions.items():
            try:
                params         = dist.fit(spy)
                ks_stat, ks_p  = stats.kstest(spy, dist.cdf,
                                               args=params)
                log_lik        = np.sum(dist.logpdf(spy, *params))
                results.append({
                    "distribution"  : name,
                    "ks_statistic"  : ks_stat,
                    "ks_pvalue"     : ks_p,
                    "log_likelihood": log_lik,
                    "params"        : str(params)
                })
                print(f"  ✓ {name}: KS={ks_stat:.4f} p={ks_p:.4f}")
            except Exception as e:
                print(f"  ✗ {name}: {e}")

        return pd.DataFrame(results), spy

    def write_results(self, ticker_stats, market_stats,
                      rolling_stats) -> None:
        print(f"\nWriting EDA results to Delta...")
        self.spark.createDataFrame(ticker_stats).write \
            .format("delta").mode("overwrite") \
            .option("overwriteSchema","true") \
            .save(f"{self.eda_path}/ticker_stats")
        print("  ✓ ticker_stats")

        market_stats.write.format("delta").mode("overwrite") \
            .option("overwriteSchema","true") \
            .partitionBy("year","month") \
            .save(f"{self.eda_path}/market_stats")
        print("  ✓ market_stats")

        rolling_stats.write.format("delta").mode("overwrite") \
            .option("overwriteSchema","true") \
            .save(f"{self.eda_path}/rolling_stats")
        print("  ✓ rolling_stats")

    def validate(self) -> None:
        print("\n" + "="*55)
        print("EDA 01 FINDINGS — Return Distribution")
        print("="*55)
        ts = self.spark.read.format("delta").load(
            f"{self.eda_path}/ticker_stats"
        ).toPandas()

        print(f"  Tickers analyzed    : {len(ts):,}")
        print(f"  Reject normality    : "
              f"{ts['jb_reject_normal'].sum():,} "
              f"({ts['jb_reject_normal'].mean()*100:.1f}%)")
        print(f"  Avg excess kurtosis : "
              f"{ts['excess_kurtosis'].mean():.2f}")
        print(f"  Avg skewness        : "
              f"{ts['skewness'].mean():.2f}")
        print(f"  Avg fat tail ratio  : "
              f"{ts['fat_tail_ratio'].mean():.2f}x")
        print(f"  Avg ann vol         : "
              f"{ts['ann_vol'].mean()*100:.1f}%")
        print(f"  Avg Sharpe          : "
              f"{ts['sharpe_ratio'].mean():.2f}")
        print(f"\n  KEY DECISION:")
        print(f"  → Returns are NOT normally distributed")
        print(f"  → Fat tails confirmed — use robust stats")
        print(f"  → Winsorize at 1/99 pct already applied ✓")
        print(f"  → Use returns (not prices) in Gold layer ✓")

    def run(self):
        print("="*55)
        print("EDA 01 — Return Distribution Analysis")
        print("="*55)
        df            = self.load()
        ticker_stats  = self.compute_ticker_stats(df)
        market_stats  = self.compute_market_stats(df)
        rolling_stats = self.compute_rolling_stats(df)
        dist_fits, spy_returns = self.fit_distributions(df)
        self.write_results(ticker_stats, market_stats, rolling_stats)
        self.validate()
        return df, ticker_stats, market_stats, rolling_stats, \
               dist_fits, spy_returns

# COMMAND ----------

class EDAReturnCharts:
    """
    Interactive Plotly charts for EDA 01.
    All charts display inline in Databricks.
    """

    COLORS = {
        "primary"  : "#2196F3",
        "secondary": "#FF5722",
        "success"  : "#4CAF50",
        "warning"  : "#FFC107",
        "purple"   : "#9C27B0",
        "dark"     : "#212121",
    }

    TEMPLATE = "plotly_dark"

    # ------------------------------------------------------------------ #
    #  Chart 1 — Return histogram + normal overlay + QQ plot
    # ------------------------------------------------------------------ #
    def chart_histogram_qq(self, returns: np.ndarray,
                            ticker: str = "SPY") -> None:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                f"{ticker} — Return Distribution vs Normal",
                f"{ticker} — QQ Plot vs Normal"
            ]
        )

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=returns * 100,
                nbinsx=120,
                name="Actual Returns",
                histnorm="probability density",
                marker_color=self.COLORS["primary"],
                opacity=0.7
            ),
            row=1, col=1
        )

        # Normal overlay
        mu, sigma = stats.norm.fit(returns)
        x = np.linspace(returns.min(), returns.max(), 300)
        fig.add_trace(
            go.Scatter(
                x=x * 100,
                y=stats.norm.pdf(x, mu, sigma),
                name=f"Normal(μ={mu*100:.3f}%, σ={sigma*100:.2f}%)",
                line=dict(color=self.COLORS["secondary"], width=2)
            ),
            row=1, col=1
        )

        # t-distribution overlay
        df_t, loc_t, scale_t = stats.t.fit(returns)
        fig.add_trace(
            go.Scatter(
                x=x * 100,
                y=stats.t.pdf(x, df_t, loc_t, scale_t),
                name=f"t-dist(df={df_t:.1f})",
                line=dict(color=self.COLORS["success"],
                          width=2, dash="dash")
            ),
            row=1, col=1
        )

        # QQ plot
        (osm, osr), (slope, intercept, r) = stats.probplot(
            returns, dist="norm"
        )
        fig.add_trace(
            go.Scatter(
                x=osm, y=osr,
                mode="markers",
                name="Sample Quantiles",
                marker=dict(color=self.COLORS["primary"],
                            size=3, opacity=0.5)
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=[osm.min(), osm.max()],
                y=[slope*osm.min()+intercept,
                   slope*osm.max()+intercept],
                name=f"Normal Line (R²={r**2:.3f})",
                line=dict(color=self.COLORS["secondary"], width=2)
            ),
            row=1, col=2
        )

        fig.update_layout(
            title=dict(
                text=f"<b>EDA 01 — {ticker} Return Distribution</b>",
                font=dict(size=18)
            ),
            template=self.TEMPLATE,
            height=500,
            showlegend=True
        )
        fig.update_xaxes(title_text="Daily Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles",
                         row=1, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)

        fig.show()

    # ------------------------------------------------------------------ #
    #  Chart 2 — Cross-ticker skewness & kurtosis
    # ------------------------------------------------------------------ #
    def chart_skew_kurt(self,
                         ticker_stats: pd.DataFrame) -> None:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Skewness Distribution Across Tickers",
                "Excess Kurtosis (Fat Tails) Across Tickers"
            ]
        )

        # Skewness histogram
        fig.add_trace(
            go.Histogram(
                x=ticker_stats["skewness"].dropna(),
                nbinsx=50,
                name="Skewness",
                marker_color=self.COLORS["secondary"],
                opacity=0.7
            ),
            row=1, col=1
        )
        fig.add_vline(
            x=0, line_dash="dash",
            line_color="white", opacity=0.5,
            annotation_text="Normal=0",
            row=1, col=1
        )
        fig.add_vline(
            x=ticker_stats["skewness"].mean(),
            line_color=self.COLORS["warning"],
            line_width=2,
            annotation_text=f"Mean={ticker_stats['skewness'].mean():.2f}",
            row=1, col=1
        )

        # Kurtosis histogram
        fig.add_trace(
            go.Histogram(
                x=ticker_stats["excess_kurtosis"].dropna().clip(-5,50),
                nbinsx=50,
                name="Excess Kurtosis",
                marker_color=self.COLORS["primary"],
                opacity=0.7
            ),
            row=1, col=2
        )
        fig.add_vline(
            x=0, line_dash="dash",
            line_color="white", opacity=0.5,
            annotation_text="Normal=0",
            row=1, col=2
        )
        fig.add_vline(
            x=ticker_stats["excess_kurtosis"].mean(),
            line_color=self.COLORS["warning"],
            line_width=2,
            annotation_text=f"Mean={ticker_stats['excess_kurtosis'].mean():.2f}",
            row=1, col=2
        )

        fig.update_layout(
            title=dict(
                text="<b>EDA 01 — Skewness & Kurtosis Across S&P 500</b>",
                font=dict(size=18)
            ),
            template=self.TEMPLATE,
            height=500,
            showlegend=False
        )
        fig.update_xaxes(title_text="Skewness", row=1, col=1)
        fig.update_xaxes(title_text="Excess Kurtosis", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.show()

    # ------------------------------------------------------------------ #
    #  Chart 3 — Tail risk analysis
    # ------------------------------------------------------------------ #
    def chart_tail_risk(self,
                         ticker_stats: pd.DataFrame) -> None:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Actual 1% VaR Distribution",
                "Fat Tail Ratio vs Normal Distribution"
            ]
        )

        p01_vals   = ticker_stats["p01"].dropna() * 100
        normal_p01 = stats.norm.ppf(0.01) * \
                     ticker_stats["std_return"].mean() * 100

        # VaR histogram
        fig.add_trace(
            go.Histogram(
                x=p01_vals,
                nbinsx=50,
                name="1% VaR",
                marker_color="darkred",
                opacity=0.7
            ),
            row=1, col=1
        )
        fig.add_vline(
            x=p01_vals.mean(),
            line_color=self.COLORS["warning"],
            line_width=2,
            annotation_text=f"Actual={p01_vals.mean():.1f}%",
            row=1, col=1
        )
        fig.add_vline(
            x=normal_p01,
            line_color=self.COLORS["primary"],
            line_dash="dash",
            line_width=2,
            annotation_text=f"Normal={normal_p01:.1f}%",
            row=1, col=1
        )

        # Fat tail ratio
        ftr = ticker_stats["fat_tail_ratio"].dropna().clip(0, 10)
        fig.add_trace(
            go.Histogram(
                x=ftr,
                nbinsx=50,
                name="Fat Tail Ratio",
                marker_color="orange",
                opacity=0.7
            ),
            row=1, col=2
        )
        fig.add_vline(
            x=1.0,
            line_color="white",
            line_dash="dash",
            line_width=2,
            annotation_text="Normal=1.0x",
            row=1, col=2
        )
        fig.add_vline(
            x=ftr.mean(),
            line_color=self.COLORS["warning"],
            line_width=2,
            annotation_text=f"Mean={ftr.mean():.2f}x",
            row=1, col=2
        )

        fig.update_layout(
            title=dict(
                text="<b>EDA 01 — Tail Risk Analysis</b>",
                font=dict(size=18)
            ),
            template=self.TEMPLATE,
            height=500,
            showlegend=False
        )
        fig.update_xaxes(title_text="1% VaR (%)", row=1, col=1)
        fig.update_xaxes(title_text="Fat Tail Ratio", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.show()

    # ------------------------------------------------------------------ #
    #  Chart 4 — Vol vs Sharpe interactive scatter
    # ------------------------------------------------------------------ #
    def chart_vol_sharpe_scatter(self,
                                  ticker_stats: pd.DataFrame) -> None:
        ts = ticker_stats.dropna(
            subset=["ann_vol","sharpe_ratio","excess_kurtosis"]
        )

        fig = px.scatter(
            ts,
            x=ts["ann_vol"] * 100,
            y="sharpe_ratio",
            color="excess_kurtosis",
            color_continuous_scale="RdYlGn_r",
            hover_data={
                "ticker"          : True,
                "ann_vol"         : ":.2%",
                "sharpe_ratio"    : ":.2f",
                "excess_kurtosis" : ":.2f",
                "skewness"        : ":.2f",
                "n_obs"           : True,
            },
            text="ticker",
            size=np.ones(len(ts)) * 8,
            template=self.TEMPLATE,
            title="<b>EDA 01 — Annualized Vol vs Sharpe Ratio</b>",
            labels={
                "x"               : "Annualized Volatility (%)",
                "sharpe_ratio"    : "Sharpe Ratio",
                "excess_kurtosis" : "Excess Kurtosis"
            },
            color_continuous_midpoint=0,
        )

        fig.update_traces(
            textposition="top center",
            textfont=dict(size=7),
            marker=dict(size=8, opacity=0.7)
        )

        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="white",
            opacity=0.3
        )
        fig.add_hline(
            y=1,
            line_dash="dash",
            line_color=self.COLORS["success"],
            opacity=0.5,
            annotation_text="Sharpe=1.0"
        )

        fig.update_layout(height=650)
        fig.show()

    # ------------------------------------------------------------------ #
    #  Chart 5 — Rolling skewness & kurtosis
    # ------------------------------------------------------------------ #
    def chart_rolling_moments(self,
                               rolling_pdf: pd.DataFrame) -> None:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=[
                "Rolling 63-Day Skewness",
                "Rolling 63-Day Excess Kurtosis"
            ],
            vertical_spacing=0.1
        )

        colors = [
            self.COLORS["primary"],
            self.COLORS["secondary"],
            self.COLORS["success"],
            self.COLORS["warning"],
            self.COLORS["purple"],
        ]

        tickers = rolling_pdf["ticker"].unique()
        for i, ticker in enumerate(tickers):
            t_data = rolling_pdf[
                rolling_pdf["ticker"] == ticker
            ].sort_values("date")

            color = colors[i % len(colors)]

            fig.add_trace(
                go.Scatter(
                    x=t_data["date"],
                    y=t_data["rolling_skew_63d"],
                    name=ticker,
                    line=dict(color=color, width=1.5),
                    showlegend=True
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=t_data["date"],
                    y=t_data["rolling_kurt_63d"],
                    name=ticker,
                    line=dict(color=color, width=1.5),
                    showlegend=False
                ),
                row=2, col=1
            )

        # Zero lines
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=1, col=1
        )
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=2, col=1
        )

        fig.update_layout(
            title=dict(
                text="<b>EDA 01 — Rolling Distribution Moments</b>",
                font=dict(size=18)
            ),
            template=self.TEMPLATE,
            height=700,
            hovermode="x unified"
        )
        fig.update_yaxes(title_text="Skewness", row=1, col=1)
        fig.update_yaxes(title_text="Excess Kurtosis", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.show()

    # ------------------------------------------------------------------ #
    #  Chart 6 — Market cross-sectional distribution
    # ------------------------------------------------------------------ #
    def chart_market_cs_distribution(self,
                                      market_pdf: pd.DataFrame) -> None:
        market_pdf = market_pdf.sort_values("date")
        market_pdf["date"] = pd.to_datetime(market_pdf["date"])

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=[
                "Cross-Sectional Return Distribution Over Time",
                "Market Dispersion (CS Std Dev)"
            ],
            vertical_spacing=0.1
        )

        # Shaded percentile bands
        fig.add_trace(
            go.Scatter(
                x=pd.concat([market_pdf["date"],
                              market_pdf["date"].iloc[::-1]]),
                y=pd.concat([market_pdf["cs_p95"],
                              market_pdf["cs_p05"].iloc[::-1]]) * 100,
                fill="toself",
                fillcolor="rgba(33,150,243,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="5-95th Pct",
                showlegend=True
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=pd.concat([market_pdf["date"],
                              market_pdf["date"].iloc[::-1]]),
                y=pd.concat([market_pdf["cs_p75"],
                              market_pdf["cs_p25"].iloc[::-1]]) * 100,
                fill="toself",
                fillcolor="rgba(33,150,243,0.3)",
                line=dict(color="rgba(255,255,255,0)"),
                name="25-75th Pct",
                showlegend=True
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=market_pdf["date"],
                y=market_pdf["cs_mean"] * 100,
                name="CS Mean",
                line=dict(color="white", width=1.5),
            ),
            row=1, col=1
        )

        # Zero line
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=1, col=1
        )

        # Dispersion
        fig.add_trace(
            go.Scatter(
                x=market_pdf["date"],
                y=market_pdf["cs_std"] * 100,
                name="CS Std Dev",
                fill="tozeroy",
                fillcolor="rgba(255,87,34,0.3)",
                line=dict(color=self.COLORS["secondary"], width=1.5),
            ),
            row=2, col=1
        )

        fig.update_layout(
            title=dict(
                text="<b>EDA 01 — Market Cross-Sectional Distribution</b>",
                font=dict(size=18)
            ),
            template=self.TEMPLATE,
            height=700,
            hovermode="x unified"
        )
        fig.update_yaxes(
            title_text="Daily Return (%)", row=1, col=1
        )
        fig.update_yaxes(
            title_text="CS Std Dev (%)", row=2, col=1
        )
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.show()

    # ------------------------------------------------------------------ #
    #  Chart 7 — Distribution fit comparison
    # ------------------------------------------------------------------ #
    def chart_distribution_fits(self, returns: np.ndarray,
                                  dist_fits: pd.DataFrame,
                                  ticker: str = "SPY") -> None:
        fig = go.Figure()

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=returns * 100,
                nbinsx=120,
                name="Actual Returns",
                histnorm="probability density",
                marker_color=self.COLORS["primary"],
                opacity=0.5
            )
        )

        x = np.linspace(returns.min(), returns.max(), 300)
        dist_objects = {
            "normal" : stats.norm,
            "t"      : stats.t,
            "laplace": stats.laplace,
            "logistic": stats.logistic,
        }
        colors = [
            self.COLORS["secondary"],
            self.COLORS["success"],
            self.COLORS["warning"],
            self.COLORS["purple"],
        ]

        for i, (name, dist) in enumerate(dist_objects.items()):
            try:
                row = dist_fits[
                    dist_fits["distribution"] == name
                ]
                if len(row) == 0:
                    continue
                params = eval(row["params"].values[0])
                ks_p   = row["ks_pvalue"].values[0]
                fig.add_trace(
                    go.Scatter(
                        x=x * 100,
                        y=dist.pdf(x, *params),
                        name=f"{name} (KS p={ks_p:.3f})",
                        line=dict(
                            color=colors[i], width=2,
                            dash="dash" if name == "normal"
                                 else "solid"
                        )
                    )
                )
            except Exception:
                continue

        fig.update_layout(
            title=dict(
                text=f"<b>EDA 01 — Distribution Fit Comparison ({ticker})</b>",
                font=dict(size=18)
            ),
            template=self.TEMPLATE,
            height=500,
            xaxis_title="Daily Return (%)",
            yaxis_title="Probability Density",
            hovermode="x unified"
        )
        fig.show()

    # ------------------------------------------------------------------ #
    #  Chart 8 — Top/bottom tickers by metric
    # ------------------------------------------------------------------ #
    def chart_top_bottom_tickers(self,
                                  ticker_stats: pd.DataFrame) -> None:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Top 20 — Highest Sharpe Ratio",
                "Top 20 — Most Negatively Skewed",
                "Top 20 — Highest Fat Tail Ratio",
                "Top 20 — Highest Annualized Vol"
            ]
        )

        # Sharpe
        top_sharpe = ticker_stats.nlargest(20, "sharpe_ratio")
        fig.add_trace(
            go.Bar(
                x=top_sharpe["ticker"],
                y=top_sharpe["sharpe_ratio"],
                marker_color=self.COLORS["success"],
                name="Sharpe",
                showlegend=False
            ),
            row=1, col=1
        )

        # Most negative skew
        neg_skew = ticker_stats.nsmallest(20, "skewness")
        fig.add_trace(
            go.Bar(
                x=neg_skew["ticker"],
                y=neg_skew["skewness"],
                marker_color=self.COLORS["secondary"],
                name="Skewness",
                showlegend=False
            ),
            row=1, col=2
        )

        # Fat tail ratio
        top_fat = ticker_stats.nlargest(20, "fat_tail_ratio")
        fig.add_trace(
            go.Bar(
                x=top_fat["ticker"],
                y=top_fat["fat_tail_ratio"],
                marker_color="orange",
                name="Fat Tail",
                showlegend=False
            ),
            row=2, col=1
        )

        # Highest vol
        top_vol = ticker_stats.nlargest(20, "ann_vol")
        fig.add_trace(
            go.Bar(
                x=top_vol["ticker"],
                y=top_vol["ann_vol"] * 100,
                marker_color=self.COLORS["purple"],
                name="Ann Vol",
                showlegend=False
            ),
            row=2, col=2
        )

        fig.update_layout(
            title=dict(
                text="<b>EDA 01 — Top Tickers by Distribution Metrics</b>",
                font=dict(size=18)
            ),
            template=self.TEMPLATE,
            height=700,
        )
        fig.show()

    # ------------------------------------------------------------------ #
    #  Run all charts
    # ------------------------------------------------------------------ #
    def run_all(self, spy_returns: np.ndarray,
                ticker_stats: pd.DataFrame,
                rolling_pdf: pd.DataFrame,
                market_pdf: pd.DataFrame,
                dist_fits: pd.DataFrame) -> None:
        print("\n" + "="*55)
        print("Generating Interactive Plotly Charts...")
        print("="*55)

        ticker = "SPY" if len(spy_returns) > 100 else "AAPL"

        print("\n[1/8] Return Histogram + QQ Plot...")
        self.chart_histogram_qq(spy_returns, ticker)

        print("[2/8] Skewness & Kurtosis Distribution...")
        self.chart_skew_kurt(ticker_stats)

        print("[3/8] Tail Risk Analysis...")
        self.chart_tail_risk(ticker_stats)

        print("[4/8] Vol vs Sharpe Scatter...")
        self.chart_vol_sharpe_scatter(ticker_stats)

        print("[5/8] Rolling Distribution Moments...")
        if len(rolling_pdf) > 0:
            self.chart_rolling_moments(rolling_pdf)

        print("[6/8] Market Cross-Sectional Distribution...")
        if len(market_pdf) > 0:
            self.chart_market_cs_distribution(market_pdf)

        print("[7/8] Distribution Fit Comparison...")
        if not dist_fits.empty:
            self.chart_distribution_fits(
                spy_returns, dist_fits, ticker
            )

        print("[8/8] Top/Bottom Tickers by Metric...")
        self.chart_top_bottom_tickers(ticker_stats)

        print("\nAll 8 charts generated ✓")

# COMMAND ----------

# Run analysis
eda = EDAReturnDistribution(
    spark       = spark,
    silver_path = SILVER_PATH,
    eda_path    = EDA_PATH
)

df, ticker_stats, market_stats, rolling_stats, \
dist_fits, spy_returns = eda.run()

# Convert Spark DFs to pandas for charts
rolling_pdf = rolling_stats.toPandas()
market_pdf  = market_stats.toPandas()

# Generate all interactive charts
charts = EDAReturnCharts()
charts.run_all(
    spy_returns  = spy_returns,
    ticker_stats = ticker_stats,
    rolling_pdf  = rolling_pdf,
    market_pdf   = market_pdf,
    dist_fits    = dist_fits
)

print("\nEDA 01 COMPLETE ✓")

# COMMAND ----------

ts = spark.read.format("delta").load(
    f"{EDA_PATH}/return_distribution/ticker_stats"
).toPandas()

print("="*50)
print("EDA 01 — Key Findings Summary")
print("="*50)
print(f"Tickers analyzed    : {len(ts):,}")
print(f"Reject normality    : "
      f"{ts['jb_reject_normal'].mean()*100:.1f}%")
print(f"Avg excess kurtosis : {ts['excess_kurtosis'].mean():.2f}")
print(f"Avg skewness        : {ts['skewness'].mean():.2f}")
print(f"Avg fat tail ratio  : {ts['fat_tail_ratio'].mean():.2f}x")
print(f"Avg ann vol         : {ts['ann_vol'].mean()*100:.1f}%")
print(f"Avg Sharpe          : {ts['sharpe_ratio'].mean():.2f}")

print(f"\nTop 5 highest Sharpe:")
print(ts.nlargest(5,"sharpe_ratio")[
    ["ticker","sharpe_ratio","ann_vol","skewness"]
].to_string(index=False))

print(f"\nTop 5 most fat-tailed:")
print(ts.nlargest(5,"excess_kurtosis")[
    ["ticker","excess_kurtosis","fat_tail_ratio","p01"]
].to_string(index=False))