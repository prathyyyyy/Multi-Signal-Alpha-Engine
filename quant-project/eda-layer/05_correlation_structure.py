# Databricks notebook source
# MAGIC %pip install arch plotly scipy pandas numpy statsmodels==0.14.5 --quiet

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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
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

print("Config loaded ✓")

# COMMAND ----------

class EDACorrelationStructure:
    """
    EDA 05 — Correlation Structure Analysis.
    Answers:
      - How correlated are stocks?
      - How does correlation change over time?
      - What are the principal risk factors?
      - Which sectors cluster together?

    Methods:
      - Rolling correlations (63d, 252d)
      - DCC-GARCH (dynamic conditional correlation)
      - PCA factor decomposition
      - Hierarchical clustering
      - Sector correlation heatmap

    Optimizations:
      - Spark computes rolling stats
      - Vectorized numpy for correlations
      - PCA on pre-computed return matrix
    """

    # Sector mapping
    SECTORS = {
        "Tech"      : ["AAPL","MSFT","NVDA","AMD","INTC",
                        "QCOM","ADBE","CRM","ORCL","IBM"],
        "Financials": ["JPM","BAC","WFC","GS","MS",
                        "C","BLK","AXP","USB","PNC"],
        "Healthcare": ["JNJ","PFE","UNH","ABBV","MRK",
                        "TMO","ABT","DHR","BMY","AMGN"],
        "Energy"    : ["XOM","CVX","COP","SLB","EOG",
                        "OXY","MPC","VLO","PSX","HAL"],
        "Consumer"  : ["AMZN","WMT","COST","TGT","HD",
                        "LOW","MCD","NKE","SBUX","TJX"],
    }

    def __init__(self, spark, silver_path, eda_path):
        self.spark       = spark
        self.silver_path = f"{silver_path}/ohlcv"
        self.eda_path    = f"{eda_path}/correlation_structure"
        print("EDACorrelationStructure ✓")

    # ------------------------------------------------------------------ #
    #  Step 1 — Load returns matrix
    # ------------------------------------------------------------------ #
    def load_returns(self,
                     min_obs: int = 500) -> pd.DataFrame:
        print("\nStep 1: Loading returns matrix...")
        start = datetime.now()

        df = self.spark.read.format("delta").load(
            self.silver_path
        )

        # Filter tickers with enough history
        ticker_counts = df.groupBy("ticker").count() \
                          .filter(F.col("count") >= min_obs)
        valid_tickers = [
            r.ticker for r in ticker_counts.collect()
        ]
        print(f"  Tickers with {min_obs}+ obs: "
              f"{len(valid_tickers):,}")

        # Get returns
        pdf = df.filter(
            F.col("ticker").isin(valid_tickers)
        ).select(
            "date","ticker","return_1d"
        ).toPandas()

        pdf["date"] = pd.to_datetime(pdf["date"])

        # Pivot to wide
        returns = pdf.pivot_table(
            index="date",
            columns="ticker",
            values="return_1d",
            aggfunc="mean"
        ).sort_index()

        returns = returns.dropna(
            axis=1, thresh=min_obs
        )

        elapsed = (datetime.now() - start).seconds
        print(f"  Return matrix : {returns.shape}")
        print(f"  Date range    : "
              f"{returns.index.min().date()} → "
              f"{returns.index.max().date()}")
        print(f"  Time elapsed  : {elapsed}s")
        return returns

    # ------------------------------------------------------------------ #
    #  Step 2 — Full correlation matrix
    # ------------------------------------------------------------------ #
    def compute_full_correlation(self,
                                  returns: pd.DataFrame
                                  ) -> pd.DataFrame:
        print("\nStep 2: Computing full correlation matrix...")

        # Fill NaN with 0 for correlation
        ret_filled = returns.fillna(0)
        corr_matrix = ret_filled.corr(method="spearman")

        # Summary stats
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1)
            .astype(bool)
        )
        vals = upper.stack().values

        print(f"  Mean correlation : {vals.mean():.3f}")
        print(f"  Std correlation  : {vals.std():.3f}")
        print(f"  Max correlation  : {vals.max():.3f}")
        print(f"  Min correlation  : {vals.min():.3f}")
        print(f"  % pairs > 0.5    : "
              f"{(vals > 0.5).mean()*100:.1f}%")
        return corr_matrix

    # ------------------------------------------------------------------ #
    #  Step 3 — Sector correlation
    # ------------------------------------------------------------------ #
    def compute_sector_correlation(self,
                                    returns: pd.DataFrame
                                    ) -> dict:
        print("\nStep 3: Computing sector correlations...")
        sector_results = {}

        for sector, tickers in self.SECTORS.items():
            available = [
                t for t in tickers
                if t in returns.columns
            ]
            if len(available) < 3:
                print(f"  {sector}: only "
                      f"{len(available)} available — skip")
                continue

            sector_ret = returns[available].dropna()
            if len(sector_ret) < 100:
                continue

            corr = sector_ret.corr()
            mean_corr = corr.where(
                np.triu(np.ones(corr.shape), k=1)
                .astype(bool)
            ).stack().mean()

            sector_results[sector] = {
                "corr_matrix" : corr,
                "mean_corr"   : mean_corr,
                "tickers"     : available,
                "n_tickers"   : len(available),
            }
            print(f"  {sector:12}: mean corr = "
                  f"{mean_corr:.3f} "
                  f"({len(available)} tickers)")

        return sector_results

    # ------------------------------------------------------------------ #
    #  Step 4 — Rolling correlation
    # ------------------------------------------------------------------ #
    def compute_rolling_correlation(self,
                                     returns: pd.DataFrame,
                                     window: int = 63
                                     ) -> pd.DataFrame:
        print(f"\nStep 4: Computing rolling correlations "
              f"({window}d window)...")

        # Compute rolling avg pairwise correlation
        # Use market-wide avg as summary
        n_sample   = min(50, returns.shape[1])
        sample_ret = returns.iloc[:, :n_sample].fillna(0)

        rolling_corrs = []
        dates         = sample_ret.index

        for i in range(window, len(dates)):
            window_ret = sample_ret.iloc[i-window:i]
            corr       = window_ret.corr()
            upper      = corr.where(
                np.triu(np.ones(corr.shape), k=1)
                .astype(bool)
            )
            vals = upper.stack().values
            rolling_corrs.append({
                "date"      : dates[i],
                "mean_corr" : vals.mean(),
                "std_corr"  : vals.std(),
                "max_corr"  : vals.max(),
                "min_corr"  : vals.min(),
                "pct_high"  : (vals > 0.5).mean(),
            })

        rolling_df = pd.DataFrame(rolling_corrs)
        print(f"  Rolling corr rows : {len(rolling_df):,}")
        return rolling_df

    # ------------------------------------------------------------------ #
    #  Step 5 — PCA factor decomposition
    # ------------------------------------------------------------------ #
    def compute_pca(self,
                    returns: pd.DataFrame,
                    n_components: int = 10) -> dict:
        print(f"\nStep 5: PCA decomposition "
              f"({n_components} components)...")

        # Use tickers with complete data
        ret_clean = returns.dropna(axis=1, thresh=1000)
        ret_filled = ret_clean.fillna(
            ret_clean.mean()
        )

        # Standardize
        scaler     = StandardScaler()
        ret_scaled = scaler.fit_transform(ret_filled)

        # PCA
        pca   = PCA(n_components=min(
            n_components, ret_filled.shape[1]
        ))
        pca.fit(ret_scaled)

        explained  = pca.explained_variance_ratio_
        cumulative = np.cumsum(explained)

        print(f"  Tickers used      : "
              f"{ret_filled.shape[1]:,}")
        print(f"  PC1 explains      : "
              f"{explained[0]*100:.1f}%")
        print(f"  PC1-3 explains    : "
              f"{cumulative[2]*100:.1f}%")
        print(f"  PC1-5 explains    : "
              f"{cumulative[4]*100:.1f}%")

        # Factor loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            index   = ret_filled.columns,
            columns = [f"PC{i+1}"
                       for i in range(pca.n_components_)]
        )

        # Factor scores (returns projected)
        scores = pd.DataFrame(
            pca.transform(ret_scaled),
            index   = ret_filled.index,
            columns = [f"PC{i+1}"
                       for i in range(pca.n_components_)]
        )

        return {
            "pca"            : pca,
            "explained"      : explained,
            "cumulative"     : cumulative,
            "loadings"       : loadings,
            "scores"         : scores,
            "n_components"   : pca.n_components_,
            "tickers"        : list(ret_filled.columns),
        }

    # ------------------------------------------------------------------ #
    #  Step 6 — DCC-GARCH (simplified)
    # ------------------------------------------------------------------ #
    def compute_dcc_correlation(self,
                                 returns: pd.DataFrame
                                 ) -> pd.DataFrame:
        """
        Simplified DCC: rolling exponentially weighted
        correlation as DCC approximation.
        Full DCC-GARCH is compute-intensive —
        this gives the same insight faster.
        """
        print("\nStep 6: Computing DCC-style "
              "dynamic correlation...")

        # Use liquid large-cap tickers
        liquid_tickers = [
            t for t in [
                "AAPL","MSFT","NVDA","JPM","BAC",
                "XOM","JNJ","AMZN","GOOGL","META"
            ] if t in returns.columns
        ]

        if len(liquid_tickers) < 3:
            liquid_tickers = list(returns.columns[:10])

        ret_subset = returns[liquid_tickers].fillna(0)

        # EWMA correlation (lambda = 0.94 — RiskMetrics)
        lam        = 0.94
        dcc_results = []

        for i in range(63, len(ret_subset)):
            window = ret_subset.iloc[max(0,i-252):i]
            # EWMA weights
            weights = np.array([
                lam**(len(window)-j-1)
                for j in range(len(window))
            ])
            weights /= weights.sum()

            # Weighted covariance
            ret_w = window.values
            mean_w = np.average(ret_w, axis=0,
                                 weights=weights)
            demeaned = ret_w - mean_w
            cov = np.dot(
                (demeaned * weights[:, None]).T,
                demeaned
            )

            # Correlation from covariance
            std_diag = np.sqrt(np.diag(cov))
            std_outer = np.outer(std_diag, std_diag)
            std_outer[std_outer == 0] = 1e-8
            corr = cov / std_outer
            np.fill_diagonal(corr, 1.0)

            # Average pairwise correlation
            upper = corr[
                np.triu(np.ones(corr.shape), k=1)
                .astype(bool)
            ]
            dcc_results.append({
                "date"     : ret_subset.index[i],
                "dcc_corr" : upper.mean(),
                "dcc_std"  : upper.std(),
            })

        dcc_df = pd.DataFrame(dcc_results)
        print(f"  DCC rows computed : {len(dcc_df):,}")
        print(f"  Avg DCC corr      : "
              f"{dcc_df['dcc_corr'].mean():.3f}")
        return dcc_df

    # ------------------------------------------------------------------ #
    #  Step 7 — Hierarchical clustering
    # ------------------------------------------------------------------ #
    def compute_clustering(self,
                            corr_matrix: pd.DataFrame,
                            n_tickers: int = 50
                            ) -> dict:
        print(f"\nStep 7: Hierarchical clustering "
              f"(top {n_tickers} tickers)...")

        # Sample top tickers
        sample = corr_matrix.iloc[
            :n_tickers, :n_tickers
        ]

        # Distance matrix (1 - correlation)
        dist_matrix = 1 - sample.fillna(0)
        np.fill_diagonal(dist_matrix.values, 0)
        dist_matrix = dist_matrix.clip(lower=0)

        # Hierarchical clustering
        linkage_matrix = linkage(
            dist_matrix.values,
            method="ward"
        )

        print(f"  Clustering done ✓")
        return {
            "linkage"   : linkage_matrix,
            "labels"    : list(sample.index),
            "dist"      : dist_matrix,
            "corr_sample": sample
        }

    # ------------------------------------------------------------------ #
    #  Write results
    # ------------------------------------------------------------------ #
    def write_results(self, corr_matrix, rolling_df,
                      pca_results, dcc_df) -> None:
        print("\nWriting results to Delta...")

        # Rolling correlation
        rolling_out = rolling_df.copy()
        rolling_out["date"] = rolling_out["date"].astype(str)
        rolling_out["year"] = pd.to_datetime(
            rolling_out["date"]
        ).dt.year
        rolling_out["month"] = pd.to_datetime(
            rolling_out["date"]
        ).dt.month

        self.spark.createDataFrame(rolling_out).write \
            .format("delta").mode("overwrite") \
            .option("overwriteSchema","true") \
            .partitionBy("year","month") \
            .save(f"{self.eda_path}/rolling_correlation")
        print("  ✓ rolling_correlation")

        # PCA explained variance
        pca_var = pd.DataFrame({
            "component": [
                f"PC{i+1}"
                for i in range(len(pca_results["explained"]))
            ],
            "explained_var" : pca_results["explained"],
            "cumulative_var": pca_results["cumulative"],
        })
        self.spark.createDataFrame(pca_var).write \
            .format("delta").mode("overwrite") \
            .option("overwriteSchema","true") \
            .save(f"{self.eda_path}/pca_variance")
        print("  ✓ pca_variance")

        # PCA loadings
        loadings = pca_results["loadings"].reset_index()
        loadings.columns = ["ticker"] + [
            c for c in loadings.columns if c != "index"
            and c != "ticker"
        ]
        self.spark.createDataFrame(loadings).write \
            .format("delta").mode("overwrite") \
            .option("overwriteSchema","true") \
            .save(f"{self.eda_path}/pca_loadings")
        print("  ✓ pca_loadings")

        # DCC correlation
        dcc_out = dcc_df.copy()
        dcc_out["date"]  = dcc_out["date"].astype(str)
        dcc_out["year"]  = pd.to_datetime(
            dcc_out["date"]
        ).dt.year
        dcc_out["month"] = pd.to_datetime(
            dcc_out["date"]
        ).dt.month

        self.spark.createDataFrame(dcc_out).write \
            .format("delta").mode("overwrite") \
            .option("overwriteSchema","true") \
            .partitionBy("year","month") \
            .save(f"{self.eda_path}/dcc_correlation")
        print("  ✓ dcc_correlation")

    # ------------------------------------------------------------------ #
    #  Validate
    # ------------------------------------------------------------------ #
    def validate(self, corr_matrix, rolling_df,
                 pca_results) -> None:
        print("\n" + "="*55)
        print("EDA 05 FINDINGS — Correlation Structure")
        print("="*55)

        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1)
            .astype(bool)
        ).stack().values

        print(f"\n  Full correlation matrix:")
        print(f"  Mean corr      : {upper.mean():.3f}")
        print(f"  Pairs > 0.5    : "
              f"{(upper>0.5).mean()*100:.1f}%")
        print(f"  Pairs > 0.7    : "
              f"{(upper>0.7).mean()*100:.1f}%")

        print(f"\n  PCA decomposition:")
        for i in range(min(5, len(
            pca_results["explained"]
        ))):
            print(f"  PC{i+1}: "
                  f"{pca_results['explained'][i]*100:.1f}% "
                  f"(cum: "
                  f"{pca_results['cumulative'][i]*100:.1f}%)")

        print(f"\n  Rolling correlation:")
        print(f"  Mean           : "
              f"{rolling_df['mean_corr'].mean():.3f}")
        print(f"  Max (crisis)   : "
              f"{rolling_df['mean_corr'].max():.3f}")
        print(f"  Min (calm)     : "
              f"{rolling_df['mean_corr'].min():.3f}")

        print(f"\n  KEY DECISIONS:")
        n_pc_80 = int(np.argmax(
            pca_results["cumulative"] >= 0.80
        )) + 1
        print(f"  → {n_pc_80} PCs explain 80% of variance")
        print(f"  → Use PCA for factor orthogonalization ✓")
        print(f"  → Rolling corr for regime detection ✓")
        print(f"  → DCC for dynamic risk model ✓")

    # ------------------------------------------------------------------ #
    #  Run
    # ------------------------------------------------------------------ #
    def run(self):
        print("="*55)
        print("EDA 05 — Correlation Structure (Optimized)")
        print("="*55)
        start = datetime.now()

        returns          = self.load_returns()
        corr_matrix      = self.compute_full_correlation(
            returns
        )
        sector_results   = self.compute_sector_correlation(
            returns
        )
        rolling_df       = self.compute_rolling_correlation(
            returns, window=63
        )
        pca_results      = self.compute_pca(returns)
        dcc_df           = self.compute_dcc_correlation(
            returns
        )
        clustering       = self.compute_clustering(
            corr_matrix
        )

        self.write_results(
            corr_matrix, rolling_df, pca_results, dcc_df
        )
        self.validate(corr_matrix, rolling_df, pca_results)

        elapsed = (datetime.now() - start).seconds / 60
        print(f"\nTotal time: {elapsed:.1f} minutes")
        print("EDA 05 COMPLETE ✓")
        return (returns, corr_matrix, sector_results,
                rolling_df, pca_results, dcc_df, clustering)

# COMMAND ----------

class EDACorrelationCharts:
    TEMPLATE = "plotly_dark"

    # ------------------------------------------------------------------ #
    #  Chart 1 — Full correlation heatmap
    # ------------------------------------------------------------------ #
    def chart_correlation_heatmap(self,
                                   corr_matrix: pd.DataFrame,
                                   n_tickers: int = 50
                                   ) -> None:
        sample = corr_matrix.iloc[
            :n_tickers, :n_tickers
        ]

        fig = go.Figure(go.Heatmap(
            z=sample.values,
            x=list(sample.columns),
            y=list(sample.index),
            colorscale="RdYlGn",
            zmid=0, zmin=-1, zmax=1,
            colorbar=dict(title="Correlation")
        ))

        fig.update_layout(
            title=dict(
                text=f"<b>EDA 05 — Correlation Matrix "
                     f"(Top {n_tickers} Tickers)</b>",
                font=dict(size=18)
            ),
            template=self.TEMPLATE,
            height=700,
            xaxis=dict(tickfont=dict(size=7)),
            yaxis=dict(tickfont=dict(size=7))
        )
        fig.show()

    # ------------------------------------------------------------------ #
    #  Chart 2 — Sector correlation heatmaps
    # ------------------------------------------------------------------ #
    def chart_sector_correlations(self,
                                   sector_results: dict
                                   ) -> None:
        n_sectors = len(sector_results)
        if n_sectors == 0:
            print("No sector data")
            return

        cols = min(3, n_sectors)
        rows = (n_sectors + cols - 1) // cols

        sector_names = list(sector_results.keys())
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[
                f"{s} (μ={sector_results[s]['mean_corr']:.2f})"
                for s in sector_names
            ]
        )

        colors = px.colors.diverging.RdYlGn

        for idx, sector in enumerate(sector_names):
            row  = idx // cols + 1
            col  = idx % cols + 1
            corr = sector_results[sector]["corr_matrix"]

            fig.add_trace(
                go.Heatmap(
                    z=corr.values,
                    x=list(corr.columns),
                    y=list(corr.index),
                    colorscale="RdYlGn",
                    zmid=0, zmin=-1, zmax=1,
                    showscale=(idx == 0),
                    colorbar=dict(
                        title="Corr",
                        len=0.3,
                        y=1 - (row-1) * (1/rows)
                    )
                ),
                row=row, col=col
            )

        fig.update_layout(
            title=dict(
                text="<b>EDA 05 — Sector Correlation Matrices</b>",
                font=dict(size=18)
            ),
            template=self.TEMPLATE,
            height=300 * rows
        )
        fig.show()

    # ------------------------------------------------------------------ #
    #  Chart 3 — Correlation distribution
    # ------------------------------------------------------------------ #
    def chart_correlation_distribution(self,
                                        corr_matrix: pd.DataFrame
                                        ) -> None:
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1)
            .astype(bool)
        ).stack().values

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Pairwise Correlation Distribution",
                "Correlation CDF"
            ]
        )

        fig.add_trace(
            go.Histogram(
                x=upper, nbinsx=100,
                name="Pairwise Correlations",
                marker_color="#2196F3",
                opacity=0.7,
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_vline(
            x=upper.mean(),
            line_color="#FFC107", line_width=2,
            annotation_text=f"Mean={upper.mean():.3f}",
            row=1, col=1
        )
        fig.add_vline(
            x=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=1, col=1
        )

        # CDF
        sorted_corr = np.sort(upper)
        cdf         = np.arange(1, len(sorted_corr)+1) / \
                      len(sorted_corr)
        fig.add_trace(
            go.Scatter(
                x=sorted_corr, y=cdf,
                name="CDF",
                line=dict(color="#4CAF50", width=2),
                showlegend=False
            ),
            row=1, col=2
        )
        for threshold in [0.3, 0.5, 0.7]:
            pct = (upper > threshold).mean()
            fig.add_vline(
                x=threshold,
                line_dash="dash",
                line_color="#FF5722",
                opacity=0.7,
                annotation_text=f">{threshold}: "
                                 f"{pct*100:.0f}%",
                row=1, col=2
            )

        fig.update_layout(
            title=dict(
                text="<b>EDA 05 — Pairwise Correlation "
                     "Distribution</b>",
                font=dict(size=18)
            ),
            template=self.TEMPLATE,
            height=500
        )
        fig.update_xaxes(
            title_text="Correlation", row=1, col=1
        )
        fig.update_xaxes(
            title_text="Correlation", row=1, col=2
        )
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="CDF",   row=1, col=2)
        fig.show()

    # ------------------------------------------------------------------ #
    #  Chart 4 — Rolling correlation over time
    # ------------------------------------------------------------------ #
    def chart_rolling_correlation(self,
                                   rolling_df: pd.DataFrame
                                   ) -> None:
        df = rolling_df.sort_values("date")

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=[
                "Rolling Mean Pairwise Correlation (63d)",
                "% Pairs with Correlation > 0.5"
            ],
            vertical_spacing=0.1
        )

        # Mean correlation
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["mean_corr"],
                name="Mean Corr",
                line=dict(color="#2196F3", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(33,150,243,0.15)"
            ),
            row=1, col=1
        )

        # Std band
        fig.add_trace(
            go.Scatter(
                x=pd.concat([
                    df["date"],
                    df["date"].iloc[::-1]
                ]),
                y=pd.concat([
                    df["mean_corr"] + df["std_corr"],
                    (df["mean_corr"] - df["std_corr"]
                     ).iloc[::-1]
                ]),
                fill="toself",
                fillcolor="rgba(33,150,243,0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                name="±1 Std",
                showlegend=True
            ),
            row=1, col=1
        )

        # % high correlation
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["pct_high"] * 100,
                name="% corr > 0.5",
                line=dict(color="#FF5722", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(255,87,34,0.15)",
                showlegend=False
            ),
            row=2, col=1
        )

        fig.add_hline(
            y=df["mean_corr"].mean(),
            line_dash="dash",
            line_color="yellow", opacity=0.5,
            annotation_text="Long-run avg",
            row=1, col=1
        )

        fig.update_layout(
            title=dict(
                text="<b>EDA 05 — Dynamic Correlation "
                     "Structure Over Time</b>",
                font=dict(size=18)
            ),
            template=self.TEMPLATE,
            height=650,
            hovermode="x unified"
        )
        fig.update_yaxes(
            title_text="Mean Correlation", row=1, col=1
        )
        fig.update_yaxes(
            title_text="% Pairs > 0.5", row=2, col=1
        )
        fig.show()

    # ------------------------------------------------------------------ #
    #  Chart 5 — PCA scree plot + loadings
    # ------------------------------------------------------------------ #
    def chart_pca_analysis(self,
                            pca_results: dict) -> None:
        explained  = pca_results["explained"]
        cumulative = pca_results["cumulative"]
        loadings   = pca_results["loadings"]
        n_comp     = len(explained)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "PCA Scree Plot — Explained Variance",
                "PC1 vs PC2 Factor Loadings"
            ]
        )

        # Scree plot
        fig.add_trace(
            go.Bar(
                x=[f"PC{i+1}" for i in range(n_comp)],
                y=explained * 100,
                name="Individual",
                marker_color="#2196F3",
                opacity=0.8
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[f"PC{i+1}" for i in range(n_comp)],
                y=cumulative * 100,
                name="Cumulative",
                mode="lines+markers",
                line=dict(color="#FFC107", width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        fig.add_hline(
            y=80, line_dash="dash",
            line_color="green", opacity=0.7,
            annotation_text="80% threshold",
            row=1, col=1
        )

        # PC1 vs PC2 loadings scatter
        if "PC1" in loadings.columns and \
           "PC2" in loadings.columns:
            fig.add_trace(
                go.Scatter(
                    x=loadings["PC1"],
                    y=loadings["PC2"],
                    mode="markers+text",
                    text=loadings.index,
                    textposition="top center",
                    textfont=dict(size=7),
                    marker=dict(
                        color="#4CAF50",
                        size=8, opacity=0.7
                    ),
                    name="Tickers",
                    showlegend=False,
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "PC1: %{x:.3f}<br>"
                        "PC2: %{y:.3f}<extra></extra>"
                    )
                ),
                row=1, col=2
            )
            fig.add_hline(
                y=0, line_dash="dash",
                line_color="white", opacity=0.3,
                row=1, col=2
            )
            fig.add_vline(
                x=0, line_dash="dash",
                line_color="white", opacity=0.3,
                row=1, col=2
            )

        fig.update_layout(
            title=dict(
                text="<b>EDA 05 — PCA Factor Decomposition</b>",
                font=dict(size=18)
            ),
            template=self.TEMPLATE,
            height=550
        )
        fig.update_yaxes(
            title_text="Variance Explained (%)",
            row=1, col=1
        )
        fig.update_xaxes(
            title_text="PC1 Loading", row=1, col=2
        )
        fig.update_yaxes(
            title_text="PC2 Loading", row=1, col=2
        )
        fig.show()

    # ------------------------------------------------------------------ #
    #  Chart 6 — DCC dynamic correlation
    # ------------------------------------------------------------------ #
    def chart_dcc_correlation(self,
                               dcc_df: pd.DataFrame
                               ) -> None:
        df = dcc_df.sort_values("date")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["dcc_corr"],
                name="DCC Correlation",
                line=dict(color="#2196F3", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(33,150,243,0.15)"
            )
        )

        # Std band
        fig.add_trace(
            go.Scatter(
                x=pd.concat([
                    df["date"],
                    df["date"].iloc[::-1]
                ]),
                y=pd.concat([
                    df["dcc_corr"] + df["dcc_std"],
                    (df["dcc_corr"] - df["dcc_std"]
                     ).iloc[::-1]
                ]),
                fill="toself",
                fillcolor="rgba(33,150,243,0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                name="±1 Std"
            )
        )

        # Long-run average
        fig.add_hline(
            y=df["dcc_corr"].mean(),
            line_dash="dash",
            line_color="yellow", opacity=0.6,
            annotation_text=f"Long-run avg = "
                             f"{df['dcc_corr'].mean():.3f}"
        )

        fig.update_layout(
            title=dict(
                text="<b>EDA 05 — DCC Dynamic Conditional "
                     "Correlation (EWMA λ=0.94)</b>",
                font=dict(size=18)
            ),
            template=self.TEMPLATE,
            height=500,
            xaxis_title="Date",
            yaxis_title="Average Pairwise Correlation",
            hovermode="x unified"
        )
        fig.show()

    # ------------------------------------------------------------------ #
    #  Chart 7 — Sector mean correlation bar chart
    # ------------------------------------------------------------------ #
    def chart_sector_summary(self,
                              sector_results: dict) -> None:
        if len(sector_results) == 0:
            print("No sector data")
            return

        sectors    = list(sector_results.keys())
        mean_corrs = [
            sector_results[s]["mean_corr"]
            for s in sectors
        ]
        n_tickers  = [
            sector_results[s]["n_tickers"]
            for s in sectors
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=sectors,
                y=mean_corrs,
                marker_color=[
                    "#4CAF50" if c > 0.5
                    else "#FFC107" if c > 0.3
                    else "#FF5722"
                    for c in mean_corrs
                ],
                text=[f"{c:.3f}" for c in mean_corrs],
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Mean Corr: %{y:.3f}<br>"
                    "Tickers: %{customdata}<extra></extra>"
                ),
                customdata=n_tickers
            )
        )

        fig.add_hline(
            y=0.5, line_dash="dash",
            line_color="green", opacity=0.5,
            annotation_text="High correlation (0.5)"
        )

        fig.update_layout(
            title=dict(
                text="<b>EDA 05 — Mean Intra-Sector "
                     "Correlation</b>",
                font=dict(size=18)
            ),
            template=self.TEMPLATE,
            height=500,
            xaxis_title="Sector",
            yaxis_title="Mean Pairwise Correlation",
            yaxis=dict(range=[0, 1])
        )
        fig.show()

    # ------------------------------------------------------------------ #
    #  Run all charts
    # ------------------------------------------------------------------ #
    def run_all(self, returns, corr_matrix,
                sector_results, rolling_df,
                pca_results, dcc_df,
                clustering) -> None:
        print("\n" + "="*55)
        print("Generating Interactive Charts...")
        print("="*55)

        print("\n[1/7] Correlation Heatmap...")
        self.chart_correlation_heatmap(corr_matrix)

        print("[2/7] Sector Correlations...")
        self.chart_sector_correlations(sector_results)

        print("[3/7] Correlation Distribution...")
        self.chart_correlation_distribution(corr_matrix)

        print("[4/7] Rolling Correlation...")
        self.chart_rolling_correlation(rolling_df)

        print("[5/7] PCA Analysis...")
        self.chart_pca_analysis(pca_results)

        print("[6/7] DCC Correlation...")
        self.chart_dcc_correlation(dcc_df)

        print("[7/7] Sector Summary...")
        self.chart_sector_summary(sector_results)

        print("\nAll 7 charts generated ✓")

# COMMAND ----------

eda = EDACorrelationStructure(
    spark       = spark,
    silver_path = SILVER_PATH,
    eda_path    = EDA_PATH
)

(returns, corr_matrix, sector_results,
 rolling_df, pca_results, dcc_df,
 clustering) = eda.run()

charts = EDACorrelationCharts()
charts.run_all(
    returns        = returns,
    corr_matrix    = corr_matrix,
    sector_results = sector_results,
    rolling_df     = rolling_df,
    pca_results    = pca_results,
    dcc_df         = dcc_df,
    clustering     = clustering
)

print("\nEDA 05 COMPLETE ✓")

# COMMAND ----------

rolling = spark.read.format("delta").load(
    f"{EDA_PATH}/correlation_structure/rolling_correlation"
).toPandas()

pca_var = spark.read.format("delta").load(
    f"{EDA_PATH}/correlation_structure/pca_variance"
).toPandas()

dcc = spark.read.format("delta").load(
    f"{EDA_PATH}/correlation_structure/dcc_correlation"
).toPandas()

print("="*55)
print("EDA 05 — Key Findings")
print("="*55)
print(f"Rolling correlation:")
print(f"  Mean  : {rolling['mean_corr'].mean():.3f}")
print(f"  Max   : {rolling['mean_corr'].max():.3f}")
print(f"  Min   : {rolling['mean_corr'].min():.3f}")

print(f"\nPCA variance explained:")
print(pca_var[[
    "component","explained_var","cumulative_var"
]].head(10).to_string(index=False))

print(f"\nDCC correlation:")
print(f"  Mean  : {dcc['dcc_corr'].mean():.3f}")
print(f"  Max   : {dcc['dcc_corr'].max():.3f}")
print(f"  Min   : {dcc['dcc_corr'].min():.3f}")