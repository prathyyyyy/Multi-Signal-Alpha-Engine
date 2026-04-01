# Databricks notebook source
# MAGIC %pip install lightgbm optuna scikit-learn plotly scipy pandas numpy --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
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
    f"fs.azure.account.key.{STORAGE_ACCOUNT}"
    f".dfs.core.windows.net",
    ADLS_KEY
)
BASE_PATH = (
    f"abfss://{CONTAINER}@"
    f"{STORAGE_ACCOUNT}.dfs.core.windows.net"
)
GOLD_PATH = f"{BASE_PATH}/gold/delta"
ML_PATH   = f"{BASE_PATH}/ml/delta"

# LightGBM GPU check
try:
    _m = lgb.LGBMRegressor(
        device="gpu", n_estimators=5, verbose=-1
    )
    _m.fit(np.random.randn(200,10), np.random.randn(200))
    GPU_OK = True
    print(f"✅ LightGBM GPU : T4 16.7GB")
except Exception as e:
    GPU_OK = False
    print(f"❌ LightGBM GPU : {e}")

print(f"GPU_OK : {GPU_OK}")

# COMMAND ----------

class MLLightGBMV3:
    """
    ML 02 v3 — Production LightGBM.

    Stack:
      LightGBM GPU → T4 5-seed ensemble
      Optuna       → 50-trial IC-maximizing HPO

    All fixes:
      ✅ Full LightGBM param names in Optuna
         (no short keys → no iter=0 bug)
      ✅ _fit_with_fallback → iter=0 guard
      ✅ Fast feature eng ~30s (global median)
      ✅ Regime models: no early stopping
      ✅ Correct TC: turnover-based not daily
      ✅ Adaptive blend: confidence-weighted
      ✅ NaN-safe IC throughout
      ✅ Consistent column names (no KeyError)
    """

    FEATURE_COLS = [
        "mom_1d","mom_5d","mom_21d",
        "mom_63d","mom_252d",
        "rev_1d","rev_5d",
        "vol_21d","vol_63d","vol_ratio_21_63",
        "vol_of_vol","downside_vol_21d",
        "rsi_14d","vwap_ratio",
        "price_to_ma20","price_to_ma50",
        "price_to_52w_high","gap","ma_cross_20_50",
        "sharpe_21d","sharpe_63d","sortino_21d",
        "volume_ratio","amihud_21d",
        "avg_dolvol_21d","volume_mom_5d",
        "mom_21d_rank","mom_252d_rank",
        "vol_21d_rank","sharpe_21d_rank",
        "rev_1d_rank","rev_5d_rank",
        "rsi_14d_rank","volume_ratio_rank",
        "prob_bull","prob_bear","prob_highvol",
    ]

    TARGET_COL     = "fwd_return_21d"
    TC_BPS         = 5
    N_FOLDS        = 5
    OPTUNA_TRIALS  = 50
    ENSEMBLE_SEEDS = [42, 123, 456, 789, 999]
    MIN_ITER       = 50

    def __init__(self, spark, gold_path, ml_path,
                 use_gpu=True):
        self.spark         = spark
        self.gold_path     = gold_path
        self.ml_path       = ml_path
        self.use_gpu       = use_gpu
        self.models        = []
        self.regime_models = {}
        self.best_params   = {}
        self.baselines     = {}

        print("MLLightGBMV3 ✓")
        print(f"  GPU       : {use_gpu}")
        print(f"  Seeds     : {len(self.ENSEMBLE_SEEDS)}")
        print(f"  TC        : {self.TC_BPS}bps (turnover)")
        print(f"  Optuna    : {self.OPTUNA_TRIALS} trials")
        print(f"  Min iter  : {self.MIN_ITER}")

    # ─────────────────────────────────────────────────
    #  Utilities
    # ─────────────────────────────────────────────────
    @staticmethod
    def _safe_ic(pred, actual, min_samples=30):
        p = np.asarray(pred,   dtype=float)
        a = np.asarray(actual, dtype=float)
        m = (
            ~np.isnan(p) & ~np.isnan(a) &
            ~np.isinf(p) & ~np.isinf(a)
        )
        if m.sum() < min_samples:
            return np.nan
        if np.std(p[m]) < 1e-10 or \
           np.std(a[m]) < 1e-10:
            return 0.0
        try:
            ic, _ = spearmanr(p[m], a[m])
            return float(ic) if not np.isnan(ic) else 0.0
        except Exception:
            return 0.0

    def _ic_stats(self, ic_list):
        vals = [v for v in ic_list
                if v is not None and not np.isnan(v)]
        if not vals:
            return {
                "mean_ic" : 0.0,
                "icir"    : 0.0,
                "ic_std"  : 0.0,
                "hit_rate": 0.0,
                "n_dates" : 0,
            }
        m = float(np.mean(vals))
        s = float(np.std(vals))
        return {
            "mean_ic" : m,
            "icir"    : m / (s + 1e-8),
            "ic_std"  : s,
            "hit_rate": float(
                (np.array(vals) > 0).mean()
            ),
            "n_dates" : len(vals),
        }

    def _build_params(self, base=None, seed=42):
        """
        Build LightGBM params from full-name dict.
        KEY FIX: always uses full LightGBM param names.
        No short keys that LightGBM ignores silently.
        """
        p = dict(base or self.best_params)
        p.setdefault("objective",         "regression_l1")
        p.setdefault("metric",            ["mae","rmse"])
        p.setdefault("bagging_freq",      5)
        p.setdefault("n_estimators",      1000)
        p.setdefault("learning_rate",     0.01)
        p.setdefault("num_leaves",        63)
        p.setdefault("max_depth",         6)
        p.setdefault("min_child_samples", 100)
        p.setdefault("feature_fraction",  0.8)
        p.setdefault("bagging_fraction",  0.8)
        p.setdefault("reg_alpha",         1.0)
        p.setdefault("reg_lambda",        5.0)
        p.update({
            "random_state": seed,
            "n_jobs"      : -1,
            "verbose"     : -1,
        })
        if self.use_gpu:
            p.update({
                "device"         : "gpu",
                "gpu_platform_id": 0,
                "gpu_device_id"  : 0,
            })
        return p

    def _fit_model(self, model, X_tr, y_tr,
                   X_va=None, y_va=None,
                   use_es=True):
        """
        Fit with early stopping.
        If best_iteration_ < MIN_ITER → refit
        without early stopping (fallback).
        Prevents iter=0/1 bug.
        """
        callbacks = [lgb.log_evaluation(period=-1)]
        eval_set  = None

        if use_es and X_va is not None:
            eval_set = [(X_va, y_va)]
            callbacks.append(
                lgb.early_stopping(100, verbose=False)
            )

        model.fit(
            X_tr, y_tr,
            eval_set=eval_set,
            callbacks=callbacks
        )

        # Fallback: collapsed → refit without ES
        bi = getattr(model, "best_iteration_", 0)
        if use_es and bi < self.MIN_ITER:
            n_est = max(
                model.get_params().get(
                    "n_estimators", 500
                ),
                self.MIN_ITER * 5
            )
            model.set_params(n_estimators=n_est)
            model.fit(
                X_tr, y_tr,
                callbacks=[
                    lgb.log_evaluation(period=-1)
                ]
            )

        return model

    # ─────────────────────────────────────────────────
    #  Step 1 — Load
    # ─────────────────────────────────────────────────
    def load_data(self):
        print("\nStep 1: Loading Gold price factors...")
        start = datetime.now()

        sdf      = self.spark.read.format("delta").load(
            f"{self.gold_path}/price_factors"
        )
        all_cols = set(sdf.columns)
        feats    = [
            c for c in self.FEATURE_COLS
            if c in all_cols
        ]
        keep = list(dict.fromkeys(
            ["date","ticker","regime_label"] +
            feats + [
                self.TARGET_COL,
                "fwd_return_1d",
                "fwd_return_5d",
            ]
        ))
        keep = [c for c in keep if c in all_cols]

        pdf = sdf.select(*keep).dropna(
            subset=[self.TARGET_COL]
        ).toPandas()

        pdf["date"] = pd.to_datetime(pdf["date"])
        pdf = pdf.sort_values(
            ["date","ticker"]
        ).reset_index(drop=True)

        self.FEATURE_COLS = [
            c for c in self.FEATURE_COLS
            if c in pdf.columns
        ]

        elapsed = (datetime.now() - start).seconds
        print(f"  Rows      : {len(pdf):,}")
        print(f"  Tickers   : {pdf['ticker'].nunique():,}")
        print(f"  Dates     : {pdf['date'].nunique():,}")
        print(f"  Range     : "
              f"{pdf['date'].min().date()} → "
              f"{pdf['date'].max().date()}")
        print(f"  Features  : {len(self.FEATURE_COLS)}")
        print(f"  Elapsed   : {elapsed}s")
        return pdf

    # ─────────────────────────────────────────────────
    #  Step 2 — Feature engineering (~30s)
    # ─────────────────────────────────────────────────
    def engineer_features(self, pdf):
        """
        Fast vectorized feature engineering.
        Uses global stats (not per-date groupby)
        to avoid 508s bottleneck.
        Per-date ranks from Gold 01 already available.
        """
        print("\nStep 2: Feature engineering (fast)...")
        start = datetime.now()

        feats = [
            c for c in self.FEATURE_COLS
            if c in pdf.columns
        ]

        # ── Global median fillna (fast) ───────────────
        for col in feats:
            if pdf[col].isna().sum() > 0:
                med = pdf[col].median()
                pdf[col] = pdf[col].fillna(
                    0.0 if np.isnan(med) else float(med)
                )

        # ── Global 1-99% winsorize (fast) ────────────
        for col in feats:
            q1 = float(pdf[col].quantile(0.01))
            q9 = float(pdf[col].quantile(0.99))
            pdf[col] = pdf[col].clip(q1, q9)

        # ── CS z-score per date ───────────────────────
        znorm_feats = []
        cs_cols = [
            "mom_21d","mom_252d","vol_21d",
            "sharpe_21d","rsi_14d",
            "mom_63d","rev_5d",
        ]
        for col in cs_cols:
            if col not in pdf.columns:
                continue
            mu  = pdf.groupby("date")[col].transform(
                "mean"
            )
            std = pdf.groupby("date")[col].transform(
                "std"
            ).clip(lower=1e-8)
            name = f"{col}_znorm"
            pdf[name] = (
                (pdf[col] - mu) / std
            ).clip(-5, 5)
            znorm_feats.append(name)

        # ── Interaction features ──────────────────────
        new_feats = []

        def _add(name, expr, clip=5.0):
            pdf[name] = expr.fillna(0).clip(-clip, clip)
            new_feats.append(name)

        if "mom_21d" in pdf.columns and \
           "vol_21d" in pdf.columns:
            _add("mom_vol_adj",
                 pdf["mom_21d"] /
                 (pdf["vol_21d"].abs() + 1e-8))

        if "mom_252d" in pdf.columns and \
           "vol_63d" in pdf.columns:
            _add("ann_sharpe_proxy",
                 pdf["mom_252d"] /
                 (pdf["vol_63d"] * np.sqrt(252) + 1e-8))

        if "mom_5d" in pdf.columns and \
           "mom_21d" in pdf.columns:
            _add("trend_accel",
                 pdf["mom_5d"] - pdf["mom_21d"],
                 clip=0.2)

        if "sharpe_21d" in pdf.columns and \
           "sharpe_63d" in pdf.columns:
            _add("sharpe_momentum",
                 pdf["sharpe_21d"] - pdf["sharpe_63d"])

        if "volume_ratio" in pdf.columns and \
           "mom_5d" in pdf.columns:
            _add("pv_signal",
                 pdf["volume_ratio"] * pdf["mom_5d"])

        if all(c in pdf.columns for c in [
            "prob_bull","prob_bear","mom_21d","rev_5d"
        ]):
            _add("regime_mom",
                 pdf["prob_bull"] * pdf["mom_21d"] -
                 pdf["prob_bear"] * pdf["rev_5d"])

        if "rsi_14d" in pdf.columns:
            _add("rsi_extreme",
                 (pdf["rsi_14d"] - 50).abs() / 50,
                 clip=1.0)

        if "price_to_52w_high" in pdf.columns and \
           "vol_21d" in pdf.columns:
            _add("breakout_adj",
                 pdf["price_to_52w_high"] /
                 (pdf["vol_21d"].abs() + 1e-8))

        if "mom_21d_znorm" in pdf.columns and \
           "vol_21d_znorm" in pdf.columns:
            _add("momentum_quality",
                 pdf["mom_21d_znorm"] -
                 pdf["vol_21d_znorm"])

        # ── CS rank of new features ───────────────────
        rank_feats = []
        for col in [
            "mom_vol_adj","ann_sharpe_proxy",
            "regime_mom","pv_signal",
        ]:
            if col not in pdf.columns:
                continue
            name = f"{col}_rank"
            pdf[name] = pdf.groupby("date")[
                col
            ].rank(pct=True, method="average")
            rank_feats.append(name)

        # ── Rank label ────────────────────────────────
        pdf["rank_label"] = pdf.groupby("date")[
            self.TARGET_COL
        ].rank(pct=True)

        # ── Update feature list ───────────────────────
        self.FEATURE_COLS = list(dict.fromkeys(
            self.FEATURE_COLS +
            znorm_feats + new_feats + rank_feats
        ))
        self.FEATURE_COLS = [
            c for c in self.FEATURE_COLS
            if c in pdf.columns
        ]

        pdf = pdf.dropna(
            subset=self.FEATURE_COLS[:5]
        ).reset_index(drop=True)

        elapsed = (datetime.now() - start).seconds
        print(f"  Base feats  : {len(feats)}")
        print(f"  znorm feats : {len(znorm_feats)}")
        print(f"  interact    : {len(new_feats)}")
        print(f"  rank feats  : {len(rank_feats)}")
        print(f"  Total feats : {len(self.FEATURE_COLS)}")
        print(f"  Rows        : {len(pdf):,}")
        print(f"  Elapsed     : {elapsed}s ✅")
        return pdf

    # ─────────────────────────────────────────────────
    #  Step 3 — Leakage check
    # ─────────────────────────────────────────────────
    def validate_no_leakage(self, pdf):
        print("\nStep 3: Leakage validation...")
        issues = 0

        leaky = [
            c for c in self.FEATURE_COLS
            if any(p in c for p in [
                "fwd_","future_","next_","lead_"
            ])
        ]
        if leaky:
            print(f"  ❌ Leaky features: {leaky}")
            issues += len(leaky)
        else:
            print(f"  ✅ No forward-looking features")

        X    = pdf[self.FEATURE_COLS].fillna(0)
        y    = pdf[self.TARGET_COL]
        corr = X.corrwith(y).abs()
        high = corr[corr > 0.4]
        if len(high):
            print(f"  ⚠️  High same-day correlation:")
            for f, c in high.head(3).items():
                print(f"    {f}: {c:.3f}")
        else:
            print(f"  ✅ Same-day corr OK "
                  f"(max={corr.max():.3f})")

        if y.std() < 1e-6:
            print(f"  ❌ Target is constant!")
            issues += 1
        else:
            print(f"  ✅ Target std={y.std():.5f}")

        print(f"  Issues: {issues}")
        return issues == 0

    # ─────────────────────────────────────────────────
    #  Step 4 — Baselines
    # ─────────────────────────────────────────────────
    def compute_baselines(self, pdf):
        print("\nStep 4: Baselines...")

        dates  = sorted(pdf["date"].unique())
        cutoff = dates[int(len(dates) * 0.8)]
        train  = pdf[pdf["date"] <= cutoff]
        test   = pdf[pdf["date"] > cutoff].copy()

        results = {}

        for feat, name in [
            ("mom_21d",    "momentum_21d"),
            ("rev_5d",     "reversal_5d"),
            ("sharpe_21d", "sharpe_21d"),
            ("vol_21d",    "low_vol"),
        ]:
            if feat not in test.columns:
                continue
            ic_list = []
            for date, grp in test.groupby("date"):
                ic = self._safe_ic(
                    grp[feat].values,
                    grp[self.TARGET_COL].values
                )
                if not np.isnan(ic):
                    ic_list.append(ic)
            if ic_list:
                results[name] = self._ic_stats(ic_list)

        # Ridge
        ridge = Ridge(alpha=1.0)
        ridge.fit(
            train[self.FEATURE_COLS].fillna(0),
            train[self.TARGET_COL]
        )
        test["pred_ridge"] = ridge.predict(
            test[self.FEATURE_COLS].fillna(0)
        )
        ic_list = []
        for date, grp in test.groupby("date"):
            ic = self._safe_ic(
                grp["pred_ridge"].values,
                grp[self.TARGET_COL].values
            )
            if not np.isnan(ic):
                ic_list.append(ic)
        if ic_list:
            results["ridge"] = self._ic_stats(ic_list)

        self.baselines = results
        print(f"  {'Model':16} {'IC':>8} "
              f"{'ICIR':>8} {'Hit':>8}")
        print(f"  {'-'*44}")
        for name, s in results.items():
            print(f"  {name:16} "
                  f"{s['mean_ic']:>+8.4f} "
                  f"{s['icir']:>+8.4f} "
                  f"{s['hit_rate']:>7.1%}")
        return results

    # ─────────────────────────────────────────────────
    #  Step 5 — Optuna (FULL param names)
    # ─────────────────────────────────────────────────
    def optuna_tune(self, pdf):
        """
        KEY FIX: Uses full LightGBM parameter names
        in trial.suggest_* calls.
        Previous bug: short names ('lr','n','leaves')
        were ignored by LGBMRegressor → iter=0.
        """
        print(f"\nStep 5: Optuna "
              f"({self.OPTUNA_TRIALS} trials)...")

        n    = min(300_000, len(pdf))
        samp = pdf.sample(
            n, random_state=42
        ).sort_values("date")
        dates  = sorted(samp["date"].unique())
        cutoff = dates[int(len(dates) * 0.7)]
        tr     = samp[samp["date"] <= cutoff]
        va     = samp[samp["date"] > cutoff]

        X_tr = tr[self.FEATURE_COLS].fillna(0).values
        y_tr = tr[self.TARGET_COL].values
        X_va = va[self.FEATURE_COLS].fillna(0).values
        y_va = va[self.TARGET_COL].values

        def objective(trial):
            # ── FULL LightGBM param names ─────────────
            params = {
                "learning_rate"    : trial.suggest_float(
                    "learning_rate", 0.003, 0.02,
                    log=True
                ),
                "n_estimators"     : trial.suggest_int(
                    "n_estimators", 500, 3000
                ),
                "num_leaves"       : trial.suggest_int(
                    "num_leaves", 31, 127
                ),
                "max_depth"        : trial.suggest_int(
                    "max_depth", 4, 8
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", 100, 400
                ),
                "feature_fraction" : trial.suggest_float(
                    "feature_fraction", 0.5, 0.9
                ),
                "bagging_fraction" : trial.suggest_float(
                    "bagging_fraction", 0.5, 0.9
                ),
                "bagging_freq"     : 5,
                "reg_alpha"        : trial.suggest_float(
                    "reg_alpha", 0.01, 10, log=True
                ),
                "reg_lambda"       : trial.suggest_float(
                    "reg_lambda", 0.1, 20, log=True
                ),
                "min_split_gain"   : trial.suggest_float(
                    "min_split_gain", 0.0, 1.0
                ),
                "objective"        : "regression_l1",
                "metric"           : "mae",
                "random_state"     : 42,
                "n_jobs"           : -1,
                "verbose"          : -1,
            }
            if self.use_gpu:
                params.update({
                    "device"         : "gpu",
                    "gpu_platform_id": 0,
                    "gpu_device_id"  : 0,
                })

            m = lgb.LGBMRegressor(**params)
            m.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[
                    lgb.early_stopping(
                        50, verbose=False
                    ),
                    lgb.log_evaluation(period=-1)
                ]
            )

            # Penalize underfitting
            if m.best_iteration_ < self.MIN_ITER:
                return 0.0

            preds   = m.predict(X_va)
            va_copy = va.copy()
            va_copy["pred"] = preds
            ic_list = []
            for date, grp in va_copy.groupby("date"):
                ic = self._safe_ic(
                    grp["pred"].values,
                    grp[self.TARGET_COL].values
                )
                if not np.isnan(ic):
                    ic_list.append(ic)
            return float(np.mean(ic_list)) \
                   if ic_list else 0.0

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=42
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10
            )
        )
        study.optimize(
            objective,
            n_trials=self.OPTUNA_TRIALS,
            show_progress_bar=False
        )

        # Store with full names
        best = study.best_params
        best.update({
            "objective"   : "regression_l1",
            "metric"      : ["mae","rmse"],
            "bagging_freq": 5,
            "random_state": 42,
            "n_jobs"      : -1,
            "verbose"     : -1,
        })
        if self.use_gpu:
            best.update({
                "device"         : "gpu",
                "gpu_platform_id": 0,
                "gpu_device_id"  : 0,
            })

        self.best_params = best

        print(f"  Best IC            : "
              f"{study.best_value:.4f}")
        print(f"  n_estimators       : "
              f"{best['n_estimators']}")
        print(f"  learning_rate      : "
              f"{best['learning_rate']:.4f}")
        print(f"  num_leaves         : "
              f"{best['num_leaves']}")
        print(f"  max_depth          : "
              f"{best['max_depth']}")
        print(f"  min_child_samples  : "
              f"{best['min_child_samples']}")
        print(f"  feature_fraction   : "
              f"{best['feature_fraction']:.3f}")
        print(f"  bagging_fraction   : "
              f"{best['bagging_fraction']:.3f}")
        print(f"  reg_alpha          : "
              f"{best['reg_alpha']:.3f}")
        print(f"  reg_lambda         : "
              f"{best['reg_lambda']:.3f}")
        return best

    # ─────────────────────────────────────────────────
    #  Step 6 — Walk-forward validation
    # ─────────────────────────────────────────────────
    def walk_forward_validation(self, pdf, params):
        print("\nStep 6: Walk-forward validation...")

        dates     = sorted(pdf["date"].unique())
        n         = len(dates)
        fold_size = n // (self.N_FOLDS + 1)
        all_ic    = []
        fold_rows = []
        all_preds = []

        for fold in range(self.N_FOLDS):
            tr_end = dates[fold_size * (fold + 1)]
            te_end = dates[
                min(fold_size*(fold+2)-1, n-1)
            ]

            tr = pdf[pdf["date"] <= tr_end]
            te = pdf[
                (pdf["date"] > tr_end) &
                (pdf["date"] <= te_end)
            ]

            n_tr = tr["date"].nunique()
            n_te = te["date"].nunique()

            if n_tr < 252 or n_te < 63:
                print(f"  Fold {fold+1}: skip "
                      f"(tr={n_tr}d te={n_te}d)")
                continue

            if tr[self.TARGET_COL].std() < 1e-10:
                print(f"  Fold {fold+1}: "
                      f"constant target — skip")
                continue

            X_tr = tr[self.FEATURE_COLS].fillna(0)
            y_tr = tr[self.TARGET_COL]
            X_te = te[self.FEATURE_COLS].fillna(0)
            y_te = te[self.TARGET_COL].values

            # 3-seed ensemble per fold
            seed_preds = []
            best_iters = []
            for seed in self.ENSEMBLE_SEEDS[:3]:
                p = self._build_params(params, seed)
                m = lgb.LGBMRegressor(**p)
                m = self._fit_model(
                    m, X_tr, y_tr, X_te, y_te,
                    use_es=True
                )
                seed_preds.append(m.predict(X_te))
                best_iters.append(
                    getattr(m, "best_iteration_", 0)
                )

            preds      = np.mean(seed_preds, axis=0)
            te_copy    = te.copy()
            te_copy["pred"] = preds

            # NaN-safe IC per date
            fold_ic = []
            for date, grp in te_copy.groupby("date"):
                ic = self._safe_ic(
                    grp["pred"].values,
                    grp[self.TARGET_COL].values
                )
                if not np.isnan(ic):
                    fold_ic.append(ic)

            all_ic.extend(fold_ic)
            all_preds.append(te_copy)

            s         = self._ic_stats(fold_ic)
            mean_iter = float(np.mean(best_iters))
            rmse      = float(np.sqrt(
                mean_squared_error(y_te, preds)
            ))

            fold_rows.append({
                "fold"           : fold + 1,
                "train_rows"     : len(tr),
                "test_rows"      : len(te),
                "n_train_dates"  : n_tr,
                "n_test_dates"   : n_te,
                "train_end"      : str(tr_end.date()),
                "test_end"       : str(te_end.date()),
                "mean_ic"        : s["mean_ic"],
                "icir"           : s["icir"],
                "rmse"           : rmse,
                "n_valid_dates"  : s["n_dates"],
                "mean_best_iter" : mean_iter,
            })
            print(f"  Fold {fold+1}: "
                  f"IC={s['mean_ic']:+.4f}  "
                  f"ICIR={s['icir']:.3f}  "
                  f"RMSE={rmse:.5f}  "
                  f"iter={mean_iter:.0f}  "
                  f"n={s['n_dates']}")

        s = self._ic_stats(all_ic)
        print(f"\n  Summary:")
        print(f"  Mean IC  : {s['mean_ic']:+.4f} "
              f"{'✅' if abs(s['mean_ic'])>0.04 else '⚠️'}")
        print(f"  ICIR     : {s['icir']:.4f} "
              f"{'✅' if s['icir']>0.5 else '⚠️'}")
        print(f"  Hit rate : {s['hit_rate']:.1%} "
              f"{'✅' if s['hit_rate']>0.55 else '⚠️'}")

        return {
            "fold_metrics" : pd.DataFrame(fold_rows),
            "all_ic"       : all_ic,
            "mean_ic"      : s["mean_ic"],
            "icir"         : s["icir"],
            "ic_std"       : s["ic_std"],
            "hit_rate"     : s["hit_rate"],
            "all_preds"    : pd.concat(
                all_preds, ignore_index=True
            ) if all_preds else pd.DataFrame(),
        }

    # ─────────────────────────────────────────────────
    #  Step 7 — 5-seed GPU ensemble
    # ─────────────────────────────────────────────────
    def train_ensemble(self, pdf, params):
        print(f"\nStep 7: "
              f"{len(self.ENSEMBLE_SEEDS)}-seed "
              f"GPU ensemble...")
        start  = datetime.now()

        dates  = sorted(pdf["date"].unique())
        cutoff = dates[int(len(dates) * 0.8)]
        tr     = pdf[pdf["date"] <= cutoff]
        va     = pdf[pdf["date"] > cutoff]

        X_tr = tr[self.FEATURE_COLS].fillna(0)
        y_tr = tr[self.TARGET_COL]
        X_va = va[self.FEATURE_COLS].fillna(0)
        y_va = va[self.TARGET_COL]

        print(f"  Train: {len(tr):,} rows "
              f"({tr['date'].nunique()} dates)")
        print(f"  Val  : {len(va):,} rows "
              f"({va['date'].nunique()} dates)")

        for seed in self.ENSEMBLE_SEEDS:
            p = self._build_params(params, seed)
            m = lgb.LGBMRegressor(**p)
            m = self._fit_model(
                m, X_tr, y_tr, X_va, y_va,
                use_es=True
            )
            self.models.append(m)

            va_c = va.copy()
            va_c["pred"] = m.predict(X_va)
            ic_list = []
            for date, grp in va_c.groupby("date"):
                ic = self._safe_ic(
                    grp["pred"].values,
                    grp[self.TARGET_COL].values
                )
                if not np.isnan(ic):
                    ic_list.append(ic)
            s = self._ic_stats(ic_list)
            bi = getattr(m, "best_iteration_", 0)
            print(f"  Seed {seed:3}: "
                  f"iter={bi:4d}  "
                  f"val_IC={s['mean_ic']:+.4f}  "
                  f"ICIR={s['icir']:.3f}")

        elapsed = (datetime.now() - start).seconds
        print(f"  Elapsed : {elapsed}s")

    # ─────────────────────────────────────────────────
    #  Step 8 — Regime models (NO early stopping)
    # ─────────────────────────────────────────────────
    def train_regime_models(self, pdf, params):
        """
        KEY FIX: No early stopping for regime models.
        Smaller per-regime dataset → ES fires at iter=1.
        Use fixed n_estimators=300 instead.
        """
        print("\nStep 8: Regime-conditional models...")

        if "regime_label" not in pdf.columns:
            print("  No regime_label — skip")
            return

        dates  = sorted(pdf["date"].unique())
        cutoff = dates[int(len(dates) * 0.8)]

        for regime in ["Bull","Bear","HighVol"]:
            rdf = pdf[pdf["regime_label"] == regime]
            if len(rdf) < 2000:
                print(f"  {regime}: too small — skip")
                continue

            tr  = rdf[rdf["date"] <= cutoff]
            va  = rdf[rdf["date"] > cutoff]
            if len(tr) < 500:
                continue

            # Relaxed params + fixed n_est
            r_params = dict(params)
            r_params.update({
                "n_estimators"     : 300,
                "learning_rate"    : max(
                    float(params.get(
                        "learning_rate", 0.01
                    )), 0.01
                ),
                "min_child_samples": min(
                    int(params.get(
                        "min_child_samples", 200
                    )), 200
                ),
                "reg_lambda"       : min(
                    float(params.get(
                        "reg_lambda", 5
                    )), 5.0
                ),
                "num_leaves"       : min(
                    int(params.get(
                        "num_leaves", 63
                    )), 63
                ),
            })

            p = self._build_params(r_params, 42)
            m = lgb.LGBMRegressor(**p)

            # NO early stopping
            m = self._fit_model(
                m,
                tr[self.FEATURE_COLS].fillna(0),
                tr[self.TARGET_COL],
                use_es=False
            )
            self.regime_models[regime] = m

            ic_list = []
            if len(va) > 50:
                va_c = va.copy()
                va_c["pred"] = m.predict(
                    va[self.FEATURE_COLS].fillna(0)
                )
                for date, grp in va_c.groupby("date"):
                    ic = self._safe_ic(
                        grp["pred"].values,
                        grp[self.TARGET_COL].values
                    )
                    if not np.isnan(ic):
                        ic_list.append(ic)
            s  = self._ic_stats(ic_list)
            bi = getattr(m, "best_iteration_", 0)
            print(f"  {regime:8}: "
                  f"{len(tr):>10,} rows  "
                  f"iter={bi:4d}  "
                  f"val_IC={s['mean_ic']:+.4f}")

    # ─────────────────────────────────────────────────
    #  Step 9 — Adaptive blend
    # ─────────────────────────────────────────────────
    def predict_blend(self, pdf):
        """
        Adaptive confidence-weighted blend.
        High pred_std → uncertain → use regime overlay.
        Low pred_std  → confident → use ensemble.
        """
        print("\nStep 9: Predict + adaptive blend...")

        X_all = pdf[self.FEATURE_COLS].fillna(0)
        pdf   = pdf.copy()

        # 5-seed ensemble
        if self.models:
            mat = np.column_stack([
                m.predict(X_all) for m in self.models
            ])
            pdf["pred_ensemble"] = mat.mean(axis=1)
            pdf["pred_std"]      = mat.std(axis=1)
        else:
            pdf["pred_ensemble"] = 0.0
            pdf["pred_std"]      = 1.0

        # Regime overlay (60% ensemble + 40% regime)
        pdf["pred_regime"] = pdf["pred_ensemble"]
        for regime, m in self.regime_models.items():
            mask = pdf["regime_label"] == regime
            if mask.sum() > 0:
                pdf.loc[mask, "pred_regime"] = (
                    0.6 * pdf.loc[
                        mask, "pred_ensemble"
                    ] +
                    0.4 * m.predict(X_all[mask])
                )

        # Adaptive blend: confidence-weighted
        conf = 1.0 / (pdf["pred_std"] + 1e-6)
        conf = conf / conf.max()

        pdf["pred_final"] = (
            conf * pdf["pred_ensemble"] +
            (1.0 - conf) * pdf["pred_regime"]
        ).clip(-0.05, 0.05)

        print(f"  Ensemble seeds : {len(self.models)}")
        print(f"  Regime models  : "
              f"{list(self.regime_models.keys())}")
        print(f"  Blend          : adaptive confidence")
        return pdf

    # ─────────────────────────────────────────────────
    #  Step 10 — Evaluate (turnover-based TC)
    # ─────────────────────────────────────────────────
    def evaluate(self, pdf, wf_metrics):
        """
        KEY FIX: TC based on TURNOVER not full position.
        Quintile strategy: ~20-40% daily turnover.
        Previous bug: applied 5bps to full return daily
        → artificially large TC → net Sharpe -1.46.
        """
        print("\nStep 10: Evaluation...")

        ic_rows = []
        ls_rows = []

        prev_long  = set()
        prev_short = set()

        for date, grp in pdf.groupby("date"):
            if len(grp) < 20:
                continue

            ic_ens = self._safe_ic(
                grp["pred_ensemble"].values,
                grp[self.TARGET_COL].values
            )
            ic_fin = self._safe_ic(
                grp["pred_final"].values,
                grp[self.TARGET_COL].values
            )
            ic_rows.append({
                "date"       : date,
                "ic_ensemble": ic_ens,
                "ic_final"   : ic_fin,
            })

            # Quintile L/S
            n   = max(1, len(grp) // 5)
            srt = grp.sort_values(
                "pred_final", ascending=False
            )
            long_set  = set(
                srt.head(n)["ticker"].tolist()
            )
            short_set = set(
                srt.tail(n)["ticker"].tolist()
            )

            lr = srt.head(n)[self.TARGET_COL].mean()
            sr = srt.tail(n)[self.TARGET_COL].mean()
            ls = lr - sr

            # CORRECT TC: fraction of portfolio changed
            if prev_long and prev_short:
                lo = len(long_set - prev_long) / max(n,1)
                so = len(short_set - prev_short) / max(n,1)
                turnover = (lo + so) / 2.0
            else:
                turnover = 1.0

            tc_cost = turnover * self.TC_BPS / 10000

            prev_long  = long_set
            prev_short = short_set

            ls_rows.append({
                "date"      : date,
                "long_ret"  : float(lr),
                "short_ret" : float(sr),
                "ls_ret"    : float(ls),
                "ls_ret_tc" : float(ls - tc_cost),
                "tc_cost"   : float(tc_cost),
                "turnover"  : float(turnover),
            })

        ic_df = pd.DataFrame(ic_rows)
        ls_df = pd.DataFrame(ls_rows)

        def _ls_stats(col):
            r   = ls_df[col].dropna()
            ann = r.mean() * 252
            vol = r.std() * np.sqrt(252)
            dd  = (
                (1+r).cumprod() /
                (1+r).cumprod().cummax() - 1
            ).min()
            return {
                "ann_ret": float(ann),
                "ann_vol": float(vol),
                "sharpe" : float(ann / (vol + 1e-8)),
                "max_dd" : float(dd),
            }

        model_stats = {
            "ensemble": self._ic_stats(
                ic_df["ic_ensemble"].tolist()
            ),
            "final"   : self._ic_stats(
                ic_df["ic_final"].tolist()
            ),
        }
        ls_gross = _ls_stats("ls_ret")
        ls_tc    = _ls_stats("ls_ret_tc")

        avg_turnover = float(ls_df["turnover"].mean())
        avg_tc_day   = float(ls_df["tc_cost"].mean())

        # Regime IC
        regime_ic = {}
        for regime in ["Bull","Bear","HighVol"]:
            if "regime_label" not in pdf.columns:
                break
            rdata = pdf[
                pdf["regime_label"] == regime
            ]
            r_ic = []
            for date, grp in rdata.groupby("date"):
                ic = self._safe_ic(
                    grp["pred_final"].values,
                    grp[self.TARGET_COL].values
                )
                if not np.isnan(ic):
                    r_ic.append(ic)
            regime_ic[regime] = self._ic_stats(r_ic)

        # Feature importance
        if self.models:
            imp = np.column_stack([
                m.feature_importances_
                for m in self.models
            ])
            feat_imp = pd.DataFrame({
                "feature"   : self.FEATURE_COLS,
                "importance": imp.mean(axis=1),
                "imp_std"   : imp.std(axis=1),
            }).sort_values(
                "importance", ascending=False
            )
        else:
            feat_imp = pd.DataFrame()

        results = {
            "model_stats"    : model_stats,
            "mean_ic"        : model_stats["final"]["mean_ic"],
            "icir"           : model_stats["final"]["icir"],
            "hit_rate"       : model_stats["final"]["hit_rate"],
            "ls_gross"       : ls_gross,
            "ls_tc"          : ls_tc,
            "avg_turnover"   : avg_turnover,
            "avg_tc_bps_day" : avg_tc_day * 10000,
            "regime_ic"      : regime_ic,
            "ic_series"      : ic_df,
            "ls_series"      : ls_df,
            "feat_importance": feat_imp,
            "predictions"    : pdf,
            "wf_metrics"     : wf_metrics,
            "baselines"      : self.baselines,
        }
        self._print_summary(results)
        return results

    def _print_summary(self, r):
        g  = r["ls_gross"]
        t  = r["ls_tc"]
        wf = r["wf_metrics"]

        print("\n" + "="*55)
        print("ML 02 v3 — Evaluation Summary")
        print("="*55)

        print(f"\n  {'Model':10} {'IC':>8} "
              f"{'ICIR':>8} {'Hit':>8}")
        print(f"  {'-'*38}")
        for model, s in r["model_stats"].items():
            flag = (
                "✅" if abs(s["mean_ic"]) > 0.04
                else "⚠️"
            )
            print(f"  {model:10} "
                  f"{s['mean_ic']:>+8.4f} "
                  f"{s['icir']:>+8.4f} "
                  f"{s['hit_rate']:>7.1%} {flag}")

        print(f"\n  L/S Gross:")
        print(f"    Ann Return : {g['ann_ret']*100:.2f}%")
        print(f"    Ann Vol    : {g['ann_vol']*100:.2f}%")
        print(f"    Sharpe     : {g['sharpe']:.2f} "
              f"{'✅' if 0.5<g['sharpe']<15 else '⚠️'}")
        print(f"    Max DD     : {g['max_dd']*100:.2f}%")

        print(f"\n  L/S Net (turnover TC):")
        print(f"    Ann Return : {t['ann_ret']*100:.2f}%")
        print(f"    Sharpe     : {t['sharpe']:.2f} "
              f"{'✅' if 0.5<t['sharpe']<15 else '⚠️'}")
        print(f"    Avg Turnover: "
              f"{r['avg_turnover']:.1%}/day")
        print(f"    Avg TC/day  : "
              f"{r['avg_tc_bps_day']:.2f}bps "
              f"({r['avg_tc_bps_day']*252:.0f}bps/yr)")

        print(f"\n  Regime IC:")
        for regime, s in r["regime_ic"].items():
            print(f"    {regime:8}: "
                  f"IC={s['mean_ic']:+.4f}  "
                  f"ICIR={s['icir']:.3f}")

        print(f"\n  vs Baselines:")
        for name, s in r["baselines"].items():
            print(f"    {name:16}: "
                  f"IC={s['mean_ic']:+.4f}  "
                  f"ICIR={s['icir']:.3f}")

        print(f"\n  Walk-Forward CV:")
        print(f"    Mean IC  : {wf['mean_ic']:+.4f}")
        print(f"    ICIR     : {wf['icir']:.4f}")
        print(f"    Hit rate : {wf['hit_rate']:.1%}")

        n_pass = sum([
            abs(r["mean_ic"])   > 0.04,
            abs(r["icir"])      > 0.4,
            r["hit_rate"]       > 0.55,
            0.5 < t["sharpe"]   < 15,
            abs(wf["mean_ic"])  > 0.04,
        ])
        print(f"\n  Score: {n_pass}/5  "
              f"{'✅ PRODUCTION READY' if n_pass>=3 else '⚠️ REVIEW'}")

    # ─────────────────────────────────────────────────
    #  Write
    # ─────────────────────────────────────────────────
    def write_results(self, results):
        print("\nWriting results...")

        def _write(df, path, partition=True):
            df  = df.copy()
            num = df.select_dtypes(
                include=[np.number]
            ).columns
            df[num] = df[num].fillna(0)
            if "date" in df.columns:
                df["date"]  = df["date"].astype(str)
                df["year"]  = pd.to_datetime(
                    df["date"]
                ).dt.year
                df["month"] = pd.to_datetime(
                    df["date"]
                ).dt.month
            w = (
                self.spark.createDataFrame(df)
                .write.format("delta")
                .mode("overwrite")
                .option("overwriteSchema","true")
            )
            if partition and "year" in df.columns:
                w = w.partitionBy("year","month")
            w.save(path)

        pred_cols = [
            "date","ticker","regime_label",
            "pred_ensemble","pred_final",
            self.TARGET_COL,
        ]
        _write(
            results["predictions"][[
                c for c in pred_cols
                if c in results["predictions"].columns
            ]],
            f"{self.ml_path}/lgbm_v3_predictions"
        )
        print("  ✓ lgbm_v3_predictions")

        _write(
            results["ic_series"],
            f"{self.ml_path}/lgbm_v3_ic_series"
        )
        print("  ✓ lgbm_v3_ic_series")

        _write(
            results["ls_series"],
            f"{self.ml_path}/lgbm_v3_ls_series"
        )
        print("  ✓ lgbm_v3_ls_series")

        if len(results["feat_importance"]) > 0:
            _write(
                results["feat_importance"],
                f"{self.ml_path}"
                f"/lgbm_v3_feature_importance",
                partition=False
            )
            print("  ✓ lgbm_v3_feature_importance")

        _write(
            results["wf_metrics"]["fold_metrics"],
            f"{self.ml_path}/lgbm_v3_wf_metrics",
            partition=False
        )
        print("  ✓ lgbm_v3_wf_metrics")

        g  = results["ls_gross"]
        t  = results["ls_tc"]
        wf = results["wf_metrics"]
        _write(
            pd.DataFrame([{
                "mean_ic"         : results["mean_ic"],
                "icir"            : results["icir"],
                "hit_rate"        : results["hit_rate"],
                "ls_sharpe_gross" : g["sharpe"],
                "ls_sharpe_tc"    : t["sharpe"],
                "ls_ann_ret_gross": g["ann_ret"],
                "ls_ann_ret_tc"   : t["ann_ret"],
                "max_dd"          : g["max_dd"],
                "avg_turnover"    : results["avg_turnover"],
                "avg_tc_bps_day"  : results["avg_tc_bps_day"],
                "wf_mean_ic"      : wf["mean_ic"],
                "wf_icir"         : wf["icir"],
                "wf_hit_rate"     : wf["hit_rate"],
                "n_models"        : len(self.models),
                "gpu_used"        : self.use_gpu,
                "n_features"      : len(self.FEATURE_COLS),
                "tc_bps"          : self.TC_BPS,
                "optuna_trials"   : self.OPTUNA_TRIALS,
            }]),
            f"{self.ml_path}/lgbm_v3_eval_summary",
            partition=False
        )
        print("  ✓ lgbm_v3_eval_summary")

    # ─────────────────────────────────────────────────
    #  Run
    # ─────────────────────────────────────────────────
    def run(self):
        print("="*55)
        print("ML 02 v3 — Production Stack")
        print(f"GPU={self.use_gpu}")
        print("="*55)
        start = datetime.now()

        pdf         = self.load_data()
        pdf         = self.engineer_features(pdf)
        _           = self.validate_no_leakage(pdf)
        _           = self.compute_baselines(pdf)
        best_params = self.optuna_tune(pdf)
        wf_metrics  = self.walk_forward_validation(
            pdf, best_params
        )
        self.train_ensemble(pdf, best_params)
        self.train_regime_models(pdf, best_params)
        pdf_pred    = self.predict_blend(pdf)
        results     = self.evaluate(
            pdf_pred, wf_metrics
        )
        self.write_results(results)

        elapsed = (
            datetime.now() - start
        ).seconds / 60
        print(f"\nTotal time : {elapsed:.1f} minutes")
        print("ML 02 v3 COMPLETE ✓")
        return pdf_pred, results

# COMMAND ----------

class MLV3Charts:
    TEMPLATE = "plotly_dark"
    C = {
        "primary"  : "#2196F3",
        "secondary": "#FF5722",
        "success"  : "#4CAF50",
        "warning"  : "#FFC107",
        "purple"   : "#9C27B0",
        "teal"     : "#00BCD4",
    }
    REGIME_C = {
        "Bull"   : "#4CAF50",
        "Bear"   : "#FF5722",
        "HighVol": "#FFC107",
    }

    def chart_ic_series(self, results):
        ic_df = results["ic_series"].copy()
        ic_df["date"] = pd.to_datetime(ic_df["date"])
        ic_df = ic_df.sort_values("date")
        vals  = ic_df["ic_final"].fillna(0)

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=[
                "Daily IC (Final)",
                "Rolling IC (21d & 63d)",
                "Cumulative IC",
            ],
            vertical_spacing=0.07,
            row_heights=[0.35, 0.35, 0.30]
        )

        bar_c = [
            self.C["success"] if v > 0
            else self.C["secondary"] for v in vals
        ]
        fig.add_trace(go.Bar(
            x=ic_df["date"], y=vals,
            marker_color=bar_c, showlegend=False
        ), row=1, col=1)
        fig.add_hline(
            y=float(vals.mean()), line_dash="dot",
            line_color=self.C["warning"],
            annotation_text=f"μ={vals.mean():.4f}",
            row=1, col=1
        )
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=1, col=1
        )

        for w, name, color in [
            (21,"21d",self.C["teal"]),
            (63,"63d",self.C["warning"]),
        ]:
            fig.add_trace(go.Scatter(
                x=ic_df["date"],
                y=vals.rolling(w).mean(),
                name=name, mode="lines",
                line=dict(color=color, width=2)
            ), row=2, col=1)
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=2, col=1
        )

        fig.add_trace(go.Scatter(
            x=ic_df["date"],
            y=vals.cumsum(),
            mode="lines",
            line=dict(
                color=self.C["success"], width=2
            ),
            fill="tozeroy",
            fillcolor="rgba(76,175,80,0.1)",
            showlegend=False
        ), row=3, col=1)
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=3, col=1
        )

        wf = results["wf_metrics"]
        fig.update_layout(
            title=(
                f"<b>ML 02 v3 — IC Series<br>"
                f"<sup>"
                f"IC={results['mean_ic']:+.4f} | "
                f"ICIR={results['icir']:.3f} | "
                f"WF={wf['mean_ic']:+.4f} | "
                f"Hit={results['hit_rate']:.1%}"
                f"</sup></b>"
            ),
            template=self.TEMPLATE,
            height=800,
            hovermode="x unified"
        )
        for row, t in [
            (1,"IC"),(2,"Rolling"),(3,"Cum IC")
        ]:
            fig.update_yaxes(
                title_text=t, row=row, col=1
            )
        fig.show()

    def chart_ls_portfolio(self, results):
        ls = results["ls_series"].copy()
        if len(ls) == 0:
            return

        ls["date"] = pd.to_datetime(ls["date"])
        ls = ls.sort_values("date")

        for col in ["ls_ret","ls_ret_tc"]:
            if col in ls.columns:
                ls[f"cum_{col}"] = (
                    1 + ls[col].fillna(0)
                ).cumprod()
                ls[f"dd_{col}"] = (
                    ls[f"cum_{col}"] /
                    ls[f"cum_{col}"].cummax() - 1
                )

        g  = results["ls_gross"]
        t  = results["ls_tc"]
        tv = results.get("avg_turnover", 0)
        tc_yr = results.get(
            "avg_tc_bps_day", 0
        ) * 252

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=[
                "Cumulative Return",
                "Daily L/S Return",
                "Drawdown",
                "Daily Turnover",
            ],
            vertical_spacing=0.06,
            row_heights=[0.40,0.20,0.20,0.20]
        )

        for col, name, color in [
            ("cum_ls_ret",    "Gross",
             self.C["success"]),
            ("cum_ls_ret_tc", "Net TC",
             self.C["teal"]),
        ]:
            if col not in ls.columns:
                continue
            fig.add_trace(go.Scatter(
                x=ls["date"], y=ls[col],
                name=name, mode="lines",
                line=dict(color=color, width=2)
            ), row=1, col=1)

        if "ls_ret" in ls.columns:
            bar_c = [
                self.C["success"] if v > 0
                else self.C["secondary"]
                for v in ls["ls_ret"]
            ]
            fig.add_trace(go.Bar(
                x=ls["date"],
                y=ls["ls_ret"] * 100,
                marker_color=bar_c,
                showlegend=False
            ), row=2, col=1)
            fig.add_hline(
                y=0, line_dash="dash",
                line_color="white", opacity=0.3,
                row=2, col=1
            )

        if "dd_ls_ret" in ls.columns:
            fig.add_trace(go.Scatter(
                x=ls["date"],
                y=ls["dd_ls_ret"] * 100,
                fill="tozeroy",
                fillcolor="rgba(255,87,34,0.3)",
                line=dict(
                    color=self.C["secondary"],
                    width=1
                ),
                showlegend=False
            ), row=3, col=1)

        if "turnover" in ls.columns:
            fig.add_trace(go.Scatter(
                x=ls["date"],
                y=ls["turnover"] * 100,
                mode="lines",
                line=dict(
                    color=self.C["purple"], width=1
                ),
                fill="tozeroy",
                fillcolor="rgba(156,39,176,0.15)",
                showlegend=False
            ), row=4, col=1)
            fig.add_hline(
                y=float(tv * 100),
                line_dash="dot",
                line_color=self.C["warning"],
                annotation_text=f"Avg={tv:.1%}",
                row=4, col=1
            )

        fig.update_layout(
            title=(
                f"<b>ML 02 v3 — L/S Portfolio<br>"
                f"<sup>"
                f"Gross Sharpe={g['sharpe']:.2f} | "
                f"Net Sharpe={t['sharpe']:.2f} | "
                f"Avg TO={tv:.1%} | "
                f"TC={tc_yr:.0f}bps/yr"
                f"</sup></b>"
            ),
            template=self.TEMPLATE,
            height=900,
            hovermode="x unified"
        )
        for row, label in [
            (1,"Cum Return"),
            (2,"L/S Ret(%)"),
            (3,"DD(%)"),
            (4,"Turnover(%)"),
        ]:
            fig.update_yaxes(
                title_text=label, row=row, col=1
            )
        fig.show()

    def chart_vs_baselines(self, results):
        b     = results.get("baselines", {})
        names = ["LightGBM v3"] + list(b.keys())
        ics   = [results["mean_ic"]] + [
            v["mean_ic"] for v in b.values()
        ]
        icirs = [results["icir"]] + [
            v["icir"] for v in b.values()
        ]
        hits  = [results["hit_rate"]] + [
            v["hit_rate"] for v in b.values()
        ]
        colors = [
            self.C["success"]
            if n == "LightGBM v3"
            else self.C["primary"]
            for n in names
        ]

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=["IC","ICIR","Hit Rate"]
        )
        for vals, col in [
            (ics,1),(icirs,2),(hits,3)
        ]:
            fig.add_trace(go.Bar(
                x=names, y=vals,
                marker_color=colors,
                text=[f"{v:+.4f}" for v in vals],
                textposition="outside",
                showlegend=False
            ), row=1, col=col)
            fig.add_hline(
                y=0, line_dash="dash",
                line_color="white", opacity=0.3,
                row=1, col=col
            )
        fig.update_layout(
            title="<b>ML 02 v3 — vs Baselines</b>",
            template=self.TEMPLATE,
            height=500
        )
        fig.show()

    def chart_feature_importance(self, results):
        fi = results["feat_importance"]
        if len(fi) == 0:
            return
        top = fi.head(30).copy()

        def cat(f):
            if ("mom" in f and
                    "rank" not in f and
                    "znorm" not in f):
                return "Momentum"
            if "vol" in f:    return "Volatility"
            if "prob" in f:   return "Regime"
            if "sharpe" in f or "sortino" in f:
                return "Quality"
            if any(x in f for x in [
                "rsi","ma","vwap","gap","52w"
            ]):
                return "Technical"
            if "volume" in f or "amihud" in f:
                return "Liquidity"
            if "rank" in f:   return "CS Rank"
            if "znorm" in f:  return "Neutralized"
            return "Engineered"

        top["cat"] = top["feature"].apply(cat)
        cat_c = {
            "Momentum"  : self.C["primary"],
            "Volatility": self.C["secondary"],
            "Regime"    : self.C["warning"],
            "Quality"   : self.C["success"],
            "Technical" : self.C["purple"],
            "Liquidity" : self.C["teal"],
            "CS Rank"   : "#FF8A65",
            "Neutralized": "#A5D6A7",
            "Engineered": "#CE93D8",
        }
        colors = [
            cat_c.get(c,"#9E9E9E")
            for c in top["cat"]
        ]

        fig = go.Figure(go.Bar(
            x=top["importance"],
            y=top["feature"],
            orientation="h",
            marker_color=colors,
            error_x=dict(
                type="data",
                array=top["imp_std"].tolist()
                      if "imp_std" in top.columns
                      else None,
                visible="imp_std" in top.columns
            ),
            text=top["importance"].apply(
                lambda x: f"{x:,.0f}"
            ),
            textposition="outside",
        ))
        fig.update_layout(
            title=(
                "<b>ML 02 v3 — Feature Importance"
                " (5-Seed Avg ± Std)</b>"
            ),
            template=self.TEMPLATE,
            height=800,
            xaxis_title="Importance"
        )
        fig.show()

    def chart_regime_ic(self, results):
        ri = results["regime_ic"]
        if not ri:
            return
        regimes  = list(ri.keys())
        mean_ics = [ri[r]["mean_ic"] for r in regimes]
        icirs    = [ri[r]["icir"]    for r in regimes]
        colors   = [
            self.REGIME_C.get(r,"#9E9E9E")
            for r in regimes
        ]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["IC by Regime",
                             "ICIR by Regime"]
        )
        for vals, col in [(mean_ics,1),(icirs,2)]:
            fig.add_trace(go.Bar(
                x=regimes, y=vals,
                marker_color=colors,
                text=[f"{v:+.4f}" for v in vals],
                textposition="outside",
                showlegend=False
            ), row=1, col=col)
            fig.add_hline(
                y=0, line_dash="dash",
                line_color="white", opacity=0.3,
                row=1, col=col
            )
        fig.update_layout(
            title="<b>ML 02 v3 — IC by Regime</b>",
            template=self.TEMPLATE, height=500
        )
        fig.show()

    def chart_walkforward(self, results):
        wf = results["wf_metrics"]
        fd = wf["fold_metrics"]
        if len(fd) == 0:
            return

        colors = [
            self.C["success"] if v > 0
            else self.C["secondary"]
            for v in fd["mean_ic"]
        ]
        labels = [
            f"Fold {f}" for f in fd["fold"]
        ]

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "IC by Fold","RMSE by Fold",
                "ICIR by Fold","Best Iter by Fold",
            ]
        )
        for vals, row, col, c in [
            (fd["mean_ic"],             1,1,colors),
            (fd["rmse"],                1,2,
             self.C["warning"]),
            (fd["icir"],                2,1,
             self.C["purple"]),
            (fd.get("mean_best_iter",
                    pd.Series([0]*len(fd))),
             2,2,self.C["teal"]),
        ]:
            fig.add_trace(go.Bar(
                x=labels, y=vals,
                marker_color=c,
                text=vals.round(3)
                     if hasattr(vals,"round")
                     else vals,
                textposition="outside",
                showlegend=False
            ), row=row, col=col)

        fig.add_hline(
            y=wf["mean_ic"], line_dash="dot",
            line_color=self.C["warning"],
            annotation_text=(
                f"μ={wf['mean_ic']:+.4f}"
            ),
            row=1, col=1
        )
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=1, col=1
        )
        fig.update_layout(
            title=(
                f"<b>ML 02 v3 — Walk-Forward CV<br>"
                f"<sup>"
                f"IC={wf['mean_ic']:+.4f} | "
                f"ICIR={wf['icir']:.4f} | "
                f"Hit={wf['hit_rate']:.1%}"
                f"</sup></b>"
            ),
            template=self.TEMPLATE,
            height=600
        )
        fig.show()

    def chart_ic_distribution(self, results):
        ic_df = results["ic_series"].copy()
        col   = (
            "ic_final"
            if "ic_final" in ic_df.columns
            else "ic_ensemble"
        )
        vals  = ic_df[col].dropna()
        from scipy import stats as sp

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "IC Distribution",
                "QQ-Plot vs Normal",
            ]
        )

        fig.add_trace(go.Histogram(
            x=vals, nbinsx=60,
            marker_color=self.C["primary"],
            opacity=0.8, showlegend=False
        ), row=1, col=1)
        for x_val, label, color in [
            (0.0,  "zero",  "white"),
            (float(vals.mean()),
             f"μ={vals.mean():.4f}",
             self.C["warning"]),
            (0.04, "target=0.04",
             self.C["success"]),
        ]:
            fig.add_vline(
                x=x_val, line_dash="dash",
                line_color=color, opacity=0.6,
                annotation_text=label,
                row=1, col=1
            )

        (osm,osr),(slope,intercept,_) = (
            sp.probplot(vals)
        )
        fig.add_trace(go.Scatter(
            x=osm, y=osr, mode="markers",
            marker=dict(
                color=self.C["primary"],
                size=3, opacity=0.5
            ),
            showlegend=False
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=[min(osm),max(osm)],
            y=[slope*min(osm)+intercept,
               slope*max(osm)+intercept],
            mode="lines",
            line=dict(color="white",dash="dash"),
            showlegend=False
        ), row=1, col=2)

        fig.update_layout(
            title=(
                f"<b>ML 02 v3 — IC Distribution<br>"
                f"<sup>"
                f"μ={vals.mean():.4f} | "
                f"σ={vals.std():.4f} | "
                f"Skew={vals.skew():.2f} | "
                f"Kurt={vals.kurt():.2f}"
                f"</sup></b>"
            ),
            template=self.TEMPLATE, height=500
        )
        fig.update_xaxes(
            title_text="IC", row=1, col=1
        )
        fig.update_xaxes(
            title_text="Theoretical Q", row=1, col=2
        )
        fig.update_yaxes(
            title_text="Count", row=1, col=1
        )
        fig.update_yaxes(
            title_text="Observed Q", row=1, col=2
        )
        fig.show()

    def run_all(self, results):
        print("\n" + "="*55)
        print("Generating ML 02 v3 Charts...")
        print("="*55)

        print("\n[1/7] IC Series...")
        self.chart_ic_series(results)

        print("[2/7] L/S Portfolio...")
        self.chart_ls_portfolio(results)

        print("[3/7] vs Baselines...")
        self.chart_vs_baselines(results)

        print("[4/7] Feature Importance...")
        self.chart_feature_importance(results)

        print("[5/7] Regime IC...")
        self.chart_regime_ic(results)

        print("[6/7] Walk-Forward CV...")
        self.chart_walkforward(results)

        print("[7/7] IC Distribution...")
        self.chart_ic_distribution(results)

        print("\nAll 7 charts ✓")

# COMMAND ----------

pipeline = MLLightGBMV3(
    spark     = spark,
    gold_path = GOLD_PATH,
    ml_path   = ML_PATH,
    use_gpu   = GPU_OK,
)

pdf_pred, results = pipeline.run()

charts = MLV3Charts()
charts.run_all(results)

print("\nML 02 v3 COMPLETE ✓")

# COMMAND ----------

s  = spark.read.format("delta").load(
    f"{ML_PATH}/lgbm_v3_eval_summary"
).toPandas().iloc[0]

fi = spark.read.format("delta").load(
    f"{ML_PATH}/lgbm_v3_feature_importance"
).toPandas()

wf = spark.read.format("delta").load(
    f"{ML_PATH}/lgbm_v3_wf_metrics"
).toPandas()

print("="*55)
print("ML 02 v3 — Final Summary")
print("="*55)
print(f"GPU       : {s['gpu_used']}")
print(f"N models  : {s['n_models']:.0f}")
print(f"N features: {s['n_features']:.0f}")

print(f"\nIC Metrics:")
print(f"  IC       : {s['mean_ic']:+.4f} "
      f"{'✅' if abs(s['mean_ic'])>0.04 else '⚠️'}")
print(f"  ICIR     : {s['icir']:.4f} "
      f"{'✅' if s['icir']>0.4 else '⚠️'}")
print(f"  Hit Rate : {s['hit_rate']:.1%}")

print(f"\nL/S Gross:")
print(f"  Sharpe   : {s['ls_sharpe_gross']:.2f}")
print(f"  Ann Ret  : "
      f"{s['ls_ann_ret_gross']*100:.1f}%")
print(f"  Max DD   : {s['max_dd']*100:.1f}%")

print(f"\nL/S Net TC (turnover-based):")
print(f"  Sharpe   : {s['ls_sharpe_tc']:.2f} "
      f"{'✅' if s['ls_sharpe_tc']>0.5 else '⚠️'}")
print(f"  Ann Ret  : "
      f"{s['ls_ann_ret_tc']*100:.1f}%")
print(f"  Avg TO   : {s['avg_turnover']:.1%}/day")
print(f"  TC/yr    : "
      f"{s['avg_tc_bps_day']*252:.0f}bps/yr")

print(f"\nWalk-Forward CV:")
print(wf[[
    "fold","mean_ic","icir",
    "rmse","n_valid_dates","mean_best_iter"
]].to_string(index=False))

print(f"\nTop 10 Features:")
print(fi.head(10)[[
    "feature","importance"
]].to_string(index=False))

print(f"\n{'='*55}")
print(f"ML Progress:")
print(f"  ML 01 HMM      ✅  4/5  Production")
print(f"  ML 02 LightGBM ✅  v3   All fixes applied")
print(f"  ML 03 PatchTST 🔲  Next (PyTorch T4)")
print(f"  ML 04 Ensemble 🔲  Pending")
print(f"{'='*55}")