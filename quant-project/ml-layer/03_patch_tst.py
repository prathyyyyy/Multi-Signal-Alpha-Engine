# Databricks notebook source
# MAGIC %pip install torch torchvision plotly scipy pandas numpy scikit-learn --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import SparkSession
from datetime import datetime
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error
)
from scipy.stats import spearmanr
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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = (
        torch.cuda.get_device_properties(0)
        .total_memory / 1e9
    )
    print(f"✅ CUDA    : {gpu} {mem:.1f}GB")
    print(f"✅ PyTorch : {torch.__version__}")
else:
    print("⚠️  CPU mode")

print(f"Device : {DEVICE}")

# COMMAND ----------

# ─────────────────────────────────────────────────────
#  RevIN
# ─────────────────────────────────────────────────────
class RevIN(nn.Module):
    """
    Reversible Instance Normalization.
    Normalizes per-sample — handles regime shifts.
    Critical for financial volatility series.
    """
    def __init__(self, n_features, eps=1e-5,
                 affine=True):
        super().__init__()
        self.eps    = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(
                torch.ones(1, 1, n_features)
            )
            self.beta  = nn.Parameter(
                torch.zeros(1, 1, n_features)
            )
        self._mean = None
        self._std  = None

    def normalize(self, x):
        # x: (B, T, C)
        self._mean = x.mean(dim=1, keepdim=True)
        self._std  = x.std(
            dim=1, keepdim=True
        ).clamp(min=self.eps)
        x = (x - self._mean) / self._std
        if self.affine:
            x = x * self.gamma + self.beta
        return x

    def denormalize(self, x):
        if self.affine:
            x = (
                x - self.beta.squeeze(1)
            ) / self.gamma.squeeze(1).clamp(
                min=self.eps
            )
        x = (
            x * self._std.squeeze(1) +
            self._mean.squeeze(1)
        )
        return x


# ─────────────────────────────────────────────────────
#  Patch Embedding
# ─────────────────────────────────────────────────────
class PatchEmbedding(nn.Module):
    def __init__(self, seq_len, patch_len,
                 stride, d_model, n_features):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        self.n_patches = (
            (seq_len - patch_len) // stride + 1
        )
        self.proj = nn.Linear(
            patch_len * n_features, d_model
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, C  = x.shape
        patches  = []
        for i in range(self.n_patches):
            s = i * self.stride
            e = s + self.patch_len
            p = x[:, s:e, :].reshape(B, -1)
            patches.append(p)
        x = torch.stack(patches, dim=1)
        return self.norm(self.proj(x))


# ─────────────────────────────────────────────────────
#  Transformer Block
# ─────────────────────────────────────────────────────
class PatchTSTBlock(nn.Module):
    def __init__(self, d_model, n_heads,
                 d_ff, dropout):
        super().__init__()
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads,
            dropout=dropout, batch_first=True
        )
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x    = self.norm1(x + self.drop(a))
        x    = self.norm2(x + self.drop(self.ff(x)))
        return x


# ─────────────────────────────────────────────────────
#  Warmup + Cosine Decay Scheduler
# ─────────────────────────────────────────────────────
class WarmupCosineScheduler:
    """Linear warmup then cosine decay."""
    def __init__(self, optimizer, warmup_steps,
                 total_steps, min_lr=1e-6):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.min_lr       = min_lr
        self.base_lrs     = [
            pg["lr"]
            for pg in optimizer.param_groups
        ]
        self._step = 0

    def step(self):
        self._step += 1
        for pg, base_lr in zip(
            self.optimizer.param_groups,
            self.base_lrs
        ):
            pg["lr"] = self._get_lr(base_lr)

    def _get_lr(self, base_lr):
        if self._step < self.warmup_steps:
            return base_lr * (
                self._step /
                max(1, self.warmup_steps)
            )
        progress = (
            self._step - self.warmup_steps
        ) / max(
            1,
            self.total_steps - self.warmup_steps
        )
        lr = self.min_lr + 0.5 * (
            base_lr - self.min_lr
        ) * (1 + math.cos(math.pi * progress))
        return max(lr, self.min_lr)

    def get_last_lr(self):
        return [
            pg["lr"]
            for pg in self.optimizer.param_groups
        ]


# ─────────────────────────────────────────────────────
#  Weighted Huber Loss
# ─────────────────────────────────────────────────────
class WeightedHuberLoss(nn.Module):
    """
    Per-target weighted Huber loss.
    Upweights fwd_vol_21d (×2.0) to force model
    to learn predictive signal over persistence.

    KEY FIX: uses .to(pred.device) in forward()
    to avoid cuda:0 vs cpu device mismatch.
    """
    def __init__(self, weights, delta=1.0):
        super().__init__()
        self.register_buffer(
            "weights",
            torch.tensor(
                weights, dtype=torch.float32
            )
        )
        self.huber = nn.HuberLoss(
            reduction="none", delta=delta
        )

    def forward(self, pred, target):
        # Always move weights to pred's device
        w    = self.weights.to(pred.device)
        loss = self.huber(pred, target)  # (B, C)
        return (loss * w).mean()


# ─────────────────────────────────────────────────────
#  PatchTST Volatility Model
# ─────────────────────────────────────────────────────
class PatchTSTVolatility(nn.Module):
    """
    Financial-aware PatchTST for vol forecasting.
    All 5 phases applied:
      Phase 2: RevIN per-sample normalization
      Phase 3: d_model=256, heads=8, layers=4
    """
    def __init__(
        self,
        seq_len    = 96,
        patch_len  = 16,
        stride     = 8,
        n_features = 20,
        d_model    = 256,
        n_heads    = 8,
        n_layers   = 4,
        d_ff       = 512,
        dropout    = 0.15,
        n_outputs  = 3,
    ):
        super().__init__()
        self.seq_len    = seq_len
        self.n_outputs  = n_outputs
        self.n_features = n_features

        self.revin       = RevIN(
            n_features, affine=True
        )
        self.patch_embed = PatchEmbedding(
            seq_len, patch_len, stride,
            d_model, n_features
        )
        n_patches    = self.patch_embed.n_patches
        self.pos_enc = nn.Parameter(
            torch.randn(1, n_patches, d_model) * 0.02
        )
        self.encoder = nn.ModuleList([
            PatchTSTBlock(
                d_model, n_heads, d_ff, dropout
            )
            for _ in range(n_layers)
        ])
        self.drop    = nn.Dropout(dropout)

        flat_dim  = n_patches * d_model
        self.head = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, n_outputs),
        )

    def forward(self, x):
        # x: (B, T, C)
        x = self.revin.normalize(x)
        x = self.patch_embed(x)
        x = x + self.pos_enc
        x = self.drop(x)
        for blk in self.encoder:
            x = blk(x)
        x = x.flatten(1)
        return self.head(x)

# COMMAND ----------

# ─────────────────────────────────────────────────────
#  Fast Vectorized Dataset
# ─────────────────────────────────────────────────────
class VolatilityDataset(Dataset):
    """
    Vectorized rolling window dataset.
    Uses numpy stride tricks — no Python loops.
    max_samples=None → full dataset (no cap).
    """
    def __init__(self, data, seq_len,
                 feature_cols, target_cols,
                 cs_norm=True,
                 max_samples=None,
                 seed=42):
        self.seq_len      = seq_len
        self.feature_cols = list(feature_cols)
        self.target_cols  = list(target_cols)

        data = data.sort_values(
            ["ticker","date"]
        ).reset_index(drop=True)

        # CS normalization per date (Phase 4)
        if cs_norm:
            for col in feature_cols:
                if col not in data.columns:
                    continue
                mu  = data.groupby("date")[
                    col
                ].transform("mean")
                std = data.groupby("date")[
                    col
                ].transform("std").clip(lower=1e-8)
                data[f"{col}_cs"] = (
                    (data[col] - mu) / std
                ).clip(-5, 5)
            self.feature_cols = [
                f"{c}_cs"
                if f"{c}_cs" in data.columns
                else c
                for c in feature_cols
            ]

        X_all = []
        y_all = []

        for _, grp in data.groupby("ticker"):
            grp = grp.sort_values("date")
            n   = len(grp)
            if n < seq_len + 1:
                continue

            X = grp[self.feature_cols].fillna(
                0
            ).values.astype(np.float32)
            y = grp[target_cols].fillna(
                0
            ).values.astype(np.float32)

            # Vectorized stride trick
            n_wins = n - seq_len
            Xw = np.lib.stride_tricks.as_strided(
                X,
                shape=(
                    n_wins, seq_len, X.shape[1]
                ),
                strides=(
                    X.strides[0],
                    X.strides[0],
                    X.strides[1],
                )
            ).copy()
            yw = y[seq_len:]

            valid = (
                ~np.isnan(Xw).any(axis=(1,2)) &
                ~np.isnan(yw).any(axis=1)
            )
            X_all.append(Xw[valid])
            y_all.append(yw[valid])

        if X_all:
            Xa = np.concatenate(X_all, axis=0)
            ya = np.concatenate(y_all, axis=0)

            # Apply sample cap only if specified
            if (max_samples is not None and
                    len(Xa) > max_samples):
                rng = np.random.default_rng(seed)
                idx = rng.choice(
                    len(Xa), max_samples,
                    replace=False
                )
                Xa = Xa[idx]
                ya = ya[idx]
                print(f"  Samples: {len(Xa):,} "
                      f"(capped from "
                      f"{len(np.concatenate(X_all)):,})")
            else:
                print(f"  Samples: {len(Xa):,} "
                      f"(full dataset ✅)")

            self.X = torch.tensor(
                Xa, dtype=torch.float32
            )
            self.y = torch.tensor(
                ya, dtype=torch.float32
            )
        else:
            self.X = torch.zeros(
                0, seq_len,
                len(self.feature_cols)
            )
            self.y = torch.zeros(
                0, len(target_cols)
            )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────
#  ML 03 Pipeline
# ─────────────────────────────────────────────────────
class MLPatchTSTVol:
    """
    ML 03 — PatchTST Vol Forecaster.
    Full dataset training (no sample cap).
    """

    FEATURE_COLS = [
        "return_1d","vol_21d","vol_63d",
        "rsi_14d","volume_ratio",
        "prob_bull","prob_bear","prob_highvol",
        "sharpe_21d","mom_21d",
        "mom_5d","mom_63d",
        "vol_ratio_21_63","vol_of_vol",
    ]
    TARGET_COLS    = [
        "fwd_vol_21d","vol_21d","vol_63d"
    ]
    TARGET_WEIGHTS = [2.0, 1.0, 1.0]

    SEQ_LEN   = 96
    PATCH_LEN = 16
    STRIDE    = 8
    D_MODEL   = 256
    N_HEADS   = 8
    N_LAYERS  = 4
    D_FF      = 512
    DROPOUT   = 0.15

    # ── KEY CHANGES FOR FULL DATASET ──────────────
    BATCH_SIZE     = 2048   # larger → fewer steps
    MAX_EPOCHS     = 60
    LR             = 3e-4
    WEIGHT_DECAY   = 1e-2
    WARMUP_PCT     = 0.05
    GRAD_CLIP      = 1.0
    PATIENCE       = 7
    ENSEMBLE_SEEDS = [42, 123, 456]
    MAX_TRAIN_SAMP = None   # None = full dataset
    MAX_VAL_SAMP   = None   # None = full val set
    INFER_BATCH    = 512
    # ──────────────────────────────────────────────

    def __init__(self, spark, gold_path,
                 ml_path, device):
        self.spark        = spark
        self.gold_path    = gold_path
        self.ml_path      = ml_path
        self.device       = device
        self.models       = []
        self.scaler_x     = StandardScaler()
        self.scaler_y     = StandardScaler()
        self.train_losses = []
        self.val_losses   = []
        self.FEATURE_COLS = list(
            dict.fromkeys(self.FEATURE_COLS)
        )

        print("MLPatchTSTVol ✓")
        print(f"  Device       : {device}")
        print(f"  Seq/Patch    : "
              f"{self.SEQ_LEN}/{self.PATCH_LEN} "
              f"stride={self.STRIDE}")
        print(f"  d_model      : {self.D_MODEL}")
        print(f"  Heads/Layers : "
              f"{self.N_HEADS}/{self.N_LAYERS}")
        print(f"  LR           : {self.LR} "
              f"(warmup "
              f"{self.WARMUP_PCT*100:.0f}%)")
        print(f"  Weight decay : {self.WEIGHT_DECAY}")
        print(f"  Grad clip    : {self.GRAD_CLIP}")
        print(f"  Patience     : {self.PATIENCE}")
        print(f"  Batch size   : {self.BATCH_SIZE}")
        print(f"  Max train    : "
              f"{'FULL DATASET ✅' if self.MAX_TRAIN_SAMP is None else f'{self.MAX_TRAIN_SAMP:,}'}")
        print(f"  Seeds        : "
              f"{self.ENSEMBLE_SEEDS}")
        print(f"  Target wts   : "
              f"{self.TARGET_WEIGHTS}")

    # ─────────────────────────────────────────────────
    #  Step 1 — Load (unchanged)
    # ─────────────────────────────────────────────────
    def load_data(self):
        print("\nStep 1: Loading data...")
        start = datetime.now()

        sdf      = self.spark.read.format(
            "delta"
        ).load(f"{self.gold_path}/price_factors")
        all_cols = set(sdf.columns)

        feat_avail = [
            c for c in self.FEATURE_COLS
            if c in all_cols
        ]
        tgt_avail  = [
            c for c in self.TARGET_COLS
            if c in all_cols
        ]
        keep = list(dict.fromkeys(
            ["date","ticker"] +
            feat_avail + tgt_avail
        ))

        pdf = sdf.select(keep).dropna(
            subset=tgt_avail
        ).toPandas()

        pdf["date"] = pd.to_datetime(pdf["date"])
        pdf = pdf.sort_values(
            ["ticker","date"]
        ).reset_index(drop=True)

        self.FEATURE_COLS = [
            c for c in self.FEATURE_COLS
            if c in pdf.columns
        ]
        self.TARGET_COLS = [
            c for c in self.TARGET_COLS
            if c in pdf.columns
        ]

        elapsed = (datetime.now()-start).seconds
        print(f"  Rows     : {len(pdf):,}")
        print(f"  Tickers  : "
              f"{pdf['ticker'].nunique():,}")
        print(f"  Dates    : "
              f"{pdf['date'].nunique():,}")
        print(f"  Range    : "
              f"{pdf['date'].min().date()} → "
              f"{pdf['date'].max().date()}")
        print(f"  Features : "
              f"{len(self.FEATURE_COLS)}")
        print(f"  Elapsed  : {elapsed}s")
        return pdf

    # ─────────────────────────────────────────────────
    #  Step 2 — Preprocess (unchanged)
    # ─────────────────────────────────────────────────
    def preprocess(self, pdf):
        print("\nStep 2: Preprocessing...")

        for col in (
            self.FEATURE_COLS + self.TARGET_COLS
        ):
            if col not in pdf.columns:
                continue
            med = pdf[col].median()
            pdf[col] = pdf[col].fillna(
                0.0 if np.isnan(med) else float(med)
            )

        for col in self.FEATURE_COLS:
            if col not in pdf.columns:
                continue
            q1 = pdf[col].quantile(0.01)
            q9 = pdf[col].quantile(0.99)
            pdf[col] = pdf[col].clip(q1, q9)

        if ("fwd_vol_21d" in pdf.columns and
                "vol_21d" in pdf.columns):
            eps = 1e-6
            pdf["log_vol_change_21d"] = np.log(
                (pdf["fwd_vol_21d"] + eps) /
                (pdf["vol_21d"] + eps)
            ).clip(-2, 2)

            pdf["vol_zscore_63d"] = (
                pdf["vol_21d"] -
                pdf.groupby("ticker")["vol_21d"]
                .transform(lambda x:
                    x.rolling(63, min_periods=10)
                    .mean()
                )
            ) / pdf.groupby("ticker")["vol_21d"] \
                .transform(lambda x:
                    x.rolling(63, min_periods=10)
                    .std()
                ).clip(lower=1e-8)

            for lag in [1, 5, 10, 21]:
                col_name = f"vol_lag_{lag}"
                pdf[col_name] = (
                    pdf.groupby("ticker")["vol_21d"]
                    .shift(lag)
                    .fillna(method="bfill")
                    .fillna(0)
                )
                if col_name not in self.FEATURE_COLS:
                    self.FEATURE_COLS.append(col_name)

            for col in [
                "log_vol_change_21d",
                "vol_zscore_63d",
            ]:
                pdf[col] = pdf[col].fillna(0)
                if col not in self.FEATURE_COLS:
                    self.FEATURE_COLS.append(col)

        for col in self.TARGET_COLS:
            if col not in pdf.columns:
                continue
            pdf[col] = pdf[col].clip(
                lower=0.0,
                upper=pdf[col].quantile(0.99)
            )

        dates      = sorted(pdf["date"].unique())
        cutoff     = dates[int(len(dates) * 0.8)]
        val_cutoff = dates[int(len(dates) * 0.9)]
        train_df   = pdf[pdf["date"] <= cutoff]
        val_df     = pdf[
            (pdf["date"] > cutoff) &
            (pdf["date"] <= val_cutoff)
        ]
        test_df    = pdf[pdf["date"] > val_cutoff]

        self.scaler_x.fit(
            train_df[self.FEATURE_COLS].fillna(0).values
        )
        self.scaler_y.fit(
            train_df[self.TARGET_COLS].fillna(0).values
        )

        def _scale(df):
            df = df.copy()
            df[self.FEATURE_COLS] = (
                self.scaler_x.transform(
                    df[self.FEATURE_COLS]
                    .fillna(0).values
                )
            )
            df[self.TARGET_COLS] = (
                self.scaler_y.transform(
                    df[self.TARGET_COLS]
                    .fillna(0).values
                )
            )
            return df

        print(f"  Features   : "
              f"{len(self.FEATURE_COLS)}")
        print(f"  Train rows : {len(train_df):,}")
        print(f"  Val rows   : {len(val_df):,}")
        print(f"  Test rows  : {len(test_df):,}")
        print(f"  Log-vol tgt: ✅")
        print(f"  Vol lags   : "
              f"{[c for c in self.FEATURE_COLS if 'lag' in c]}")
        return (
            _scale(train_df), _scale(val_df),
            _scale(test_df), pdf
        )

    # ─────────────────────────────────────────────────
    #  Step 3 — DataLoaders (no sample cap)
    # ─────────────────────────────────────────────────
    def build_loaders(self, train_df, val_df,
                       test_df):
        print("\nStep 3: Building DataLoaders "
              "(full dataset)...")
        pin = self.device.type == "cuda"

        print("  Train dataset...")
        train_ds = VolatilityDataset(
            train_df, self.SEQ_LEN,
            self.FEATURE_COLS, self.TARGET_COLS,
            cs_norm=True,
            max_samples=self.MAX_TRAIN_SAMP, # None
            seed=42,
        )
        print("  Val dataset...")
        val_ds   = VolatilityDataset(
            val_df, self.SEQ_LEN,
            self.FEATURE_COLS, self.TARGET_COLS,
            cs_norm=True,
            max_samples=self.MAX_VAL_SAMP,   # None
            seed=42,
        )
        print("  Test dataset...")
        test_ds  = VolatilityDataset(
            test_df, self.SEQ_LEN,
            self.FEATURE_COLS, self.TARGET_COLS,
            cs_norm=True,
            max_samples=self.MAX_VAL_SAMP,   # None
            seed=42,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=pin,
            drop_last=True,
        )
        val_loader   = DataLoader(
            val_ds,
            batch_size=self.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=pin,
        )
        test_loader  = DataLoader(
            test_ds,
            batch_size=self.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=pin,
        )

        n_train = len(train_ds)
        n_val   = len(val_ds)
        n_test  = len(test_ds)
        print(f"\n  Train samples : {n_train:,}")
        print(f"  Val samples   : {n_val:,}")
        print(f"  Test samples  : {n_test:,}")
        print(f"  Train batches : {len(train_loader)}")
        print(f"  Val batches   : {len(val_loader)}")
        print(f"  Test batches  : {len(test_loader)}")
        print(f"\n  Est. time/epoch: "
              f"~{len(train_loader)*2//60} min")
        print(f"  Est. total (3 seeds): "
              f"~{len(train_loader)*2*20//60} min "
              f"(~20 epochs avg)")
        return train_loader, val_loader, test_loader

    # ─────────────────────────────────────────────────
    #  Step 4 — Build model (unchanged)
    # ─────────────────────────────────────────────────
    def build_model(self, seed=42):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = PatchTSTVolatility(
            seq_len    = self.SEQ_LEN,
            patch_len  = self.PATCH_LEN,
            stride     = self.STRIDE,
            n_features = len(self.FEATURE_COLS),
            d_model    = self.D_MODEL,
            n_heads    = self.N_HEADS,
            n_layers   = self.N_LAYERS,
            d_ff       = self.D_FF,
            dropout    = self.DROPOUT,
            n_outputs  = len(self.TARGET_COLS),
        ).to(self.device)

        n = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
        print(f"  Params   : {n:,}")
        print(f"  Patches  : "
              f"{model.patch_embed.n_patches}")
        print(f"  Input    : "
              f"(B,{self.SEQ_LEN},"
              f"{len(self.FEATURE_COLS)})")
        return model

    # ─────────────────────────────────────────────────
    #  Step 5 — Train one seed (device fix)
    # ─────────────────────────────────────────────────
    def _train_one(self, model, train_loader,
                    val_loader):
        total_steps  = self.MAX_EPOCHS * len(
            train_loader
        )
        warmup_steps = int(
            total_steps * self.WARMUP_PCT
        )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.LR,
            weight_decay=self.WEIGHT_DECAY,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_steps,
            total_steps, min_lr=1e-6
        )

        # KEY FIX: .to(self.device) prevents
        # cuda:0 vs cpu device mismatch
        criterion = WeightedHuberLoss(
            weights=self.TARGET_WEIGHTS, delta=1.0
        ).to(self.device)

        use_amp   = self.device.type == "cuda"
        scaler    = torch.cuda.amp.GradScaler(
            enabled=use_amp
        )

        best_val   = float("inf")
        best_state = None
        patience   = 0
        t_losses   = []
        v_losses   = []

        print(f"\n  Steps  : {total_steps:,} "
              f"| Warmup: {warmup_steps:,}")
        print(f"  AMP    : {use_amp}")
        print(f"\n  {'Ep':>4} {'Train':>9} "
              f"{'Val':>9} {'LR':>9} {'★':>4}")
        print(f"  {'-'*40}")

        for epoch in range(1, self.MAX_EPOCHS + 1):

            model.train()
            t_loss = 0.0
            n_tr   = 0

            for X_b, y_b in train_loader:
                X_b = X_b.to(
                    self.device, non_blocking=True
                )
                y_b = y_b.to(
                    self.device, non_blocking=True
                )
                optimizer.zero_grad(
                    set_to_none=True
                )

                with torch.cuda.amp.autocast(
                    enabled=use_amp
                ):
                    pred = model(X_b)
                    loss = criterion(pred, y_b)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.GRAD_CLIP
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                t_loss += loss.item()
                n_tr   += 1

            t_loss /= max(n_tr, 1)

            model.eval()
            v_loss = 0.0
            n_va   = 0

            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b = X_b.to(self.device)
                    y_b = y_b.to(self.device)
                    with torch.cuda.amp.autocast(
                        enabled=use_amp
                    ):
                        pred = model(X_b)
                        loss = criterion(pred, y_b)
                    v_loss += loss.item()
                    n_va   += 1

            v_loss /= max(n_va, 1)
            t_losses.append(t_loss)
            v_losses.append(v_loss)

            is_best = v_loss < best_val
            if is_best:
                best_val   = v_loss
                best_state = {
                    k: v.clone()
                    for k, v in
                    model.state_dict().items()
                }
                patience   = 0
                marker     = "✅"
            else:
                patience  += 1
                marker     = ""

            cur_lr = scheduler.get_last_lr()[0]
            if (epoch % 5 == 0 or
                    is_best or epoch == 1):
                print(f"  {epoch:>4} "
                      f"{t_loss:>9.5f} "
                      f"{v_loss:>9.5f} "
                      f"{cur_lr:>9.2e} "
                      f"{marker}")

            if patience >= self.PATIENCE:
                print(
                    f"\n  Early stop "
                    f"@ epoch {epoch}"
                )
                break

        if best_state:
            model.load_state_dict(best_state)
        print(f"  Best val : {best_val:.5f}")
        return model, best_val, t_losses, v_losses

    # ─────────────────────────────────────────────────
    #  Steps 6-8 + write + run (all unchanged)
    # ─────────────────────────────────────────────────
    def train_ensemble(self, train_loader,
                        val_loader):
        print(f"\nStep 6: Training "
              f"{len(self.ENSEMBLE_SEEDS)}-seed "
              f"ensemble (full data)...")

        for i, seed in enumerate(self.ENSEMBLE_SEEDS):
            print(f"\n── Seed {seed} "
                  f"({i+1}/"
                  f"{len(self.ENSEMBLE_SEEDS)}) ──")
            model = self.build_model(seed=seed)
            model, bv, tl, vl = self._train_one(
                model, train_loader, val_loader
            )
            self.models.append(model)
            self.train_losses = tl
            self.val_losses   = vl
            print(f"Seed {seed}: {bv:.5f} ✓")

        print(f"\nEnsemble size: {len(self.models)}")

    def evaluate_ensemble(self, test_loader):
        print("\nStep 7: Evaluating ensemble...")

        seed_preds = []
        trues_list = []

        for model in self.models:
            model.eval()
            sp = []
            tl = []
            with torch.no_grad():
                for X_b, y_b in test_loader:
                    X_b = X_b.to(self.device)
                    with torch.cuda.amp.autocast(
                        enabled=(
                            self.device.type == "cuda"
                        )
                    ):
                        pred = model(X_b)
                    sp.append(pred.cpu().numpy())
                    tl.append(y_b.numpy())
            seed_preds.append(np.vstack(sp))
            trues_list = tl

        trues_sc = np.vstack(trues_list)
        ens_sc   = np.mean(seed_preds, axis=0)

        preds = self.scaler_y.inverse_transform(
            ens_sc
        )
        trues = self.scaler_y.inverse_transform(
            trues_sc
        )

        metrics = {}
        for i, col in enumerate(self.TARGET_COLS):
            p    = preds[:, i]
            t    = trues[:, i]
            rmse = float(np.sqrt(
                mean_squared_error(t, p)
            ))
            mae  = float(mean_absolute_error(t, p))
            corr = float(np.corrcoef(p, t)[0, 1])
            ic,_ = spearmanr(p, t)
            metrics[col] = {
                "rmse"    : rmse,
                "mae"     : mae,
                "corr"    : corr,
                "spearman": float(ic),
            }
            flag = (
                "✅" if corr > 0.5 else
                "⚠️" if corr > 0.3 else "❌"
            )
            print(f"  {col:20}: "
                  f"RMSE={rmse:.4f}  "
                  f"Corr={corr:.4f}  "
                  f"IC={ic:.4f} {flag}")

        vol_signal = pd.DataFrame({
            "pred_vol_21d": preds[:, 0],
            "true_vol_21d": trues[:, 0],
        })
        return metrics, preds, trues, vol_signal

    def predict_full(self, pdf):
        print("\nStep 8: Fast batch predictions...")
        start     = datetime.now()
        all_preds = []

        for m in self.models:
            m.eval()

        tickers = pdf["ticker"].unique()
        n_total = len(tickers)

        for t_idx, ticker in enumerate(tickers):
            if t_idx % 100 == 0:
                elapsed = (
                    datetime.now()-start
                ).seconds
                pct = t_idx / max(1, n_total) * 100
                print(f"  [{t_idx:>4}/{n_total}] "
                      f"{pct:.0f}% "
                      f"| {elapsed}s elapsed")

            grp = pdf[
                pdf["ticker"] == ticker
            ].sort_values("date")
            n   = len(grp)
            if n < self.SEQ_LEN + 1:
                continue

            X_raw    = grp[
                self.FEATURE_COLS
            ].fillna(0).values.astype(np.float32)
            X_scaled = self.scaler_x.transform(
                X_raw
            ).astype(np.float32)

            n_wins = n - self.SEQ_LEN
            Xw     = np.lib.stride_tricks.as_strided(
                X_scaled,
                shape=(
                    n_wins, self.SEQ_LEN,
                    X_scaled.shape[1]
                ),
                strides=(
                    X_scaled.strides[0],
                    X_scaled.strides[0],
                    X_scaled.strides[1],
                )
            ).copy()

            dates_w = grp["date"].iloc[
                self.SEQ_LEN:
            ].values

            X_t      = torch.tensor(
                Xw, dtype=torch.float32
            ).to(self.device)
            ens_outs = []

            with torch.no_grad():
                for s in range(
                    0, len(X_t), self.INFER_BATCH
                ):
                    xb       = X_t[
                        s:s+self.INFER_BATCH
                    ]
                    seed_out = []
                    for model in self.models:
                        with torch.cuda.amp.autocast(
                            enabled=(
                                self.device.type
                                == "cuda"
                            )
                        ):
                            out = model(xb)
                        seed_out.append(
                            out.cpu().numpy()
                        )
                    ens_outs.append(
                        np.mean(seed_out, axis=0)
                    )

            ens_preds = np.vstack(ens_outs)
            preds_inv = self.scaler_y \
                .inverse_transform(ens_preds)

            for j in range(len(dates_w)):
                row = {
                    "date"  : dates_w[j],
                    "ticker": ticker,
                }
                for k, col in enumerate(
                    self.TARGET_COLS
                ):
                    row[f"pred_{col}"] = float(
                        preds_inv[j, k]
                    )
                row["vol_position_signal"] = float(
                    np.clip(
                        1.0 / (
                            preds_inv[j, 0] * 20 + 1
                        ),
                        0.5, 1.0
                    )
                )
                all_preds.append(row)

        pred_df = pd.DataFrame(all_preds)
        elapsed  = (datetime.now()-start).seconds
        print(f"\n  ✅ Done")
        print(f"  Rows    : {len(pred_df):,}")
        print(f"  Tickers : "
              f"{pred_df['ticker'].nunique():,}")
        print(f"  Elapsed : {elapsed}s "
              f"({elapsed/60:.1f} min)")

        pc = "pred_fwd_vol_21d"
        if pc in pred_df.columns:
            print(f"\n  Pred vol stats:")
            print(f"  Mean : "
                  f"{pred_df[pc].mean():.4f}")
            print(f"  Std  : "
                  f"{pred_df[pc].std():.4f}")
            print(f"  P25  : "
                  f"{pred_df[pc].quantile(0.25):.4f}")
            print(f"  P75  : "
                  f"{pred_df[pc].quantile(0.75):.4f}")
        return pred_df

    def write_results(self, pred_df, metrics,
                       vol_signal):
        print("\nWriting results...")

        def _write(df, path, partition=True):
            df   = df.copy()
            nums = df.select_dtypes(
                include=[np.number]
            ).columns
            df[nums] = df[nums].fillna(0)
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

        _write(
            pred_df,
            f"{self.ml_path}"
            f"/patchtst_vol_predictions"
        )
        print("  ✓ patchtst_vol_predictions")

        _write(
            pd.DataFrame({
                "epoch"     : range(
                    1, len(self.train_losses)+1
                ),
                "train_loss": self.train_losses,
                "val_loss"  : self.val_losses,
            }),
            f"{self.ml_path}"
            f"/patchtst_training_curve",
            partition=False
        )
        print("  ✓ patchtst_training_curve")

        _write(
            pd.DataFrame([
                {"target": k, **v}
                for k, v in metrics.items()
            ]),
            f"{self.ml_path}/patchtst_metrics",
            partition=False
        )
        print("  ✓ patchtst_metrics")

        _write(
            vol_signal,
            f"{self.ml_path}/patchtst_vol_signal",
            partition=False
        )
        print("  ✓ patchtst_vol_signal")

    def run(self):
        print("="*55)
        print("ML 03 — PatchTST (FULL DATASET)")
        print(f"All 5 phases | {self.device}")
        print("="*55)
        start = datetime.now()

        pdf = self.load_data()
        train_df, val_df, test_df, pdf_raw = (
            self.preprocess(pdf)
        )
        train_loader, val_loader, test_loader = (
            self.build_loaders(
                train_df, val_df, test_df
            )
        )
        self.train_ensemble(train_loader, val_loader)
        metrics, preds, trues, vol_signal = (
            self.evaluate_ensemble(test_loader)
        )
        pred_df = self.predict_full(pdf_raw)
        self.write_results(
            pred_df, metrics, vol_signal
        )

        elapsed = (
            datetime.now()-start
        ).seconds / 60
        print(f"\nTotal time : {elapsed:.1f} min")
        print("ML 03 COMPLETE ✓")
        return pred_df, metrics, preds, trues
                

# COMMAND ----------

class MLPatchTSTCharts:
    TEMPLATE = "plotly_dark"
    C = {
        "primary"  : "#2196F3",
        "secondary": "#FF5722",
        "success"  : "#4CAF50",
        "warning"  : "#FFC107",
        "purple"   : "#9C27B0",
        "teal"     : "#00BCD4",
    }

    def chart_training_curve(self, pipeline):
        tl     = pipeline.train_losses
        vl     = pipeline.val_losses
        epochs = list(range(1, len(tl)+1))
        best_e = int(np.argmin(vl)) + 1

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Train vs Val Loss",
                "Overfit Gap (Val - Train)",
            ]
        )
        fig.add_trace(go.Scatter(
            x=epochs, y=tl, name="Train",
            line=dict(
                color=self.C["primary"], width=2
            )
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=epochs, y=vl, name="Val",
            line=dict(
                color=self.C["secondary"], width=2
            )
        ), row=1, col=1)
        fig.add_vline(
            x=best_e, line_dash="dot",
            line_color=self.C["success"],
            annotation_text=f"Best={best_e}",
            row=1, col=1
        )

        gap   = [v-t for t,v in zip(tl,vl)]
        bar_c = [
            self.C["secondary"] if g > 0
            else self.C["success"] for g in gap
        ]
        fig.add_trace(go.Bar(
            x=epochs, y=gap,
            marker_color=bar_c,
            showlegend=False
        ), row=1, col=2)
        fig.add_hline(
            y=0, line_dash="dash",
            line_color="white", opacity=0.3,
            row=1, col=2
        )

        fig.update_layout(
            title=(
                f"<b>ML 03 — Training Curve<br>"
                f"<sup>"
                f"Best val={min(vl):.5f} "
                f"@ epoch {best_e} | "
                f"Seeds="
                f"{len(pipeline.ENSEMBLE_SEEDS)}"
                f"</sup></b>"
            ),
            template=self.TEMPLATE, height=500,
            hovermode="x unified"
        )
        fig.show()

    def chart_pred_vs_actual(self, preds, trues,
                              target_cols):
        n   = min(len(target_cols), 3)
        fig = make_subplots(
            rows=1, cols=n,
            subplot_titles=target_cols[:n]
        )
        for i in range(n):
            p    = preds[:, i]
            t    = trues[:, i]
            corr = float(np.corrcoef(p, t)[0, 1])
            ic,_ = spearmanr(p, t)

            fig.add_trace(go.Scatter(
                x=t, y=p, mode="markers",
                marker=dict(
                    color=self.C["primary"],
                    size=2, opacity=0.3
                ),
                showlegend=False
            ), row=1, col=i+1)

            mn = float(min(t.min(), p.min()))
            mx = float(max(t.max(), p.max()))
            fig.add_trace(go.Scatter(
                x=[mn,mx], y=[mn,mx],
                mode="lines",
                line=dict(
                    color="white", dash="dash"
                ),
                showlegend=False
            ), row=1, col=i+1)

            fig.update_xaxes(
                title_text=(
                    f"Actual "
                    f"(Corr={corr:.3f} "
                    f"IC={ic:.3f})"
                ),
                row=1, col=i+1
            )
            fig.update_yaxes(
                title_text="Predicted",
                row=1, col=i+1
            )

        fig.update_layout(
            title=(
                "<b>ML 03 — "
                "Ensemble Pred vs Actual</b>"
            ),
            template=self.TEMPLATE, height=500
        )
        fig.show()

    def chart_vol_forecast(self, pred_df,
                            target_cols):
        pc = f"pred_{target_cols[0]}"
        if pc not in pred_df.columns:
            return

        pred_df = pred_df.copy()
        pred_df["date"] = pd.to_datetime(
            pred_df["date"]
        )
        daily = pred_df.groupby("date").agg(
            mean_vol=(pc, "mean"),
            p10=(pc, lambda x: x.quantile(0.1)),
            p90=(pc, lambda x: x.quantile(0.9)),
        ).reset_index().sort_values("date")

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=[
                "Mean Predicted Vol (P10-P90)",
                "Vol Dispersion (P90-P10)",
            ],
            vertical_spacing=0.1,
            row_heights=[0.6, 0.4]
        )

        fig.add_trace(go.Scatter(
            x=pd.concat([
                daily["date"],
                daily["date"].iloc[::-1]
            ]),
            y=pd.concat([
                daily["p90"],
                daily["p10"].iloc[::-1]
            ]),
            fill="toself",
            fillcolor="rgba(33,150,243,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="P10-P90"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=daily["date"],
            y=daily["mean_vol"],
            name="Mean Pred Vol",
            mode="lines",
            line=dict(
                color=self.C["primary"], width=2
            )
        ), row=1, col=1)

        disp = daily["p90"] - daily["p10"]
        fig.add_trace(go.Scatter(
            x=daily["date"], y=disp,
            mode="lines", showlegend=False,
            line=dict(
                color=self.C["warning"], width=1.5
            ),
            fill="tozeroy",
            fillcolor="rgba(255,193,7,0.15)"
        ), row=2, col=1)

        fig.update_layout(
            title=(
                "<b>ML 03 — "
                "Volatility Forecast</b>"
            ),
            template=self.TEMPLATE, height=650,
            hovermode="x unified"
        )
        fig.update_yaxes(
            title_text="Pred Vol", row=1, col=1
        )
        fig.update_yaxes(
            title_text="Dispersion", row=2, col=1
        )
        fig.show()

    def chart_metrics(self, metrics):
        targets = list(metrics.keys())
        colors  = [
            self.C["primary"],
            self.C["success"],
            self.C["warning"],
        ][:len(targets)]

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "RMSE","Pearson Corr",
                "Spearman IC","MAE",
            ]
        )
        for vals, row, col in [
            ([metrics[t]["rmse"]
              for t in targets], 1, 1),
            ([metrics[t]["corr"]
              for t in targets], 1, 2),
            ([metrics[t]["spearman"]
              for t in targets], 2, 1),
            ([metrics[t]["mae"]
              for t in targets], 2, 2),
        ]:
            fig.add_trace(go.Bar(
                x=targets, y=vals,
                marker_color=colors,
                text=[f"{v:.4f}" for v in vals],
                textposition="outside",
                showlegend=False
            ), row=row, col=col)
            fig.add_hline(
                y=0, line_dash="dash",
                line_color="white", opacity=0.3,
                row=row, col=col
            )
        fig.update_layout(
            title=(
                "<b>ML 03 — "
                "Ensemble Metrics</b>"
            ),
            template=self.TEMPLATE, height=600
        )
        fig.show()

    def chart_position_signal(self, pred_df,
                               target_cols):
        if ("vol_position_signal"
                not in pred_df.columns):
            return

        pred_df = pred_df.copy()
        pred_df["date"] = pd.to_datetime(
            pred_df["date"]
        )
        pc    = f"pred_{target_cols[0]}"
        aggs  = {"signal": (
            "vol_position_signal","mean"
        )}
        if pc in pred_df.columns:
            aggs["vol"] = (pc,"mean")

        daily = pred_df.groupby("date").agg(
            **aggs
        ).reset_index().sort_values("date")

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=[
                "Position Size Signal → ML 04",
                "Mean Predicted Vol",
            ],
            vertical_spacing=0.1,
            row_heights=[0.5, 0.5]
        )

        colors = [
            self.C["success"] if v > 0.7
            else self.C["warning"] if v > 0.55
            else self.C["secondary"]
            for v in daily["signal"]
        ]
        fig.add_trace(go.Bar(
            x=daily["date"],
            y=daily["signal"],
            marker_color=colors,
            showlegend=False
        ), row=1, col=1)
        for y_val, label, color in [
            (0.7,"Reduce",self.C["warning"]),
            (0.55,"Cut",  self.C["secondary"]),
        ]:
            fig.add_hline(
                y=y_val, line_dash="dash",
                line_color=color, opacity=0.6,
                annotation_text=label,
                row=1, col=1
            )

        if "vol" in daily.columns:
            fig.add_trace(go.Scatter(
                x=daily["date"],
                y=daily["vol"],
                mode="lines", showlegend=False,
                line=dict(
                    color=self.C["secondary"],
                    width=1.5
                ),
                fill="tozeroy",
                fillcolor="rgba(255,87,34,0.15)"
            ), row=2, col=1)

        fig.update_layout(
            title=(
                "<b>ML 03 — "
                "Vol Position Signal (→ ML 04)</b>"
            ),
            template=self.TEMPLATE, height=650,
            hovermode="x unified"
        )
        fig.update_yaxes(
            title_text="Position Size",
            range=[0, 1.1], row=1, col=1
        )
        fig.update_yaxes(
            title_text="Pred Vol", row=2, col=1
        )
        fig.show()

    def run_all(self, pipeline, preds, trues,
                pred_df, metrics):
        print("\n" + "="*55)
        print("ML 03 Charts...")
        print("="*55)

        print("\n[1/5] Training Curve...")
        self.chart_training_curve(pipeline)

        print("[2/5] Pred vs Actual...")
        self.chart_pred_vs_actual(
            preds, trues, pipeline.TARGET_COLS
        )

        print("[3/5] Vol Forecast...")
        self.chart_vol_forecast(
            pred_df, pipeline.TARGET_COLS
        )

        print("[4/5] Metrics...")
        self.chart_metrics(metrics)

        print("[5/5] Position Signal...")
        self.chart_position_signal(
            pred_df, pipeline.TARGET_COLS
        )

        print("\nAll 5 charts ✓")

# COMMAND ----------

pipeline = MLPatchTSTVol(
    spark     = spark,
    gold_path = GOLD_PATH,
    ml_path   = ML_PATH,
    device    = DEVICE,
)

pdf = pipeline.load_data()

train_df, val_df, test_df, pdf_raw = (
    pipeline.preprocess(pdf)
)
train_loader, val_loader, test_loader = (
    pipeline.build_loaders(
        train_df, val_df, test_df
    )
)

pipeline.train_ensemble(train_loader, val_loader)

metrics, preds, trues, vol_signal = (
    pipeline.evaluate_ensemble(test_loader)
)

pred_df = pipeline.predict_full(pdf_raw)
pipeline.write_results(pred_df, metrics, vol_signal)

charts = MLPatchTSTCharts()
charts.run_all(
    pipeline = pipeline,
    preds    = preds,
    trues    = trues,
    pred_df  = pred_df,
    metrics  = metrics,
)
print("\nML 03 COMPLETE ✓")

# COMMAND ----------

m_df  = spark.read.format("delta").load(
    f"{ML_PATH}/patchtst_metrics"
).toPandas()
curve = spark.read.format("delta").load(
    f"{ML_PATH}/patchtst_training_curve"
).toPandas()
psdf  = spark.read.format("delta").load(
    f"{ML_PATH}/patchtst_vol_predictions"
)

n_params = sum(
    p.numel()
    for m in pipeline.models
    for p in m.parameters()
) // max(1, len(pipeline.models))

print("="*55)
print("ML 03 PatchTST — Final Summary")
print("="*55)
print(f"Device      : {DEVICE}")
print(f"Ensemble    : {len(pipeline.models)} seeds")
print(f"Params/model: {n_params:,}")
print(f"Best epoch  : "
      f"{curve['val_loss'].idxmin()+1}")
print(f"Best val    : "
      f"{curve['val_loss'].min():.5f}")

print(f"\nEnsemble Metrics:")
print(m_df[[
    "target","rmse","mae","corr","spearman"
]].to_string(index=False))

print(f"\nPrediction Coverage:")
print(f"  Rows    : {psdf.count():,}")
print(f"  Tickers : "
      f"{psdf.select('ticker').distinct().count():,}")

print(f"\n{'='*55}")
print(f"ML Progress:")
print(f"  ML 01 HMM      ✅ 4/5 Production")
print(f"  ML 02 LightGBM ✅ IC=0.093 Sharpe=7.3")
print(f"  ML 03 PatchTST ✅ Device fix applied")
print(f"  ML 04 Ensemble 🔲 Next")
print(f"{'='*55}")