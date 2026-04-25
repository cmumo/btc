"""
Bitcoin 5-Minute Prediction Terminal
=====================================
- TradingView chart (BITSTAMP:BTCUSD) + Coinbase WebSocket for live prices
- Hybrid ensemble: LSTM + GRU-Attention (TFT-inspired) + XGBoost
- SQLite database for persistent storage
- GMT+3 timezone, prices with 2 decimal places
- Fully responsive for mobile devices

ML Architecture
---------------
1. LSTM (2-layer, hidden=64)
     Captures sequential price memory across time steps.

2. GRU-Attention / TFT-inspired (2-layer GRU + 4-head self-attention + residual norm)
     Learns WHICH past timesteps matter most for the current prediction.
     This is the core idea behind Temporal Fusion Transformers.

3. XGBoost (n=150, depth=4)
     Handles non-sequential feature interactions and momentum signals.

4. Weighted soft voting ensemble
     LSTM 35% · GRU-Attn 35% · XGBoost 30%
     Adaptive: weights shift toward whichever model is available each cycle.

Feature Set (27 engineered features per candle)
-----------------------------------------------
Price structure : ret, hl, oc, body_ratio, upper_wick, lower_wick, is_bullish
Trend           : ema_diff (EMA9-EMA21), price_vs_ema9, ma_diff (MA3-MA5), price_vs_ma10
Momentum        : RSI-14 (normalised), MACD, MACD histogram, ROC-5, momentum-3
Volatility      : ATR-10, Bollinger %B, Bollinger width, rolling_vol-5
Volume          : vol_ratio (vs 3-bar MA), vol_trend (3-bar vs 10-bar MA)
Lags            : ret_lag1, ret_lag2, ret_lag3
Time            : hour_sin, hour_cos  (intraday seasonality)
"""

import asyncio
import json
import logging
import os
import random
import sqlite3
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from queue import Empty, Queue
from typing import Deque, List, Optional, Tuple

import numpy as np
import pandas as pd
import pickle
import websocket as ws_client
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================
WINDOW_SECONDS     = 300
HISTORY_LIMIT      = 200      # More history for better training
MIN_CANDLES_TRAIN  = 30       # Minimum for XGBoost
MIN_CANDLES_DEEP   = 55       # Minimum for LSTM / GRU-Attn (need SEQ_LEN warm-up)
RETRAIN_EVERY      = 3        # Retrain more frequently to stay adaptive
SEQ_LEN            = 20       # Lookback window: 20 x 5 min = 100 min
N_FEATURES         = 27       # Number of engineered features (must match FEATURE_COLS)
LSTM_HIDDEN        = 64
LSTM_LAYERS        = 2
GRU_HIDDEN         = 64
GRU_LAYERS         = 2
ATTN_HEADS         = 4
DROPOUT            = 0.25
TRAIN_EPOCHS       = 120
TRAIN_PATIENCE     = 15
TRAIN_LR           = 0.001
TRAIN_BATCH        = 16
TZ                 = timezone(timedelta(hours=3))  # GMT+3

# Signal thresholds — higher bar = fewer but higher-quality signals
SIGNAL_UP_THRESH   = 0.60
SIGNAL_DOWN_THRESH = 0.40

# Base ensemble weights [lstm, gru_attn, xgb]
BASE_WEIGHTS: List[float] = [0.35, 0.35, 0.30]

# Database
DB_PATH = os.environ.get("DATABASE_PATH", "/data/predictions.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
logger.info(f"Database path: {DB_PATH}")

torch.set_num_threads(2)   # Polite CPU usage in shared environments

# ============================================================
# DATABASE SETUP
# ============================================================
def init_database():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS candles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL UNIQUE, open REAL, high REAL, low REAL, close REAL, volume REAL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        window_start TEXT, window_end TEXT,
        predicted_signal TEXT, confidence INTEGER,
        actual_signal TEXT, actual_open REAL, actual_close REAL,
        result TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS model (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_data BLOB, trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        wins INTEGER, losses INTEGER
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS stats (
        key TEXT PRIMARY KEY, value INTEGER
    )''')
    conn.commit()
    conn.close()
    logger.info(f"Database initialised at {DB_PATH}")

def save_candle(candle: dict):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            'INSERT OR REPLACE INTO candles (timestamp,open,high,low,close,volume) VALUES (?,?,?,?,?,?)',
            (candle["ts"], candle["open"], candle["high"],
             candle["low"], candle["close"], candle["volume"]))
        conn.commit()
    except Exception as e:
        logger.error(f"save_candle: {e}")
    finally:
        conn.close()

def load_candles(limit: int = HISTORY_LIMIT) -> List[dict]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        'SELECT timestamp,open,high,low,close,volume FROM candles ORDER BY timestamp DESC LIMIT ?',
        (limit,)).fetchall()
    conn.close()
    return [{"ts": r[0], "open": r[1], "high": r[2],
             "low": r[3], "close": r[4], "volume": r[5]}
            for r in reversed(rows)]

def save_prediction(p: dict):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            'INSERT INTO predictions (window_start,window_end,predicted_signal,confidence,actual_signal,actual_open,actual_close,result) VALUES (?,?,?,?,?,?,?,?)',
            (p["window_start"], p["window_end"], p["predicted"], p["confidence"],
             p.get("actual", "⏳"), p.get("act_open", 0),
             p.get("act_close", 0), p.get("result", "⏳")))
        conn.commit()
    except Exception as e:
        logger.error(f"save_prediction: {e}")
    finally:
        conn.close()

def update_prediction_result(ws, we, actual, ao, ac, result):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            'UPDATE predictions SET actual_signal=?,actual_open=?,actual_close=?,result=? WHERE window_start=? AND window_end=?',
            (actual, ao, ac, result, ws, we))
        conn.commit()
    except Exception as e:
        logger.error(f"update_prediction_result: {e}")
    finally:
        conn.close()

def load_predictions(limit: int = 5) -> List[dict]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        'SELECT window_start,window_end,predicted_signal,confidence,actual_signal,actual_open,actual_close,result FROM predictions ORDER BY id DESC LIMIT ?',
        (limit,)).fetchall()
    conn.close()
    return [{
        "window_start": r[0], "window_end": r[1],
        "window": f"{r[0]}-{r[1]}",
        "predicted": r[2], "confidence": r[3],
        "actual": r[4] if r[4] else "⏳",
        "act_open": r[5] or 0, "act_close": r[6] or 0,
        "result": r[7] if r[7] else "⏳",
    } for r in rows]

def save_model(obj, wins: int, losses: int):
    conn = sqlite3.connect(DB_PATH)
    try:
        blob = pickle.dumps(obj)
        conn.execute(
            'INSERT OR REPLACE INTO model (id,model_data,wins,losses) VALUES (1,?,?,?)',
            (blob, wins, losses))
        conn.commit()
        logger.info(f"Model saved — {wins}W / {losses}L")
    except Exception as e:
        logger.error(f"save_model: {e}")
    finally:
        conn.close()

def load_model():
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute('SELECT model_data,wins,losses FROM model WHERE id=1').fetchone()
    conn.close()
    if row:
        try:
            obj = pickle.loads(row[0])
            logger.info(f"Model loaded — {row[1]}W / {row[2]}L")
            return obj, row[1], row[2]
        except Exception as e:
            logger.error(f"load_model: {e}")
    return None, 0, 0

def save_stats(wins: int, losses: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('INSERT OR REPLACE INTO stats (key,value) VALUES (?,?)', ("wins",   wins))
    conn.execute('INSERT OR REPLACE INTO stats (key,value) VALUES (?,?)', ("losses", losses))
    conn.commit()
    conn.close()

def load_stats():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute('SELECT key,value FROM stats').fetchall()
    conn.close()
    d = dict(rows)
    return d.get("wins", 0), d.get("losses", 0)

# ============================================================
# SHARED STATE
# ============================================================
state_lock             = threading.Lock()
completed_candles:     Deque[dict]     = deque(maxlen=HISTORY_LIMIT)
history_rows:          List[dict]      = []
live_candle:           Optional[dict]  = None
live_window_start:     Optional[float] = None
wins:                  int = 0
losses:                int = 0
candles_since_retrain: int = 0

model      = None
model_lock = threading.Lock()
price_queue: Queue = Queue()

_clients:      List[WebSocket]                     = []
_clients_lock: Optional[asyncio.Lock]              = None
_main_loop:    Optional[asyncio.AbstractEventLoop] = None

# ============================================================
# HELPERS
# ============================================================
def get_window_time_range(timestamp: float) -> Tuple[str, str]:
    dt = datetime.fromtimestamp(timestamp, tz=TZ)
    return dt.strftime("%H:%M"), (dt + timedelta(minutes=5)).strftime("%H:%M")

# ============================================================
# SYNTHETIC HISTORY (for cold start)
# ============================================================
def generate_synthetic_history(n: int = 80) -> List[dict]:
    candles, price = [], 65_000.0
    t = time.time() - n * WINDOW_SECONDS
    for i in range(n):
        o  = price + random.gauss(0, 50)
        c  = o     + random.gauss(0, 120)
        h  = max(o, c) + abs(random.gauss(0, 25))
        lo = min(o, c) - abs(random.gauss(0, 25))
        candles.append({
            "ts": t + i * WINDOW_SECONDS,
            "open": round(o, 2), "high": round(h, 2),
            "low": round(lo, 2), "close": round(c, 2),
            "volume": round(random.uniform(5, 50), 4),
        })
        price = c
    return candles

# ============================================================
# ENHANCED FEATURE ENGINEERING  (27 features)
# ============================================================
FEATURE_COLS = [
    # Price structure
    "ret", "hl", "oc", "body_ratio", "upper_wick", "lower_wick", "is_bullish",
    # Trend
    "ema_diff", "price_vs_ema9", "ma_diff", "price_vs_ma10",
    # Oscillators
    "rsi_norm", "macd", "macd_hist",
    # Volatility / bands
    "bb_pct", "bb_width", "atr",
    # Volume
    "vol_ratio", "vol_trend",
    # Lagged returns & momentum
    "ret_lag1", "ret_lag2", "ret_lag3",
    "momentum3_norm", "roc5", "volatility",
    # Intraday seasonality
    "hour_sin", "hour_cos",
]
assert len(FEATURE_COLS) == N_FEATURES, \
    f"FEATURE_COLS has {len(FEATURE_COLS)} items, expected {N_FEATURES}"


def make_features(candles: List[dict]) -> Optional[pd.DataFrame]:
    """
    Build 27 engineered features. Returns a DataFrame with FEATURE_COLS columns,
    NaN rows already dropped. Returns None if too few candles remain.
    """
    if len(candles) < 30:
        return None
    df = pd.DataFrame(candles)

    # ── Price structure ──────────────────────────────────────────
    df["ret"] = df["close"].pct_change()
    df["hl"]  = (df["high"] - df["low"]) / (df["close"].clip(lower=1e-9))
    df["oc"]  = (df["close"] - df["open"]) / (df["open"].clip(lower=1e-9))

    body = (df["close"] - df["open"]).abs()
    rng  = (df["high"]  - df["low"]).clip(lower=1e-9)
    df["body_ratio"]  = body / rng
    df["upper_wick"]  = (df["high"] - df[["close","open"]].max(axis=1)) / rng
    df["lower_wick"]  = (df[["close","open"]].min(axis=1) - df["low"])   / rng
    df["is_bullish"]  = (df["close"] > df["open"]).astype(float)

    # ── EMA trend ───────────────────────────────────────────────
    ema9  = df["close"].ewm(span=9,  adjust=False).mean()
    ema21 = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_diff"]      = (ema9 - ema21) / df["close"].clip(lower=1e-9)
    df["price_vs_ema9"] = (df["close"] - ema9) / ema9.clip(lower=1e-9)

    ma3  = df["close"].rolling(3).mean()
    ma5  = df["close"].rolling(5).mean()
    ma10 = df["close"].rolling(10).mean()
    df["ma_diff"]       = (ma3 - ma5)  / df["close"].clip(lower=1e-9)
    df["price_vs_ma10"] = (df["close"] - ma10) / ma10.clip(lower=1e-9)

    # ── RSI-14 (normalised to −1 … +1) ──────────────────────────
    delta    = df["close"].diff()
    avg_gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rs       = avg_gain / (avg_loss + 1e-9)
    df["rsi_norm"] = (100 - (100 / (1 + rs)) - 50) / 50

    # ── MACD (12/26/9) ───────────────────────────────────────────
    ema12     = df["close"].ewm(span=12, adjust=False).mean()
    ema26     = df["close"].ewm(span=26, adjust=False).mean()
    macd_line = (ema12 - ema26) / df["close"].clip(lower=1e-9)
    macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
    df["macd"]      = macd_line
    df["macd_hist"] = macd_line - macd_sig

    # ── Bollinger Bands (10-period) ──────────────────────────────
    bb_mid = df["close"].rolling(10).mean()
    bb_std = df["close"].rolling(10).std().clip(lower=1e-9)
    bb_up  = bb_mid + 2 * bb_std
    bb_lo  = bb_mid - 2 * bb_std
    df["bb_pct"]   = (df["close"] - bb_lo) / (bb_up - bb_lo + 1e-9)
    df["bb_width"] = (bb_up - bb_lo) / bb_mid.clip(lower=1e-9)

    # ── ATR-10 (normalised) ──────────────────────────────────────
    hl_r = df["high"] - df["low"]
    hc   = (df["high"] - df["close"].shift()).abs()
    lc   = (df["low"]  - df["close"].shift()).abs()
    tr   = pd.concat([hl_r, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=10, adjust=False).mean() / df["close"].clip(lower=1e-9)

    # ── Volume ───────────────────────────────────────────────────
    vm3  = df["volume"].rolling(3).mean()
    vm10 = df["volume"].rolling(10).mean()
    df["vol_ratio"] = df["volume"] / (vm3  + 1e-9)
    df["vol_trend"] = (vm3 - vm10) / (vm10 + 1e-9)

    # ── Lagged returns & momentum ────────────────────────────────
    df["ret_lag1"]       = df["ret"].shift(1)
    df["ret_lag2"]       = df["ret"].shift(2)
    df["ret_lag3"]       = df["ret"].shift(3)
    df["momentum3_norm"] = df["close"].pct_change(3)
    df["roc5"]           = df["close"].pct_change(5)
    df["volatility"]     = df["ret"].rolling(5).std()

    # ── Intraday time features ───────────────────────────────────
    try:
        hours = (pd.to_datetime(df["ts"], unit="s")
                   .dt.tz_localize("UTC")
                   .dt.tz_convert("Africa/Nairobi")
                   .dt.hour)
    except Exception:
        hours = pd.to_datetime(df["ts"], unit="s").dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    if len(df) < 1:
        return None
    return df[FEATURE_COLS]


# ============================================================
# PYTORCH MODEL DEFINITIONS
# ============================================================
class LSTMPredictor(nn.Module):
    """
    Stacked LSTM for sequential BTC candle data.
    Input : (B, SEQ_LEN, N_FEATURES)
    Output: (B, 2)  raw logits
    """
    def __init__(self, n_feat=N_FEATURES, hidden=LSTM_HIDDEN,
                 layers=LSTM_LAYERS, drop=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers,
                            batch_first=True, dropout=drop)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 32),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)          # (B, T, H)
        return self.head(out[:, -1, :])


class GRUAttentionPredictor(nn.Module):
    """
    TFT-inspired architecture:
      1. Linear input projection → hidden dim
      2. 2-layer GRU encoder
      3. Multi-head self-attention over all timesteps  ← the TFT insight
      4. Residual + LayerNorm  (add & norm, like a Transformer block)
      5. Feed-forward + residual + LayerNorm
      6. MLP head on the last timestep

    The attention mechanism lets the model focus on the candles that matter
    most right now (e.g. the spike 3 candles ago), rather than treating all
    past steps equally like a plain GRU.

    Input : (B, SEQ_LEN, N_FEATURES)
    Output: (B, 2)  raw logits
    """
    def __init__(self, n_feat=N_FEATURES, hidden=GRU_HIDDEN,
                 layers=GRU_LAYERS, heads=ATTN_HEADS, drop=DROPOUT):
        super().__init__()
        self.proj  = nn.Linear(n_feat, hidden)
        self.gru   = nn.GRU(hidden, hidden, layers,
                             batch_first=True, dropout=drop)
        self.attn  = nn.MultiheadAttention(hidden, heads,
                                            dropout=drop, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn   = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden * 2, hidden),
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.head  = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                           # (B, T, H)
        g, _ = self.gru(x)                         # (B, T, H)
        a, _ = self.attn(g, g, g)                  # multi-head attention
        x2   = self.norm1(g + a)                   # residual + norm
        x3   = self.norm2(x2 + self.ffn(x2))      # FFN residual + norm
        return self.head(x3[:, -1, :])             # last timestep → logits


# ============================================================
# HYBRID ENSEMBLE
# ============================================================
class HybridEnsemble:
    """
    Soft-voting ensemble of LSTM, GRU-Attention and XGBoost.

    Design notes
    ─────────────
    • scaler is fitted once on the training split and applied consistently
      to both training and inference data, preventing leakage.
    • Weights adapt to model availability each training cycle.
    • predict_proba() is intentionally fast (no grad, eval mode, CPU).
    """

    def __init__(self):
        self.scaler:   Optional[StandardScaler]        = None
        self.lstm:     Optional[LSTMPredictor]         = None
        self.gru_attn: Optional[GRUAttentionPredictor] = None
        self.xgb:      Optional[xgb.XGBClassifier]    = None
        self.weights:  List[float]                     = list(BASE_WEIGHTS)

    def predict_proba(self, X_seq: np.ndarray,
                       X_flat: np.ndarray) -> np.ndarray:
        """
        X_seq  : (1, SEQ_LEN, N_FEATURES) for LSTM / GRU-Attn
        X_flat : (1, N_FEATURES)          for XGBoost
        Returns: (1, 2)  [P(down), P(up)]
        """
        probs, used_w = [], []

        if self.lstm is not None:
            p = _torch_infer(self.lstm, X_seq)
            if p is not None:
                probs.append(p); used_w.append(self.weights[0])

        if self.gru_attn is not None:
            p = _torch_infer(self.gru_attn, X_seq)
            if p is not None:
                probs.append(p); used_w.append(self.weights[1])

        if self.xgb is not None:
            try:
                p = self.xgb.predict_proba(X_flat)
                probs.append(p); used_w.append(self.weights[2])
            except Exception as e:
                logger.error(f"XGB infer: {e}")

        if not probs:
            return np.array([[0.5, 0.5]])

        total = sum(used_w)
        return sum(p * w for p, w in zip(probs, used_w)) / total

    def update_weights(self) -> None:
        """Redistribute BASE_WEIGHTS among available sub-models."""
        avail = [self.lstm is not None,
                 self.gru_attn is not None,
                 self.xgb is not None]
        if all(avail):
            self.weights = list(BASE_WEIGHTS)
        else:
            total = sum(BASE_WEIGHTS[i] for i, a in enumerate(avail) if a)
            self.weights = [BASE_WEIGHTS[i] / total if avail[i] else 0.0
                            for i in range(3)]
        logger.info(
            f"Ensemble weights → LSTM:{self.weights[0]:.2f} "
            f"GRU-Attn:{self.weights[1]:.2f} XGB:{self.weights[2]:.2f}"
        )


def _torch_infer(model: nn.Module, X_seq: np.ndarray) -> Optional[np.ndarray]:
    try:
        model.eval()
        with torch.no_grad():
            logits = model(torch.FloatTensor(X_seq))
            return F.softmax(logits, dim=-1).numpy()
    except Exception as e:
        logger.error(f"_torch_infer: {e}")
        return None


# ============================================================
# PYTORCH TRAINING HELPER
# ============================================================
def _train_torch(model: nn.Module, X: np.ndarray, y: np.ndarray) -> nn.Module:
    """
    Trains a PyTorch model with:
      • Chronological 80/20 train–val split (no look-ahead leakage)
      • AdamW + ReduceLROnPlateau
      • Gradient clipping (max_norm=1.0)
      • Early stopping on validation loss
    """
    X_t = torch.FloatTensor(X)
    y_t = torch.LongTensor(y)

    split      = max(int(len(X_t) * 0.8), 1)
    train_dl   = DataLoader(TensorDataset(X_t[:split], y_t[:split]),
                             batch_size=TRAIN_BATCH, shuffle=True, drop_last=False)
    val_dl     = DataLoader(TensorDataset(X_t[split:], y_t[split:]),
                             batch_size=TRAIN_BATCH, drop_last=False)

    opt       = torch.optim.AdamW(model.parameters(), lr=TRAIN_LR, weight_decay=1e-4)
    sched     = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, patience=5, factor=0.5, min_lr=1e-5)
    crit      = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_sd   = None
    wait      = 0

    for epoch in range(TRAIN_EPOCHS):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if len(val_dl.dataset) == 0:
            break

        model.eval()
        v = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                v += crit(model(xb), yb).item()
        v /= max(len(val_dl), 1)
        sched.step(v)

        if v < best_loss - 1e-6:
            best_loss = v
            best_sd   = {k: t.clone() for k, t in model.state_dict().items()}
            wait      = 0
        else:
            wait += 1
            if wait >= TRAIN_PATIENCE:
                logger.info(f"  early-stop @ epoch {epoch+1}, val_loss={best_loss:.4f}")
                break

    if best_sd:
        model.load_state_dict(best_sd)
    model.eval()
    return model


def _build_sequences(feat: np.ndarray, labels: np.ndarray,
                      seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window sequences.
    feat  : (N, F)  scaled features
    labels: (N,)
    Returns X_seq (M, seq_len, F) and y (M,),  M = N − seq_len + 1
    """
    X, y = [], []
    for i in range(seq_len - 1, min(len(feat), len(labels))):
        X.append(feat[i - seq_len + 1: i + 1])
        y.append(labels[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ============================================================
# FULL MODEL TRAINING
# ============================================================
def train_model_from_candles(candles: List[dict]) -> None:
    global model

    n = len(candles)
    logger.info(f"Training on {n} candles …")
    if n < MIN_CANDLES_TRAIN + 2:
        logger.warning(f"Too few candles ({n}) — need {MIN_CANDLES_TRAIN+2}")
        return

    try:
        # Features for all candles except the last (no label yet)
        feat_df = make_features(candles[:-1])
        if feat_df is None or len(feat_df) == 0:
            return

        nf     = len(feat_df)
        offset = (n - 1) - nf

        # Build next-candle direction labels
        labels = []
        for i in range(nf):
            idx = offset + i + 1
            if idx >= n:
                break
            nxt = candles[idx]
            labels.append(1 if nxt["close"] >= nxt["open"] else 0)

        if len(labels) < MIN_CANDLES_TRAIN:
            logger.warning(f"Only {len(labels)} labelled samples — skip")
            return

        feat_df = feat_df.iloc[:len(labels)]
        raw     = feat_df.values.astype(np.float32)
        y_arr   = np.array(labels, dtype=np.int64)

        # Fit scaler on training portion only (80 % chronological split)
        split  = max(int(len(raw) * 0.8), 1)
        scaler = StandardScaler()
        scaler.fit(raw[:split])
        scaled = scaler.transform(raw)

        # Recency weighting for XGBoost
        # Most-recent sample → weight 1.0; oldest → ~0.05
        ns = len(y_arr)
        sw = np.array([0.95 ** (ns - 1 - i) for i in range(ns)], dtype=np.float32)

        ens = HybridEnsemble()
        ens.scaler = scaler

        # ── XGBoost ──────────────────────────────────────────────────
        logger.info("  Training XGBoost …")
        xgb_model = xgb.XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", verbosity=0,
        )
        xgb_model.fit(scaled, y_arr, sample_weight=sw)
        ens.xgb = xgb_model
        logger.info("  XGBoost ✓")

        # ── LSTM + GRU-Attention ─────────────────────────────────────
        if n >= MIN_CANDLES_DEEP:
            X_seq, y_seq = _build_sequences(scaled, y_arr, SEQ_LEN)
            logger.info(f"  Sequence dataset: {len(X_seq)} × {SEQ_LEN} × {N_FEATURES}")

            if len(X_seq) >= 20:
                logger.info("  Training LSTM …")
                lstm = LSTMPredictor(N_FEATURES, LSTM_HIDDEN, LSTM_LAYERS, DROPOUT)
                ens.lstm = _train_torch(lstm, X_seq, y_seq)
                logger.info("  LSTM ✓")

                logger.info("  Training GRU-Attention (TFT-inspired) …")
                gru = GRUAttentionPredictor(N_FEATURES, GRU_HIDDEN,
                                             GRU_LAYERS, ATTN_HEADS, DROPOUT)
                ens.gru_attn = _train_torch(gru, X_seq, y_seq)
                logger.info("  GRU-Attention ✓")
            else:
                logger.warning(f"  Not enough sequences ({len(X_seq)}) for deep learning")
        else:
            logger.info(f"  {n} candles < {MIN_CANDLES_DEEP} — XGBoost-only this cycle")

        ens.update_weights()

        with model_lock:
            model = ens

        save_model(ens, wins, losses)
        logger.info("  Ensemble saved to DB ✓")

    except Exception as exc:
        logger.exception(f"Training error: {exc}")


# ============================================================
# PREDICTION
# ============================================================
def predict_from_candles(candles: List[dict]) -> dict:
    default = {"signal": "HOLD", "confidence": 0, "next_window": ""}

    with model_lock:
        ens = model

    if ens is None or len(candles) < 6:
        return default

    try:
        feat_df = make_features(candles)
        if feat_df is None or len(feat_df) == 0:
            return default

        scaled = ens.scaler.transform(feat_df.values.astype(np.float32))

        # Build sequence for LSTM / GRU-Attn
        if len(scaled) >= SEQ_LEN:
            seq = scaled[-SEQ_LEN:]
        else:
            pad = np.zeros((SEQ_LEN - len(scaled), N_FEATURES), dtype=np.float32)
            seq = np.vstack([pad, scaled])

        X_seq  = seq[np.newaxis, :, :]   # (1, SEQ_LEN, F)
        X_flat = scaled[[-1], :]         # (1, F)

        prob = ens.predict_proba(X_seq, X_flat)   # (1, 2)
        up_p = float(prob[0, 1])
        conf = int(round(max(up_p, 1 - up_p) * 100))

        if   up_p >= SIGNAL_UP_THRESH:   signal = "UP"
        elif up_p <= SIGNAL_DOWN_THRESH: signal = "DOWN"
        else:                            signal = "HOLD"

        now       = datetime.now(TZ)
        nm        = ((now.minute // 5) + 1) * 5
        ns        = now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=nm)
        ne        = ns + timedelta(minutes=5)
        nxt_range = f"{ns.strftime('%H:%M')}-{ne.strftime('%H:%M')}"

        return {"signal": signal, "confidence": conf, "next_window": nxt_range}

    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        return default


# ============================================================
# CANDLE BUILDING (FIXED - Proper window detection)
# ============================================================
def process_price_tick(price: float, trade_ts: float) -> None:
    global live_candle, live_window_start, candles_since_retrain, wins, losses

    with state_lock:
        # Calculate which 5-minute window this timestamp belongs to
        current_window_start = int(trade_ts // WINDOW_SECONDS) * WINDOW_SECONDS
        
        # Case 1: No live candle exists yet - initialize
        if live_window_start is None:
            live_window_start = current_window_start
            live_candle = _new_candle(live_window_start, price)
            return
        
        # Case 2: Current tick belongs to a NEWER window than our live candle
        if current_window_start > live_window_start:
            # Close the old window first
            if live_candle is not None:
                _close_current_window_locked()
            # Start new window
            live_window_start = current_window_start
            live_candle = _new_candle(live_window_start, price)
            return
        
        # Case 3: Current tick belongs to the SAME window as our live candle
        if live_candle is not None:
            # Check if the window should have expired by time
            window_end_time = live_candle["ts"] + WINDOW_SECONDS
            if trade_ts >= window_end_time:
                # Window has expired - close it and start new
                _close_current_window_locked()
                live_window_start = current_window_start
                live_candle = _new_candle(live_window_start, price)
            else:
                # Update current candle prices
                live_candle["high"] = max(live_candle["high"], price)
                live_candle["low"] = min(live_candle["low"], price)
                live_candle["close"] = price


def _new_candle(ts: float, price: float) -> dict:
    return {"ts": ts, "open": price, "high": price, "low": price,
            "close": price, "volume": 0.0}


def _close_current_window_locked() -> None:
    global live_candle, candles_since_retrain, wins, losses

    candle = dict(live_candle)
    live_candle = None
    completed_candles.append(candle)
    candles_since_retrain += 1
    save_candle(candle)

    ws_str, we_str = get_window_time_range(candle["ts"])

    for row in history_rows:
        if row.get("actual") == "⏳" and row.get("window_start") == ws_str:
            actual = "UP" if candle["close"] > candle["open"] else "DOWN"
            row.update(actual=actual, act_open=candle["open"],
                       act_close=candle["close"], window_end=we_str,
                       window=f"{ws_str}-{we_str}")
            if row["predicted"] == actual:
                row["result"] = "✅"; wins   += 1
            else:
                row["result"] = "❌"; losses += 1

            update_prediction_result(ws_str, we_str, actual,
                                     candle["open"], candle["close"], row["result"])
            save_stats(wins, losses)
            logger.info(f"Window {ws_str}-{we_str}: "
                        f"Pred={row['predicted']} Act={actual} {row['result']}")
            break

    snap = list(completed_candles)
    pred = predict_from_candles(snap)

    ns_dt = datetime.fromtimestamp(candle["ts"] + WINDOW_SECONDS, tz=TZ)
    ne_dt = ns_dt + timedelta(minutes=5)
    nws, nwe = ns_dt.strftime("%H:%M"), ne_dt.strftime("%H:%M")

    rec = {
        "window_start": nws, "window_end": nwe, "window": f"{nws}-{nwe}",
        "predicted": pred["signal"], "confidence": pred["confidence"],
        "act_open": 0.0, "act_close": 0.0, "actual": "⏳", "result": "⏳",
    }
    history_rows.insert(0, rec)
    save_prediction(rec)
    while len(history_rows) > 5:
        history_rows.pop()

    if candles_since_retrain >= RETRAIN_EVERY:
        candles_since_retrain = 0
        threading.Thread(
            target=train_model_from_candles,
            args=(list(completed_candles),), daemon=True,
        ).start()


# ============================================================
# COINBASE WEBSOCKET  (unchanged from original)
# ============================================================
def _coinbase_thread() -> None:
    url, retry = "wss://ws-feed.exchange.coinbase.com", 2
    while True:
        logger.info("Connecting to Coinbase WS …")
        try:
            def on_open(w):
                nonlocal retry; retry = 2
                w.send(json.dumps({"type": "subscribe",
                                   "product_ids": ["BTC-USD"],
                                   "channels": ["ticker", "matches"]}))
                logger.info("Coinbase WS connected")

            def on_message(w, raw):
                try:
                    d = json.loads(raw)
                    if d.get("type") in ["ticker", "match"]:
                        price = float(d.get("price", 0))
                        if price > 0:
                            price_queue.put_nowait({"price": price, "ts": time.time()})
                except Exception:
                    pass

            ws_client.WebSocketApp(
                url, on_open=on_open, on_message=on_message,
                on_error=lambda w, e: logger.error(f"WS err: {e}"),
                on_close=lambda w, c, m: logger.warning("WS closed"),
            ).run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            logger.error(f"Coinbase thread: {e}")
        logger.warning(f"Reconnecting in {retry}s …")
        time.sleep(retry)
        retry = min(retry * 2, 60)


# ============================================================
# PRICE PROCESSOR  (unchanged from original)
# ============================================================
def price_processor_thread() -> None:
    while True:
        try:
            item = price_queue.get(timeout=1.0)
            process_price_tick(item["price"], item["ts"])
        except Empty:
            continue
        except Exception as e:
            logger.error(f"Price processor: {e}")


# ============================================================
# BROADCAST  (unchanged from original)
# ============================================================
async def _periodic_broadcast() -> None:
    while True:
        await asyncio.sleep(1)
        await _broadcast_state()

async def _broadcast_state() -> None:
    msg  = json.dumps(_build_state_payload())
    async with _clients_lock:
        snap = list(_clients)
    dead = []
    for ws in snap:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    if dead:
        async with _clients_lock:
            for ws in dead:
                if ws in _clients:
                    _clients.remove(ws)

def _build_state_payload() -> dict:
    with state_lock:
        snap = list(completed_candles)
        lc   = dict(live_candle) if live_candle else None
        w, l = wins, losses
        rows = list(history_rows[:5])

    pred       = predict_from_candles(snap) if snap else \
                 {"signal": "HOLD", "confidence": 0, "next_window": ""}
    ohlc       = ({k: round(lc[k], 2) for k in ("open","high","low","close")} if lc else {})
    live_price = lc["close"] if lc else (snap[-1]["close"] if snap else 0.0)

    with model_lock:
        model_ready = model is not None

    return {
        "price": round(live_price, 2),
        "signal": pred["signal"], "confidence": pred["confidence"],
        "next_window": pred["next_window"],
        "wins": w, "losses": l,
        "ohlc": ohlc, "table": rows,
        "model_ready": model_ready,
    }


# ============================================================
# HTML FRONTEND  (unchanged from original)
# ============================================================
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=yes">
<title>BTC 5-Min Predictor</title>
<style>
  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }
  
  body {
    background: #080C14;
    color: #C8D8EF;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
    min-height: 100vh;
    padding: 0;
    margin: 0;
  }
  
  .container {
    max-width: 1280px;
    margin: 0 auto;
    padding: 12px;
  }
  
  .header {
    margin-bottom: 12px;
    padding: 8px 0;
    text-align: center;
  }
  
  .header h1 {
    font-size: 1.4rem;
    font-weight: 700;
    color: #F7931A;
    letter-spacing: .02em;
    padding: 4px 0;
  }
  
  .status-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
    font-size: 0.8rem;
    color: #4A6080;
    flex-wrap: wrap;
  }
  
  .status-bar #clock-gmt3 {
    color: #FFFFFF;
    margin-left: auto;
  }
  
  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
  }
  
  .dot-ok {
    background: #00E5A0;
    box-shadow: 0 0 6px #00E5A0;
  }
  
  .dot-bad {
    background: #FF4560;
  }
  
  .dot-wait {
    background: #F7931A;
    animation: pulse 1.2s infinite;
  }
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }
  
  .main-grid {
    display: grid;
    grid-template-columns: 1fr 340px;
    gap: 12px;
    margin-bottom: 12px;
    align-items: stretch;
  }
  
  #tv-chart {
    background: #0D1421;
    border-radius: 10px;
    border: 1px solid #1E2D45;
    height: 380px;
    overflow: hidden;
  }
  
  .sidebar {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  
  .sidebar .card:first-child {
    flex: 0 0 auto;
  }
  
  .sidebar .card:nth-child(2) {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  
  .sidebar .card:nth-child(3) {
    flex: 0 0 auto;
  }
  
  .card {
    background: #0D1421;
    border: 1px solid #1E2D45;
    border-radius: 10px;
    padding: 14px;
  }
  
  .card-title {
    font-size: 0.85rem;
    color: #4A6080;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
  }
  
  .price {
    font-size: 2rem;
    font-weight: 700;
    color: #F7931A;
  }
  
  .pchange {
    font-size: 0.82rem;
    margin-left: 6px;
  }
  
  .pred-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 4px;
  }
  
  .pred-arrow {
    font-size: 2.4rem;
    line-height: 1;
  }
  
  .pred-dir {
    font-size: 1.3rem;
    font-weight: 700;
  }
  
  .conf-bar {
    background: #1E2D45;
    border-radius: 4px;
    height: 6px;
    margin-top: 8px;
    overflow: hidden;
  }
  
  .conf-fill {
    height: 100%;
    width: 0%;
    transition: width 0.4s ease;
  }
  
  .countdown {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  
  .cd-ring {
    width: 58px;
    height: 58px;
    transform: rotate(-90deg);
  }
  
  .cd-text {
    font-size: 1.8rem;
    font-weight: 700;
    color: #F7931A;
  }
  
  .bottom-row {
    display: grid;
    gap: 12px;
    grid-template-columns: 1fr 1fr;
    margin-bottom: 12px;
  }
  
  .ohlc-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 6px;
  }
  
  .ohlc-cell {
    background: #0F1623;
    border-radius: 6px;
    padding: 7px 10px;
    border: 1px solid #1E2D45;
  }
  
  .ohlc-cell .lbl {
    font-size: 0.6rem;
    color: #4A6080;
  }
  
  .ohlc-cell .val {
    font-size: 0.88rem;
    font-weight: 700;
    margin-top: 2px;
  }
  
  .perf-row {
    display: flex;
    gap: 10px;
    margin-top: 8px;
  }
  
  .perf-stat {
    flex: 1;
    text-align: center;
    background: #0F1623;
    border-radius: 7px;
    padding: 8px;
    border: 1px solid #1E2D45;
  }
  
  .perf-num {
    font-size: 1.35rem;
    font-weight: 700;
  }
  
  .perf-lbl {
    font-size: 0.6rem;
    color: #4A6080;
    margin-top: 2px;
  }
  
  .table-wrapper {
    overflow-x: auto;
    margin-top: 10px;
  }
  
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.75rem;
  }
  
  th, td {
    padding: 8px;
    text-align: left;
    border-bottom: 1px solid #1E2D45;
  }
  
  th {
    color: #4A6080;
    font-weight: 600;
  }
  
  .up {
    color: #00E5A0;
  }
  
  .down {
    color: #FF4560;
  }
  
  .disclaimer {
    background: #0F1623;
    border-radius: 8px;
    padding: 10px;
    font-size: 0.7rem;
    text-align: center;
    color: #FACC15;
    border: 1px solid rgba(250, 204, 21, 0.13);
    margin-top: 12px;
  }
  
  .flash-up {
    animation: flashUp 0.4s ease;
  }
  
  .flash-dn {
    animation: flashDn 0.4s ease;
  }
  
  @keyframes flashUp {
    0%, 100% { color: #C8D8EF; }
    50% { color: #00E5A0; }
  }
  
  @keyframes flashDn {
    0%, 100% { color: #C8D8EF; }
    50% { color: #FF4560; }
  }
  
  @media (max-width: 900px) {
    .main-grid {
      grid-template-columns: 1fr;
    }
    
    #tv-chart {
      height: 320px;
    }
    
    .sidebar {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
    }
    
    .sidebar .card:first-child,
    .sidebar .card:nth-child(2),
    .sidebar .card:nth-child(3) {
      flex: auto;
    }
    
    .bottom-row {
      grid-template-columns: 1fr;
    }
    
    .header h1 {
      font-size: 1.2rem;
    }
  }
  
  @media (max-width: 600px) {
    .container {
      padding: 8px;
    }
    
    .header h1 {
      font-size: 1.1rem;
    }
    
    .status-bar {
      font-size: 0.7rem;
      gap: 6px;
    }
    
    .sidebar {
      grid-template-columns: 1fr;
      gap: 10px;
    }
    
    #tv-chart {
      height: 260px;
    }
    
    .card {
      padding: 10px;
    }
    
    .card-title {
      font-size: 0.75rem;
    }
    
    .price {
      font-size: 1.5rem;
    }
    
    .pred-arrow {
      font-size: 1.8rem;
    }
    
    .pred-dir {
      font-size: 1.1rem;
    }
    
    .cd-text {
      font-size: 1.4rem;
    }
    
    .cd-ring {
      width: 48px;
      height: 48px;
    }
    
    .perf-num {
      font-size: 1.1rem;
    }
    
    table {
      font-size: 0.65rem;
    }
    
    th, td {
      padding: 6px 4px;
    }
    
    .ohlc-cell {
      padding: 5px 8px;
    }
    
    .ohlc-cell .val {
      font-size: 0.75rem;
    }
    
    .bottom-row {
      gap: 10px;
    }
  }
  
  @media (max-width: 380px) {
    .container {
      padding: 6px;
    }
    
    .header h1 {
      font-size: 1rem;
    }
    
    .status-bar {
      font-size: 0.65rem;
    }
    
    .pred-row {
      gap: 8px;
    }
    
    .countdown {
      gap: 10px;
    }
    
    table {
      font-size: 0.6rem;
    }
    
    th, td {
      padding: 4px 3px;
    }
  }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>₿ Bitcoin Price Trend Predictor</h1>
  </div>
  <div class="status-bar">
    <div class="dot dot-wait" id="ws-dot"></div>
    <span id="ws-txt">Connecting...</span>
    <span id="clock-gmt3">--:--:-- GMT+3</span>
  </div>

  <div class="main-grid">
    <div id="tv-chart"><div id="tv-widget" style="width:100%;height:100%"></div></div>
    <div class="sidebar">
      <div class="card">
        <div class="card-title">Live BTC / USD</div>
        <div><span class="price" id="price-val">$---.--</span><span class="pchange" id="price-change">--</span></div>
      </div>
      <div class="card">
        <div class="card-title">Next 5-Min Prediction</div>
        <div class="pred-row">
          <span class="pred-arrow" id="pred-arrow" style="color:#4A6080">◆</span>
          <span class="pred-dir" id="pred-dir" style="color:#4A6080">HOLD</span>
          <span id="conf-pct" style="font-size:.85rem;color:#4A6080">--%</span>
        </div>
        <div class="conf-bar"><div class="conf-fill" id="conf-bar"></div></div>
        <div id="pred-window" style="margin-top:6px;font-size:.7rem;color:#4A6080;"></div>
      </div>
      <div class="card">
        <div class="card-title">Next window opens in</div>
        <div class="countdown">
          <svg class="cd-ring" viewBox="0 0 72 72">
            <circle cx="36" cy="36" r="32" stroke="#1E2D45" stroke-width="5" fill="none"/>
            <circle cx="36" cy="36" r="32" stroke="#F7931A" stroke-width="5" fill="none"
              stroke-dasharray="201" stroke-dashoffset="201" id="cd-ring" stroke-linecap="round"/>
          </svg>
          <div class="cd-text" id="cd-val">5:00</div>
        </div>
      </div>
    </div>
  </div>

  <div class="bottom-row">
    <div class="card">
      <div class="card-title">Current Window</div>
      <div class="ohlc-grid">
        <div class="ohlc-cell"><div class="lbl">Open</div><div class="val" id="o-open">--</div></div>
        <div class="ohlc-cell"><div class="lbl">High</div><div class="val up" id="o-high">--</div></div>
        <div class="ohlc-cell"><div class="lbl">Low</div><div class="val down" id="o-low">--</div></div>
        <div class="ohlc-cell"><div class="lbl">Close</div><div class="val" id="o-close">--</div></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Performance</div>
      <div class="perf-row">
        <div class="perf-stat"><div class="perf-num up" id="p-wins">0</div><div class="perf-lbl">Wins</div></div>
        <div class="perf-stat"><div class="perf-num down" id="p-losses">0</div><div class="perf-lbl">Losses</div></div>
        <div class="perf-stat"><div class="perf-num" id="p-acc" style="color:#F7931A">--%</div><div class="perf-lbl">Accuracy</div></div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Last 5 Predictions</div>
    <div class="table-wrapper">
      <table>
        <thead>
          <tr>
            <th>Window</th><th>Prediction</th><th>Conf</th><th>Act.Open</th><th>Act.Close</th><th>Actual</th><th>Result</th>
          </tr>
        </thead>
        <tbody id="history-body">
          <tr><td colspan="7" style="text-align:center;padding:20px;">Waiting for data...</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <div class="disclaimer">⚠️ For educational purposes only. Past accuracy does not guarantee future results.</div>
</div>

<script src="https://s3.tradingview.com/tv.js"></script>
<script>
  new TradingView.widget({
    container_id:'tv-widget',symbol:'BITSTAMP:BTCUSD',interval:'5',
    theme:'dark',style:'1',locale:'en',toolbar_bg:'#080C14',
    enable_publishing:false,autosize:true
  });

  let ws, firstPrice=null, prevPrice=null;

  function updateClock(){
    const d=new Date();
    document.getElementById('clock-gmt3').textContent=
      d.toLocaleTimeString('en-US',{timeZone:'Africa/Nairobi',hour12:false})+' GMT+3';
  }
  updateClock(); setInterval(updateClock,1000);

  function updateCountdown(){
    const now=new Date();
    const gmt3=new Date(now.getTime()+3*3600000);
    const minutes = gmt3.getUTCMinutes();
    const seconds = gmt3.getUTCSeconds();
    const remainingSeconds = (5 - (minutes % 5)) * 60 - seconds;
    const remaining = Math.max(0, remainingSeconds);
    const mm = Math.floor(remaining / 60);
    const ss = String(remaining % 60).padStart(2,'0');
    document.getElementById('cd-val').textContent = `${mm}:${ss}`;
    const circ=201;
    const percent = remaining / 300;
    document.getElementById('cd-ring').setAttribute('stroke-dashoffset',
      String(circ * (1 - percent)));
  }
  updateCountdown(); setInterval(updateCountdown, 1000);

  function fmt(n){
    if(n==null||n===0)return'--';
    return'$'+Number(n).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
  }

  function connect(){
    const wsUrl=(location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws';
    ws=new WebSocket(wsUrl);
    ws.onopen=function(){
      document.getElementById('ws-dot').className='dot dot-ok';
      document.getElementById('ws-txt').textContent='Live';
    };
    ws.onclose=function(){
      document.getElementById('ws-dot').className='dot dot-bad';
      document.getElementById('ws-txt').textContent='Disconnected';
      setTimeout(connect,3000);
    };
    ws.onerror=function(){
      document.getElementById('ws-dot').className='dot dot-bad';
    };
    ws.onmessage=function(e){
      try{handleState(JSON.parse(e.data));}catch(err){}
    };
  }

  function handleState(d){
    if(d.type==='ping')return;

    const p=d.price;
    if(p&&p>0){
      const priceEl=document.getElementById('price-val');
      if(prevPrice!==null){
        priceEl.className='price '+(p>prevPrice?'flash-up':p<prevPrice?'flash-dn':'');
        setTimeout(function(){priceEl.className='price';},400);
      }
      priceEl.textContent=fmt(p);
      prevPrice=p;
      if(firstPrice===null)firstPrice=p;
      const chg=p-firstPrice;
      const pct=(chg/firstPrice*100).toFixed(2);
      const chgEl=document.getElementById('price-change');
      chgEl.textContent=(chg>=0?'+':'')+fmt(chg)+' ('+pct+'%)';
      chgEl.style.color=chg>=0?'#00E5A0':'#FF4560';
    }

    const sig=d.signal||'HOLD';
    const isUp=sig==='UP',isDn=sig==='DOWN';
    const col=isUp?'#00E5A0':isDn?'#FF4560':'#4A6080';
    document.getElementById('pred-arrow').textContent=isUp?'▲':isDn?'▼':'◆';
    document.getElementById('pred-arrow').style.color=col;
    document.getElementById('pred-dir').textContent=isUp?'UP':isDn?'DOWN':'HOLD';
    document.getElementById('pred-dir').style.color=col;
    document.getElementById('conf-pct').textContent=(isUp||isDn)?d.confidence+'%':'--%';
    document.getElementById('conf-pct').style.color=col;
    document.getElementById('conf-bar').style.width=(d.confidence||0)+'%';
    document.getElementById('conf-bar').style.background=col;
    document.getElementById('pred-window').textContent=d.next_window?'Next: '+d.next_window:'';

    if(d.ohlc){
      document.getElementById('o-open').textContent=fmt(d.ohlc.open);
      document.getElementById('o-high').textContent=fmt(d.ohlc.high);
      document.getElementById('o-low').textContent=fmt(d.ohlc.low);
      document.getElementById('o-close').textContent=fmt(d.ohlc.close);
    }

    const w=d.wins||0,l=d.losses||0,tot=w+l;
    document.getElementById('p-wins').textContent=w;
    document.getElementById('p-losses').textContent=l;
    document.getElementById('p-acc').textContent=tot?(w/tot*100).toFixed(1)+'%':'--%';

    const tbody=document.getElementById('history-body');
    if(d.table&&d.table.length){
      tbody.innerHTML=d.table.map(function(r){
        const predCls=r.predicted==='UP'?'up':r.predicted==='DOWN'?'down':'';
        const actCls=r.actual==='UP'?'up':r.actual==='DOWN'?'down':'';
        const predTxt=r.predicted==='UP'?'▲ UP':r.predicted==='DOWN'?'▼ DOWN':r.predicted;
        const actTxt=r.actual==='⏳'?'--':r.actual==='UP'?'▲ UP':r.actual==='DOWN'?'▼ DOWN':r.actual;
        return '<tr>'
          +'<td style="color:#4A6080">'+r.window+'</td>'
          +'<td class="'+predCls+'">'+predTxt+'</td>'
          +'<td style="color:#F7931A">'+r.confidence+'%</td>'
          +'<td>'+fmt(r.act_open)+'</td>'
          +'<td>'+fmt(r.act_close)+'</td>'
          +'<td class="'+actCls+'">'+actTxt+'</td>'
          +'<td>'+(r.result==='⏳'?'--':r.result)+'</td>'
          +'</tr>';
      }).join('');
    }else{
      tbody.innerHTML='<tr><td colspan="7" style="text-align:center;padding:20px;">Waiting for data...</td></tr>';
    }
  }

  connect();
</script>
</body>
</html>"""


# ============================================================
# FASTAPI APPLICATION  (unchanged from original)
# ============================================================
@asynccontextmanager
async def lifespan(application: FastAPI):
    global _main_loop, _clients_lock, model, wins, losses, completed_candles, history_rows

    _main_loop    = asyncio.get_running_loop()
    _clients_lock = asyncio.Lock()

    init_database()

    db_wins, db_losses = load_stats()
    if db_wins > 0 or db_losses > 0:
        wins, losses = db_wins, db_losses
        logger.info(f"Loaded stats: {wins}W / {losses}L")

    saved_model, mw, ml = load_model()
    if saved_model:
        with model_lock:
            model = saved_model
        if mw > wins:
            wins, losses = mw, ml
        logger.info(f"Loaded hybrid ensemble — {wins}W / {losses}L")

    saved_preds = load_predictions(5)
    if saved_preds:
        with state_lock:
            history_rows = saved_preds
        logger.info(f"Loaded {len(saved_preds)} predictions from DB")

    saved_candles = load_candles(HISTORY_LIMIT)
    if saved_candles:
        with state_lock:
            for c in saved_candles:
                completed_candles.append(c)
        logger.info(f"Loaded {len(saved_candles)} candles from DB")

    if model is None:
        logger.info("No saved model — seeding with synthetic data and training …")
        synth = generate_synthetic_history(80)
        with state_lock:
            for c in synth:
                completed_candles.append(c)
                save_candle(c)
        threading.Thread(
            target=train_model_from_candles,
            args=(list(completed_candles),), daemon=True, name="init-train",
        ).start()
    else:
        logger.info(f"Continuing with existing ensemble ({len(completed_candles)} candles loaded)")

    save_stats(wins, losses)

    threading.Thread(target=price_processor_thread, name="price-proc", daemon=True).start()
    threading.Thread(target=_coinbase_thread,        name="coinbase-ws", daemon=True).start()

    asyncio.create_task(_periodic_broadcast())

    logger.info("=" * 60)
    logger.info("BTC Hybrid Predictor — ready")
    logger.info(f"Models   : LSTM + GRU-Attention (TFT-inspired) + XGBoost")
    logger.info(f"Features : {N_FEATURES}  |  Seq len : {SEQ_LEN}  |  History : {HISTORY_LIMIT}")
    logger.info(f"Thresholds: UP ≥ {SIGNAL_UP_THRESH}  DOWN ≤ {SIGNAL_DOWN_THRESH}")
    logger.info(f"Stats    : {wins}W / {losses}L")
    logger.info("=" * 60)
    yield

    with model_lock:
        mdl = model
    if mdl:
        save_model(mdl, wins, losses)
    save_stats(wins, losses)
    logger.info(f"Shutdown — state saved ({wins}W / {losses}L)")


app = FastAPI(lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_CONTENT

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    async with _clients_lock:
        _clients.append(websocket)
    logger.info(f"WS client connected — total: {len(_clients)}")
    try:
        await websocket.send_text(json.dumps(_build_state_payload()))
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=25.0)
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("text") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except asyncio.TimeoutError:
                try:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                except Exception:
                    break
            except WebSocketDisconnect:
                break
            except Exception:
                break
    except Exception:
        pass
    finally:
        async with _clients_lock:
            if websocket in _clients:
                _clients.remove(websocket)
        logger.info(f"WS client disconnected — total: {len(_clients)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port,
                ws="websockets", log_level="info")
