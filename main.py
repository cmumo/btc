"""
Bitcoin (BTC/USD) 5-Minute Prediction Terminal  ·  v4.0  (Ultimate Ensemble)
====================================================================
Data Source : Binance BTC/USDT  (real‑time trades)

Architecture
─────────────
1. BiLSTM       – bidirectional 2-layer LSTM, hidden=64
2. AttentionGRU – TFT-inspired: 2-layer GRU + 4-head self-attention + residual
3. TCN          – Temporal Convolutional Network, 4 dilated causal residual blocks
4. XGBoost      – n=300, depth=6, recency-weighted sample weights
5. RandomForest – n=300 trees, max_depth=8
6. LightGBM     – n=400, num_leaves=63

Adaptive Ensemble
──────────────────
Each model's voting weight = rolling accuracy over last 20 resolved UP/DOWN
predictions.  Best-performing model carries more weight in real time.
Base weights are used until 20 resolved predictions exist.

HOLD zone   : 0.49 – 0.51  (ultra-tight 2% neutral band → maximum directional signals)
HOLD effect : confidence shown as "--"; NEVER counted in wins/losses.

Feature Set  (42 engineered features per candle)
─────────────────────────────────────────────────
Price structure  (7) : ret, hl, oc, body_ratio, upper_wick, lower_wick, is_bullish
Trend            (6) : ema_diff, price_vs_ema9, ma_diff, price_vs_ma10, ema_slope, dema_diff
Oscillators      (8) : rsi_norm, macd, macd_hist, roc5, roc3, momentum3_norm, stoch_k_norm, cci_norm
Volatility       (5) : atr, bb_pct, bb_width, volatility, keltner_pct
Volume           (2) : vol_ratio, vol_trend
Directional      (3) : adx_norm, di_plus_norm, di_minus_norm
Lags             (5) : ret_lag1 – ret_lag5
Time             (2) : hour_sin, hour_cos
VMD-inspired     (4) : trend_component, residual_component, trend_slope, residual_energy

Sliding Window
───────────────
History table is duplicate-free: current pending row always at top; oldest
window drops after the table reaches 5 unique entries (v2 logic merged in).
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
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logging.warning("LightGBM not installed — will be skipped in ensemble")

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================
WINDOW_SECONDS     = 300
HISTORY_LIMIT      = 300
MIN_CANDLES_TRAIN  = 35
MIN_CANDLES_DEEP   = 60
RETRAIN_EVERY      = 3
SEQ_LEN            = 20
N_FEATURES         = 42          # Must match FEATURE_COLS length exactly
LSTM_HIDDEN        = 64
LSTM_LAYERS        = 2
GRU_HIDDEN         = 64
GRU_LAYERS         = 2
ATTN_HEADS         = 4
TCN_HIDDEN         = 64
TCN_LEVELS         = 4
TCN_KERNEL         = 3
DROPOUT            = 0.25
TRAIN_EPOCHS       = 120
TRAIN_PATIENCE     = 15
TRAIN_LR           = 0.001
TRAIN_BATCH        = 16
TZ                 = timezone(timedelta(hours=3))   # GMT+3

# Ultra-tight HOLD zone — only 2 % of probability space triggers HOLD
SIGNAL_UP_THRESH   = 0.51
SIGNAL_DOWN_THRESH = 0.49

# Base weights  [bilstm, attn_gru, tcn, xgb, rf, lgb]
BASE_WEIGHTS: List[float] = [0.20, 0.20, 0.15, 0.15, 0.15, 0.15]
ADAPTIVE_WINDOW    = 20

DB_PATH = os.environ.get("DATABASE_PATH", "/data/predictions.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
logger.info(f"Database path: {DB_PATH}")

torch.set_num_threads(2)

# ============================================================
# DATABASE
# ============================================================
def init_database():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS candles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL UNIQUE, open REAL, high REAL,
        low REAL, close REAL, volume REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        window_start TEXT, window_end TEXT,
        predicted_signal TEXT, confidence INTEGER,
        actual_signal TEXT, actual_open REAL, actual_close REAL,
        result TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS model (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_data BLOB, trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        wins INTEGER, losses INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS stats (
        key TEXT PRIMARY KEY, value INTEGER)''')
    conn.commit()
    conn.close()
    logger.info(f"Database initialised at {DB_PATH}")


def save_candle(candle: dict):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            'INSERT OR REPLACE INTO candles '
            '(timestamp,open,high,low,close,volume) VALUES (?,?,?,?,?,?)',
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
        'SELECT timestamp,open,high,low,close,volume FROM candles '
        'ORDER BY timestamp DESC LIMIT ?', (limit,)).fetchall()
    conn.close()
    return [{"ts": r[0], "open": r[1], "high": r[2],
             "low": r[3], "close": r[4], "volume": r[5]}
            for r in reversed(rows)]


def save_prediction(p: dict):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            'INSERT INTO predictions '
            '(window_start,window_end,predicted_signal,confidence,'
            'actual_signal,actual_open,actual_close,result) VALUES (?,?,?,?,?,?,?,?)',
            (p["window_start"], p["window_end"], p["predicted"],
             p["confidence"], p.get("actual", "⏳"),
             p.get("act_open", 0), p.get("act_close", 0),
             p.get("result", "⏳")))
        conn.commit()
    except Exception as e:
        logger.error(f"save_prediction: {e}")
    finally:
        conn.close()


def update_prediction_result(ws, we, actual, ao, ac, result):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            'UPDATE predictions SET actual_signal=?,actual_open=?,'
            'actual_close=?,result=? WHERE window_start=? AND window_end=?',
            (actual, ao, ac, result, ws, we))
        conn.commit()
    except Exception as e:
        logger.error(f"update_prediction_result: {e}")
    finally:
        conn.close()


def load_predictions(limit: int = 15) -> List[dict]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        'SELECT window_start,window_end,predicted_signal,confidence,'
        'actual_signal,actual_open,actual_close,result FROM predictions '
        'ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
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
            'INSERT OR REPLACE INTO model (id,model_data,wins,losses) '
            'VALUES (1,?,?,?)', (blob, wins, losses))
        conn.commit()
        logger.info(f"Model saved — {wins}W / {losses}L")
    except Exception as e:
        logger.error(f"save_model: {e}")
    finally:
        conn.close()


def load_model():
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        'SELECT model_data,wins,losses FROM model WHERE id=1').fetchone()
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
    conn.execute('INSERT OR REPLACE INTO stats (key,value) VALUES (?,?)',
                 ("wins", wins))
    conn.execute('INSERT OR REPLACE INTO stats (key,value) VALUES (?,?)',
                 ("losses", losses))
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
history_rows:          List[dict]      = []   # newest first, max 5 unique windows
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

_last_per_model_preds: List[Optional[int]] = [None] * 6
_last_per_model_lock  = threading.Lock()


# ============================================================
# HELPERS
# ============================================================
def get_window_time_range(timestamp: float) -> Tuple[str, str]:
    dt = datetime.fromtimestamp(timestamp, tz=TZ)
    return dt.strftime("%H:%M"), (dt + timedelta(minutes=5)).strftime("%H:%M")


# ============================================================
# SYNTHETIC HISTORY  (Bitcoin ~ $65,000)
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
# VMD-INSPIRED DECOMPOSITION
# ============================================================
def _vmd_decompose(prices: np.ndarray,
                   window: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    trend    = pd.Series(prices).rolling(window, min_periods=1).mean().values
    residual = prices - trend
    return trend, residual


# ============================================================
# FEATURE ENGINEERING  (42 features)
# ============================================================
FEATURE_COLS = [
    # Price structure (7)
    "ret", "hl", "oc", "body_ratio", "upper_wick", "lower_wick", "is_bullish",
    # Trend (6)
    "ema_diff", "price_vs_ema9", "ma_diff", "price_vs_ma10",
    "ema_slope", "dema_diff",
    # Oscillators (8)
    "rsi_norm", "macd", "macd_hist", "roc5", "roc3",
    "momentum3_norm", "stoch_k_norm", "cci_norm",
    # Volatility (5)
    "atr", "bb_pct", "bb_width", "volatility", "keltner_pct",
    # Volume (2)
    "vol_ratio", "vol_trend",
    # Directional (3)
    "adx_norm", "di_plus_norm", "di_minus_norm",
    # Lags (5)
    "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4", "ret_lag5",
    # Time (2)
    "hour_sin", "hour_cos",
    # VMD-inspired (4)
    "trend_component", "residual_component", "trend_slope", "residual_energy",
]
assert len(FEATURE_COLS) == N_FEATURES, \
    f"FEATURE_COLS length {len(FEATURE_COLS)} != N_FEATURES {N_FEATURES}"


def make_features(candles: List[dict]) -> Optional[pd.DataFrame]:
    if len(candles) < 30:
        return None
    df = pd.DataFrame(candles)

    # ── Price structure ──────────────────────────────────────
    df["ret"] = df["close"].pct_change()
    df["hl"]  = (df["high"] - df["low"])  / df["close"].clip(lower=1e-9)
    df["oc"]  = (df["close"] - df["open"]) / df["open"].clip(lower=1e-9)

    body = (df["close"] - df["open"]).abs()
    rng  = (df["high"]  - df["low"]).clip(lower=1e-9)
    df["body_ratio"] = body / rng
    df["upper_wick"] = (df["high"] - df[["close","open"]].max(axis=1)) / rng
    df["lower_wick"] = (df[["close","open"]].min(axis=1) - df["low"])  / rng
    df["is_bullish"] = (df["close"] > df["open"]).astype(float)

    # ── Trend / EMA ───────────────────────────────────────────
    ema9  = df["close"].ewm(span=9,  adjust=False).mean()
    ema21 = df["close"].ewm(span=21, adjust=False).mean()
    ma3   = df["close"].rolling(3).mean()
    ma5   = df["close"].rolling(5).mean()
    ma10  = df["close"].rolling(10).mean()

    df["ema_diff"]      = (ema9 - ema21) / df["close"].clip(lower=1e-9)
    df["price_vs_ema9"] = (df["close"] - ema9) / ema9.clip(lower=1e-9)
    df["ma_diff"]       = (ma3 - ma5) / df["close"].clip(lower=1e-9)
    df["price_vs_ma10"] = (df["close"] - ma10) / ma10.clip(lower=1e-9)
    df["ema_slope"]     = ema9.diff(3) / (df["close"].clip(lower=1e-9) * 3)

    # Double EMA (DEMA) — more responsive than single EMA
    ema9_2      = ema9.ewm(span=9, adjust=False).mean()
    dema        = 2 * ema9 - ema9_2
    df["dema_diff"] = (df["close"] - dema) / df["close"].clip(lower=1e-9)

    # ── RSI-14 ────────────────────────────────────────────────
    delta    = df["close"].diff()
    avg_gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rs       = avg_gain / (avg_loss + 1e-9)
    df["rsi_norm"] = (100 - 100 / (1 + rs) - 50) / 50

    # ── MACD ─────────────────────────────────────────────────
    ema12     = df["close"].ewm(span=12, adjust=False).mean()
    ema26     = df["close"].ewm(span=26, adjust=False).mean()
    macd_line = (ema12 - ema26) / df["close"].clip(lower=1e-9)
    macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
    df["macd"]      = macd_line
    df["macd_hist"] = macd_line - macd_sig

    # ── ROC / Momentum ────────────────────────────────────────
    df["roc5"]           = df["close"].pct_change(5)
    df["roc3"]           = df["close"].pct_change(3)
    df["momentum3_norm"] = df["close"].pct_change(3)

    # ── Stochastic %K (14-period, normalised -1→1) ────────────
    low14  = df["low"].rolling(14, min_periods=1).min()
    high14 = df["high"].rolling(14, min_periods=1).max()
    df["stoch_k_norm"] = (
        (df["close"] - low14) / (high14 - low14 + 1e-9) - 0.5
    ) * 2

    # ── CCI-14 ────────────────────────────────────────────────
    tp     = (df["high"] + df["low"] + df["close"]) / 3
    tp_ma  = tp.rolling(14, min_periods=1).mean()
    tp_md  = tp.rolling(14, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci_norm"] = ((tp - tp_ma) / (0.015 * tp_md + 1e-9)) / 200.0

    # ── True Range (shared by ATR, Keltner, ADX) ─────────────
    hl_r = df["high"] - df["low"]
    hc   = (df["high"] - df["close"].shift()).abs()
    lc   = (df["low"]  - df["close"].shift()).abs()
    tr   = pd.concat([hl_r, hc, lc], axis=1).max(axis=1)

    # ── ATR-10 ────────────────────────────────────────────────
    atr10 = tr.ewm(span=10, adjust=False).mean()
    df["atr"]        = atr10 / df["close"].clip(lower=1e-9)
    df["volatility"] = df["ret"].rolling(5).std()

    # ── Bollinger Bands ───────────────────────────────────────
    bb_mid = df["close"].rolling(10).mean()
    bb_std = df["close"].rolling(10).std().clip(lower=1e-9)
    bb_up  = bb_mid + 2 * bb_std
    bb_lo  = bb_mid - 2 * bb_std
    df["bb_pct"]   = (df["close"] - bb_lo) / (bb_up - bb_lo + 1e-9)
    df["bb_width"] = (bb_up - bb_lo) / bb_mid.clip(lower=1e-9)

    # ── Keltner Channel ───────────────────────────────────────
    kc_basis = df["close"].ewm(span=20, adjust=False).mean()
    kc_range = tr.ewm(span=10, adjust=False).mean() * 2.0
    kc_upper = kc_basis + kc_range
    kc_lower = kc_basis - kc_range
    df["keltner_pct"] = (df["close"] - kc_lower) / (
        kc_upper - kc_lower + 1e-9)

    # ── Volume ────────────────────────────────────────────────
    vm3  = df["volume"].rolling(3).mean()
    vm10 = df["volume"].rolling(10).mean()
    df["vol_ratio"] = df["volume"] / (vm3  + 1e-9)
    df["vol_trend"] = (vm3 - vm10) / (vm10 + 1e-9)

    # ── ADX / Directional Movement (14-period) ─────────────────
    up_move   = df["high"].diff()
    down_move = -df["low"].diff()
    dm_plus   = np.where((up_move > down_move)   & (up_move   > 0),
                         up_move,   0.0)
    dm_minus  = np.where((down_move > up_move)   & (down_move > 0),
                         down_move, 0.0)

    adx_period = 14
    atr14      = tr.ewm(span=adx_period, adjust=False).mean()
    di_plus    = (100 * pd.Series(dm_plus,  index=df.index)
                  .ewm(span=adx_period, adjust=False).mean()
                  / (atr14 + 1e-9))
    di_minus   = (100 * pd.Series(dm_minus, index=df.index)
                  .ewm(span=adx_period, adjust=False).mean()
                  / (atr14 + 1e-9))
    di_diff    = (di_plus - di_minus).abs()
    di_sum     = (di_plus + di_minus).clip(lower=1e-9)
    dx         = 100 * di_diff / di_sum
    adx        = dx.ewm(span=adx_period, adjust=False).mean()

    df["adx_norm"]     = adx     / 100.0
    df["di_plus_norm"] = di_plus / 100.0
    df["di_minus_norm"]= di_minus/ 100.0

    # ── Lagged returns ────────────────────────────────────────
    for lag in range(1, 6):
        df[f"ret_lag{lag}"] = df["ret"].shift(lag)

    # ── Intraday seasonality ──────────────────────────────────
    try:
        hours = (pd.to_datetime(df["ts"], unit="s")
                   .dt.tz_localize("UTC")
                   .dt.tz_convert("Africa/Nairobi")
                   .dt.hour)
    except Exception:
        hours = pd.to_datetime(df["ts"], unit="s").dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    # ── VMD-inspired decomposition ────────────────────────────
    prices_arr       = df["close"].values.astype(np.float64)
    trend, residual  = _vmd_decompose(prices_arr, window=10)

    df["trend_component"]   = (trend - prices_arr) / (prices_arr + 1e-9)
    df["residual_component"]= residual / (prices_arr + 1e-9)
    df["trend_slope"]       = (
        pd.Series(trend).diff(3) / (prices_arr + 1e-9)).values
    df["residual_energy"]   = (
        pd.Series(residual).rolling(5).std() / (prices_arr + 1e-9)).values

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    if len(df) < 1:
        return None
    return df[FEATURE_COLS]


# ============================================================
# PYTORCH MODEL DEFINITIONS
# ============================================================

class BiLSTMPredictor(nn.Module):
    """Bidirectional stacked LSTM — captures temporal patterns in both directions."""
    def __init__(self, n_feat=N_FEATURES, hidden=LSTM_HIDDEN,
                 layers=LSTM_LAYERS, drop=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers,
                            batch_first=True, dropout=drop, bidirectional=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, 64),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class AttentionGRUPredictor(nn.Module):
    """TFT-inspired: linear projection → GRU encoder → multi-head attention
    → residual + FFN → MLP head on the last timestep."""
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
            nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden * 2, hidden),
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.head  = nn.Sequential(
            nn.Linear(hidden, 32), nn.GELU(),
            nn.Dropout(drop), nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = self.proj(x)
        g, _ = self.gru(x)
        a, _ = self.attn(g, g, g)
        x2   = self.norm1(g + a)
        x3   = self.norm2(x2 + self.ffn(x2))
        return self.head(x3[:, -1, :])


class _TCNResBlock(nn.Module):
    """Single TCN residual block with causal (padding + chomp) convolutions."""
    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int, dilation: int, drop: float):
        super().__init__()
        self._pad  = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel,
                               dilation=dilation, padding=self._pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel,
                               dilation=dilation, padding=self._pad)
        self.drop  = nn.Dropout(drop)
        self.skip  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def _causal(self, x: torch.Tensor, conv) -> torch.Tensor:
        o = conv(x)
        return o[:, :, :-self._pad].contiguous() if self._pad > 0 else o

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.drop(F.gelu(self._causal(x, self.conv1)))
        o = self.drop(F.gelu(self._causal(o, self.conv2)))
        r = x if self.skip is None else self.skip(x)
        return F.gelu(o + r)


class TCNPredictor(nn.Module):
    """
    Temporal Convolutional Network with exponentially dilated causal convolutions.
    Receptive field: sum(2*(k-1)*2^i for i in 0..levels-1) covers entire SEQ_LEN.
    Input : (B, SEQ_LEN, N_FEATURES)
    Output: (B, 2)  raw logits
    """
    def __init__(self, n_feat=N_FEATURES, hidden=TCN_HIDDEN,
                 levels=TCN_LEVELS, kernel=TCN_KERNEL, drop=DROPOUT):
        super().__init__()
        self.proj   = nn.Conv1d(n_feat, hidden, 1)
        self.blocks = nn.ModuleList([
            _TCNResBlock(hidden, hidden, kernel, 2 ** i, drop)
            for i in range(levels)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.GELU(),
            nn.Dropout(drop), nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) → project → (B, H, T)
        x = self.proj(x.transpose(1, 2))
        for block in self.blocks:
            x = block(x)
        return self.head(x[:, :, -1])    # last timestep


# ============================================================
# ADAPTIVE ENSEMBLE  (6 models)
# ============================================================
class AdaptiveEnsemble:
    """
    Six-model soft-voting ensemble with adaptive weight calculation.

    Models  : BiLSTM · AttentionGRU · TCN · XGBoost · RandomForest · LightGBM
    Weights : rolling accuracy over last ADAPTIVE_WINDOW resolved UP/DOWN preds.
    HOLD    : never counted in wins/losses; never used to update weights.
    """

    def __init__(self):
        self.scaler:      Optional[StandardScaler]         = None
        self.bilstm:      Optional[BiLSTMPredictor]        = None
        self.attn_gru:    Optional[AttentionGRUPredictor]  = None
        self.tcn:         Optional[TCNPredictor]           = None
        self.xgb_clf:     Optional[xgb.XGBClassifier]     = None
        self.rf_clf:      Optional[RandomForestClassifier] = None
        self.lgb_clf                                       = None
        self.result_history: List[List[Tuple[int, int]]] = [[] for _ in range(6)]
        self.weights: List[float] = list(BASE_WEIGHTS)

    def _available(self) -> List[bool]:
        return [
            self.bilstm   is not None,
            self.attn_gru is not None,
            self.tcn      is not None,
            self.xgb_clf  is not None,
            self.rf_clf   is not None,
            self.lgb_clf  is not None,
        ]

    def update_adaptive_weights(self) -> None:
        avail = self._available()
        min_hist = min(
            (len(self.result_history[i]) for i, a in enumerate(avail) if a),
            default=0)
        if min_hist < ADAPTIVE_WINDOW:
            total = sum(BASE_WEIGHTS[i] for i, a in enumerate(avail) if a) or 1e-9
            self.weights = [
                BASE_WEIGHTS[i] / total if avail[i] else 0.0
                for i in range(6)]
        else:
            accs = []
            for i, a in enumerate(avail):
                if not a:
                    accs.append(0.0)
                    continue
                recent = self.result_history[i][-ADAPTIVE_WINDOW:]
                accs.append(sum(p == q for p, q in recent) / ADAPTIVE_WINDOW)
            total = sum(accs) or 1e-9
            self.weights = [acc / total for acc in accs]
        logger.info(
            f"Weights → BiLSTM:{self.weights[0]:.2f} "
            f"AttnGRU:{self.weights[1]:.2f} TCN:{self.weights[2]:.2f} "
            f"XGB:{self.weights[3]:.2f} RF:{self.weights[4]:.2f} "
            f"LGB:{self.weights[5]:.2f}")

    def record_result(self, per_model_preds: List[Optional[int]],
                      actual_class: int) -> None:
        for i, pred in enumerate(per_model_preds):
            if pred is not None:
                self.result_history[i].append((pred, actual_class))
                if len(self.result_history[i]) > ADAPTIVE_WINDOW * 3:
                    self.result_history[i] = (
                        self.result_history[i][-ADAPTIVE_WINDOW:])
        self.update_adaptive_weights()

    def predict_proba(self, X_seq: np.ndarray,
                      X_flat: np.ndarray
                      ) -> Tuple[np.ndarray, List[Optional[int]]]:
        probs:   List[np.ndarray]    = []
        used_w:  List[float]         = []
        pmp:     List[Optional[int]] = [None] * 6

        def _add(idx: int, p: Optional[np.ndarray]) -> None:
            if p is not None:
                probs.append(p)
                used_w.append(self.weights[idx])
                pmp[idx] = int(np.argmax(p[0]))

        # Deep models
        if self.bilstm:
            _add(0, _torch_infer(self.bilstm,   X_seq))
        if self.attn_gru:
            _add(1, _torch_infer(self.attn_gru, X_seq))
        if self.tcn:
            _add(2, _torch_infer(self.tcn,      X_seq))

        # Tree models
        for idx, clf in [(3, self.xgb_clf),
                         (4, self.rf_clf),
                         (5, self.lgb_clf)]:
            if clf is not None:
                try:
                    _add(idx, clf.predict_proba(X_flat))
                except Exception as e:
                    logger.error(f"Model[{idx}] infer: {e}")

        if not probs:
            return np.array([[0.5, 0.5]]), pmp

        total    = sum(used_w) or 1e-9
        combined = sum(p * w for p, w in zip(probs, used_w)) / total
        return combined, pmp


def _torch_infer(model: nn.Module,
                 X_seq: np.ndarray) -> Optional[np.ndarray]:
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
def _train_torch(model: nn.Module,
                 X: np.ndarray, y: np.ndarray) -> nn.Module:
    X_t    = torch.FloatTensor(X)
    y_t    = torch.LongTensor(y)
    split  = max(int(len(X_t) * 0.8), 1)
    tr_dl  = DataLoader(TensorDataset(X_t[:split], y_t[:split]),
                        batch_size=TRAIN_BATCH, shuffle=True, drop_last=False)
    va_dl  = DataLoader(TensorDataset(X_t[split:], y_t[split:]),
                        batch_size=TRAIN_BATCH, drop_last=False)

    opt      = torch.optim.AdamW(model.parameters(),
                                  lr=TRAIN_LR, weight_decay=1e-4)
    sched    = torch.optim.lr_scheduler.ReduceLROnPlateau(
                   opt, patience=5, factor=0.5, min_lr=1e-5)
    crit     = nn.CrossEntropyLoss()
    best     = float("inf")
    best_sd  = None
    wait     = 0

    for epoch in range(TRAIN_EPOCHS):
        model.train()
        for xb, yb in tr_dl:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if len(va_dl.dataset) == 0:
            break

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in va_dl:
                vl += crit(model(xb), yb).item()
        vl /= max(len(va_dl), 1)
        sched.step(vl)

        if vl < best - 1e-6:
            best    = vl
            best_sd = {k: t.clone() for k, t in model.state_dict().items()}
            wait    = 0
        else:
            wait += 1
            if wait >= TRAIN_PATIENCE:
                logger.info(f"  Early-stop @ epoch {epoch+1}, val_loss={best:.4f}")
                break

    if best_sd:
        model.load_state_dict(best_sd)
    model.eval()
    return model


def _build_sequences(feat: np.ndarray, labels: np.ndarray,
                     seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
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
        logger.warning(f"Too few candles ({n}) — need {MIN_CANDLES_TRAIN + 2}")
        return

    try:
        feat_df = make_features(candles[:-1])
        if feat_df is None or len(feat_df) == 0:
            return

        nf     = len(feat_df)
        offset = (n - 1) - nf
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

        split  = max(int(len(raw) * 0.8), 1)
        scaler = StandardScaler()
        scaler.fit(raw[:split])
        scaled = scaler.transform(raw)

        # Recency-weighted sample weights (recent candles matter more)
        ns = len(y_arr)
        sw = np.array([0.97 ** (ns - 1 - i) for i in range(ns)],
                      dtype=np.float32)

        # Carry over adaptive history from existing ensemble
        with model_lock:
            old_ens = model

        ens        = AdaptiveEnsemble()
        ens.scaler = scaler
        if (old_ens is not None
                and isinstance(old_ens, AdaptiveEnsemble)
                and hasattr(old_ens, "result_history")
                and len(old_ens.result_history) == 6):
            ens.result_history = old_ens.result_history
            ens.weights        = old_ens.weights

        # ── XGBoost ──────────────────────────────────────────
        logger.info("  Training XGBoost …")
        xgb_mdl = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.75,
            min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", verbosity=0,
        )
        xgb_mdl.fit(scaled, y_arr, sample_weight=sw)
        ens.xgb_clf = xgb_mdl
        logger.info("  XGBoost ✓")

        # ── RandomForest ─────────────────────────────────────
        logger.info("  Training RandomForest …")
        rf_mdl = RandomForestClassifier(
            n_estimators=300, max_depth=8,
            min_samples_leaf=3, max_features="sqrt",
            n_jobs=2, random_state=42,
        )
        rf_mdl.fit(scaled, y_arr, sample_weight=sw)
        ens.rf_clf = rf_mdl
        logger.info("  RandomForest ✓")

        # ── LightGBM ─────────────────────────────────────────
        if HAS_LGB:
            logger.info("  Training LightGBM …")
            lgb_mdl = lgb.LGBMClassifier(
                n_estimators=400, num_leaves=63, max_depth=7,
                learning_rate=0.04, subsample=0.8, colsample_bytree=0.75,
                min_child_samples=5, reg_alpha=0.1, reg_lambda=1.0,
                verbosity=-1, n_jobs=2,
            )
            lgb_mdl.fit(scaled, y_arr, sample_weight=sw)
            ens.lgb_clf = lgb_mdl
            logger.info("  LightGBM ✓")
        else:
            logger.info("  LightGBM skipped (not installed)")

        # ── Deep sequence models ──────────────────────────────
        if n >= MIN_CANDLES_DEEP:
            X_seq, y_seq = _build_sequences(scaled, y_arr, SEQ_LEN)
            logger.info(
                f"  Sequence dataset: {len(X_seq)} × {SEQ_LEN} × {N_FEATURES}")

            if len(X_seq) >= 20:
                logger.info("  Training BiLSTM …")
                bilstm = BiLSTMPredictor(N_FEATURES, LSTM_HIDDEN,
                                         LSTM_LAYERS, DROPOUT)
                ens.bilstm = _train_torch(bilstm, X_seq, y_seq)
                logger.info("  BiLSTM ✓")

                logger.info("  Training AttentionGRU …")
                attn_gru = AttentionGRUPredictor(
                    N_FEATURES, GRU_HIDDEN, GRU_LAYERS, ATTN_HEADS, DROPOUT)
                ens.attn_gru = _train_torch(attn_gru, X_seq, y_seq)
                logger.info("  AttentionGRU ✓")

                logger.info("  Training TCN …")
                tcn = TCNPredictor(N_FEATURES, TCN_HIDDEN,
                                   TCN_LEVELS, TCN_KERNEL, DROPOUT)
                ens.tcn = _train_torch(tcn, X_seq, y_seq)
                logger.info("  TCN ✓")
            else:
                logger.warning(
                    f"  Not enough sequences ({len(X_seq)}) for deep models")
        else:
            logger.info(
                f"  {n} candles < {MIN_CANDLES_DEEP} — tree-only this cycle")

        ens.update_adaptive_weights()

        with model_lock:
            model = ens

        save_model(ens, wins, losses)
        logger.info("  Adaptive ensemble saved ✓")

    except Exception as exc:
        logger.exception(f"Training error: {exc}")


# ============================================================
# PREDICTION
# ============================================================
def predict_from_candles(candles: List[dict]) -> dict:
    global _last_per_model_preds
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

        if len(scaled) >= SEQ_LEN:
            seq = scaled[-SEQ_LEN:]
        else:
            pad = np.zeros((SEQ_LEN - len(scaled), N_FEATURES),
                           dtype=np.float32)
            seq = np.vstack([pad, scaled])

        X_seq  = seq[np.newaxis, :, :]
        X_flat = scaled[[-1], :]

        prob, pmp = ens.predict_proba(X_seq, X_flat)
        up_p = float(prob[0, 1])
        conf = int(round(max(up_p, 1 - up_p) * 100))

        if   up_p >= SIGNAL_UP_THRESH:   signal = "UP"
        elif up_p <= SIGNAL_DOWN_THRESH: signal = "DOWN"
        else:                            signal = "HOLD"

        with _last_per_model_lock:
            _last_per_model_preds = pmp

        now  = datetime.now(TZ)
        nm   = ((now.minute // 5) + 1) * 5
        ns_  = now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=nm)
        ne_  = ns_ + timedelta(minutes=5)
        nxt  = f"{ns_.strftime('%H:%M')}-{ne_.strftime('%H:%M')}"

        if signal == "HOLD":
            conf = 0

        return {"signal": signal, "confidence": conf, "next_window": nxt,
                "per_model": pmp}

    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        return default


# ============================================================
# CANDLE BUILDING  (v2 duplicate-free sliding window)
# ============================================================
def _seed_pending_row_locked(window_start_ts: float) -> None:
    """Seed a pending row for this window ONLY if one doesn't already exist."""
    ws_str, we_str = get_window_time_range(window_start_ts)
    for row in history_rows:
        if row.get("window_start") == ws_str:
            return   # already present

    snap = list(completed_candles)
    pred = (predict_from_candles(snap) if snap
            else {"signal": "HOLD", "confidence": 0})
    rec = {
        "window_start": ws_str, "window_end": we_str,
        "window": f"{ws_str}-{we_str}",
        "predicted": pred["signal"], "confidence": pred["confidence"],
        "act_open": 0.0, "act_close": 0.0,
        "actual": "⏳", "result": "⏳",
    }
    history_rows.insert(0, rec)
    save_prediction(rec)

    # Keep max 5 unique windows, newest first
    seen, unique = set(), []
    for row in history_rows:
        k = row["window_start"]
        if k not in seen:
            seen.add(k)
            unique.append(row)
    history_rows[:] = unique[:5]


def _new_candle(ts: float, price: float) -> dict:
    return {"ts": ts, "open": price, "high": price,
            "low": price, "close": price, "volume": 0.0}


def _close_current_window_locked() -> None:
    global live_candle, candles_since_retrain, wins, losses

    candle = dict(live_candle)
    live_candle = None
    completed_candles.append(candle)
    candles_since_retrain += 1
    save_candle(candle)

    ws_str, we_str = get_window_time_range(candle["ts"])

    # Resolve the pending row for this window
    for row in history_rows:
        if row.get("actual") == "⏳" and row.get("window_start") == ws_str:
            actual = "UP" if candle["close"] > candle["open"] else "DOWN"
            row.update(actual=actual,
                       act_open=candle["open"], act_close=candle["close"],
                       window_end=we_str, window=f"{ws_str}-{we_str}")
            predicted = row["predicted"]

            if predicted == "HOLD":
                row["result"] = "—"
                update_prediction_result(ws_str, we_str, actual,
                                         candle["open"], candle["close"], "—")
                logger.info(f"Window {ws_str}: Pred=HOLD (not evaluated)")
            else:
                if predicted == actual:
                    row["result"] = "✅"; wins += 1
                else:
                    row["result"] = "❌"; losses += 1

                update_prediction_result(ws_str, we_str, actual,
                                         candle["open"], candle["close"],
                                         row["result"])
                save_stats(wins, losses)

                actual_class = 1 if actual == "UP" else 0
                with _last_per_model_lock:
                    pmp = list(_last_per_model_preds)
                with model_lock:
                    ens = model
                if ens is not None and isinstance(ens, AdaptiveEnsemble):
                    ens.record_result(pmp, actual_class)

                logger.info(
                    f"Window {ws_str}: Pred={predicted} "
                    f"Act={actual} {row['result']}")
            break
    else:
        logger.warning(f"No pending row found for {ws_str}")

    # Retrain if due
    if candles_since_retrain >= RETRAIN_EVERY:
        candles_since_retrain = 0
        threading.Thread(
            target=train_model_from_candles,
            args=(list(completed_candles),), daemon=True).start()


def process_price_tick(price: float, trade_ts: float) -> None:
    global live_candle, live_window_start

    with state_lock:
        current_ws = int(trade_ts // WINDOW_SECONDS) * WINDOW_SECONDS

        if live_window_start is None:
            live_window_start = current_ws
            live_candle = _new_candle(live_window_start, price)
            _seed_pending_row_locked(live_window_start)
            return

        if current_ws > live_window_start:
            if live_candle is not None:
                _close_current_window_locked()
            live_window_start = current_ws
            live_candle = _new_candle(live_window_start, price)
            _seed_pending_row_locked(live_window_start)
            return

        if live_candle is not None:
            if trade_ts >= live_candle["ts"] + WINDOW_SECONDS:
                _close_current_window_locked()
                live_window_start = current_ws
                live_candle = _new_candle(live_window_start, price)
                _seed_pending_row_locked(live_window_start)
            else:
                live_candle["high"]  = max(live_candle["high"],  price)
                live_candle["low"]   = min(live_candle["low"],   price)
                live_candle["close"] = price


# ============================================================
# BINANCE WEBSOCKET  (BTC/USDT real‑time trades)
# ============================================================
def _binance_thread() -> None:
    urls  = [
        "wss://stream.binance.com:9443/ws/btcusdt@aggTrade",
        "wss://stream.binance.com:443/ws/btcusdt@aggTrade",
    ]
    retry = 2

    while True:
        for url in urls:
            logger.info(f"Connecting to Binance WS: {url}")
            try:
                def on_open(w):
                    nonlocal retry
                    retry = 2
                    logger.info("Binance WS connected (BTC/USDT)")

                def on_message(w, raw):
                    try:
                        d = json.loads(raw)
                        if d.get("e") == "aggTrade":
                            price = float(d.get("p", 0))
                            ts    = float(d.get("T", time.time() * 1000)) / 1000.0
                            if price > 0:
                                price_queue.put_nowait(
                                    {"price": price, "ts": ts})
                    except Exception:
                        pass

                ws_client.WebSocketApp(
                    url,
                    on_open=on_open,
                    on_message=on_message,
                    on_error=lambda w, e: logger.error(f"Binance WS err: {e}"),
                    on_close=lambda w, c, m: logger.warning("Binance WS closed"),
                ).run_forever(ping_interval=20, ping_timeout=10)

            except Exception as e:
                logger.error(f"Binance thread ({url}): {e}")

        logger.warning(f"Reconnecting in {retry}s …")
        time.sleep(retry)
        retry = min(retry * 2, 60)


# ============================================================
# PRICE PROCESSOR
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
# BROADCAST
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

    pred       = (predict_from_candles(snap) if snap
                  else {"signal": "HOLD", "confidence": 0, "next_window": ""})
    ohlc       = ({k: round(lc[k], 2)
                   for k in ("open", "high", "low", "close")} if lc else {})
    live_price = (lc["close"] if lc
                  else (snap[-1]["close"] if snap else 0.0))

    with model_lock:
        model_ready = model is not None

    return {
        "price":       round(live_price, 2),
        "signal":      pred["signal"],
        "confidence":  pred["confidence"],
        "next_window": pred["next_window"],
        "wins":  w, "losses": l,
        "ohlc":  ohlc,
        "table": rows,
        "model_ready": model_ready,
    }


# ============================================================
# HTML FRONTEND  (Bitcoin theme, 2 decimal places)
# ============================================================
HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=yes">
<title>Bitcoin 5-Min Predictor</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#080C14;color:#C8D8EF;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Roboto','Helvetica Neue',sans-serif;min-height:100vh}
  .container{max-width:1280px;margin:0 auto;padding:12px}
  .header{margin-bottom:12px;padding:8px 0;text-align:center}
  .header h1{font-size:1.4rem;font-weight:700;color:#F7931A;letter-spacing:.03em}
  .header p{font-size:.72rem;color:#4A6080;margin-top:3px}
  .status-bar{display:flex;align-items:center;gap:10px;margin-bottom:12px;font-size:.8rem;color:#4A6080;flex-wrap:wrap}
  .status-bar #clock-gmt3{color:#fff;margin-left:auto}
  .dot{width:8px;height:8px;border-radius:50%;display:inline-block;flex-shrink:0}
  .dot-ok{background:#00E5A0;box-shadow:0 0 6px #00E5A0}
  .dot-bad{background:#FF4560}
  .dot-wait{background:#F7931A;animation:pulse 1.2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
  .main-grid{display:grid;grid-template-columns:1fr 340px;gap:12px;margin-bottom:12px;align-items:stretch}
  #tv-chart{background:#0D1421;border-radius:10px;border:1px solid #1E2D45;height:380px;overflow:hidden}
  .sidebar{display:flex;flex-direction:column;gap:12px}
  .card{background:#0D1421;border:1px solid #1E2D45;border-radius:10px;padding:14px}
  .card-title{font-size:.85rem;color:#4A6080;text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px}
  .price{font-size:2rem;font-weight:700;color:#F7931A}
  .pchange{font-size:.82rem;margin-left:6px}
  .pred-row{display:flex;align-items:center;gap:12px;margin-top:4px}
  .pred-arrow{font-size:2.4rem;line-height:1}
  .pred-dir{font-size:1.3rem;font-weight:700}
  .conf-bar{background:#1E2D45;border-radius:4px;height:6px;margin-top:8px;overflow:hidden}
  .conf-fill{height:100%;width:0%;transition:width .4s ease}
  .countdown{display:flex;align-items:center;gap:14px}
  .cd-ring{width:58px;height:58px;transform:rotate(-90deg)}
  .cd-text{font-size:1.8rem;font-weight:700;color:#F7931A}
  .bottom-row{display:grid;gap:12px;grid-template-columns:1fr 1fr;margin-bottom:12px}
  .ohlc-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:6px}
  .ohlc-cell{background:#0F1623;border-radius:6px;padding:7px 10px;border:1px solid #1E2D45}
  .ohlc-cell .lbl{font-size:.6rem;color:#4A6080}
  .ohlc-cell .val{font-size:.88rem;font-weight:700;margin-top:2px}
  .perf-row{display:flex;gap:10px;margin-top:8px}
  .perf-stat{flex:1;text-align:center;background:#0F1623;border-radius:7px;padding:8px;border:1px solid #1E2D45}
  .perf-num{font-size:1.35rem;font-weight:700}
  .perf-lbl{font-size:.6rem;color:#4A6080;margin-top:2px}
  .table-wrapper{overflow-x:auto;margin-top:10px}
  table{width:100%;border-collapse:collapse;font-size:.75rem}
  th,td{padding:8px;text-align:left;border-bottom:1px solid #1E2D45}
  th{color:#4A6080;font-weight:600}
  .up{color:#00E5A0}.down{color:#FF4560}.hold-clr{color:#4A6080}
  .disclaimer{background:#0F1623;border-radius:8px;padding:10px;font-size:.7rem;text-align:center;color:#FACC15;border:1px solid rgba(250,204,21,.13);margin-top:12px}
  .model-badge{display:inline-block;background:#0F1623;border:1px solid #1E2D45;border-radius:4px;padding:2px 6px;font-size:.6rem;color:#4A6080;margin-top:6px;margin-right:4px}
  .flash-up{animation:flashUp .4s ease}.flash-dn{animation:flashDn .4s ease}
  @keyframes flashUp{0%,100%{color:#C8D8EF}50%{color:#00E5A0}}
  @keyframes flashDn{0%,100%{color:#C8D8EF}50%{color:#FF4560}}
  @media(max-width:900px){
    .main-grid{grid-template-columns:1fr}
    #tv-chart{height:300px}
    .sidebar{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
    .bottom-row{grid-template-columns:1fr}
  }
  @media(max-width:600px){
    .container{padding:8px}
    .header h1{font-size:1.1rem}
    .status-bar{font-size:.7rem;gap:6px}
    .sidebar{grid-template-columns:1fr;gap:10px}
    #tv-chart{height:240px}
    .card{padding:10px}
    .price{font-size:1.5rem}
    .pred-arrow{font-size:1.8rem}
    .cd-text{font-size:1.4rem}
    .cd-ring{width:48px;height:48px}
    table{font-size:.65rem}
    th,td{padding:6px 4px}
  }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>₿ Bitcoin Price Trend Predictor</h1>
    <p>BTC/USD &middot; 5-Minute Windows &middot; BiLSTM + AttentionGRU + TCN + XGBoost + RF + LightGBM</p>
  </div>
  <div class="status-bar">
    <div class="dot dot-wait" id="ws-dot"></div>
    <span id="ws-txt">Connecting...</span>
    <span id="model-txt" style="color:#4A6080">Model: loading…</span>
    <span id="clock-gmt3">--:--:-- GMT+3</span>
  </div>

  <div class="main-grid">
    <div id="tv-chart"><div id="tv-widget" style="width:100%;height:100%"></div></div>
    <div class="sidebar">
      <div class="card">
        <div class="card-title">Live BTC / USD</div>
        <div>
          <span class="price" id="price-val">$---.--</span>
          <span class="pchange" id="price-change">--</span>
        </div>
        <div style="margin-top:6px;font-size:.65rem;color:#4A6080">Binance BTC/USDT &middot; Real‑time trades</div>
      </div>
      <div class="card">
        <div class="card-title">Next 5-Min Prediction</div>
        <div class="pred-row">
          <span class="pred-arrow" id="pred-arrow" style="color:#4A6080">&#9670;</span>
          <span class="pred-dir"   id="pred-dir"   style="color:#4A6080">HOLD</span>
          <span id="conf-pct" style="font-size:.85rem;color:#4A6080">--</span>
        </div>
        <div class="conf-bar"><div class="conf-fill" id="conf-bar"></div></div>
        <div id="pred-window" style="margin-top:6px;font-size:.7rem;color:#4A6080"></div>
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
      <div class="card-title">Current 5-Min Window</div>
      <div class="ohlc-grid">
        <div class="ohlc-cell"><div class="lbl">Open</div> <div class="val"       id="o-open">--</div></div>
        <div class="ohlc-cell"><div class="lbl">High</div> <div class="val up"    id="o-high">--</div></div>
        <div class="ohlc-cell"><div class="lbl">Low</div>  <div class="val down"  id="o-low">--</div></div>
        <div class="ohlc-cell"><div class="lbl">Close</div><div class="val"       id="o-close">--</div></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Performance <span style="font-size:.65rem;color:#4A6080">(UP/DOWN only — HOLD excluded)</span></div>
      <div class="perf-row">
        <div class="perf-stat"><div class="perf-num up"   id="p-wins">0</div>  <div class="perf-lbl">Wins</div></div>
        <div class="perf-stat"><div class="perf-num down" id="p-losses">0</div><div class="perf-lbl">Losses</div></div>
        <div class="perf-stat"><div class="perf-num" id="p-acc" style="color:#F7931A">--</div><div class="perf-lbl">Accuracy</div></div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Last 5 Predictions</div>
    <div class="table-wrapper">
      <table>
        <thead>
          <tr><th>Window</th><th>Prediction</th><th>Conf</th>
              <th>Act.Open</th><th>Act.Close</th><th>Actual</th><th>Result</th></tr>
        </thead>
        <tbody id="history-body">
          <tr><td colspan="7" style="text-align:center;padding:20px">Waiting for data…</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <div class="disclaimer">⚠️ For educational purposes only. Past accuracy does not guarantee future results. Not financial advice.</div>
</div>

<script src="https://s3.tradingview.com/tv.js"></script>
<script>
  new TradingView.widget({
    container_id:'tv-widget', symbol:'BITSTAMP:BTCUSD', interval:'5',
    theme:'dark', style:'1', locale:'en', toolbar_bg:'#080C14',
    enable_publishing:false, autosize:true
  });

  let ws, firstPrice=null, prevPrice=null;

  function updateClock(){
    document.getElementById('clock-gmt3').textContent =
      new Date().toLocaleTimeString('en-US',{timeZone:'Africa/Nairobi',hour12:false})+' GMT+3';
  }
  updateClock(); setInterval(updateClock,1000);

  function updateCountdown(){
    const now=new Date(), gmt3=new Date(now.getTime()+3*3600000);
    const rem=Math.max(0,(5-(gmt3.getUTCMinutes()%5))*60-gmt3.getUTCSeconds());
    document.getElementById('cd-val').textContent=
      `${Math.floor(rem/60)}:${String(rem%60).padStart(2,'0')}`;
    document.getElementById('cd-ring').setAttribute(
      'stroke-dashoffset', String(201*(1-rem/300)));
  }
  updateCountdown(); setInterval(updateCountdown,1000);

  function fmt(n,dp=2){
    if(n==null||n===0)return'--';
    return'$'+Number(n).toLocaleString('en-US',{minimumFractionDigits:dp,maximumFractionDigits:dp});
  }

  function connect(){
    const url=(location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws';
    ws=new WebSocket(url);
    ws.onopen=()=>{
      document.getElementById('ws-dot').className='dot dot-ok';
      document.getElementById('ws-txt').textContent='Live';
    };
    ws.onclose=()=>{
      document.getElementById('ws-dot').className='dot dot-bad';
      document.getElementById('ws-txt').textContent='Disconnected';
      setTimeout(connect,3000);
    };
    ws.onerror=()=>{ document.getElementById('ws-dot').className='dot dot-bad'; };
    ws.onmessage=(e)=>{ try{ handleState(JSON.parse(e.data)); }catch(err){} };
  }

  function handleState(d){
    if(d.type==='ping')return;

    // Price
    const p=d.price;
    if(p&&p>0){
      const el=document.getElementById('price-val');
      if(prevPrice!==null){
        el.className='price '+(p>prevPrice?'flash-up':p<prevPrice?'flash-dn':'');
        setTimeout(()=>{el.className='price';},400);
      }
      el.textContent=fmt(p,2); prevPrice=p;
      if(firstPrice===null)firstPrice=p;
      const chg=p-firstPrice, pct=(chg/firstPrice*100).toFixed(2);
      const ce=document.getElementById('price-change');
      ce.textContent=(chg>=0?'+':'')+fmt(chg,2)+' ('+pct+'%)';
      ce.style.color=chg>=0?'#00E5A0':'#FF4560';
    }

    // Model status
    document.getElementById('model-txt').textContent=
      d.model_ready?'Model: ready':'Model: training…';

    // Prediction
    const sig=d.signal||'HOLD';
    const isUp=sig==='UP', isDn=sig==='DOWN', isHold=sig==='HOLD';
    const col=isUp?'#00E5A0':isDn?'#FF4560':'#4A6080';
    document.getElementById('pred-arrow').textContent=isUp?'▲':isDn?'▼':'◆';
    document.getElementById('pred-arrow').style.color=col;
    document.getElementById('pred-dir').textContent=isUp?'UP':isDn?'DOWN':'HOLD';
    document.getElementById('pred-dir').style.color=col;
    const ce2=document.getElementById('conf-pct');
    if(isHold){ce2.textContent='--';ce2.style.color='#4A6080';}
    else{ce2.textContent=d.confidence+'%';ce2.style.color=col;}
    document.getElementById('conf-bar').style.width=isHold?'0%':(d.confidence||0)+'%';
    document.getElementById('conf-bar').style.background=col;
    document.getElementById('pred-window').textContent=d.next_window?'Next: '+d.next_window:'';

    // OHLC
    if(d.ohlc){
      document.getElementById('o-open').textContent=fmt(d.ohlc.open,2);
      document.getElementById('o-high').textContent=fmt(d.ohlc.high,2);
      document.getElementById('o-low').textContent=fmt(d.ohlc.low,2);
      document.getElementById('o-close').textContent=fmt(d.ohlc.close,2);
    }

    // Performance
    const w=d.wins||0, l=d.losses||0, tot=w+l;
    document.getElementById('p-wins').textContent=w;
    document.getElementById('p-losses').textContent=l;
    document.getElementById('p-acc').textContent=tot?(w/tot*100).toFixed(1)+'%':'--';

    // Table
    const tbody=document.getElementById('history-body');
    if(d.table&&d.table.length){
      tbody.innerHTML=d.table.map(r=>{
        const isH=r.predicted==='HOLD';
        const pc=r.predicted==='UP'?'up':r.predicted==='DOWN'?'down':'hold-clr';
        const ac=r.actual==='UP'?'up':r.actual==='DOWN'?'down':'';
        const pt=r.predicted==='UP'?'▲ UP':r.predicted==='DOWN'?'▼ DOWN':'◆ HOLD';
        const at=r.actual==='⏳'?'--':r.actual==='UP'?'▲ UP':r.actual==='DOWN'?'▼ DOWN':r.actual;
        const ct=isH?'--':r.confidence+'%';
        const cc=isH?'#4A6080':'#F7931A';
        const rt=r.result==='⏳'?'⏳':r.result==='—'?'—':r.result;
        return `<tr>
          <td style="color:#4A6080">${r.window}</td>
          <td class="${pc}">${pt}</td>
          <td style="color:${cc}">${ct}</td>
          <td>${fmt(r.act_open,2)}</td>
          <td>${fmt(r.act_close,2)}</td>
          <td class="${ac}">${at}</td>
          <td>${rt}</td>
        </tr>`;
      }).join('');
    }else{
      tbody.innerHTML='<tr><td colspan="7" style="text-align:center;padding:20px">Waiting for data…</td></tr>';
    }
  }
  connect();
</script>
</body>
</html>"""


# ============================================================
# FASTAPI APPLICATION
# ============================================================
@asynccontextmanager
async def lifespan(application: FastAPI):
    global _main_loop, _clients_lock, model, wins, losses
    global completed_candles, history_rows, live_candle, live_window_start

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
        logger.info(f"Loaded ensemble — {wins}W / {losses}L")

    # Load and deduplicate saved predictions
    saved_preds = load_predictions(20)
    unique: dict = {}
    for p in saved_preds:
        k = p["window_start"]
        if k not in unique:
            unique[k] = p
    dedup = sorted(unique.values(),
                   key=lambda x: x["window_start"], reverse=True)
    with state_lock:
        history_rows = dedup[:5]
    logger.info(f"Loaded {len(history_rows)} unique predictions")

    saved_candles = load_candles(HISTORY_LIMIT)
    if saved_candles:
        with state_lock:
            for c in saved_candles:
                completed_candles.append(c)
        logger.info(f"Loaded {len(saved_candles)} candles")

    if model is None:
        logger.info("No saved model — seeding synthetic history and training …")
        synth = generate_synthetic_history(80)
        with state_lock:
            for c in synth:
                completed_candles.append(c)
                save_candle(c)
        threading.Thread(
            target=train_model_from_candles,
            args=(list(completed_candles),),
            daemon=True, name="init-train").start()
    else:
        logger.info(f"Existing ensemble ready ({len(completed_candles)} candles)")

    # Seed pending row for the current window if missing
    now_ts     = time.time()
    cur_ws     = int(now_ts // WINDOW_SECONDS) * WINDOW_SECONDS
    ws_str, _  = get_window_time_range(cur_ws)
    with state_lock:
        exists = any(r.get("window_start") == ws_str for r in history_rows)
        if not exists and completed_candles:
            _seed_pending_row_locked(cur_ws)
            logger.info(f"Seeded initial pending row for {ws_str}")

    save_stats(wins, losses)

    threading.Thread(target=price_processor_thread,
                     name="price-proc",  daemon=True).start()
    threading.Thread(target=_binance_thread,
                     name="binance-ws",  daemon=True).start()
    asyncio.create_task(_periodic_broadcast())

    lgb_note = "LightGBM" if HAS_LGB else "LightGBM (missing — install lightgbm)"
    logger.info("=" * 64)
    logger.info("Bitcoin Predictor v4.0 — Ultimate Ensemble — ready")
    logger.info(f"Models   : BiLSTM · AttentionGRU · TCN · XGBoost · RF · {lgb_note}")
    logger.info(f"Features : {N_FEATURES}  |  Seq len : {SEQ_LEN}  |  History : {HISTORY_LIMIT}")
    logger.info(f"HOLD zone: {SIGNAL_DOWN_THRESH} – {SIGNAL_UP_THRESH} (ultra-tight 2 %)")
    logger.info(f"Stats    : {wins}W / {losses}L")
    logger.info("=" * 64)
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
