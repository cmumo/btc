"""
Bitcoin 5-Minute Prediction Terminal  ·  v4.0  (Supabase)
==========================================================
Backend  : Supabase (PostgreSQL via REST API) — no SQLite
Models   : BiLSTM · AttentionGRU (TFT-inspired) · XGBoost · RandomForest · LightGBM
Ensemble : Adaptive soft-voting — weights shift toward the best-performing model
           using rolling accuracy over the last 20 resolved UP/DOWN predictions
HOLD     : Never counted in wins/losses/accuracy; shown as "--" confidence
Features : 31 engineered features including VMD-inspired decomposition
History  : Last 5 predictions always loaded from Supabase on startup
           so the table is never empty when a new visitor opens the page
"""

import asyncio
import base64
import json
import logging
import os
import random
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from queue import Empty, Queue
from typing import Deque, List, Optional, Tuple

import httpx
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

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# SUPABASE CONFIG
# ============================================================
SUPABASE_URL = os.environ.get(
    "SUPABASE_URL", "https://qmisavuhrtqzoctrhmiw.supabase.co")
SUPABASE_KEY = os.environ.get(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFtaXNhdnVocnRxem9jdHJobWl3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzcxODQwNTMsImV4cCI6MjA5Mjc2MDA1M30.NVdBqqsqt7KWkbyZQvvk14MSh2a6phSaLnnBh9M2fe8")

SB_HEADERS = {
    "apikey":        SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type":  "application/json",
}
REST = f"{SUPABASE_URL}/rest/v1"

# Single persistent HTTP client (connection-pooled, thread-safe for reads)
_http = httpx.Client(timeout=12.0)


def _sb_get(table: str, params: dict) -> list:
    try:
        r = _http.get(f"{REST}/{table}", headers=SB_HEADERS, params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"SB GET {table}: {e}")
        return []


def _sb_upsert(table: str, payload) -> bool:
    try:
        data = [payload] if isinstance(payload, dict) else payload
        hdrs = {**SB_HEADERS, "Prefer": "resolution=merge-duplicates"}
        r = _http.post(f"{REST}/{table}", headers=hdrs, content=json.dumps(data))
        r.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"SB UPSERT {table}: {e}")
        return False


def _sb_patch(table: str, match: dict, payload: dict) -> bool:
    try:
        params = {k: f"eq.{v}" for k, v in match.items()}
        r = _http.patch(f"{REST}/{table}", headers=SB_HEADERS,
                        params=params, content=json.dumps(payload))
        r.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"SB PATCH {table}: {e}")
        return False


# ============================================================
# CONSTANTS
# ============================================================
WINDOW_SECONDS     = 300
HISTORY_LIMIT      = 250
MIN_CANDLES_TRAIN  = 35
MIN_CANDLES_DEEP   = 60
RETRAIN_EVERY      = 3
SEQ_LEN            = 20
N_FEATURES         = 31
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
TZ                 = timezone(timedelta(hours=3))   # GMT+3

SIGNAL_UP_THRESH   = 0.60
SIGNAL_DOWN_THRESH = 0.40

# Base weights [bilstm, attn_gru, xgb, rf, lgb]
BASE_WEIGHTS: List[float] = [0.25, 0.25, 0.20, 0.15, 0.15]
ADAPTIVE_WINDOW = 20

torch.set_num_threads(2)

# ============================================================
# SUPABASE DB HELPERS
# ============================================================

def save_candle(candle: dict) -> None:
    _sb_upsert("candles", {
        "timestamp": candle["ts"],
        "open":  candle["open"],  "high":   candle["high"],
        "low":   candle["low"],   "close":  candle["close"],
        "volume": candle["volume"],
    })


def load_candles(limit: int = HISTORY_LIMIT) -> List[dict]:
    rows = _sb_get("candles", {
        "select": "timestamp,open,high,low,close,volume",
        "order":  "timestamp.desc",
        "limit":  limit,
    })
    return [{"ts": r["timestamp"], "open": r["open"], "high": r["high"],
             "low": r["low"], "close": r["close"], "volume": r["volume"]}
            for r in reversed(rows)]


def save_prediction(p: dict) -> None:
    _sb_upsert("predictions", {
        "window_start":     p["window_start"],
        "window_end":       p["window_end"],
        "predicted_signal": p["predicted"],
        "confidence":       p["confidence"],
        "actual_signal":    p.get("actual", "\u23f3"),
        "actual_open":      p.get("act_open", 0),
        "actual_close":     p.get("act_close", 0),
        "result":           p.get("result", "\u23f3"),
    })


def update_prediction_result(ws: str, we: str, actual: str,
                              ao: float, ac: float, result: str) -> None:
    _sb_patch("predictions", {"window_start": ws}, {
        "window_end":    we,
        "actual_signal": actual,
        "actual_open":   ao,
        "actual_close":  ac,
        "result":        result,
    })


def load_predictions(limit: int = 5) -> List[dict]:
    """
    Load the most recent `limit` predictions from Supabase, newest first.
    Always returns rows regardless of resolved state — guarantees the
    history table is populated for any visitor, including on first load.
    """
    rows = _sb_get("predictions", {
        "select": "window_start,window_end,predicted_signal,confidence,"
                  "actual_signal,actual_open,actual_close,result",
        "order":  "id.desc",
        "limit":  limit,
    })
    out = []
    for r in rows:
        out.append({
            "window_start": r["window_start"] or "",
            "window_end":   r["window_end"]   or "",
            "window":       f"{r['window_start']}-{r['window_end'] or ''}",
            "predicted":    r["predicted_signal"] or "HOLD",
            "confidence":   r["confidence"] or 0,
            "actual":       r["actual_signal"] or "\u23f3",
            "act_open":     r["actual_open"]  or 0,
            "act_close":    r["actual_close"] or 0,
            "result":       r["result"] or "\u23f3",
        })
    return out


def save_model(obj, wins: int, losses: int) -> None:
    try:
        blob_b64 = base64.b64encode(pickle.dumps(obj)).decode()
        _sb_upsert("model_store", {
            "id": 1, "model_data": blob_b64,
            "wins": wins, "losses": losses,
        })
        logger.info(f"Model saved — {wins}W / {losses}L")
    except Exception as e:
        logger.error(f"save_model: {e}")


def load_model():
    rows = _sb_get("model_store", {"select": "model_data,wins,losses", "id": "eq.1"})
    if rows:
        try:
            obj = pickle.loads(base64.b64decode(rows[0]["model_data"]))
            w, l = rows[0].get("wins", 0) or 0, rows[0].get("losses", 0) or 0
            logger.info(f"Model loaded — {w}W / {l}L")
            return obj, w, l
        except Exception as e:
            logger.error(f"load_model: {e}")
    return None, 0, 0


def save_stats(wins: int, losses: int) -> None:
    _sb_upsert("stats", {"key": "wins",   "value": wins})
    _sb_upsert("stats", {"key": "losses", "value": losses})


def recompute_stats_from_db() -> Tuple[int, int]:
    """Count wins/losses directly from predictions — authoritative source."""
    rows = _sb_get("predictions", {"select": "result"})
    w = sum(1 for r in rows if r.get("result") == "\u2705")
    l = sum(1 for r in rows if r.get("result") == "\u274c")
    return w, l


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

_last_per_model_preds: List[Optional[int]] = [None] * 5
_last_per_model_lock  = threading.Lock()

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
# SYNTHETIC HISTORY  (cold-start only)
# ============================================================
def generate_synthetic_history(n: int = 80) -> List[dict]:
    candles, price = [], 65_000.0
    t = time.time() - n * WINDOW_SECONDS
    for i in range(n):
        o  = price + random.gauss(0, 50)
        c  = o     + random.gauss(0, 120)
        h  = max(o, c) + abs(random.gauss(0, 25))
        lo = min(o, c) - abs(random.gauss(0, 25))
        candles.append({"ts": t + i * WINDOW_SECONDS,
                        "open": round(o, 2), "high": round(h, 2),
                        "low": round(lo, 2), "close": round(c, 2),
                        "volume": round(random.uniform(5, 50), 4)})
        price = c
    return candles


# ============================================================
# VMD-INSPIRED DECOMPOSITION
# ============================================================
def _vmd_decompose(prices: np.ndarray, window: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    trend    = pd.Series(prices).rolling(window, min_periods=1).mean().values
    residual = prices - trend
    return trend, residual


# ============================================================
# FEATURE ENGINEERING  (31 features)
# ============================================================
FEATURE_COLS = [
    "ret","hl","oc","body_ratio","upper_wick","lower_wick","is_bullish",
    "ema_diff","price_vs_ema9","ma_diff","price_vs_ma10",
    "rsi_norm","macd","macd_hist","roc5","momentum3_norm",
    "atr","bb_pct","bb_width","volatility",
    "vol_ratio","vol_trend",
    "ret_lag1","ret_lag2","ret_lag3",
    "hour_sin","hour_cos",
    "trend_component","residual_component","trend_slope","residual_energy",
]
assert len(FEATURE_COLS) == N_FEATURES, f"Feature count mismatch: {len(FEATURE_COLS)}"


def make_features(candles: List[dict]) -> Optional[pd.DataFrame]:
    if len(candles) < 30:
        return None
    df = pd.DataFrame(candles)

    df["ret"] = df["close"].pct_change()
    df["hl"]  = (df["high"] - df["low"]) / df["close"].clip(lower=1e-9)
    df["oc"]  = (df["close"] - df["open"]) / df["open"].clip(lower=1e-9)
    body = (df["close"] - df["open"]).abs()
    rng  = (df["high"]  - df["low"]).clip(lower=1e-9)
    df["body_ratio"]  = body / rng
    df["upper_wick"]  = (df["high"] - df[["close","open"]].max(axis=1)) / rng
    df["lower_wick"]  = (df[["close","open"]].min(axis=1) - df["low"])  / rng
    df["is_bullish"]  = (df["close"] > df["open"]).astype(float)

    ema9  = df["close"].ewm(span=9,  adjust=False).mean()
    ema21 = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_diff"]      = (ema9 - ema21) / df["close"].clip(lower=1e-9)
    df["price_vs_ema9"] = (df["close"] - ema9) / ema9.clip(lower=1e-9)
    ma3  = df["close"].rolling(3).mean()
    ma5  = df["close"].rolling(5).mean()
    ma10 = df["close"].rolling(10).mean()
    df["ma_diff"]       = (ma3 - ma5)  / df["close"].clip(lower=1e-9)
    df["price_vs_ma10"] = (df["close"] - ma10) / ma10.clip(lower=1e-9)

    delta    = df["close"].diff()
    avg_gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rs       = avg_gain / (avg_loss + 1e-9)
    df["rsi_norm"] = (100 - (100 / (1 + rs)) - 50) / 50

    ema12     = df["close"].ewm(span=12, adjust=False).mean()
    ema26     = df["close"].ewm(span=26, adjust=False).mean()
    macd_line = (ema12 - ema26) / df["close"].clip(lower=1e-9)
    macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
    df["macd"]           = macd_line
    df["macd_hist"]      = macd_line - macd_sig
    df["roc5"]           = df["close"].pct_change(5)
    df["momentum3_norm"] = df["close"].pct_change(3)

    bb_mid = df["close"].rolling(10).mean()
    bb_std = df["close"].rolling(10).std().clip(lower=1e-9)
    bb_up  = bb_mid + 2 * bb_std
    bb_lo  = bb_mid - 2 * bb_std
    df["bb_pct"]   = (df["close"] - bb_lo) / (bb_up - bb_lo + 1e-9)
    df["bb_width"] = (bb_up - bb_lo) / bb_mid.clip(lower=1e-9)
    hl_r = df["high"] - df["low"]
    hc   = (df["high"] - df["close"].shift()).abs()
    lc   = (df["low"]  - df["close"].shift()).abs()
    tr   = pd.concat([hl_r, hc, lc], axis=1).max(axis=1)
    df["atr"]        = tr.ewm(span=10, adjust=False).mean() / df["close"].clip(lower=1e-9)
    df["volatility"] = df["ret"].rolling(5).std()

    vm3  = df["volume"].rolling(3).mean()
    vm10 = df["volume"].rolling(10).mean()
    df["vol_ratio"] = df["volume"] / (vm3  + 1e-9)
    df["vol_trend"] = (vm3 - vm10) / (vm10 + 1e-9)

    df["ret_lag1"] = df["ret"].shift(1)
    df["ret_lag2"] = df["ret"].shift(2)
    df["ret_lag3"] = df["ret"].shift(3)

    try:
        hours = (pd.to_datetime(df["ts"], unit="s")
                   .dt.tz_localize("UTC")
                   .dt.tz_convert("Africa/Nairobi")
                   .dt.hour)
    except Exception:
        hours = pd.to_datetime(df["ts"], unit="s").dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    prices_arr = df["close"].values.astype(np.float64)
    trend, residual = _vmd_decompose(prices_arr, window=10)
    df["trend_component"]    = (trend - prices_arr) / (prices_arr + 1e-9)
    df["residual_component"] = residual / (prices_arr + 1e-9)
    trend_s = pd.Series(trend)
    df["trend_slope"]     = (trend_s.diff(3) / (prices_arr + 1e-9)).values
    df["residual_energy"] = (pd.Series(residual).rolling(5).std() / (prices_arr + 1e-9)).values

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df[FEATURE_COLS] if len(df) > 0 else None


# ============================================================
# PYTORCH MODELS
# ============================================================
class BiLSTMPredictor(nn.Module):
    def __init__(self, n_feat=N_FEATURES, hidden=LSTM_HIDDEN,
                 layers=LSTM_LAYERS, drop=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True,
                            dropout=drop, bidirectional=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, 64),
            nn.GELU(), nn.Dropout(drop), nn.Linear(64, 2))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class AttentionGRUPredictor(nn.Module):
    def __init__(self, n_feat=N_FEATURES, hidden=GRU_HIDDEN,
                 layers=GRU_LAYERS, heads=ATTN_HEADS, drop=DROPOUT):
        super().__init__()
        self.proj  = nn.Linear(n_feat, hidden)
        self.gru   = nn.GRU(hidden, hidden, layers, batch_first=True, dropout=drop)
        self.attn  = nn.MultiheadAttention(hidden, heads, dropout=drop, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn   = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(),
                                    nn.Dropout(drop), nn.Linear(hidden * 2, hidden))
        self.norm2 = nn.LayerNorm(hidden)
        self.head  = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(),
                                    nn.Dropout(drop), nn.Linear(32, 2))

    def forward(self, x):
        x     = self.proj(x)
        g, _  = self.gru(x)
        a, _  = self.attn(g, g, g)
        x2    = self.norm1(g + a)
        x3    = self.norm2(x2 + self.ffn(x2))
        return self.head(x3[:, -1, :])


# ============================================================
# ADAPTIVE ENSEMBLE
# ============================================================
class AdaptiveEnsemble:
    """
    Five-model soft-voting ensemble.
    Weights adapt to each model's rolling accuracy over the last
    ADAPTIVE_WINDOW resolved UP/DOWN predictions.
    HOLD predictions are never scored.
    """

    def __init__(self):
        self.scaler:   Optional[StandardScaler]        = None
        self.bilstm:   Optional[BiLSTMPredictor]       = None
        self.attn_gru: Optional[AttentionGRUPredictor] = None
        self.xgb_clf:  Optional[xgb.XGBClassifier]    = None
        self.rf_clf:   Optional[RandomForestClassifier]= None
        self.lgb_clf                                    = None
        self.result_history: List[List[Tuple[int,int]]] = [[] for _ in range(5)]
        self.weights: List[float] = list(BASE_WEIGHTS)

    def _available(self) -> List[bool]:
        return [self.bilstm   is not None, self.attn_gru is not None,
                self.xgb_clf  is not None, self.rf_clf   is not None,
                self.lgb_clf  is not None]

    def update_adaptive_weights(self) -> None:
        avail    = self._available()
        min_hist = min((len(self.result_history[i])
                        for i, a in enumerate(avail) if a), default=0)
        if min_hist < ADAPTIVE_WINDOW:
            total = sum(BASE_WEIGHTS[i] for i, a in enumerate(avail) if a) or 1e-9
            self.weights = [BASE_WEIGHTS[i]/total if avail[i] else 0.0 for i in range(5)]
        else:
            accs = []
            for i, a in enumerate(avail):
                if not a: accs.append(0.0); continue
                recent = self.result_history[i][-ADAPTIVE_WINDOW:]
                accs.append(sum(p == q for p, q in recent) / ADAPTIVE_WINDOW)
            total = sum(accs) or 1e-9
            self.weights = [acc / total for acc in accs]
        logger.info(
            f"Weights → BiLSTM:{self.weights[0]:.2f} AttnGRU:{self.weights[1]:.2f} "
            f"XGB:{self.weights[2]:.2f} RF:{self.weights[3]:.2f} LGB:{self.weights[4]:.2f}")

    def record_result(self, per_model_preds: List[Optional[int]],
                      actual_class: int) -> None:
        for i, pred in enumerate(per_model_preds):
            if pred is not None:
                self.result_history[i].append((pred, actual_class))
                if len(self.result_history[i]) > ADAPTIVE_WINDOW * 3:
                    self.result_history[i] = self.result_history[i][-ADAPTIVE_WINDOW:]
        self.update_adaptive_weights()

    def predict_proba(self, X_seq: np.ndarray,
                      X_flat: np.ndarray) -> Tuple[np.ndarray, List[Optional[int]]]:
        probs, used_w, per_model = [], [], [None] * 5

        def _try_torch(m, idx):
            p = _torch_infer(m, X_seq)
            if p is not None:
                probs.append(p); used_w.append(self.weights[idx])
                per_model[idx] = int(np.argmax(p[0]))

        if self.bilstm   is not None: _try_torch(self.bilstm,   0)
        if self.attn_gru is not None: _try_torch(self.attn_gru, 1)

        for idx, clf, name in [(2, self.xgb_clf, "XGB"),
                                (3, self.rf_clf,  "RF"),
                                (4, self.lgb_clf, "LGB")]:
            if clf is not None:
                try:
                    p = clf.predict_proba(X_flat)
                    probs.append(p); used_w.append(self.weights[idx])
                    per_model[idx] = int(np.argmax(p[0]))
                except Exception as e:
                    logger.error(f"{name}: {e}")

        if not probs:
            return np.array([[0.5, 0.5]]), per_model
        total = sum(used_w) or 1e-9
        return sum(p * w for p, w in zip(probs, used_w)) / total, per_model


def _torch_infer(m: nn.Module, X_seq: np.ndarray) -> Optional[np.ndarray]:
    try:
        m.eval()
        with torch.no_grad():
            return F.softmax(m(torch.FloatTensor(X_seq)), dim=-1).numpy()
    except Exception as e:
        logger.error(f"torch_infer: {e}"); return None


# ============================================================
# TRAINING
# ============================================================
def _train_torch(m: nn.Module, X: np.ndarray, y: np.ndarray) -> nn.Module:
    Xt, yt   = torch.FloatTensor(X), torch.LongTensor(y)
    split    = max(int(len(Xt) * 0.8), 1)
    tr_dl    = DataLoader(TensorDataset(Xt[:split], yt[:split]),
                           batch_size=TRAIN_BATCH, shuffle=True, drop_last=False)
    va_dl    = DataLoader(TensorDataset(Xt[split:], yt[split:]),
                           batch_size=TRAIN_BATCH, drop_last=False)
    opt      = torch.optim.AdamW(m.parameters(), lr=TRAIN_LR, weight_decay=1e-4)
    sched    = torch.optim.lr_scheduler.ReduceLROnPlateau(
                   opt, patience=5, factor=0.5, min_lr=1e-5)
    crit     = nn.CrossEntropyLoss()
    best_loss, best_sd, wait = float("inf"), None, 0

    for epoch in range(TRAIN_EPOCHS):
        m.train()
        for xb, yb in tr_dl:
            opt.zero_grad(); loss = crit(m(xb), yb)
            loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if not len(va_dl.dataset): break
        m.eval(); v = 0.0
        with torch.no_grad():
            for xb, yb in va_dl: v += crit(m(xb), yb).item()
        v /= max(len(va_dl), 1); sched.step(v)
        if v < best_loss - 1e-6:
            best_loss = v
            best_sd   = {k: t.clone() for k, t in m.state_dict().items()}
            wait      = 0
        else:
            wait += 1
            if wait >= TRAIN_PATIENCE:
                logger.info(f"  early-stop @ {epoch+1}, val={best_loss:.4f}"); break
    if best_sd: m.load_state_dict(best_sd)
    m.eval(); return m


def _build_sequences(feat, labels, seq_len):
    X, y = [], []
    for i in range(seq_len - 1, min(len(feat), len(labels))):
        X.append(feat[i - seq_len + 1: i + 1]); y.append(labels[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def train_model_from_candles(candles: List[dict]) -> None:
    global model
    n = len(candles)
    logger.info(f"Training on {n} candles …")
    if n < MIN_CANDLES_TRAIN + 2: return
    try:
        feat_df = make_features(candles[:-1])
        if feat_df is None or len(feat_df) == 0: return
        nf     = len(feat_df); offset = (n - 1) - nf
        labels = []
        for i in range(nf):
            idx = offset + i + 1
            if idx >= n: break
            nxt = candles[idx]
            labels.append(1 if nxt["close"] >= nxt["open"] else 0)
        if len(labels) < MIN_CANDLES_TRAIN: return

        feat_df = feat_df.iloc[:len(labels)]
        raw     = feat_df.values.astype(np.float32)
        y_arr   = np.array(labels, dtype=np.int64)
        split   = max(int(len(raw) * 0.8), 1)
        scaler  = StandardScaler(); scaler.fit(raw[:split]); scaled = scaler.transform(raw)
        ns      = len(y_arr)
        sw      = np.array([0.95 ** (ns-1-i) for i in range(ns)], dtype=np.float32)

        with model_lock: old_ens = model
        ens = AdaptiveEnsemble(); ens.scaler = scaler
        if old_ens and isinstance(old_ens, AdaptiveEnsemble):
            ens.result_history = old_ens.result_history
            ens.weights        = old_ens.weights

        logger.info("  Training XGBoost …")
        xm = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
             subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
             reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss", verbosity=0)
        xm.fit(scaled, y_arr, sample_weight=sw); ens.xgb_clf = xm
        logger.info("  XGBoost ✓")

        logger.info("  Training RandomForest …")
        rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=3,
             max_features="sqrt", n_jobs=-1, random_state=42)
        rf.fit(scaled, y_arr, sample_weight=sw); ens.rf_clf = rf
        logger.info("  RandomForest ✓")

        if HAS_LGB:
            logger.info("  Training LightGBM …")
            lm = lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                 reg_alpha=0.1, reg_lambda=1.0, verbosity=-1, n_jobs=2)
            lm.fit(scaled, y_arr, sample_weight=sw); ens.lgb_clf = lm
            logger.info("  LightGBM ✓")

        if n >= MIN_CANDLES_DEEP:
            X_seq, y_seq = _build_sequences(scaled, y_arr, SEQ_LEN)
            if len(X_seq) >= 20:
                logger.info("  Training BiLSTM …")
                ens.bilstm = _train_torch(
                    BiLSTMPredictor(N_FEATURES, LSTM_HIDDEN, LSTM_LAYERS, DROPOUT),
                    X_seq, y_seq)
                logger.info("  BiLSTM ✓")
                logger.info("  Training AttentionGRU …")
                ens.attn_gru = _train_torch(
                    AttentionGRUPredictor(N_FEATURES, GRU_HIDDEN, GRU_LAYERS,
                                          ATTN_HEADS, DROPOUT),
                    X_seq, y_seq)
                logger.info("  AttentionGRU ✓")

        ens.update_adaptive_weights()
        with model_lock: model = ens
        save_model(ens, wins, losses)
        logger.info("  Ensemble saved to Supabase ✓")
    except Exception as exc:
        logger.exception(f"Training error: {exc}")


# ============================================================
# PREDICTION
# ============================================================
def predict_from_candles(candles: List[dict]) -> dict:
    default = {"signal": "HOLD", "confidence": 0, "next_window": "", "per_model": [None]*5}
    with model_lock: ens = model
    if ens is None or len(candles) < 6: return default
    try:
        feat_df = make_features(candles)
        if feat_df is None: return default
        scaled = ens.scaler.transform(feat_df.values.astype(np.float32))
        if len(scaled) >= SEQ_LEN:
            seq = scaled[-SEQ_LEN:]
        else:
            pad = np.zeros((SEQ_LEN - len(scaled), N_FEATURES), dtype=np.float32)
            seq = np.vstack([pad, scaled])
        X_seq  = seq[np.newaxis, :, :]
        X_flat = scaled[[-1], :]
        prob, per_model = ens.predict_proba(X_seq, X_flat)
        up_p = float(prob[0, 1])
        if   up_p >= SIGNAL_UP_THRESH:   signal = "UP"
        elif up_p <= SIGNAL_DOWN_THRESH: signal = "DOWN"
        else:                            signal = "HOLD"
        # HOLD → confidence 0 so UI renders "--"
        conf = 0 if signal == "HOLD" else int(round(max(up_p, 1 - up_p) * 100))
        with _last_per_model_lock: _last_per_model_preds[:] = per_model
        now = datetime.now(TZ)
        nm  = ((now.minute // 5) + 1) * 5
        ns  = now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=nm)
        ne  = ns + timedelta(minutes=5)
        return {"signal": signal, "confidence": conf,
                "next_window": f"{ns.strftime('%H:%M')}-{ne.strftime('%H:%M')}",
                "per_model": per_model}
    except Exception as exc:
        logger.error(f"Prediction error: {exc}"); return default


# ============================================================
# CANDLE BUILDING
# ============================================================
def _new_candle(ts: float, price: float) -> dict:
    return {"ts": ts, "open": price, "high": price,
            "low": price, "close": price, "volume": 0.0}


def _seed_pending_row_locked(window_ts: float) -> None:
    """
    Create and persist a pending prediction row for the window that is
    just opening. Idempotent — skipped if the window_start already exists.

    This is the key fix for "table empty on load":
      • Called when a new 5-min window starts
      • Saved to Supabase immediately
      • Any visitor who loads the page BEFORE the first window closes
        will still see this pending row in the table
    """
    ws_dt  = datetime.fromtimestamp(window_ts, tz=TZ)
    we_dt  = ws_dt + timedelta(minutes=5)
    ws_str = ws_dt.strftime("%H:%M")
    we_str = we_dt.strftime("%H:%M")

    # Idempotency: skip if already in memory
    for row in history_rows:
        if row.get("window_start") == ws_str:
            return

    snap = list(completed_candles)
    pred = predict_from_candles(snap) if snap else \
           {"signal": "HOLD", "confidence": 0, "next_window": "", "per_model": [None]*5}

    rec = {
        "window_start": ws_str, "window_end": we_str,
        "window":       f"{ws_str}-{we_str}",
        "predicted":    pred["signal"],
        "confidence":   pred["confidence"],
        "act_open":     0.0, "act_close": 0.0,
        "actual":       "\u23f3", "result": "\u23f3",
    }
    history_rows.insert(0, rec)
    while len(history_rows) > 5:
        history_rows.pop()
    # Persist immediately — new visitors see this row right away
    threading.Thread(target=save_prediction, args=(rec,), daemon=True).start()


def _close_current_window_locked() -> None:
    global live_candle, candles_since_retrain, wins, losses

    candle       = dict(live_candle)
    live_candle  = None
    completed_candles.append(candle)
    candles_since_retrain += 1
    threading.Thread(target=save_candle, args=(candle,), daemon=True).start()

    ws_str, we_str = get_window_time_range(candle["ts"])
    actual       = "UP" if candle["close"] > candle["open"] else "DOWN"
    actual_open  = candle["open"]
    actual_close = candle["close"]

    # ── Step 1: resolve the matching pending row ─────────────────────────
    matched = False
    for row in history_rows:
        if row.get("actual") == "\u23f3" and row.get("window_start") == ws_str:
            matched    = True
            row["actual"]     = actual
            row["act_open"]   = actual_open
            row["act_close"]  = actual_close
            row["window_end"] = we_str
            row["window"]     = f"{ws_str}-{we_str}"
            predicted         = row["predicted"]

            if predicted == "HOLD":
                row["result"] = "\u2014"   # em-dash
                threading.Thread(target=update_prediction_result,
                    args=(ws_str, we_str, actual, actual_open, actual_close, "\u2014"),
                    daemon=True).start()
                logger.info(f"Window {ws_str}: HOLD — skipped")
            else:
                if predicted == actual:
                    row["result"] = "\u2705"; wins   += 1
                else:
                    row["result"] = "\u274c"; losses += 1

                res_val = row["result"]
                threading.Thread(target=update_prediction_result,
                    args=(ws_str, we_str, actual, actual_open, actual_close, res_val),
                    daemon=True).start()
                threading.Thread(target=save_stats, args=(wins, losses),
                    daemon=True).start()

                actual_class = 1 if actual == "UP" else 0
                with _last_per_model_lock: pmp = list(_last_per_model_preds)
                with model_lock: ens = model
                if ens and isinstance(ens, AdaptiveEnsemble):
                    ens.record_result(pmp, actual_class)

                logger.info(f"Window {ws_str}: {predicted} vs {actual} → {res_val}")
            break

    if not matched:
        logger.debug(f"No pending row for {ws_str} — will catch next cycle")

    # ── Step 2: seed pending row for the NEXT window ─────────────────────
    _seed_pending_row_locked(candle["ts"] + WINDOW_SECONDS)

    # ── Step 3: retrain if due ────────────────────────────────────────────
    if candles_since_retrain >= RETRAIN_EVERY:
        candles_since_retrain = 0
        threading.Thread(target=train_model_from_candles,
                          args=(list(completed_candles),), daemon=True).start()


def process_price_tick(price: float, trade_ts: float) -> None:
    global live_candle, live_window_start

    with state_lock:
        current_window_start = int(trade_ts // WINDOW_SECONDS) * WINDOW_SECONDS

        if live_window_start is None:
            live_window_start = current_window_start
            live_candle = _new_candle(live_window_start, price)
            _seed_pending_row_locked(live_window_start)
            return

        if current_window_start > live_window_start:
            if live_candle is not None:
                _close_current_window_locked()
            live_window_start = current_window_start
            live_candle = _new_candle(live_window_start, price)
            return

        if live_candle is not None:
            if trade_ts >= live_candle["ts"] + WINDOW_SECONDS:
                _close_current_window_locked()
                live_window_start = current_window_start
                live_candle = _new_candle(live_window_start, price)
            else:
                live_candle["high"]  = max(live_candle["high"], price)
                live_candle["low"]   = min(live_candle["low"],  price)
                live_candle["close"] = price


# ============================================================
# COINBASE WEBSOCKET
# ============================================================
def _coinbase_thread() -> None:
    url, retry = "wss://ws-feed.exchange.coinbase.com", 2
    while True:
        try:
            def on_open(w):
                nonlocal retry; retry = 2
                w.send(json.dumps({"type": "subscribe", "product_ids": ["BTC-USD"],
                                   "channels": ["ticker", "matches"]}))
                logger.info("Coinbase WS connected")

            def on_message(w, raw):
                try:
                    d = json.loads(raw)
                    if d.get("type") in ["ticker", "match"]:
                        p = float(d.get("price", 0))
                        if p > 0: price_queue.put_nowait({"price": p, "ts": time.time()})
                except Exception: pass

            ws_client.WebSocketApp(
                url, on_open=on_open, on_message=on_message,
                on_error=lambda w, e: logger.error(f"WS err: {e}"),
                on_close=lambda w, c, m: logger.warning("WS closed"),
            ).run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e: logger.error(f"Coinbase thread: {e}")
        logger.warning(f"Reconnecting in {retry}s …")
        time.sleep(retry); retry = min(retry * 2, 60)


# ============================================================
# PRICE PROCESSOR
# ============================================================
def price_processor_thread() -> None:
    while True:
        try:
            item = price_queue.get(timeout=1.0)
            process_price_tick(item["price"], item["ts"])
        except Empty: continue
        except Exception as e: logger.error(f"Price processor: {e}")


# ============================================================
# BROADCAST
# ============================================================
async def _periodic_broadcast() -> None:
    while True:
        await asyncio.sleep(1)
        await _broadcast_state()


async def _broadcast_state() -> None:
    msg = json.dumps(_build_state_payload())
    async with _clients_lock: snap = list(_clients)
    dead = []
    for ws in snap:
        try: await ws.send_text(msg)
        except Exception: dead.append(ws)
    if dead:
        async with _clients_lock:
            for ws in dead:
                if ws in _clients: _clients.remove(ws)


def _build_state_payload() -> dict:
    with state_lock:
        snap = list(completed_candles)
        lc   = dict(live_candle) if live_candle else None
        w, l = wins, losses
        rows = list(history_rows[:5])

    pred       = predict_from_candles(snap) if snap else \
                 {"signal": "HOLD", "confidence": 0, "next_window": ""}
    ohlc       = {k: round(lc[k], 2) for k in ("open","high","low","close")} if lc else {}
    live_price = lc["close"] if lc else (snap[-1]["close"] if snap else 0.0)
    with model_lock: model_ready = model is not None
    return {
        "price": round(live_price, 2),
        "signal": pred["signal"], "confidence": pred["confidence"],
        "next_window": pred["next_window"],
        "wins": w, "losses": l,
        "ohlc": ohlc, "table": rows, "model_ready": model_ready,
    }


# ============================================================
# HTML  (original UI — pixel-perfect, untouched)
# ============================================================
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=yes">
<title>BTC 5-Min Predictor</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #080C14; color: #C8D8EF;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
    min-height: 100vh;
  }
  .container { max-width: 1280px; margin: 0 auto; padding: 12px; }
  .header { margin-bottom: 12px; padding: 8px 0; text-align: center; }
  .header h1 { font-size: 1.4rem; font-weight: 700; color: #F7931A; letter-spacing: .02em; }
  .status-bar {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 12px; font-size: 0.8rem; color: #4A6080; flex-wrap: wrap;
  }
  .status-bar #clock-gmt3 { color: #FFFFFF; margin-left: auto; }
  .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
  .dot-ok  { background: #00E5A0; box-shadow: 0 0 6px #00E5A0; }
  .dot-bad { background: #FF4560; }
  .dot-wait { background: #F7931A; animation: pulse 1.2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
  .main-grid {
    display: grid; grid-template-columns: 1fr 340px;
    gap: 12px; margin-bottom: 12px; align-items: stretch;
  }
  #tv-chart {
    background: #0D1421; border-radius: 10px;
    border: 1px solid #1E2D45; height: 380px; overflow: hidden;
  }
  .sidebar { display: flex; flex-direction: column; gap: 12px; }
  .card { background: #0D1421; border: 1px solid #1E2D45; border-radius: 10px; padding: 14px; }
  .card-title { font-size: 0.85rem; color: #4A6080; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px; }
  .price { font-size: 2rem; font-weight: 700; color: #F7931A; }
  .pchange { font-size: 0.82rem; margin-left: 6px; }
  .pred-row { display: flex; align-items: center; gap: 12px; margin-top: 4px; }
  .pred-arrow { font-size: 2.4rem; line-height: 1; }
  .pred-dir { font-size: 1.3rem; font-weight: 700; }
  .conf-bar { background: #1E2D45; border-radius: 4px; height: 6px; margin-top: 8px; overflow: hidden; }
  .conf-fill { height: 100%; width: 0%; transition: width 0.4s ease; }
  .countdown { display: flex; align-items: center; gap: 14px; }
  .cd-ring { width: 58px; height: 58px; transform: rotate(-90deg); }
  .cd-text { font-size: 1.8rem; font-weight: 700; color: #F7931A; }
  .bottom-row { display: grid; gap: 12px; grid-template-columns: 1fr 1fr; margin-bottom: 12px; }
  .ohlc-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 6px; }
  .ohlc-cell { background: #0F1623; border-radius: 6px; padding: 7px 10px; border: 1px solid #1E2D45; }
  .ohlc-cell .lbl { font-size: 0.6rem; color: #4A6080; }
  .ohlc-cell .val { font-size: 0.88rem; font-weight: 700; margin-top: 2px; }
  .perf-row { display: flex; gap: 10px; margin-top: 8px; }
  .perf-stat { flex:1; text-align:center; background:#0F1623; border-radius:7px; padding:8px; border:1px solid #1E2D45; }
  .perf-num { font-size: 1.35rem; font-weight: 700; }
  .perf-lbl { font-size: 0.6rem; color: #4A6080; margin-top: 2px; }
  .table-wrapper { overflow-x: auto; margin-top: 10px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.75rem; }
  th, td { padding: 8px; text-align: left; border-bottom: 1px solid #1E2D45; }
  th { color: #4A6080; font-weight: 600; }
  .up   { color: #00E5A0; }
  .down { color: #FF4560; }
  .hold-clr { color: #4A6080; }
  .disclaimer {
    background: #0F1623; border-radius: 8px; padding: 10px;
    font-size: 0.7rem; text-align: center; color: #FACC15;
    border: 1px solid rgba(250,204,21,.13); margin-top: 12px;
  }
  .flash-up { animation: flashUp 0.4s ease; }
  .flash-dn { animation: flashDn 0.4s ease; }
  @keyframes flashUp { 0%,100%{color:#C8D8EF} 50%{color:#00E5A0} }
  @keyframes flashDn { 0%,100%{color:#C8D8EF} 50%{color:#FF4560} }
  @media (max-width:900px) {
    .main-grid { grid-template-columns: 1fr; }
    #tv-chart { height: 320px; }
    .sidebar { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; }
    .bottom-row { grid-template-columns: 1fr; }
  }
  @media (max-width:600px) {
    .container { padding: 8px; }
    .header h1 { font-size: 1.1rem; }
    .status-bar { font-size: 0.7rem; gap: 6px; }
    .sidebar { grid-template-columns: 1fr; gap: 10px; }
    #tv-chart { height: 260px; }
    .card { padding: 10px; }
    .price { font-size: 1.5rem; }
    .pred-arrow { font-size: 1.8rem; }
    .cd-text { font-size: 1.4rem; }
    .cd-ring { width: 48px; height: 48px; }
    table { font-size: 0.65rem; }
    th, td { padding: 6px 4px; }
  }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>&#x20BF; Bitcoin Price Trend Predictor</h1>
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
        <div>
          <span class="price" id="price-val">$---.--</span>
          <span class="pchange" id="price-change">--</span>
        </div>
      </div>
      <div class="card">
        <div class="card-title">Next 5-Min Prediction</div>
        <div class="pred-row">
          <span class="pred-arrow" id="pred-arrow" style="color:#4A6080">&#9670;</span>
          <span class="pred-dir"   id="pred-dir"   style="color:#4A6080">HOLD</span>
          <span id="conf-pct" style="font-size:.85rem;color:#4A6080">--</span>
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
        <div class="ohlc-cell"><div class="lbl">Open</div> <div class="val"      id="o-open">--</div></div>
        <div class="ohlc-cell"><div class="lbl">High</div> <div class="val up"   id="o-high">--</div></div>
        <div class="ohlc-cell"><div class="lbl">Low</div>  <div class="val down" id="o-low">--</div></div>
        <div class="ohlc-cell"><div class="lbl">Close</div><div class="val"      id="o-close">--</div></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Performance <span style="font-size:.65rem;color:#4A6080">(UP/DOWN only)</span></div>
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
          <tr>
            <th>Window</th><th>Prediction</th><th>Conf</th>
            <th>Act.Open</th><th>Act.Close</th><th>Actual</th><th>Result</th>
          </tr>
        </thead>
        <tbody id="history-body">
          <tr><td colspan="7" style="text-align:center;padding:20px;">Waiting for data...</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <div class="disclaimer">&#9888;&#65039; For educational purposes only. Past accuracy does not guarantee future results.</div>
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
    const d=new Date();
    document.getElementById('clock-gmt3').textContent=
      d.toLocaleTimeString('en-US',{timeZone:'Africa/Nairobi',hour12:false})+' GMT+3';
  }
  updateClock(); setInterval(updateClock,1000);

  function updateCountdown(){
    const now=new Date();
    const gmt3=new Date(now.getTime()+3*3600000);
    const minutes=gmt3.getUTCMinutes(), seconds=gmt3.getUTCSeconds();
    const remaining=Math.max(0,(5-(minutes%5))*60-seconds);
    const mm=Math.floor(remaining/60), ss=String(remaining%60).padStart(2,'0');
    document.getElementById('cd-val').textContent=mm+':'+ss;
    document.getElementById('cd-ring').setAttribute('stroke-dashoffset',
      String(201*(1-remaining/300)));
  }
  updateCountdown(); setInterval(updateCountdown,1000);

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
    ws.onerror=function(){ document.getElementById('ws-dot').className='dot dot-bad'; };
    ws.onmessage=function(e){
      try{ handleState(JSON.parse(e.data)); }catch(err){}
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
      const chg=p-firstPrice, pct=(chg/firstPrice*100).toFixed(2);
      const chgEl=document.getElementById('price-change');
      chgEl.textContent=(chg>=0?'+':'')+fmt(chg)+' ('+pct+'%)';
      chgEl.style.color=chg>=0?'#00E5A0':'#FF4560';
    }

    const sig=d.signal||'HOLD';
    const isUp=sig==='UP', isDn=sig==='DOWN', isHold=sig==='HOLD';
    const col=isUp?'#00E5A0':isDn?'#FF4560':'#4A6080';
    document.getElementById('pred-arrow').textContent=isUp?'\u25B2':isDn?'\u25BC':'\u25C6';
    document.getElementById('pred-arrow').style.color=col;
    document.getElementById('pred-dir').textContent=isUp?'UP':isDn?'DOWN':'HOLD';
    document.getElementById('pred-dir').style.color=col;
    const confEl=document.getElementById('conf-pct');
    if(isHold){ confEl.textContent='--'; confEl.style.color='#4A6080'; }
    else{ confEl.textContent=d.confidence+'%'; confEl.style.color=col; }
    document.getElementById('conf-bar').style.width=isHold?'0%':(d.confidence||0)+'%';
    document.getElementById('conf-bar').style.background=col;
    document.getElementById('pred-window').textContent=d.next_window?'Next: '+d.next_window:'';

    if(d.ohlc){
      document.getElementById('o-open').textContent =fmt(d.ohlc.open);
      document.getElementById('o-high').textContent =fmt(d.ohlc.high);
      document.getElementById('o-low').textContent  =fmt(d.ohlc.low);
      document.getElementById('o-close').textContent=fmt(d.ohlc.close);
    }

    const w=d.wins||0, l=d.losses||0, tot=w+l;
    document.getElementById('p-wins').textContent  =w;
    document.getElementById('p-losses').textContent=l;
    document.getElementById('p-acc').textContent   =tot?(w/tot*100).toFixed(1)+'%':'--';

    const tbody=document.getElementById('history-body');
    if(d.table&&d.table.length){
      tbody.innerHTML=d.table.map(function(r){
        const isHoldPred=r.predicted==='HOLD';
        const predCls=r.predicted==='UP'?'up':r.predicted==='DOWN'?'down':'hold-clr';
        const actCls =r.actual==='UP'   ?'up':r.actual==='DOWN'    ?'down':'';
        const predTxt=r.predicted==='UP'?'\u25B2 UP':r.predicted==='DOWN'?'\u25BC DOWN':'\u25C6 HOLD';
        const actTxt =r.actual==='\u23F3'?'--':r.actual==='UP'?'\u25B2 UP':r.actual==='DOWN'?'\u25BC DOWN':r.actual;
        const confTxt=isHoldPred?'--':r.confidence+'%';
        const confCol=isHoldPred?'#4A6080':'#F7931A';
        const resTxt =r.result==='\u23F3'?'\u23F3':r.result;
        return'<tr>'
          +'<td style="color:#4A6080">'+r.window+'</td>'
          +'<td class="'+predCls+'">'+predTxt+'</td>'
          +'<td style="color:'+confCol+'">'+confTxt+'</td>'
          +'<td>'+fmt(r.act_open)+'</td>'
          +'<td>'+fmt(r.act_close)+'</td>'
          +'<td class="'+actCls+'">'+actTxt+'</td>'
          +'<td>'+resTxt+'</td>'
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
# FASTAPI
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _main_loop, _clients_lock, model, wins, losses

    _main_loop    = asyncio.get_running_loop()
    _clients_lock = asyncio.Lock()

    # 1. Candles first — needed for training and prediction
    saved_candles = load_candles(HISTORY_LIMIT)
    if saved_candles:
        for c in saved_candles:
            completed_candles.append(c)
        logger.info(f"Loaded {len(saved_candles)} candles from Supabase")

    # 2. Model
    saved_model, _, _ = load_model()
    if saved_model:
        with model_lock: model = saved_model

    # 3. Last 5 predictions → history_rows
    #    Loaded BEFORE any price tick arrives so new visitors see data immediately
    saved_preds = load_predictions(5)
    if saved_preds:
        history_rows[:] = saved_preds
        logger.info(f"Loaded {len(saved_preds)} predictions — table pre-populated")

    # 4. Authoritative win/loss count straight from DB
    wins, losses = recompute_stats_from_db()
    logger.info(f"Stats recomputed from DB: {wins}W / {losses}L")

    # 5. Cold start
    if not saved_candles:
        logger.info("Cold start — seeding synthetic history …")
        synth = generate_synthetic_history(80)
        for c in synth: completed_candles.append(c)
        threading.Thread(target=lambda: [save_candle(c) for c in synth],
                         daemon=True, name="seed-candles").start()
        threading.Thread(target=train_model_from_candles,
                         args=(list(completed_candles),),
                         daemon=True, name="init-train").start()
    elif model is None:
        logger.info("Candles present but no model — training …")
        threading.Thread(target=train_model_from_candles,
                         args=(list(completed_candles),),
                         daemon=True, name="init-train").start()

    threading.Thread(target=price_processor_thread, name="price-proc",  daemon=True).start()
    threading.Thread(target=_coinbase_thread,        name="coinbase-ws", daemon=True).start()
    asyncio.create_task(_periodic_broadcast())

    lgb_info = "LightGBM" if HAS_LGB else "LightGBM(missing)"
    logger.info("=" * 60)
    logger.info("BTC Adaptive Predictor v4.0 — Supabase backend — ready")
    logger.info(f"Models   : BiLSTM · AttentionGRU · XGBoost · RF · {lgb_info}")
    logger.info(f"Features : {N_FEATURES}  |  SeqLen : {SEQ_LEN}  |  History : {HISTORY_LIMIT}")
    logger.info(f"Stats    : {wins}W / {losses}L")
    logger.info("=" * 60)
    yield

    with model_lock: mdl = model
    if mdl: save_model(mdl, wins, losses)
    save_stats(wins, losses)
    logger.info(f"Shutdown — {wins}W / {losses}L")


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_CONTENT


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    async with _clients_lock: _clients.append(websocket)
    logger.info(f"WS client connected — total: {len(_clients)}")
    try:
        await websocket.send_text(json.dumps(_build_state_payload()))
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=25.0)
                if msg.get("type") == "websocket.disconnect": break
                if msg.get("text") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except asyncio.TimeoutError:
                try: await websocket.send_text(json.dumps({"type": "ping"}))
                except Exception: break
            except WebSocketDisconnect: break
            except Exception: break
    except Exception: pass
    finally:
        async with _clients_lock:
            if websocket in _clients: _clients.remove(websocket)
        logger.info(f"WS client disconnected — total: {len(_clients)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port,
                ws="websockets", log_level="info")
