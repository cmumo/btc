"""
Bitcoin 5-Minute Prediction Terminal
=====================================
- TradingView WebSocket (BITSTAMP:BTCUSD)
- XGBoost model, 5-min candles, win/loss tracking
- GMT+3 timezone, prices with 2 decimal places
- No TP/SL, no extra text, clean UI
"""

import asyncio
import json
import logging
import random
import string
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from queue import Empty, Queue
from typing import Deque, Optional

import numpy as np
import pandas as pd
import websocket as ws_client
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# === GMT+3 timezone offset ===
TZ_OFFSET = timedelta(hours=3)

def now_gmt3() -> datetime:
    return datetime.utcnow() + TZ_OFFSET

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SYMBOL          = "BITSTAMP:BTCUSD"
WINDOW_SECONDS  = 300
MAX_BUFFER      = 500
RETRAIN_EVERY   = 10
MIN_TRAIN_ROWS  = 40
HISTORICAL_ROWS = 800
HISTORY_LIMIT   = 5   # rows shown in table

# ─────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────
state_lock = threading.Lock()
completed_candles: deque = deque(maxlen=MAX_BUFFER)
live_candle: Optional[dict] = None
live_window_start: Optional[int] = None
pending_predictions: dict = {}
history_rows: list = []
wins = 0
losses = 0
latest_price = 0.0

# ─────────────────────────────────────────────
# ML MODEL (XGBoost)
# ─────────────────────────────────────────────
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    tree_method="hist",
    verbosity=0,
)
scaler = StandardScaler()
trained = False
candles_since_retrain = 0

# ─────────────────────────────────────────────
# ASYNC / QUEUE / CLIENTS
# ─────────────────────────────────────────────
_main_loop: Optional[asyncio.AbstractEventLoop] = None
price_queue: Queue = Queue(maxsize=5000)
ws_clients: list = []
ws_clients_lock = threading.Lock()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def aligned_window(ts: float) -> int:
    return int(ts) - (int(ts) % WINDOW_SECONDS)

def window_label(start_ts: int) -> str:
    dt = datetime.utcfromtimestamp(start_ts) + TZ_OFFSET
    end = dt + timedelta(seconds=WINDOW_SECONDS)
    return f"{dt.strftime('%H:%M')}-{end.strftime('%H:%M')}"

def fmt_price(value: Optional[float]) -> str:
    if value is None:
        return "--"
    return f"${value:,.2f}"

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def build_feature_df(candles: list) -> pd.DataFrame:
    if len(candles) < 30:
        return pd.DataFrame()
    df = pd.DataFrame(candles).sort_values("window_start").reset_index(drop=True)
    df["returns"] = df["close"].pct_change()
    for p in [5, 10, 20, 50]:
        df[f"sma_{p}"] = df["close"].rolling(p).mean()
        df[f"ema_{p}"] = df["close"].ewm(span=p).mean()
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    macd = MACD(df["close"])
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"]   = macd.macd_diff()
    bb = BollingerBands(df["close"])
    df["bb_width"]    = (bb.bollinger_hband() - bb.bollinger_lband()) / (bb.bollinger_mavg() + 1e-9)
    df["bb_position"] = (df["close"] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-9)
    df["atr"]         = AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
    stoch = StochasticOscillator(df["high"], df["low"], df["close"])
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    adx = ADXIndicator(df["high"], df["low"], df["close"])
    df["adx"]     = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()
    df["volume_sma"]   = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_sma"] + 1e-9)
    for p in [5, 10, 20]:
        df[f"momentum_{p}"] = df["close"] - df["close"].shift(p)
        df[f"roc_{p}"]      = df["close"].pct_change(p) * 100
    df["volatility_20"] = df["returns"].rolling(20).std()
    df["high_20"]       = df["high"].rolling(20).max()
    df["low_20"]        = df["low"].rolling(20).min()
    df["price_range"]   = (df["close"] - df["low_20"]) / (df["high_20"] - df["low_20"] + 1e-9)
    df["body_ratio"]    = (df["close"] - df["open"]).abs() / (df["high"] - df["low"] + 1e-9)
    df["upper_wick"]    = (df["high"] - df[["close", "open"]].max(axis=1)) / (df["high"] - df["low"] + 1e-9)
    df["lower_wick"]    = (df[["close", "open"]].min(axis=1) - df["low"]) / (df["high"] - df["low"] + 1e-9)
    for p in [5, 10, 20, 50]:
        df[f"sma_{p}"] = df[f"sma_{p}"] / df["close"] - 1
        df[f"ema_{p}"] = df[f"ema_{p}"] / df["close"] - 1
    return df

FEATURE_COLS = [
    "returns", "rsi", "macd", "macd_signal", "macd_diff",
    "bb_width", "bb_position", "atr", "stoch_k", "stoch_d",
    "adx", "adx_pos", "adx_neg", "volume_ratio", "volatility_20",
    "price_range", "momentum_5", "momentum_10", "momentum_20",
    "roc_5", "roc_10", "roc_20", "sma_5", "sma_10", "sma_20", "sma_50",
    "ema_5", "ema_10", "ema_20", "ema_50", "body_ratio", "upper_wick", "lower_wick"
]

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLS].copy().ffill().fillna(0)

def create_labels(df: pd.DataFrame, future_periods: int = 1) -> pd.Series:
    future_ret = df["close"].shift(-future_periods) / df["close"] - 1
    return (future_ret > 0.0005).astype(int)

# ============================================================
# SYNTHETIC HISTORY
# ============================================================
def generate_synthetic_history(n: int = HISTORICAL_ROWS, base_price: float = 65000.0) -> list:
    logger.info(f"Generating {n} synthetic candles...")
    now_ts = int(time.time())
    current_window = aligned_window(now_ts)
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0001, 0.004, n)
    trend   = np.linspace(0, 0.03, n)
    cycle   = 0.015 * np.sin(np.linspace(0, 6 * np.pi, n))
    returns = returns + trend / n + cycle / n
    closes  = base_price * np.exp(np.cumsum(returns))
    candles = []
    for i in range(n):
        ts  = current_window - (n - i) * WINDOW_SECONDS
        c   = float(closes[i])
        o   = float(closes[i - 1]) if i > 0 else c
        noise_h = abs(float(rng.normal(0, 0.0008))) * c
        noise_l = abs(float(rng.normal(0, 0.0008))) * c
        h = max(o, c) + noise_h
        l = min(o, c) - noise_l
        v = float(rng.integers(500, 8000))
        candles.append({
            "window_start": ts,
            "label":        window_label(ts),
            "open":         round(o, 2),
            "high":         round(h, 2),
            "low":          round(l, 2),
            "close":        round(c, 2),
            "volume":       round(v, 2),
        })
    return candles

# ============================================================
# TRAIN / PREDICT
# ============================================================
def train_model_from_candles(candles: list) -> None:
    global model, scaler, trained
    if len(candles) < MIN_TRAIN_ROWS + 10:
        return
    df = build_feature_df(candles)
    if df.empty:
        return
    df["label"] = create_labels(df)
    valid = df[FEATURE_COLS].notna().all(axis=1) & df["label"].notna()
    df_v = df[valid].copy()
    if len(df_v) < MIN_TRAIN_ROWS:
        return
    X = prepare_features(df_v)
    y = df_v["label"]
    try:
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)
        acc = model.score(X_scaled, y)
        trained = True
        logger.info(f"Model trained on {len(X)} candles, train-acc: {acc:.2%}")
    except Exception as exc:
        logger.error(f"Training error: {exc}")

def predict_from_candles(candles: list) -> dict:
    default = {"signal": "HOLD", "direction": "HOLD", "confidence": 0.0}
    if not trained or len(candles) < 30:
        return default
    df = build_feature_df(candles)
    if df.empty:
        return default
    X = prepare_features(df).ffill().fillna(0)
    if X.empty:
        return default
    try:
        X_scaled  = scaler.transform(X.iloc[-1:])
        prob_up   = float(model.predict_proba(X_scaled)[0][1])
        confidence = abs(prob_up - 0.5) * 200
        if prob_up > 0.62 and confidence > 20:
            signal, direction = "BUY", "UP"
        elif prob_up < 0.38 and confidence > 20:
            signal, direction = "SELL", "DOWN"
        else:
            signal, direction = "HOLD", "HOLD"
            confidence = 0.0
        return {
            "signal":     signal,
            "direction":  direction,
            "confidence": round(min(confidence, 100), 1),
        }
    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        return default

# ============================================================
# CANDLE BUILDING
# ============================================================
def process_price_tick(price: float, trade_ts: float) -> None:
    global live_candle, live_window_start, wins, losses, candles_since_retrain
    ws = aligned_window(trade_ts)
    with state_lock:
        if live_candle is None:
            live_window_start = ws
            live_candle = _new_candle(ws, price)
            return
        if ws > live_window_start:
            _close_current_window()
            live_window_start = ws
            prev_close = completed_candles[-1]["close"] if completed_candles else price
            live_candle = _new_candle(ws, prev_close)
        live_candle["high"]   = max(live_candle["high"], price)
        live_candle["low"]    = min(live_candle["low"], price)
        live_candle["close"]  = price
        live_candle["ticks"]  = live_candle.get("ticks", 0) + 1
        live_candle["volume"] = live_candle.get("volume", 0.0) + 1.0

def _new_candle(window_start: int, open_price: float) -> dict:
    return {
        "window_start": window_start,
        "label":        window_label(window_start),
        "open":         round(open_price, 2),
        "high":         round(open_price, 2),
        "low":          round(open_price, 2),
        "close":        round(open_price, 2),
        "volume":       0.0,
        "ticks":        0,
    }

def _close_current_window() -> None:
    global wins, losses, candles_since_retrain
    candle = {**live_candle}
    candle["volume"] = max(candle.get("volume", 0), 1.0)
    completed_candles.append(candle)
    candles_since_retrain += 1
    ws_start = candle["window_start"]
    logger.info(f"Window closed: {candle['label']} | O={candle['open']} C={candle['close']}")

    if ws_start in pending_predictions:
        pred   = pending_predictions.pop(ws_start)
        actual = "UP" if candle["close"] >= candle["open"] else "DOWN"
        is_hold = pred["direction"] == "HOLD"
        correct = (not is_hold) and (pred["direction"] == actual)
        if not is_hold:
            if correct:
                wins += 1
            else:
                losses += 1
        for row in history_rows:
            if row["predicted_window_start"] == ws_start:
                row["actual"] = actual
                row["result"] = "-" if is_hold else ("✅" if correct else "❌")
                row["act_open"]  = candle["open"]
                row["act_close"] = candle["close"]
                break
        total = wins + losses
        if total and not is_hold:
            logger.info(
                f"Outcome {candle['label']}: pred={pred['direction']} actual={actual} "
                f"{'✅' if correct else '❌'} W={wins} L={losses} Acc={wins/total*100:.1f}%"
            )

    if candles_since_retrain >= RETRAIN_EVERY:
        candles_since_retrain = 0
        threading.Thread(
            target=train_model_from_candles,
            args=(list(completed_candles),),
            daemon=True,
        ).start()

    next_ws      = ws_start + WINDOW_SECONDS
    pred_result  = predict_from_candles(list(completed_candles))
    pending_predictions[next_ws] = {
        "direction":  pred_result["direction"],
        "confidence": pred_result["confidence"],
    }
    history_rows.insert(0, {
        "window_label":           candle["label"],
        "predicted_window_start": next_ws,
        "predicted_window_label": window_label(next_ws),
        "predicted":              pred_result["direction"],
        "confidence":             pred_result["confidence"],
        "signal":                 pred_result["signal"],
        "actual":   "⏳",
        "result":   "⏳",
        "act_open":  None,
        "act_close": None,
    })
    if len(history_rows) > HISTORY_LIMIT:
        history_rows[:] = history_rows[:HISTORY_LIMIT]

# ============================================================
# TRADINGVIEW WEBSOCKET
# ─────────────────────────────────────────────────────────────
# FIX: The original handshake was double-encoding the heartbeat.
#      TradingView's protocol uses ~m~{len}~m~{payload} framing.
#      The heartbeat must be sent as a raw framed string, NOT as
#      a JSON object wrapped inside another ~m~ envelope.
# ============================================================
def _tv_frame(payload: str) -> str:
    """Wrap a raw string payload in TradingView's ~m~len~m~payload framing."""
    return f"~m~{len(payload)}~m~{payload}"

def _tv_json_frame(payload: dict) -> str:
    """Serialize a dict to JSON then wrap in TV framing."""
    raw = json.dumps(payload)
    return _tv_frame(raw)

def tradingview_ws_thread() -> None:
    retry_delay = 2
    while True:
        logger.info("Connecting to TradingView WebSocket...")

        def on_open(ws_app):
            nonlocal retry_delay
            retry_delay = 2

            session = "qs_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=12))

            # FIX: heartbeat is a plain framed string, not a JSON object
            ws_app.send(_tv_frame("~h~1"))

            ws_app.send(_tv_json_frame({"m": "set_locale",           "p": ["en", "US"]}))
            ws_app.send(_tv_json_frame({"m": "quote_create_session", "p": [session]}))
            ws_app.send(_tv_json_frame({"m": "quote_add_symbols",    "p": [session, SYMBOL]}))
            logger.info(f"TradingView WS connected, session={session}")

        def on_message(ws_app, raw):
            try:
                for seg in raw.split("~m~"):
                    if not seg or seg[0] != "{":
                        continue
                    data = json.loads(seg)
                    if data.get("m") == "qsd":
                        p = data.get("p", [])
                        if len(p) >= 2:
                            v = p[1].get("v", {})
                            price = v.get("lp")
                            if price:
                                ts = v.get("lp_time", time.time())
                                price_queue.put_nowait({"price": float(price), "ts": float(ts)})
            except Exception:
                pass

        def on_error(ws_app, err):
            logger.error(f"TradingView WS error: {err}")

        def on_close(ws_app, code, msg):
            logger.warning(f"TradingView WS closed (code={code}). Reconnecting in {retry_delay}s...")

        try:
            app_ws = ws_client.WebSocketApp(
                "wss://data.tradingview.com/socket.io/websocket",
                header={"Origin": "https://www.tradingview.com"},
                on_open=on_open, on_message=on_message,
                on_error=on_error, on_close=on_close,
            )
            app_ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as exc:
            logger.error(f"WS run error: {exc}")

        time.sleep(retry_delay)
        retry_delay = min(retry_delay * 2, 60)

# ============================================================
# BACKGROUND PROCESSOR & BROADCAST
# ============================================================
def price_processor_thread() -> None:
    global latest_price
    last_broadcast = 0.0
    while True:
        try:
            tick = price_queue.get(timeout=0.1)
            latest_price = tick["price"]
            process_price_tick(tick["price"], tick["ts"])
            now = time.time()
            if now - last_broadcast >= 0.2:
                last_broadcast = now
                _schedule_broadcast()
        except Empty:
            pass
        except Exception as exc:
            logger.error(f"Processor error: {exc}")

def _schedule_broadcast() -> None:
    if _main_loop and not _main_loop.is_closed():
        asyncio.run_coroutine_threadsafe(_broadcast_state(), _main_loop)

async def _broadcast_state() -> None:
    payload = _build_state_payload()
    msg = json.dumps(payload)
    with ws_clients_lock:
        clients = list(ws_clients)
    dead = []
    for client in clients:
        try:
            await client.send_text(msg)
        except Exception:
            dead.append(client)
    if dead:
        with ws_clients_lock:
            for d in dead:
                if d in ws_clients:
                    ws_clients.remove(d)

async def _periodic_broadcast() -> None:
    while True:
        await asyncio.sleep(1)
        await _broadcast_state()

def _build_state_payload() -> dict:
    with state_lock:
        if live_window_start is not None:
            next_ws  = live_window_start + WINDOW_SECONDS
            pred_now = pending_predictions.get(next_ws, {})
        else:
            pred_now = {}
        total = wins + losses
        acc   = round(wins / total * 100, 1) if total > 0 else None
        now_ts = time.time()
        if live_window_start is not None:
            window_end  = live_window_start + WINDOW_SECONDS
            secs_left   = max(0.0, window_end - now_ts)
            window_pct  = (WINDOW_SECONDS - secs_left) / WINDOW_SECONDS
            next_label  = window_label(live_window_start + WINDOW_SECONDS)
        else:
            secs_left, window_pct, next_label = float(WINDOW_SECONDS), 0.0, "--"
        lc = live_candle or {}
        table = [
            {
                "window":     r["predicted_window_label"],
                "predicted":  r["predicted"],
                "confidence": r["confidence"],
                "actual":     r["actual"],
                "result":     r["result"],
                "act_open":   r["act_open"],
                "act_close":  r["act_close"],
            }
            for r in history_rows[:HISTORY_LIMIT]
        ]
        return {
            "price":      round(latest_price, 2),
            "next_window": next_label,
            "secs_left":  round(secs_left, 1),
            "window_pct": round(window_pct * 100, 1),
            "signal":     pred_now.get("direction", "HOLD"),
            "confidence": pred_now.get("confidence", 0),
            "wins":       wins,
            "losses":     losses,
            "accuracy":   acc,
            "trained":    trained,
            "ohlc": {
                "open":  lc.get("open",  0),
                "high":  lc.get("high",  0),
                "low":   lc.get("low",   0),
                "close": lc.get("close", 0),
            },
            "table": table,
        }

# ============================================================
# HTML FRONTEND
# ─────────────────────────────────────────────────────────────
# FIX 1: .mono class and all inline font-family declarations
#         changed from 'Serif';; → 'Space Mono', monospace
# FIX 2: Font updated to Gill Sans throughout
# ============================================================
HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>BTC 5-Min Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
  *{margin:0;padding:0;box-sizing:border-box;}
  :root{
    --bg:#080C14;--surface:#0F1623;--card:#141D2E;--border:#1E2D45;
    --text:#C8D8EF;--muted:#4A6080;
    --green:#00E5A0;--red:#FF4560;--orange:#F7931A;--blue:#38BDF8;
    --green-dim:#00E5A015;--red-dim:#FF456015;--orange-dim:#F7931A15;
  }
  body{background:var(--bg);font-family:'Inter',sans-serif;color:var(--text);padding:20px;min-height:100vh;}

  /* FIX: was 'Serif';; — now correctly references the loaded Google Font */
  .mono{font-family:'Inter',sans-serif;}

  /* ── Header ── */
  .header{
    display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px;
    background:var(--surface);padding:14px 22px;border-radius:10px;
    border:1px solid var(--border);margin-bottom:18px;
  }
  .header h1{font-size:1.25rem;font-weight:800;color:var(--orange);letter-spacing:1px;}
  .stat{
    background:var(--card);padding:15px 14px;border-radius:10px;
    font-size:0.85rem;font-weight:700;font-family:'Inter',sans-serif;
    border:1px solid var(--border);
  }
  .stat.win{color:var(--green);}
  .stat.loss{color:var(--red);}
  .stat.acc{color:var(--orange);}

  /* ── Status bar ── */
  .status-bar{
    display:flex;align-items:center;gap:10px;
    background:var(--surface);padding:10px 16px;border-radius:2px;
    border:1px solid var(--border);margin-bottom:18px;font-size:0.72rem;
  }
  .dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;}
  .dot-ok{background:var(--green);box-shadow:0 0 6px var(--green);animation:pulse 2s infinite;}
  .dot-bad{background:var(--red);}
  .dot-wait{background:var(--orange);animation:pulse 0.8s infinite;}
  #clock-gmt3{margin-left:auto;font-family:'Inter',sans-serif;color:white;font-size:0.85rem;}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.35}}

  /* ── Main grid ── */
  .main-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px;}
  @media(max-width:820px){.main-grid{grid-template-columns:1fr;}}
  #tv-chart{height:400px;border-radius:10px;overflow:hidden;background:var(--card);border:1px solid var(--border);}

  /* ── Sidebar ── */
  .sidebar{display:flex;flex-direction:column;gap:14px;}
  .card{background:var(--card);border-radius:4px;padding:16px;border:1px solid var(--border);}
  .card-title{
    font-size:0.85rem;text-transform:uppercase;letter-spacing:2px;
    color:var(--muted);margin-bottom:10px;font-weight:600;
  }

  /* ── Price card ── */
  .price{font-size:2rem;font-weight:700;font-family:'Inter',sans-serif;margin-bottom:5px;}
  .pchange{font-size:0.78rem;padding:3px 10px;border-radius:5px;display:inline-block;font-family:'Inter',sans-serif;}

  /* ── Prediction card ── */
  .pred-row{display:flex;align-items:baseline;justify-content:space-between;margin:10px 0 8px;}
  .pred-arrow{font-size:2rem;line-height:1;}
  .pred-dir{font-size:1.7rem;font-weight:800;letter-spacing:2px;}
  .conf-bar{height:3px;background:var(--border);border-radius:2px;overflow:hidden;margin-top:4px;}
  .conf-fill{height:100%;width:0%;transition:width 0.4s ease;}

  /* ── Countdown ── */
  .countdown{display:flex;align-items:center;gap:14px;}
  .cd-ring{width:58px;height:58px;transform:rotate(-90deg);}
  .cd-text{font-size:1.8rem;font-weight:700;color:var(--orange);font-family:'Inter',sans-serif;}

  /* ── OHLC + Performance row ── */
  .bottom-row{display:grid;gap:14px;grid-template-columns:1fr 1fr;margin-bottom:14px;}
  @media(max-width:600px){.bottom-row{grid-template-columns:1fr;}}
  .ohlc-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:6px;}
  .ohlc-cell{background:var(--surface);border-radius:6px;padding:7px 10px;border:1px solid var(--border);}
  .ohlc-cell .lbl{font-size:0.6rem;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;}
  .ohlc-cell .val{font-size:0.88rem;font-weight:700;margin-top:2px;font-family:'Inter',sans-serif;}
  .perf-row{display:flex;gap:10px;margin-top:8px;}
  .perf-stat{flex:1;text-align:center;background:var(--surface);border-radius:7px;padding:8px;border:1px solid var(--border);}
  .perf-num{font-size:1.35rem;font-weight:700;font-family:'Inter',sans-serif;}
  .perf-lbl{font-size:0.6rem;color:var(--muted);letter-spacing:1.5px;margin-top:2px;text-transform:uppercase;}

  /* ── History table ── */
  .table-wrapper{overflow-x:auto;margin-top:10px;}
  table{width:100%;border-collapse:collapse;font-size:0.78rem;}
  th,td{padding:8px 8px;text-align:left;border-bottom:1px solid var(--border);}
  th{color:var(--muted);font-weight:600;font-size:0.65rem;text-transform:uppercase;letter-spacing:1px;}
  td{font-family:'Inter',sans-serif;}
  .up{color:var(--green);}
  .down{color:var(--red);}
  tr:last-child td{border-bottom:none;}
  tr:hover td{background:#ffffff04;}



  .disclaimer{
    background:var(--surface);border-radius:2px;padding:10px;
    font-size:1rem;text-align:center;color:#FACC15;
    border:1px solid #FACC1520;
  }

  /* ── Flash animations ── */
  .flash-up{animation:flashUp 0.4s ease;}
  .flash-dn{animation:flashDn 0.4s ease;}
  @keyframes flashUp{0%,100%{color:var(--text)}50%{color:var(--green)}}
  @keyframes flashDn{0%,100%{color:var(--text)}50%{color:var(--red)}}

</style>
</head>
<body>
<div class="container" style="max-width:1400px;margin:0 auto;">
  <!-- Header -->
  <div class="header">
    <h1>₿ Bitcoin Price Trend Predictor</h1>
  </div>

  <!-- Status bar -->
  <div class="status-bar">
    <div class="dot dot-wait" id="ws-dot"></div>
    <span id="ws-txt">Connecting…</span>
    <span id="clock-gmt3">--:--:-- GMT+3</span>
  </div>

  <!-- Main grid: chart + sidebar -->
  <div class="main-grid">
    <div id="tv-chart"><div id="tv-widget" style="width:100%;height:100%"></div></div>
    <div class="sidebar">
      <!-- Price -->
      <div class="card">
        <div class="card-title">Live BTC / USD</div>
        <div class="price mono" id="price-val">$---.--</div>
        <span class="pchange mono" id="price-change">--%</span>
      </div>
      <!-- Prediction -->
      <div class="card">
        <div class="card-title">Next 5-Min Prediction</div>
        <div class="pred-row">
          <span class="pred-arrow" id="pred-arrow">◆</span>
          <span class="pred-dir"   id="pred-dir">HOLD</span>
          <span class="mono" id="conf-pct" style="font-size:1.15rem;">--%</span>
        </div>
        <div class="conf-bar"><div class="conf-fill" id="conf-bar"></div></div>
        <div id="pred-window" style="margin-top:6px;font-size:0.68rem;color:var(--muted);"></div>
      </div>
      <!-- Countdown -->
      <div class="card">
        <div class="card-title">Next window opens in</div>
        <div class="countdown">
          <svg class="cd-ring" viewBox="0 0 72 72">
            <circle cx="36" cy="36" r="32" stroke="var(--border)" stroke-width="5" fill="none"/>
            <circle cx="36" cy="36" r="32" stroke="var(--orange)" stroke-width="5" fill="none"
              stroke-dasharray="201" stroke-dashoffset="201" id="cd-ring" stroke-linecap="round"/>
          </svg>
          <div class="cd-text" id="cd-val">5:00</div>
        </div>
      </div>
    </div>
  </div>

  <!-- OHLC + Performance -->
  <div class="bottom-row">
    <div class="card">
      <div class="card-title">Current Window</div>
      <div class="ohlc-grid">
        <div class="ohlc-cell"><div class="lbl">Open</div> <div class="val" id="o-open">--</div></div>
        <div class="ohlc-cell"><div class="lbl">High</div> <div class="val up" id="o-high">--</div></div>
        <div class="ohlc-cell"><div class="lbl">Low</div>  <div class="val down" id="o-low">--</div></div>
        <div class="ohlc-cell"><div class="lbl">Close</div><div class="val" id="o-close">--</div></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Performance</div>
      <div class="perf-row">
        <div class="perf-stat">
          <div class="perf-num up"   id="p-wins">0</div>
          <div class="perf-lbl">Wins</div>
        </div>
        <div class="perf-stat">
          <div class="perf-num down" id="p-losses">0</div>
          <div class="perf-lbl">Losses</div>
        </div>
        <div class="perf-stat">
          <div class="perf-num" id="p-acc" style="color:var(--orange);">--%</div>
          <div class="perf-lbl">Accuracy</div>
        </div>
      </div>
    </div>
  </div>

  <!-- History table -->
  <div class="card" style="margin-bottom:14px;">
    <div class="card-title">Prediction History (last 5)</div>
    <div class="table-wrapper">
      <table>
        <thead>
          <tr>
            <th>Window</th><th>Prediction</th><th>Conf</th>
            <th>Act. Open</th><th>Act. Close</th><th>Actual</th><th>Result</th>
          </tr>
        </thead>
        <tbody id="history-body">
          <tr><td colspan="7" style="text-align:center;color:var(--muted);padding:18px;">Waiting for data…</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <div class="disclaimer">⚠️ For educational purpose only. Past accuracy does not guarantee future results.</div>
</div>

<script src="https://s3.tradingview.com/tv.js"></script>
<script>
  // ── TradingView widget ──────────────────────────────────────
  new TradingView.widget({
    container_id: 'tv-widget', symbol: 'BITSTAMP:BTCUSD', interval: '5',
    theme: 'dark', style: '1', locale: 'en', toolbar_bg: '#080C14',
    enable_publishing: false, autosize: true, hide_side_toolbar: false
  });

  // ── State ───────────────────────────────────────────────────
  let ws, firstPrice = null, prevPrice = null;
  const CIRC = 2 * Math.PI * 32;

  // ── GMT+3 clock ─────────────────────────────────────────────
  function updateClock() {
    const gmt3dt = new Date(Date.now() + 3 * 60 * 60 * 1000);
    const hh = String(gmt3dt.getUTCHours()).padStart(2, '0');
    const mm = String(gmt3dt.getUTCMinutes()).padStart(2, '0');
    const ss = String(gmt3dt.getUTCSeconds()).padStart(2, '0');
    document.getElementById('clock-gmt3').textContent = `${hh}:${mm}:${ss} GMT+3`;
  }
  updateClock();
  setInterval(updateClock, 1000);

  // ── Format price ─────────────────────────────────────────────
  function fmt(n) {
    if (n == null || n === '--' || n === 0) return '--';
    return '$' + Number(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  // ── WebSocket ───────────────────────────────────────────────
  function connect() {
    ws = new WebSocket(`ws://${location.host}/ws`);
    ws.onopen  = () => {
      document.getElementById('ws-dot').className = 'dot dot-ok';
      document.getElementById('ws-txt').textContent = 'Live';
    };
    ws.onclose = () => {
      document.getElementById('ws-dot').className = 'dot dot-bad';
      document.getElementById('ws-txt').textContent = 'Disconnected';
      setTimeout(connect, 3000);
    };
    ws.onerror = () => {
      document.getElementById('ws-dot').className = 'dot dot-bad';
      document.getElementById('ws-txt').textContent = 'Error';
    };
    ws.onmessage = e => render(JSON.parse(e.data));
  }

  // ── Render ──────────────────────────────────────────────────
  function render(d) {
    if (d.price != null) {
      if (firstPrice == null) firstPrice = d.price;
      const chg    = (d.price - firstPrice) / firstPrice * 100;
      const priceEl = document.getElementById('price-val');
      if (prevPrice != null) {
        priceEl.classList.remove('flash-up', 'flash-dn');
        void priceEl.offsetWidth;
        priceEl.classList.add(d.price > prevPrice ? 'flash-up' : 'flash-dn');
      }
      priceEl.textContent = fmt(d.price);
      const pch = document.getElementById('price-change');
      pch.textContent = (chg >= 0 ? '+' : '') + chg.toFixed(2) + '%';
      pch.style.background = chg >= 0 ? 'var(--green-dim)' : 'var(--red-dim)';
      pch.style.color       = chg >= 0 ? 'var(--green)'    : 'var(--red)';
      prevPrice = d.price;
    }

    const secs = d.secs_left || 0;
    const m = Math.floor(secs / 60), s = Math.floor(secs % 60);
    document.getElementById('cd-val').textContent = `${m}:${String(s).padStart(2, '0')}`;
    const pct = Math.min(1, (d.window_pct || 0) / 100);
    document.getElementById('cd-ring').style.strokeDashoffset = CIRC * (1 - pct);

    const sig  = d.signal || 'HOLD';
    const isUp = sig === 'UP', isDn = sig === 'DOWN';
    const col  = isUp ? 'var(--green)' : isDn ? 'var(--red)' : 'var(--muted)';
    document.getElementById('pred-arrow').textContent = isUp ? '▲' : isDn ? '▼' : '◆';
    document.getElementById('pred-arrow').style.color  = col;
    document.getElementById('pred-dir').textContent   = isUp ? 'UP' : isDn ? 'DOWN' : 'HOLD';
    document.getElementById('pred-dir').style.color   = col;
    document.getElementById('conf-pct').textContent   = (isUp || isDn) ? d.confidence + '%' : '--%';
    document.getElementById('conf-pct').style.color   = col;
    document.getElementById('conf-bar').style.width      = (d.confidence || 0) + '%';
    document.getElementById('conf-bar').style.background = col;
    document.getElementById('pred-window').textContent = d.next_window ? `Next: ${d.next_window}` : '';

    if (d.ohlc) {
      document.getElementById('o-open').textContent  = fmt(d.ohlc.open);
      document.getElementById('o-high').textContent  = fmt(d.ohlc.high);
      document.getElementById('o-low').textContent   = fmt(d.ohlc.low);
      document.getElementById('o-close').textContent = fmt(d.ohlc.close);
    }

    const wins   = d.wins   || 0;
    const losses = d.losses || 0;
    const total  = wins + losses;
    const acc    = total ? (wins / total * 100).toFixed(1) + '%' : '--%';
    document.getElementById('p-wins').textContent        = wins;
    document.getElementById('p-losses').textContent      = losses;
    document.getElementById('p-acc').textContent         = acc;

    const tbody = document.getElementById('history-body');
    if (d.table && d.table.length) {
      tbody.innerHTML = d.table.map(r => {
        const predCls = r.predicted === 'UP' ? 'up' : r.predicted === 'DOWN' ? 'down' : '';
        const actCls  = r.actual    === 'UP' ? 'up' : r.actual    === 'DOWN' ? 'down' : '';
        const predTxt = r.predicted === 'UP' ? '▲ UP' : r.predicted === 'DOWN' ? '▼ DOWN' : r.predicted;
        const actTxt  = r.actual === '⏳'    ? '--'   : r.actual  === 'UP'    ? '▲ UP'   : r.actual === 'DOWN' ? '▼ DOWN' : r.actual;
        const resTxt  = r.result === '⏳'    ? '--'   : r.result;
        return `<tr>
          <td style="color:var(--muted)">${r.window}</td>
          <td class="${predCls}">${predTxt}</td>
          <td style="color:var(--orange)">${r.confidence}%</td>
          <td>${fmt(r.act_open)}</td>
          <td>${fmt(r.act_close)}</td>
          <td class="${actCls}">${actTxt}</td>
          <td style="font-size:1rem;">${resTxt}</td>
        </tr>`;
      }).join('');
    } else {
      tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--muted);padding:18px;">Waiting for data…</td></tr>';
    }
  }



  connect();
</script>
</body>
</html>
"""

# ============================================================
# FASTAPI APP
# ============================================================
@asynccontextmanager
async def lifespan(application: FastAPI):
    global _main_loop
    _main_loop = asyncio.get_running_loop()

    synth = generate_synthetic_history()
    with state_lock:
        for c in synth:
            completed_candles.append(c)
    threading.Thread(target=train_model_from_candles, args=(list(completed_candles),), daemon=True).start()

    threading.Thread(target=tradingview_ws_thread, daemon=True).start()
    threading.Thread(target=price_processor_thread, daemon=True).start()

    asyncio.create_task(_periodic_broadcast())

    logger.info("=" * 52)
    logger.info("BTC Predictor ready → http://localhost:8000")
    logger.info("Feed: TradingView BITSTAMP:BTCUSD | Clock: GMT+3")
    logger.info("=" * 52)

    yield


app = FastAPI(title="BTC Predictor", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_CONTENT


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    with ws_clients_lock:
        ws_clients.append(websocket)
    logger.info(f"Client connected. Total={len(ws_clients)}")
    try:
        await websocket.send_text(json.dumps(_build_state_payload()))
    except Exception:
        pass
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        with ws_clients_lock:
            if websocket in ws_clients:
                ws_clients.remove(websocket)
        logger.info(f"Client disconnected. Total={len(ws_clients)}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)