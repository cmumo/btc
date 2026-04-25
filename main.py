"""
Bitcoin 5-Minute Prediction Terminal
=====================================
- TradingView chart (BITSTAMP:BTCUSD) + Coinbase WebSocket for live prices
- XGBoost model, 5-min candles, win/loss tracking
- SQLite database for persistent storage
- GMT+3 timezone, prices with 2 decimal places
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
from typing import Deque, List, Optional

import numpy as np
import pandas as pd
import pickle
import websocket as ws_client
import xgboost as xgb
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
WINDOW_SECONDS    = 300
HISTORY_LIMIT     = 100
MIN_CANDLES_TRAIN = 12
RETRAIN_EVERY     = 5
TZ                = timezone(timedelta(hours=3))  # GMT+3

# Database setup
DB_PATH = os.environ.get("DATABASE_PATH", "/data/predictions.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# ============================================================
# DATABASE SETUP
# ============================================================
def init_database():
    """Initialize SQLite database tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table for completed candles
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS candles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL UNIQUE,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        )
    ''')
    
    # Table for predictions history
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            window_time TEXT,
            predicted_signal TEXT,
            confidence INTEGER,
            actual_signal TEXT,
            actual_open REAL,
            actual_close REAL,
            result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table for model storage (binary)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_data BLOB,
            trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            wins INTEGER,
            losses INTEGER
        )
    ''')
    
    # Table for stats
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stats (
            key TEXT PRIMARY KEY,
            value INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")

def save_candle(candle: dict):
    """Save a completed candle to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO candles (timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (candle["ts"], candle["open"], candle["high"], candle["low"], candle["close"], candle["volume"]))
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to save candle: {e}")
    finally:
        conn.close()

def load_candles(limit: int = HISTORY_LIMIT) -> List[dict]:
    """Load recent candles from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT timestamp, open, high, low, close, volume 
        FROM candles 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    candles = []
    for row in reversed(rows):  # Oldest first
        candles.append({
            "ts": row[0],
            "open": row[1],
            "high": row[2],
            "low": row[3],
            "close": row[4],
            "volume": row[5]
        })
    return candles

def save_prediction(prediction: dict):
    """Save a prediction to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO predictions (window_time, predicted_signal, confidence, actual_signal, actual_open, actual_close, result)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (prediction["window"], prediction["predicted"], prediction["confidence"], 
              prediction.get("actual", "⏳"), prediction.get("act_open", 0), 
              prediction.get("act_close", 0), prediction.get("result", "⏳")))
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to save prediction: {e}")
    finally:
        conn.close()

def update_prediction_result(window_time: str, actual_signal: str, actual_open: float, actual_close: float, result: str):
    """Update prediction with actual result"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            UPDATE predictions 
            SET actual_signal = ?, actual_open = ?, actual_close = ?, result = ?
            WHERE window_time = ?
        ''', (actual_signal, actual_open, actual_close, result, window_time))
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to update prediction: {e}")
    finally:
        conn.close()

def load_predictions(limit: int = 5) -> List[dict]:
    """Load recent predictions from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT window_time, predicted_signal, confidence, actual_signal, actual_open, actual_close, result
        FROM predictions 
        ORDER BY id DESC 
        LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    predictions = []
    for row in reversed(rows):  # Chronological order
        predictions.append({
            "window": row[0],
            "predicted": row[1],
            "confidence": row[2],
            "actual": row[3] if row[3] else "⏳",
            "act_open": row[4] or 0,
            "act_close": row[5] or 0,
            "result": row[6] if row[6] else "⏳"
        })
    return predictions

def save_model(model_obj, wins: int, losses: int):
    """Save model to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        model_bytes = pickle.dumps(model_obj)
        cursor.execute('''
            INSERT OR REPLACE INTO model (id, model_data, wins, losses)
            VALUES (1, ?, ?, ?)
        ''', (model_bytes, wins, losses))
        conn.commit()
        logger.info(f"Model saved to database - Wins: {wins}, Losses: {losses}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
    finally:
        conn.close()

def load_model():
    """Load model from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT model_data, wins, losses FROM model WHERE id = 1')
    row = cursor.fetchone()
    conn.close()
    
    if row:
        try:
            model_obj = pickle.loads(row[0])
            wins = row[1]
            losses = row[2]
            logger.info(f"Model loaded from database - Wins: {wins}, Losses: {losses}")
            return model_obj, wins, losses
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    return None, 0, 0

def save_stats(wins: int, losses: int):
    """Save win/loss stats to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT OR REPLACE INTO stats (key, value) VALUES (?, ?)', ("wins", wins))
    cursor.execute('INSERT OR REPLACE INTO stats (key, value) VALUES (?, ?)', ("losses", losses))
    conn.commit()
    conn.close()

def load_stats():
    """Load win/loss stats from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT key, value FROM stats')
    rows = cursor.fetchall()
    conn.close()
    
    wins = 0
    losses = 0
    for key, value in rows:
        if key == "wins":
            wins = value
        elif key == "losses":
            losses = value
    return wins, losses

# ============================================================
# SHARED STATE
# ============================================================
state_lock             = threading.Lock()
completed_candles:     Deque[dict]    = deque(maxlen=HISTORY_LIMIT)
history_rows:          List[dict]     = []
live_candle:           Optional[dict] = None
live_window_start:     Optional[float] = None
wins:                  int = 0
losses:                int = 0
candles_since_retrain: int = 0

model      = None
model_lock = threading.Lock()

price_queue: Queue = Queue()

_clients:      List[WebSocket]                    = []
_clients_lock: Optional[asyncio.Lock]             = None
_main_loop:    Optional[asyncio.AbstractEventLoop] = None

# ============================================================
# SYNTHETIC HISTORY
# ============================================================
def generate_synthetic_history(n: int = 60) -> List[dict]:
    candles, price = [], 65_000.0
    t = time.time() - n * WINDOW_SECONDS
    for i in range(n):
        o  = price + random.gauss(0, 50)
        c  = o     + random.gauss(0, 120)
        h  = max(o, c) + abs(random.gauss(0, 25))
        lo = min(o, c) - abs(random.gauss(0, 25))
        candles.append({
            "ts":     t + i * WINDOW_SECONDS,
            "open":   round(o,  2),
            "high":   round(h,  2),
            "low":    round(lo, 2),
            "close":  round(c,  2),
            "volume": round(random.uniform(5, 50), 4),
        })
        price = c
    return candles

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def make_features(candles: List[dict]) -> Optional[pd.DataFrame]:
    if len(candles) < 6:
        return None
    df = pd.DataFrame(candles)
    df["ret"]       = df["close"].pct_change()
    df["hl"]        = (df["high"] - df["low"]) / df["close"]
    df["oc"]        = (df["close"] - df["open"]) / df["open"]
    df["ma3"]       = df["close"].rolling(3).mean()
    df["ma5"]       = df["close"].rolling(5).mean()
    df["ma_diff"]   = (df["ma3"] - df["ma5"]) / df["close"]
    df["vol_ma3"]   = df["volume"].rolling(3).mean()
    df["vol_ratio"] = df["volume"] / (df["vol_ma3"] + 1e-9)
    df["ret_lag1"]  = df["ret"].shift(1)
    df["ret_lag2"]  = df["ret"].shift(2)
    df["momentum"]  = df["close"] - df["close"].shift(3)
    df.dropna(inplace=True)
    cols = ["ret", "hl", "oc", "ma_diff", "vol_ratio", "ret_lag1", "ret_lag2", "momentum"]
    return df[cols] if len(df) >= 1 else None

# ============================================================
# MODEL TRAINING
# ============================================================
def train_model_from_candles(candles: List[dict]) -> None:
    global model
    if len(candles) < MIN_CANDLES_TRAIN + 2:
        return
    try:
        features_df = make_features(candles[:-1])
        if features_df is None or len(features_df) == 0:
            return
        n      = len(features_df)
        offset = (len(candles) - 1) - n
        labels = []
        for i in range(n):
            nxt_idx = offset + i + 1
            if nxt_idx >= len(candles):
                break
            nxt = candles[nxt_idx]
            labels.append(1 if nxt["close"] >= nxt["open"] else 0)
        if len(labels) < MIN_CANDLES_TRAIN:
            return
        features_df = features_df.iloc[:len(labels)]
        clf = xgb.XGBClassifier(
            n_estimators=60, max_depth=3,
            learning_rate=0.1, eval_metric="logloss", verbosity=0,
        )
        clf.fit(features_df.values, labels)
        with model_lock:
            model = clf
        # Save model to database
        save_model(clf, wins, losses)
        logger.info(f"Model trained on {len(labels)} samples and saved to DB")
    except Exception as exc:
        logger.error(f"Training error: {exc}")

# ============================================================
# PREDICTION
# ============================================================
def predict_from_candles(candles: List[dict]) -> dict:
    default = {"signal": "HOLD", "confidence": 0, "next_window": ""}
    with model_lock:
        mdl = model
    if mdl is None or len(candles) < 6:
        return default
    try:
        features_df = make_features(candles)
        if features_df is None or len(features_df) == 0:
            return default
        prob = mdl.predict_proba(features_df.iloc[[-1]].values)[0]
        up_p = float(prob[1])
        conf = int(round(max(up_p, 1 - up_p) * 100))
        if   up_p > 0.55: signal = "UP"
        elif up_p < 0.45: signal = "DOWN"
        else:             signal = "HOLD"
        now      = datetime.now(TZ)
        next_min = ((now.minute // 5) + 1) * 5
        delta    = timedelta(
            minutes      = next_min - now.minute,
            seconds      = -now.second,
            microseconds = -now.microsecond,
        )
        nxt_time = (now + delta).strftime("%H:%M")
        return {"signal": signal, "confidence": conf, "next_window": nxt_time}
    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        return default

# ============================================================
# CANDLE BUILDING
# ============================================================
def process_price_tick(price: float, trade_ts: float) -> None:
    global live_candle, live_window_start, candles_since_retrain, wins, losses
    with state_lock:
        if live_window_start is None:
            live_window_start = trade_ts - (trade_ts % WINDOW_SECONDS)
        if trade_ts >= live_window_start + WINDOW_SECONDS:
            if live_candle is not None:
                _close_current_window_locked()
            live_window_start = trade_ts - (trade_ts % WINDOW_SECONDS)
            live_candle = _new_candle(live_window_start, price)
        elif live_candle is None:
            live_candle = _new_candle(live_window_start, price)
        else:
            live_candle["high"]  = max(live_candle["high"],  price)
            live_candle["low"]   = min(live_candle["low"],   price)
            live_candle["close"] = price

def _new_candle(ts: float, price: float) -> dict:
    return {"ts": ts, "open": price, "high": price, "low": price, "close": price, "volume": 0.0}

def _close_current_window_locked() -> None:
    global live_candle, candles_since_retrain, wins, losses
    candle       = dict(live_candle)
    live_candle  = None
    completed_candles.append(candle)
    candles_since_retrain += 1
    
    # Save candle to database
    save_candle(candle)

    # Update prediction with actual result
    for row in reversed(history_rows):
        if row.get("actual") == "\u23f3":
            actual           = "UP" if candle["close"] > candle["open"] else "DOWN"
            row["actual"]    = actual
            row["act_open"]  = candle["open"]
            row["act_close"] = candle["close"]
            if row["predicted"] == actual:
                row["result"] = "\u2705"
                wins += 1
            else:
                row["result"] = "\u274c"
                losses += 1
            
            # Update in database
            update_prediction_result(row["window"], actual, candle["open"], candle["close"], row["result"])
            save_stats(wins, losses)
            break

    snap = list(completed_candles)
    pred = predict_from_candles(snap)
    dt   = datetime.fromtimestamp(candle["ts"], tz=TZ).strftime("%H:%M")
    
    prediction_record = {
        "window": dt,
        "predicted": pred["signal"],
        "confidence": pred["confidence"],
        "act_open": 0.0,
        "act_close": 0.0,
        "actual": "⏳",
        "result": "⏳"
    }
    history_rows.append(prediction_record)
    
    # Save prediction to database
    save_prediction(prediction_record)
    
    # Trim history
    del history_rows[:-HISTORY_LIMIT]

    if candles_since_retrain >= RETRAIN_EVERY:
        candles_since_retrain = 0
        threading.Thread(
            target=train_model_from_candles, args=(snap,), daemon=True
        ).start()

# ============================================================
# COINBASE WEBSOCKET
# ============================================================
def _coinbase_thread() -> None:
    url = "wss://ws-feed.exchange.coinbase.com"
    retry_delay = 2
    
    while True:
        logger.info("Connecting to Coinbase WebSocket...")
        try:
            def on_open(ws_obj):
                nonlocal retry_delay
                retry_delay = 2
                subscribe_msg = json.dumps({
                    "type": "subscribe",
                    "product_ids": ["BTC-USD"],
                    "channels": ["matches"]
                })
                ws_obj.send(subscribe_msg)
                logger.info("Coinbase WS connected - receiving BTC-USD trades")

            def on_message(ws_obj, raw):
                try:
                    data = json.loads(raw)
                    if data.get("type") == "match":
                        price_queue.put_nowait({
                            "price": float(data["price"]),
                            "ts":    time.time(),
                        })
                except Exception:
                    pass

            def on_error(ws_obj, err):
                logger.error(f"Coinbase WS error: {err}")

            def on_close(ws_obj, code, msg):
                logger.warning("Coinbase WS closed")

            app_ws = ws_client.WebSocketApp(
                url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            app_ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as exc:
            logger.error(f"Coinbase thread exception: {exc}")

        logger.warning(f"Coinbase WS reconnecting in {retry_delay}s...")
        time.sleep(retry_delay)
        retry_delay = min(retry_delay * 2, 60)

# ============================================================
# PRICE PROCESSOR THREAD
# ============================================================
def price_processor_thread() -> None:
    while True:
        try:
            item = price_queue.get(timeout=1.0)
            process_price_tick(item["price"], item["ts"])
        except Empty:
            continue
        except Exception as exc:
            logger.error(f"Price processor error: {exc}")

# ============================================================
# BROADCAST
# ============================================================
async def _periodic_broadcast() -> None:
    while True:
        await asyncio.sleep(1)
        await _broadcast_state()

async def _broadcast_state() -> None:
    payload  = _build_state_payload()
    msg      = json.dumps(payload)
    async with _clients_lock:
        snapshot = list(_clients)
    dead = []
    for ws in snapshot:
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
        rows = list(history_rows[-5:])

    pred       = predict_from_candles(snap) if snap else \
                 {"signal": "HOLD", "confidence": 0, "next_window": ""}
    ohlc       = ({k: round(lc[k], 2) for k in ("open", "high", "low", "close")} if lc else {})
    live_price = lc["close"] if lc else (snap[-1]["close"] if snap else 0.0)

    with model_lock:
        model_ready = model is not None

    return {
        "price":       round(live_price, 2),
        "signal":      pred["signal"],
        "confidence":  pred["confidence"],
        "next_window": pred["next_window"],
        "wins":        w,
        "losses":      l,
        "ohlc":        ohlc,
        "table":       rows,
        "model_ready": model_ready,
    }

# ============================================================
# HTML FRONTEND (same as before - omitted for brevity)
# ============================================================
# [Your existing HTML_CONTENT here - unchanged]

# ============================================================
# FASTAPI APPLICATION
# ============================================================
@asynccontextmanager
async def lifespan(application: FastAPI):
    global _main_loop, _clients_lock, model, wins, losses, completed_candles, history_rows
    
    _main_loop    = asyncio.get_running_loop()
    _clients_lock = asyncio.Lock()
    
    # Initialize database
    init_database()
    
    # Load saved model and stats from database
    saved_model, saved_wins, saved_losses = load_model()
    if saved_model:
        with model_lock:
            model = saved_model
        wins = saved_wins
        losses = saved_losses
        logger.info(f"Loaded saved model with {wins} wins, {losses} losses")
    
    # Load saved predictions history
    saved_predictions = load_predictions(HISTORY_LIMIT)
    if saved_predictions:
        with state_lock:
            history_rows = saved_predictions
        logger.info(f"Loaded {len(saved_predictions)} predictions from database")
    
    # Load saved candles
    saved_candles = load_candles(HISTORY_LIMIT)
    if saved_candles:
        with state_lock:
            for candle in saved_candles:
                completed_candles.append(candle)
        logger.info(f"Loaded {len(saved_candles)} candles from database")
    
    # If no model exists, generate synthetic data and train
    if model is None:
        logger.info("No saved model found, generating synthetic data...")
        synth = generate_synthetic_history()
        with state_lock:
            for c in synth:
                completed_candles.append(c)
                save_candle(c)  # Save synthetic candles too
        threading.Thread(
            target=train_model_from_candles,
            args=(list(completed_candles),),
            daemon=True, name="initial-train",
        ).start()

    # Start threads
    threading.Thread(target=price_processor_thread, name="price-proc", daemon=True).start()
    threading.Thread(target=_coinbase_thread, name="coinbase-ws", daemon=True).start()
    
    # Periodic broadcast
    asyncio.create_task(_periodic_broadcast())

    port = os.environ.get("PORT", "8000")
    logger.info("=" * 52)
    logger.info(f"BTC Predictor ready — Coinbase live feed — Database at {DB_PATH}")
    logger.info("=" * 52)
    yield
    
    # Save final state on shutdown
    if model:
        save_model(model, wins, losses)
    save_stats(wins, losses)
    logger.info("Final state saved to database")

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
                text = msg.get("text")
                if text == "ping":
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

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        ws="websockets",
        log_level="info",
    )
