"""
Bitcoin 5-Minute Prediction Terminal
=====================================
- TradingView WebSocket (unofficial) for live BTC price ticks
- TradingView chart widget (BINANCE:BTCUSDT) — same source as backend
- XGBoost model, 5-min candles, win/loss tracking
- GMT+3 timezone, prices with 2 decimal places
- Correct port binding for Render ($PORT)
- WebSocket keepalive to prevent Render proxy timeout
"""

import asyncio
import json
import logging
import os
import random
import re
import string
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from queue import Empty, Queue
from typing import Deque, List, Optional

import numpy as np
import pandas as pd
import websocket as ws_client
import xgboost as xgb
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================
WINDOW_SECONDS    = 300          # 5-minute candles
HISTORY_LIMIT     = 100
MIN_CANDLES_TRAIN = 12
RETRAIN_EVERY     = 5
TZ                = timezone(timedelta(hours=3))   # GMT+3

# TradingView symbol — must match the chart widget below
TV_SYMBOL  = "BINANCE:BTCUSDT"
TV_WS_URL  = "wss://data.tradingview.com/socket.io/websocket?from=chart%2F&date="
TV_HEADERS = {
    "Origin":     "https://www.tradingview.com",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# ============================================================
# SHARED STATE  (all mutation under state_lock)
# ============================================================
state_lock            = threading.Lock()
completed_candles:    Deque[dict]   = deque(maxlen=HISTORY_LIMIT)
history_rows:         List[dict]    = []
live_candle:          Optional[dict] = None
live_window_start:    Optional[float] = None
wins:                 int = 0
losses:               int = 0
candles_since_retrain: int = 0

model      = None
model_lock = threading.Lock()

price_queue: Queue = Queue()

_clients:      List[WebSocket]                      = []
_clients_lock: Optional[asyncio.Lock]               = None   # created in lifespan
_main_loop:    Optional[asyncio.AbstractEventLoop]  = None

# ============================================================
# TRADINGVIEW WEBSOCKET HELPERS
# ============================================================
def _rand_str(n: int = 12) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def _pack(text: str) -> str:
    """Wrap a string in TradingView wire format:  ~m~<len>~m~<text>"""
    return f"~m~{len(text)}~m~{text}"


def _tv_msg(func: str, args: list) -> str:
    """Serialise and pack a TradingView JSON message."""
    body = json.dumps({"m": func, "p": args}, separators=(",", ":"))
    return _pack(body)


def _parse_packets(raw: str) -> List[str]:
    """
    Extract all payloads from a raw TV frame.
    Wire format: ~m~<len>~m~<payload>  (may repeat multiple times per recv)
    """
    packets, i = [], 0
    while i < len(raw):
        if raw[i: i + 3] != "~m~":
            i += 1
            continue
        try:
            end_len = raw.index("~m~", i + 3)
            length  = int(raw[i + 3: end_len])
            start   = end_len + 3
            packets.append(raw[start: start + length])
            i = start + length
        except (ValueError, IndexError):
            break
    return packets


# ============================================================
# TRADINGVIEW WEBSOCKET THREAD  (replaces Binance WS)
# ============================================================
def tradingview_ws_thread() -> None:
    """
    Connects to TradingView's real-time quote feed, subscribes to
    TV_SYMBOL, and pushes every last-price tick into price_queue.

    TradingView wire protocol:
      - Messages are packed as  ~m~<len>~m~<json>
      - Server sends heartbeats like  ~h~N  — we must echo them back
      - Quote updates arrive as  {"m":"qsd","p":[session, symbol, {"v":{"lp":<price>,...}}]}
    """
    retry_delay = 2

    while True:
        logger.info("Connecting to TradingView WebSocket ...")
        try:
            ws = ws_client.create_connection(
                TV_WS_URL,
                header=[f"{k}: {v}" for k, v in TV_HEADERS.items()],
                timeout=15,
            )

            qs = "qs_" + _rand_str()   # unique quote-session id

            # Handshake sequence
            ws.send(_tv_msg("set_auth_token",       ["unauthorized_user_token"]))
            ws.send(_tv_msg("quote_create_session",  [qs]))
            ws.send(_tv_msg("quote_set_fields",      [
                qs, "lp", "lp_time", "volume", "ch", "chp", "bid", "ask",
            ]))
            ws.send(_tv_msg("quote_add_symbols",     [
                qs, TV_SYMBOL, {"flags": ["force_permission"]},
            ]))
            ws.send(_tv_msg("quote_fast_symbols",    [qs, TV_SYMBOL]))

            logger.info(f"TradingView WS connected (session={qs})")
            retry_delay = 2   # reset backoff on successful connect

            while True:
                raw = ws.recv()
                if not raw:
                    continue

                packets = _parse_packets(raw)

                for packet in packets:
                    if not packet:
                        continue

                    # Echo heartbeats so the server keeps the connection alive
                    if packet.startswith("~h~"):
                        ws.send(_pack(packet))
                        continue

                    try:
                        msg = json.loads(packet)
                    except (json.JSONDecodeError, ValueError):
                        continue

                    # Real-time quote update
                    if msg.get("m") == "qsd":
                        p = msg.get("p", [])
                        if len(p) >= 2:
                            v  = p[1].get("v", {})
                            lp = v.get("lp")
                            if lp is not None:
                                ts = float(v["lp_time"]) if v.get("lp_time") else time.time()
                                price_queue.put_nowait({"price": float(lp), "ts": ts})

        except ws_client.WebSocketConnectionClosedException:
            logger.warning("TradingView WS connection closed")
        except Exception as exc:
            logger.error(f"TradingView WS error: {exc}")

        logger.info(f"TradingView WS reconnecting in {retry_delay}s ...")
        time.sleep(retry_delay)
        retry_delay = min(retry_delay * 2, 60)


# ============================================================
# SYNTHETIC HISTORY  (warm-start the model before live data)
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
    cols = ["ret", "hl", "oc", "ma_diff", "vol_ratio",
            "ret_lag1", "ret_lag2", "momentum"]
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
        logger.info(f"Model trained on {len(labels)} samples")
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
    return {"ts": ts, "open": price, "high": price,
            "low": price, "close": price, "volume": 0.0}


def _close_current_window_locked() -> None:
    """Must be called with state_lock held."""
    global live_candle, candles_since_retrain, wins, losses

    candle       = dict(live_candle)
    live_candle  = None
    completed_candles.append(candle)
    candles_since_retrain += 1

    # Score the most recent pending prediction
    for row in reversed(history_rows):
        if row.get("actual") == "⏳":
            actual           = "UP" if candle["close"] > candle["open"] else "DOWN"
            row["actual"]    = actual
            row["act_open"]  = candle["open"]
            row["act_close"] = candle["close"]
            if row["predicted"] == actual:
                row["result"] = "✅"; wins   += 1
            else:
                row["result"] = "❌"; losses += 1
            break

    # New prediction for the window that just opened
    snap = list(completed_candles)
    pred = predict_from_candles(snap)
    dt   = datetime.fromtimestamp(candle["ts"], tz=TZ).strftime("%H:%M")

    history_rows.append({
        "window":     dt,
        "predicted":  pred["signal"],
        "confidence": pred["confidence"],
        "act_open":   0.0,
        "act_close":  0.0,
        "actual":     "⏳",
        "result":     "⏳",
    })
    del history_rows[:-HISTORY_LIMIT]

    if candles_since_retrain >= RETRAIN_EVERY:
        candles_since_retrain = 0
        threading.Thread(
            target=train_model_from_candles,
            args=(snap,), daemon=True,
        ).start()


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
# WEBSOCKET BROADCAST
# ============================================================
async def _periodic_broadcast() -> None:
    while True:
        await asyncio.sleep(1)
        await _broadcast_state()


async def _broadcast_state() -> None:
    payload = _build_state_payload()
    msg     = json.dumps(payload)
    async with _clients_lock:
        dead = []
        for ws in list(_clients):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
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
    ohlc       = ({k: round(lc[k], 2) for k in ("open", "high", "low", "close")}
                  if lc else {})
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
# HTML FRONTEND
# ============================================================
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC 5-Min Predictor</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#080C14;color:#E0E6F0;font-family:'Segoe UI',sans-serif;min-height:100vh}
  .container{max-width:1200px;margin:0 auto;padding:16px}
  .header{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}
  .logo{font-size:1.3rem;font-weight:700;color:#F7931A}
  .status{display:flex;align-items:center;gap:8px;font-size:.85rem}
  .dot{width:8px;height:8px;border-radius:50%;display:inline-block}
  .dot-ok{background:#00E5A0;box-shadow:0 0 6px #00E5A0}
  .dot-bad{background:#FF4560}
  .clock{color:#4A6080;font-size:.85rem}
  .grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:12px}
  .grid-2{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
  .card{background:#0D1421;border:1px solid #1A2540;border-radius:10px;padding:16px}
  .card-title{font-size:.75rem;color:#4A6080;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px}
  #tv-widget{height:320px;border-radius:8px;overflow:hidden;margin-bottom:12px;border:1px solid #1A2540}
  .price-big{font-size:2.2rem;font-weight:700;color:#F7931A;letter-spacing:.02em}
  .price-change{font-size:.85rem;margin-top:4px}
  .pred-box{text-align:center}
  .pred-arrow{font-size:3rem;line-height:1}
  .pred-dir{font-size:1.4rem;font-weight:700;margin-top:4px}
  .conf-label{font-size:.75rem;color:#4A6080;margin-top:8px}
  .conf-bar-bg{background:#1A2540;border-radius:4px;height:6px;margin-top:6px}
  .conf-bar{height:6px;border-radius:4px;transition:width .5s}
  .pred-window{font-size:.8rem;color:#4A6080;margin-top:6px}
  .ohlc-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  .ohlc-cell{background:#080C14;border-radius:6px;padding:8px;text-align:center}
  .lbl{font-size:.7rem;color:#4A6080;margin-bottom:4px}
  .up{color:#00E5A0}.down{color:#FF4560}
  .perf-row{display:flex;justify-content:space-around;text-align:center}
  .perf-num{font-size:1.6rem;font-weight:700}
  .perf-lbl{font-size:.75rem;color:#4A6080;margin-top:4px}
  .table-wrapper{overflow-x:auto}
  table{width:100%;border-collapse:collapse;font-size:.82rem}
  th{color:#4A6080;font-weight:500;text-align:left;padding:6px 8px;border-bottom:1px solid #1A2540}
  td{padding:6px 8px;border-bottom:1px solid #0D1421}
  .disclaimer{text-align:center;color:#4A6080;font-size:.75rem;margin-top:12px;padding:8px}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div class="logo">&#8383; BTC Predictor</div>
    <div class="status">
      <span class="dot dot-bad" id="ws-dot"></span>
      <span id="ws-txt">Connecting...</span>
    </div>
    <div class="clock" id="clock">--:--:--</div>
  </div>

  <div id="tv-widget"></div>

  <div class="grid-3">
    <div class="card">
      <div class="card-title">Live Price</div>
      <div class="price-big" id="live-price">$--</div>
      <div class="price-change" id="price-change"></div>
    </div>
    <div class="card pred-box">
      <div class="card-title">Next 5-Min Signal</div>
      <div class="pred-arrow" id="pred-arrow">&#9670;</div>
      <div class="pred-dir" id="pred-dir" style="color:#4A6080">HOLD</div>
      <div class="conf-label">Confidence: <span id="conf-pct">--%</span></div>
      <div class="conf-bar-bg"><div class="conf-bar" id="conf-bar" style="width:0%"></div></div>
      <div class="pred-window" id="pred-window"></div>
    </div>
    <div class="card">
      <div class="card-title">Current Window</div>
      <div class="ohlc-grid">
        <div class="ohlc-cell"><div class="lbl">Open</div><div id="o-open">--</div></div>
        <div class="ohlc-cell"><div class="lbl">High</div><div id="o-high" class="up">--</div></div>
        <div class="ohlc-cell"><div class="lbl">Low</div><div id="o-low" class="down">--</div></div>
        <div class="ohlc-cell"><div class="lbl">Close</div><div id="o-close">--</div></div>
      </div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-title">Performance</div>
      <div class="perf-row">
        <div class="perf-stat"><div class="perf-num up" id="p-wins">0</div><div class="perf-lbl">Wins</div></div>
        <div class="perf-stat"><div class="perf-num down" id="p-losses">0</div><div class="perf-lbl">Losses</div></div>
        <div class="perf-stat"><div class="perf-num" id="p-acc" style="color:#F7931A">--%</div><div class="perf-lbl">Accuracy</div></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Data Source</div>
      <div id="model-status" style="color:#F7931A;font-size:.85rem">&#8987; Collecting data...</div>
      <div style="color:#4A6080;font-size:.75rem;margin-top:8px;">&#128250; TradingView &mdash; BINANCE:BTCUSDT</div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Prediction History (last 5)</div>
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

  <div class="disclaimer">&#9888;&#65039; For educational purpose only. Past accuracy does not guarantee future results.</div>
</div>

<script src="https://s3.tradingview.com/tv.js"></script>
<script>
  // Chart widget uses same symbol as backend data feed
  new TradingView.widget({
    container_id:'tv-widget', symbol:'BINANCE:BTCUSDT', interval:'5',
    theme:'dark', style:'1', locale:'en', toolbar_bg:'#080C14',
    enable_publishing:false, autosize:true
  });

  var ws, firstPrice = null;

  function updateClock() {
    var d = new Date();
    document.getElementById('clock').textContent =
      d.toLocaleTimeString('en-US', {timeZone:'Africa/Nairobi', hour12:false});
  }
  updateClock(); setInterval(updateClock, 1000);

  function fmt(n) {
    if (n == null || n === 0) return '--';
    return '$' + Number(n).toLocaleString('en-US',
      {minimumFractionDigits:2, maximumFractionDigits:2});
  }

  function connect() {
    var wsUrl = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';
    ws = new WebSocket(wsUrl);
    ws.onopen  = function() {
      document.getElementById('ws-dot').className = 'dot dot-ok';
      document.getElementById('ws-txt').textContent = 'Live';
    };
    ws.onclose = function() {
      document.getElementById('ws-dot').className = 'dot dot-bad';
      document.getElementById('ws-txt').textContent = 'Disconnected';
      setTimeout(connect, 3000);
    };
    ws.onerror = function() {
      document.getElementById('ws-dot').className = 'dot dot-bad';
    };
    ws.onmessage = function(e) {
      try { handleState(JSON.parse(e.data)); } catch(err) {}
    };
  }

  function handleState(d) {
    if (d.type === 'ping') return;

    var p = d.price;
    if (p && p > 0) {
      document.getElementById('live-price').textContent = fmt(p);
      if (firstPrice === null) firstPrice = p;
      var chg = p - firstPrice;
      var pct = (chg / firstPrice * 100).toFixed(2);
      var chgEl = document.getElementById('price-change');
      chgEl.textContent = (chg >= 0 ? '+' : '') + fmt(chg) + ' (' + pct + '%)';
      chgEl.style.color = chg >= 0 ? '#00E5A0' : '#FF4560';
    }

    var sig  = d.signal || 'HOLD';
    var isUp = sig === 'UP', isDn = sig === 'DOWN';
    var col  = isUp ? '#00E5A0' : isDn ? '#FF4560' : '#4A6080';
    document.getElementById('pred-arrow').textContent = isUp ? '▲' : isDn ? '▼' : '◆';
    document.getElementById('pred-arrow').style.color = col;
    document.getElementById('pred-dir').textContent   = isUp ? 'UP' : isDn ? 'DOWN' : 'HOLD';
    document.getElementById('pred-dir').style.color   = col;
    document.getElementById('conf-pct').textContent   = (isUp||isDn) ? d.confidence+'%' : '--%';
    document.getElementById('conf-pct').style.color   = col;
    document.getElementById('conf-bar').style.width   = (d.confidence||0) + '%';
    document.getElementById('conf-bar').style.background = col;
    document.getElementById('pred-window').textContent =
      d.next_window ? 'Next: ' + d.next_window : '';

    if (d.ohlc) {
      document.getElementById('o-open').textContent  = fmt(d.ohlc.open);
      document.getElementById('o-high').textContent  = fmt(d.ohlc.high);
      document.getElementById('o-low').textContent   = fmt(d.ohlc.low);
      document.getElementById('o-close').textContent = fmt(d.ohlc.close);
    }

    var w = d.wins||0, l = d.losses||0, tot = w+l;
    document.getElementById('p-wins').textContent   = w;
    document.getElementById('p-losses').textContent = l;
    document.getElementById('p-acc').textContent    = tot ? (w/tot*100).toFixed(1)+'%' : '--%';

    var msEl = document.getElementById('model-status');
    if (d.model_ready) {
      msEl.textContent = '&#10003; Model active';
      msEl.style.color = '#00E5A0';
    } else {
      msEl.textContent = '&#8987; Collecting data...';
      msEl.style.color = '#F7931A';
    }

    var tbody = document.getElementById('history-body');
    if (d.table && d.table.length) {
      tbody.innerHTML = d.table.map(function(r) {
        var predCls = r.predicted==='UP'?'up':r.predicted==='DOWN'?'down':'';
        var actCls  = r.actual==='UP'?'up':r.actual==='DOWN'?'down':'';
        var predTxt = r.predicted==='UP'?'&#9650; UP':r.predicted==='DOWN'?'&#9660; DOWN':r.predicted;
        var actTxt  = r.actual==='⏳'?'--':r.actual==='UP'?'&#9650; UP':r.actual==='DOWN'?'&#9660; DOWN':r.actual;
        return '<tr>' +
          '<td style="color:#4A6080">'   + r.window      + '</td>' +
          '<td class="' + predCls + '">' + predTxt       + '</td>' +
          '<td style="color:#F7931A">'   + r.confidence  + '%</td>' +
          '<td>'                         + fmt(r.act_open)  + '</td>' +
          '<td>'                         + fmt(r.act_close) + '</td>' +
          '<td class="' + actCls + '">'  + actTxt        + '</td>' +
          '<td>'                         + (r.result==='⏳'?'--':r.result) + '</td>' +
          '</tr>';
      }).join('');
    } else {
      tbody.innerHTML =
        '<tr><td colspan="7" style="text-align:center;padding:20px;">Waiting for data...</td></tr>';
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
    global _main_loop, _clients_lock
    _main_loop    = asyncio.get_running_loop()
    _clients_lock = asyncio.Lock()          # created inside running loop

    # Warm-start: synthetic candles + first model train
    synth = generate_synthetic_history()
    with state_lock:
        for c in synth:
            completed_candles.append(c)
    threading.Thread(
        target=train_model_from_candles,
        args=(list(completed_candles),),
        daemon=True, name="initial-train",
    ).start()

    # Background threads
    threading.Thread(target=tradingview_ws_thread,  name="tv-ws",     daemon=True).start()
    threading.Thread(target=price_processor_thread, name="price-proc", daemon=True).start()

    # Async broadcast task
    asyncio.create_task(_periodic_broadcast())

    logger.info("=" * 54)
    logger.info("BTC Predictor ready  —  TradingView live feed active")
    logger.info(f"Symbol: {TV_SYMBOL}")
    logger.info("=" * 54)
    yield


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
            await asyncio.sleep(20)
            await websocket.send_text(json.dumps({"type": "ping"}))   # Render keepalive
    except WebSocketDisconnect:
        pass
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
    uvicorn.run(app, host="0.0.0.0", port=port)