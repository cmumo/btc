"""
Bitcoin 5-Minute Prediction Terminal
=====================================
- TradingView chart (BITSTAMP:BTCUSD) + Binance WebSocket for live prices
- XGBoost model, 5-min candles, win/loss tracking
- GMT+3 timezone, prices with 2 decimal places
- Render-compatible: plain daemon threads, no anyio, correct $PORT binding
"""

import asyncio
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

# ============================================================
# SHARED STATE  (all mutation under state_lock)
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
# SYNTHETIC HISTORY  (warm-start model before live data)
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
    return {"ts": ts, "open": price, "high": price, "low": price, "close": price, "volume": 0.0}

def _close_current_window_locked() -> None:
    global live_candle, candles_since_retrain, wins, losses
    candle       = dict(live_candle)
    live_candle  = None
    completed_candles.append(candle)
    candles_since_retrain += 1

    for row in reversed(history_rows):
        if row.get("actual") == "\u23f3":
            actual           = "UP" if candle["close"] > candle["open"] else "DOWN"
            row["actual"]    = actual
            row["act_open"]  = candle["open"]
            row["act_close"] = candle["close"]
            if row["predicted"] == actual:
                row["result"] = "\u2705"; wins   += 1
            else:
                row["result"] = "\u274c"; losses += 1
            break

    snap = list(completed_candles)
    pred = predict_from_candles(snap)
    dt   = datetime.fromtimestamp(candle["ts"], tz=TZ).strftime("%H:%M")
    history_rows.append({
        "window":     dt,
        "predicted":  pred["signal"],
        "confidence": pred["confidence"],
        "act_open":   0.0,
        "act_close":  0.0,
        "actual":     "\u23f3",
        "result":     "\u23f3",
    })
    del history_rows[:-HISTORY_LIMIT]

    if candles_since_retrain >= RETRAIN_EVERY:
        candles_since_retrain = 0
        threading.Thread(
            target=train_model_from_candles, args=(snap,), daemon=True
        ).start()

# ============================================================
# BINANCE WEBSOCKET — plain daemon thread, NO anyio
# ============================================================
def _binance_thread() -> None:
    url         = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    retry_delay = 2
    while True:
        logger.info("Connecting to Binance WebSocket...")
        try:
            def on_open(ws_obj):
                nonlocal retry_delay
                retry_delay = 2
                logger.info("Binance WS connected")

            def on_message(ws_obj, raw):
                try:
                    data = json.loads(raw)
                    if data.get("e") == "trade":
                        price_queue.put_nowait({
                            "price": float(data["p"]),
                            "ts":    float(data["T"]) / 1000.0,
                        })
                except Exception:
                    pass

            def on_error(ws_obj, err):
                logger.error(f"Binance WS error: {err}")

            def on_close(ws_obj, code, msg):
                logger.warning("Binance WS closed")

            app_ws = ws_client.WebSocketApp(
                url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            app_ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as exc:
            logger.error(f"Binance thread exception: {exc}")

        logger.warning(f"Binance WS reconnecting in {retry_delay}s...")
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
    # Snapshot clients before releasing lock — never hold lock during async sends
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
# HTML FRONTEND  (original sidebar design)
# ============================================================
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC 5-Min Predictor</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#080C14;color:#C8D8EF;font-family:'Segoe UI',sans-serif;min-height:100vh;}
  .container{max-width:1280px;margin:0 auto;padding:14px;}
  .header{margin-bottom:10px;}
  .header h1{font-size:1.2rem;font-weight:700;color:#F7931A;letter-spacing:.04em;}
  .status-bar{display:flex;align-items:center;gap:10px;margin-bottom:12px;font-size:.8rem;color:#4A6080;}
  .dot{width:8px;height:8px;border-radius:50%;display:inline-block;flex-shrink:0;}
  .dot-ok{background:#00E5A0;box-shadow:0 0 6px #00E5A0;}
  .dot-bad{background:#FF4560;}
  .dot-wait{background:#F7931A;animation:pulse 1.2s infinite;}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
  .main-grid{display:grid;grid-template-columns:1fr 340px;gap:14px;margin-bottom:14px;}
  @media(max-width:900px){.main-grid{grid-template-columns:1fr;}}
  #tv-chart{background:#0D1421;border-radius:10px;border:1px solid #1E2D45;height:380px;overflow:hidden;}
  .sidebar{display:flex;flex-direction:column;gap:12px;}
  .card{background:#0D1421;border:1px solid #1E2D45;border-radius:10px;padding:14px;}
  .card-title{font-size:.68rem;color:#4A6080;text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px;}
  .price{font-size:2rem;font-weight:700;color:#F7931A;}
  .pchange{font-size:.82rem;margin-left:6px;}
  .pred-row{display:flex;align-items:center;gap:12px;margin-top:4px;}
  .pred-arrow{font-size:2.4rem;line-height:1;}
  .pred-dir{font-size:1.3rem;font-weight:700;}
  .conf-bar{background:#1E2D45;border-radius:4px;height:6px;margin-top:8px;overflow:hidden;}
  .conf-fill{height:100%;width:0%;transition:width 0.4s ease;}
  .countdown{display:flex;align-items:center;gap:14px;}
  .cd-ring{width:58px;height:58px;transform:rotate(-90deg);}
  .cd-text{font-size:1.8rem;font-weight:700;color:#F7931A;}
  .bottom-row{display:grid;gap:14px;grid-template-columns:1fr 1fr;margin-bottom:14px;}
  @media(max-width:600px){.bottom-row{grid-template-columns:1fr;}}
  .ohlc-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:6px;}
  .ohlc-cell{background:#0F1623;border-radius:6px;padding:7px 10px;border:1px solid #1E2D45;}
  .ohlc-cell .lbl{font-size:.6rem;color:#4A6080;}
  .ohlc-cell .val{font-size:.88rem;font-weight:700;margin-top:2px;}
  .perf-row{display:flex;gap:10px;margin-top:8px;}
  .perf-stat{flex:1;text-align:center;background:#0F1623;border-radius:7px;padding:8px;border:1px solid #1E2D45;}
  .perf-num{font-size:1.35rem;font-weight:700;}
  .perf-lbl{font-size:.6rem;color:#4A6080;margin-top:2px;}
  .table-wrapper{overflow-x:auto;margin-top:10px;}
  table{width:100%;border-collapse:collapse;font-size:.75rem;}
  th,td{padding:8px;text-align:left;border-bottom:1px solid #1E2D45;}
  th{color:#4A6080;font-weight:600;}
  .up{color:#00E5A0;}.down{color:#FF4560;}
  .disclaimer{background:#0F1623;border-radius:8px;padding:10px;font-size:.7rem;text-align:center;color:#FACC15;border:1px solid rgba(250,204,21,.13);margin-top:14px;}
  .flash-up{animation:flashUp .4s ease;}.flash-dn{animation:flashDn .4s ease;}
  @keyframes flashUp{0%,100%{color:#C8D8EF}50%{color:#00E5A0}}
  @keyframes flashDn{0%,100%{color:#C8D8EF}50%{color:#FF4560}}
</style>
</head>
<body>
<div class="container">
  <div class="header"><h1>&#x20BF; Bitcoin Price Trend Predictor</h1></div>
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
          <span class="pred-arrow" id="pred-arrow" style="color:#4A6080">&#x25C6;</span>
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
    <div class="card-title">Prediction History (last 5)</div>
    <div class="table-wrapper">
      <table>
        <thead><tr><th>Window</th><th>Prediction</th><th>Conf</th><th>Act.Open</th><th>Act.Close</th><th>Actual</th><th>Result</th></tr></thead>
        <tbody id="history-body"><tr><td colspan="7" style="text-align:center;padding:20px;">Waiting for data...</td></tr></tbody>
      </table>
    </div>
  </div>

  <div class="disclaimer">&#x26A0;&#xFE0F; For educational purposes only. Past accuracy does not guarantee future results.</div>
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
    const sec=gmt3.getUTCSeconds()+gmt3.getUTCMinutes()%5*60;
    const remaining=300-sec%300;
    const mm=Math.floor(remaining/60);
    const ss=String(remaining%60).padStart(2,'0');
    document.getElementById('cd-val').textContent=mm+':'+ss;
    const circ=201;
    document.getElementById('cd-ring').setAttribute('stroke-dashoffset',
      String(circ*(1-(remaining/300))));
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
    document.getElementById('pred-arrow').textContent=isUp?'\u25b2':isDn?'\u25bc':'\u25c6';
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
        const predTxt=r.predicted==='UP'?'\u25b2 UP':r.predicted==='DOWN'?'\u25bc DOWN':r.predicted;
        const actTxt=r.actual==='\u23f3'?'--':r.actual==='UP'?'\u25b2 UP':r.actual==='DOWN'?'\u25bc DOWN':r.actual;
        return '<tr>'
          +'<td style="color:#4A6080">'+r.window+'</td>'
          +'<td class="'+predCls+'">'+predTxt+'</td>'
          +'<td style="color:#F7931A">'+r.confidence+'%</td>'
          +'<td>'+fmt(r.act_open)+'</td>'
          +'<td>'+fmt(r.act_close)+'</td>'
          +'<td class="'+actCls+'">'+actTxt+'</td>'
          +'<td>'+(r.result==='\u23f3'?'--':r.result)+'</td>'
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
# FASTAPI APPLICATION
# ============================================================
@asynccontextmanager
async def lifespan(application: FastAPI):
    global _main_loop, _clients_lock
    _main_loop    = asyncio.get_running_loop()
    _clients_lock = asyncio.Lock()

    # Warm-start with synthetic candles + initial model train
    synth = generate_synthetic_history()
    with state_lock:
        for c in synth:
            completed_candles.append(c)
    threading.Thread(
        target=train_model_from_candles,
        args=(list(completed_candles),),
        daemon=True, name="initial-train",
    ).start()

    # Price processor thread
    threading.Thread(target=price_processor_thread, name="price-proc", daemon=True).start()

    # Binance: plain daemon thread — keeps asyncio event loop completely free
    threading.Thread(target=_binance_thread, name="binance-ws", daemon=True).start()

    # Periodic broadcast to all connected WebSocket clients
    asyncio.create_task(_periodic_broadcast())

    logger.info("=" * 52)
    logger.info("BTC Predictor ready")
    logger.info("=" * 52)
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
        # Send current state immediately on connect
        await websocket.send_text(json.dumps(_build_state_payload()))
        # Keep alive: actively receive frames with a timeout.
        # Render's HTTP proxy kills idle WebSockets after ~55s.
        # On TimeoutError we send a ping to satisfy the proxy.
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=25.0)
                if data == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping"}))
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