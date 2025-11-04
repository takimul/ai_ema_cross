import os
import json
import requests
import numpy as np
import asyncio
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, Request
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

load_dotenv()  # load BOT_TOKEN from .env

BOT_TOKEN = os.getenv("BOT_TOKEN")
SYMBOLS = ["OPUSDT", "RENDERUSDT", "BTCUSDT"]
TIMEFRAMES = ["5m", "15m"]
user_ids = set()
sent_signals = set()

app = FastAPI()

# ---------------- Telegram ----------------
def send_telegram(chat_id, msg):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": msg}, timeout=5)
    except Exception as e:
        print("Telegram send error:", e)

def broadcast(msg):
    for uid in user_ids:
        send_telegram(uid, msg)

# ---------------- EMA ----------------
def get_ema(values, period):
    if len(values) < period:
        return None
    weights = np.exp(np.linspace(-1.,0.,period))
    weights /= weights.sum()
    ema = np.convolve(values, weights, mode="full")[:len(values)]
    ema[:period] = ema[period]
    return float(np.round(ema[-1],8))

# ---------------- AI Predictor ----------------
def predict_trend_probability(closes):
    try:
        if len(closes) < 50:
            return None
        X, y = [], []
        for i in range(30, len(closes)-5):
            ema9 = get_ema(closes[:i], 9)
            ema26 = get_ema(closes[:i], 26)
            if ema9 is None or ema26 is None:
                continue
            X.append([ema9, ema26, closes[i]])
            future = closes[i+5]
            y.append(1 if future>closes[i] else 0)
        if len(X) < 20:
            return None
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=500)
        model.fit(Xs, y)
        last_feat = scaler.transform([[get_ema(closes,9), get_ema(closes,26), closes[-1]]])
        prob = model.predict_proba(last_feat)[0][1]
        return round(prob*100,2)
    except Exception as e:
        print("AI prediction error:", e)
        return None

# ---------------- Binance REST ----------------
def fetch_klines(symbol, interval, limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return []

# ---------------- Process EMA ----------------
def process_candles(symbol, interval, closes):
    prev_ema9 = get_ema(closes[:-1], 9)
    prev_ema26 = get_ema(closes[:-1], 26)
    ema9 = get_ema(closes, 9)
    ema26 = get_ema(closes, 26)
    if prev_ema9 is None or prev_ema26 is None:
        return
    last_close_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    key = (symbol, interval, last_close_time)
    if key in sent_signals:
        return
    if prev_ema9 < prev_ema26 and ema9 >= ema26:
        prob = predict_trend_probability(closes)
        msg = f"📈 {symbol} ({interval}) EMA9 CROSSED ABOVE EMA26 — BUY 💰\nPrice: {closes[-1]}"
        if prob is not None:
            msg += f"\n🤖 AI Uptrend Probability: {prob}%"
        broadcast(msg)
        sent_signals.add(key)
    elif prev_ema9 > prev_ema26 and ema9 <= ema26:
        prob = predict_trend_probability(closes)
        msg = f"📉 {symbol} ({interval}) EMA9 CROSSED BELOW EMA26 — SELL ⚠️\nPrice: {closes[-1]}"
        if prob is not None:
            msg += f"\n🤖 AI Downtrend Probability: {100-prob}%"
        broadcast(msg)
        sent_signals.add(key)

# ---------------- Webhook ----------------
@app.post("/webhook/{token}")
async def telegram_webhook(token: str, request: Request):
    if token != BOT_TOKEN:
        return {"status": "unauthorized"}
    data = await request.json()
    if "message" in data:
        chat_id = data["message"]["chat"]["id"]
        user_ids.add(chat_id)
        send_telegram(chat_id, f"✅ Subscribed to EMA alerts!\nTracking: {', '.join(SYMBOLS)}")
    return {"ok": True}

# ---------------- Background tasks ----------------
async def monitor_interval(interval):
    while True:
        for symbol in SYMBOLS:
            klines = fetch_klines(symbol, interval, limit=60)
            closes = [float(k[4]) for k in klines]
            process_candles(symbol, interval, closes)
        await asyncio.sleep(interval_to_seconds(interval))

def interval_to_seconds(interval):
    if interval.endswith("m"):
        return int(interval[:-1])*60
    if interval.endswith("h"):
        return int(interval[:-1])*3600
    return 60

async def monitor_hourly():
    while True:
        for symbol in SYMBOLS:
            klines = fetch_klines(symbol, "1h", limit=2)
            if klines:
                close_price = float(klines[-1][4])
                ts = datetime.fromtimestamp(int(klines[-1][6])/1000, tz=timezone.utc)
                msg = f"🕐 {symbol} 1H close ({ts.strftime('%Y-%m-%d %H:%M UTC')})\nClose: {close_price}"
                broadcast(msg)
        await asyncio.sleep(3600)

@app.on_event("startup")
async def startup_event():
    for tf in TIMEFRAMES:
        asyncio.create_task(monitor_interval(tf))
    asyncio.create_task(monitor_hourly())
