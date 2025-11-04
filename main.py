import os
import json
import requests
import asyncio
import numpy as np
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Request
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BOT_TOKEN = os.getenv("BOT_TOKEN")
SYMBOLS = ["OPUSDT", "RENDERUSDT", "BTCUSDT"]
TIMEFRAMES = ["5m", "15m"]
HISTORICAL_CANDLES = 300
PRED_LOOKAHEAD = 5

app = FastAPI()
user_ids = set()

# ---------------- Telegram Helpers ----------------
def send_telegram(chat_id, msg):
    if not BOT_TOKEN:
        print("BOT_TOKEN not set — would send:", msg)
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": msg}, timeout=5)
    except Exception as e:
        print("Telegram send error:", e)

def broadcast(msg):
    for uid in list(user_ids):
        send_telegram(uid, msg)

# ---------------- EMA / AI ----------------
def get_ema(values, period):
    if len(values) < period:
        return None
    weights = np.exp(np.linspace(-1., 0., period))
    weights /= weights.sum()
    ema = np.convolve(values, weights, mode="full")[:len(values)]
    ema[:period] = ema[period]
    return float(np.round(ema[-1], 8))

def predict_trend_probability(closes):
    try:
        if len(closes) < 50:
            return None
        X, y = [], []
        for i in range(30, len(closes) - PRED_LOOKAHEAD):
            ema9 = get_ema(closes[:i], 9)
            ema26 = get_ema(closes[:i], 26)
            if ema9 is None or ema26 is None:
                continue
            X.append([ema9, ema26, closes[i]])
            future = closes[i + PRED_LOOKAHEAD]
            y.append(1 if future > closes[i] else 0)
        if len(X) < 20:
            return None
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=500)
        model.fit(Xs, y)
        last_feat = scaler.transform([[get_ema(closes, 9), get_ema(closes, 26), closes[-1]]])
        prob = model.predict_proba(last_feat)[0][1]
        return round(prob * 100, 2)
    except Exception as e:
        print("AI prediction error:", e)
        return None

# ---------------- Binance Helper ----------------
def fetch_klines(symbol: str, interval: str, limit: int = 100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[REST] Error fetching klines {symbol} {interval}: {e}")
        return []

# ---------------- Telegram Webhook ----------------
@app.post("/webhook/{token}")
async def telegram_webhook(token: str, request: Request):
    if token != BOT_TOKEN:
        return {"ok": False}
    data = await request.json()
    if "message" in data:
        chat_id = data["message"]["chat"]["id"]
        text = data["message"].get("text", "")
        if chat_id not in user_ids:
            user_ids.add(chat_id)
            send_telegram(chat_id, f"✅ Subscribed to EMA alerts!\nTracking: {', '.join(SYMBOLS)}")
        if text.lower() == "/start":
            send_telegram(chat_id, "👋 Welcome! EMA + AI alerts are active.")
    return {"ok": True}

# ---------------- EMA Monitor Task ----------------
async def monitor_ema(symbol, interval):
    closes = [float(k[4]) for k in fetch_klines(symbol, interval, limit=120)]
    prev_ema9 = get_ema(closes, 9)
    prev_ema26 = get_ema(closes, 26)

    while True:
        await asyncio.sleep(5)  # poll every 5 sec (optional)
        klines = fetch_klines(symbol, interval, limit=2)
        if not klines:
            continue
        close_price = float(klines[-1][4])
        ema9 = get_ema(closes + [close_price], 9)
        ema26 = get_ema(closes + [close_price], 26)
        if prev_ema9 and prev_ema26 and ema9 and ema26:
            if prev_ema9 < prev_ema26 and ema9 >= ema26:
                prob = predict_trend_probability(closes + [close_price])
                msg = f"📈 {symbol} ({interval}) EMA9 CROSSED ABOVE EMA26 — BUY 💰\nPrice: {close_price}"
                if prob: msg += f"\n🤖 Uptrend Probability: {prob}%"
                broadcast(msg)
            elif prev_ema9 > prev_ema26 and ema9 <= ema26:
                prob = predict_trend_probability(closes + [close_price])
                msg = f"📉 {symbol} ({interval}) EMA9 CROSSED BELOW EMA26 — SELL ⚠️\nPrice: {close_price}"
                if prob: msg += f"\n🤖 Downtrend Probability: {100-prob}%"
                broadcast(msg)
        prev_ema9, prev_ema26 = ema9, ema26

# ---------------- Startup ----------------
@app.on_event("startup")
async def startup_event():
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            asyncio.create_task(monitor_ema(symbol, tf))
