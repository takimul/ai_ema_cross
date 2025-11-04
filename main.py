import asyncio
import json
import requests
import numpy as np
import os
from datetime import datetime, timedelta, timezone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from flask import Flask, request
import threading

# ---------------------------
# CONFIG
# ---------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
SYMBOLS = ["OPUSDT", "RENDERUSDT", "BTCUSDT"]
TIMEFRAMES = ["5m", "15m"]
HISTORICAL_CANDLES = 300
PRED_LOOKAHEAD = 5
user_ids = set()
sent_signals = set()

# ---------------------------
# TELEGRAM HELPERS
# ---------------------------
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

# ---------------------------
# EMA + AI
# ---------------------------
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

# ---------------------------
# BINANCE HELPERS
# ---------------------------
def fetch_klines(symbol, interval, limit=100):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[REST] Error fetching {symbol} {interval}: {e}")
        return []

# ---------------------------
# EMA CROSS + SIGNAL LOGIC
# ---------------------------
def process_closed_kline(symbol, interval, kline):
    try:
        close_time_ms = int(kline[6])
        key = (symbol, interval, close_time_ms)
        if key in sent_signals:
            return
        klines = fetch_klines(symbol, interval, limit=120)
        closes = [float(k[4]) for k in klines]
        if len(closes) < 26:
            return

        prev_ema9 = get_ema(closes[:-1], 9)
        prev_ema26 = get_ema(closes[:-1], 26)
        ema9 = get_ema(closes, 9)
        ema26 = get_ema(closes, 26)

        if prev_ema9 is None or prev_ema26 is None:
            return

        if prev_ema9 < prev_ema26 and ema9 >= ema26:
            prob = predict_trend_probability(closes)
            msg = f"📈 {symbol} ({interval}) EMA9 CROSSED ABOVE EMA26 — BUY 💰\nPrice: {closes[-1]}"
            if prob is not None:
                msg += f"\n🤖 AI Uptrend Probability: {prob}%"
            broadcast(msg)
            print("[ALERT]", msg)
        elif prev_ema9 > prev_ema26 and ema9 <= ema26:
            prob = predict_trend_probability(closes)
            msg = f"📉 {symbol} ({interval}) EMA9 CROSSED BELOW EMA26 — SELL ⚠️\nPrice: {closes[-1]}"
            if prob is not None:
                msg += f"\n🤖 AI Downtrend Probability: {100 - prob}%"
            broadcast(msg)
            print("[ALERT]", msg)
        sent_signals.add(key)
    except Exception as e:
        print("process_closed_kline error:", e)

def interval_to_minutes(interval):
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    return None

def seconds_until_next_close(interval_minutes):
    now = datetime.now(timezone.utc)
    next_minute = ((now.minute // interval_minutes) + 1) * interval_minutes
    next_hour = now.replace(minute=0, second=0, microsecond=0)
    if next_minute >= 60:
        next_minute -= 60
        next_hour += timedelta(hours=1)
    next_close = next_hour.replace(minute=next_minute, second=5)
    wait = (next_close - now).total_seconds()
    return max(wait, 1)

# ---------------------------
# MAIN TASKS
# ---------------------------
async def monitor_interval(interval):
    minutes = interval_to_minutes(interval)
    for symbol in SYMBOLS:
        klines = fetch_klines(symbol, interval, limit=120)
        for k in klines[-5:]:
            process_closed_kline(symbol, interval, k)

    while True:
        wait = seconds_until_next_close(minutes)
        print(f"[SYNC] waiting {int(wait)}s until next {interval} close...")
        await asyncio.sleep(wait)
        for symbol in SYMBOLS:
            k = fetch_klines(symbol, interval, limit=1)[-1]
            process_closed_kline(symbol, interval, k)
        await asyncio.sleep(1)

async def monitor_hourly():
    for symbol in SYMBOLS:
        klines = fetch_klines(symbol, "1h", limit=24)
        for k in klines[-3:]:
            close_time_ms = int(k[6])
            key = (symbol, "1h", close_time_ms)
            if key not in sent_signals:
                price = float(k[4])
                ts = datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc)
                msg = f"🕐 {symbol} 1H close ({ts.strftime('%H:%M UTC')})\nClose: {price}"
                broadcast(msg)
                sent_signals.add(key)

    while True:
        wait = seconds_until_next_close(60)
        print(f"[SYNC] waiting {int(wait)}s until next 1h close...")
        await asyncio.sleep(wait)
        for symbol in SYMBOLS:
            k = fetch_klines(symbol, "1h", limit=1)[-1]
            close_time_ms = int(k[6])
            if (symbol, "1h", close_time_ms) not in sent_signals:
                price = float(k[4])
                ts = datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc)
                msg = f"🕐 {symbol} 1H close ({ts.strftime('%H:%M UTC')})\nClose: {price}"
                broadcast(msg)
                sent_signals.add((symbol, "1h", close_time_ms))
        await asyncio.sleep(1)

async def main():
    tasks = []
    for tf in TIMEFRAMES:
        tasks.append(monitor_interval(tf))
    tasks.append(monitor_hourly())
    await asyncio.gather(*tasks)

# ---------------------------
# FLASK WEBHOOK SERVER
# ---------------------------
app = Flask(__name__)

@app.route(f"/webhook/{BOT_TOKEN}", methods=["POST"])
def telegram_webhook():
    data = request.get_json()
    if not data or "message" not in data:
        return "no message", 200
    chat_id = data["message"]["chat"]["id"]
    text = data["message"].get("text", "").strip()
    if chat_id not in user_ids:
        user_ids.add(chat_id)
    if text == "/start":
        send_telegram(chat_id, f"✅ Subscribed to EMA alerts!\nTracking: {', '.join(SYMBOLS)}")
    return "ok", 200

def run_async_tasks():
    asyncio.run(main())

threading.Thread(target=run_async_tasks, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
