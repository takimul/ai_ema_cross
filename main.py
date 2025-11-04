import asyncio
import json
import requests
import websockets
import numpy as np
import os
from datetime import datetime, timedelta, timezone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import deque

# ---------------------------
# CONFIG
# ---------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
SYMBOLS = ["OPUSDT", "RENDERUSDT", "BTCUSDT"]  # uppercase symbols for REST
TIMEFRAMES = ["5m", "15m"]
HISTORICAL_CANDLES = 300
PRED_LOOKAHEAD = 5  # lookahead candles for AI label
MAX_STORE = 1000

# runtime state
user_ids = set()
sent_signals = set()  # tuple (symbol, interval, close_time_ms) to prevent duplicates

# ---------------------------
# Helpers: Telegram
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
# Helpers: Binance REST
# ---------------------------
def fetch_klines(symbol: str, interval: str, limit: int = 100, startTime: int = None):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
    if startTime:
        url += f"&startTime={int(startTime)}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[REST] Error fetching klines {symbol} {interval}: {e}")
        return []

# ---------------------------
# EMA calculation (same style used earlier)
# ---------------------------
def get_ema(values, period):
    if len(values) < period:
        return None
    weights = np.exp(np.linspace(-1., 0., period))
    weights /= weights.sum()
    ema = np.convolve(values, weights, mode="full")[:len(values)]
    ema[:period] = ema[period]
    return float(np.round(ema[-1], 8))

# ---------------------------
# AI predictor (lightweight on-demand training)
# ---------------------------
def predict_trend_probability(closes):
    try:
        if len(closes) < 50:
            return None
        X, y = [], []
        # build small dataset from historical closes
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
# Utilities: time alignment
# ---------------------------
def seconds_until_next_close(interval_minutes: int):
    now = datetime.now(timezone.utc)
    minute = now.minute
    # compute next multiple
    next_min = ((minute // interval_minutes) + 1) * interval_minutes
    # handle wrap-around to next hour
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=0)
    # if next_min >= 60, add an hour and set minute = next_min - 60
    if next_min >= 60:
        next_min -= 60
        next_hour = next_hour + timedelta(hours=1)
    next_close = next_hour.replace(minute=next_min, second=5, microsecond=0)
    wait = (next_close - now).total_seconds()
    return max(wait, 1)

# convert interval string to minutes
def interval_to_minutes(interval: str):
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    return None

# ---------------------------
# Process a closed candle (kline)
# ---------------------------
def process_closed_kline(symbol, interval, kline):
    """
    kline: list as returned by Binance klines
    uses close_time (index 6) as unique id
    """
    try:
        close_time_ms = int(kline[6])
        key = (symbol, interval, close_time_ms)
        if key in sent_signals:
            return
        # fetch recent closes to compute prev and current EMAs
        # get 60 candles for safe EMA
        klines = fetch_klines(symbol, interval, limit=120)
        closes = [float(k[4]) for k in klines]
        if len(closes) < 26:
            return
        # find index of this close_time in fetched
        idx = None
        for i, k in enumerate(klines):
            if int(k[6]) == close_time_ms:
                idx = i
                break
        if idx is None:
            # fallback: use last candle
            idx = len(closes) - 1
        # need previous closed candle index for prev EMA: use portion up to idx-1
        if idx < 1:
            return
        closes_up_to_prev = closes[:idx]  # up to previous
        closes_up_to_prev = closes_up_to_prev if len(closes_up_to_prev) > 0 else closes[:-1]
        prev_ema9 = get_ema(closes_up_to_prev, 9)
        prev_ema26 = get_ema(closes_up_to_prev, 26)
        # current EMAs using up to idx (including current close)
        closes_up_to_current = closes[:idx + 1]
        ema9 = get_ema(closes_up_to_current, 9)
        ema26 = get_ema(closes_up_to_current, 26)
        if prev_ema9 is None or prev_ema26 is None or ema9 is None or ema26 is None:
            return
        # real crossover detection
        if prev_ema9 < prev_ema26 and ema9 >= ema26:
            # bullish cross
            prob = predict_trend_probability(closes_up_to_current)
            price = float(closes_up_to_current[-1])
            msg = f"📈 {symbol} ({interval}) EMA9 CROSSED ABOVE EMA26 — BUY 💰\nPrice: {price}"
            if prob is not None:
                msg += f"\n🤖 AI Uptrend Probability: {prob}%"
            broadcast(msg)
            sent_signals.add(key)
            print("[ALERT]", msg)
        elif prev_ema9 > prev_ema26 and ema9 <= ema26:
            prob = predict_trend_probability(closes_up_to_current)
            price = float(closes_up_to_current[-1])
            msg = f"📉 {symbol} ({interval}) EMA9 CROSSED BELOW EMA26 — SELL ⚠️\nPrice: {price}"
            if prob is not None:
                msg += f"\n🤖 AI Downtrend Probability: {100 - prob}%"
            broadcast(msg)
            sent_signals.add(key)
            print("[ALERT]", msg)
    except Exception as e:
        print("Error in process_closed_kline:", e)

# ---------------------------
# Initial backfill: send missed closed candles for interval
# ---------------------------
def backfill_and_mark(symbol, interval, lookback=120):
    klines = fetch_klines(symbol, interval, limit=lookback)
    if not isinstance(klines, list):
        return
    # ensure ordered oldest -> newest
    for k in klines:
        process_closed_kline(symbol, interval, k)

# ---------------------------
# Monitors (synchronized to Binance)
# ---------------------------
async def monitor_interval(interval):
    minutes = interval_to_minutes(interval)
    # initial backfill for each symbol
    for symbol in SYMBOLS:
        print(f"[BACKFILL] {symbol} {interval} recent candles...")
        backfill_and_mark(symbol, interval, lookback=120)
    # loop: wait until next close (actual Binance close time), then process latest candle(s)
    while True:
        wait = seconds_until_next_close(minutes)
        print(f"[SYNC] waiting {int(wait)}s until next {interval} close...")
        await asyncio.sleep(wait)
        # when close occurs, fetch last closed candle per symbol and process
        for symbol in SYMBOLS:
            klines = fetch_klines(symbol, interval, limit=2)
            if isinstance(klines, list) and len(klines) > 0:
                # last item is latest closed candle
                latest = klines[-1]
                process_closed_kline(symbol, interval, latest)
        # small sleep to avoid tight loop around boundary
        await asyncio.sleep(1)

# ---------------------------
# Hourly monitor (with backfill of missed hours)
# ---------------------------
async def monitor_hourly():
    # backfill several recent hours
    for symbol in SYMBOLS:
        try:
            klines = fetch_klines(symbol, "1h", limit=24)
            if isinstance(klines, list):
                for k in klines:
                    close_time_ms = int(k[6])
                    key = (symbol, "1h", close_time_ms)
                    if key not in sent_signals:
                        # for hourly we only broadcast close price (no EMA)
                        close_price = float(k[4])
                        ts = datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc)
                        msg = f"🕐 {symbol} 1H close ({ts.strftime('%Y-%m-%d %H:%M UTC')})\nClose: {close_price}"
                        broadcast(msg)
                        sent_signals.add(key)
        except Exception as e:
            print("Hourly backfill error:", e)

    # then sync to actual hour close times
    while True:
        wait = seconds_until_next_close(60)
        print(f"[SYNC] waiting {int(wait)}s until next 1h close...")
        await asyncio.sleep(wait)
        for symbol in SYMBOLS:
            klines = fetch_klines(symbol, "1h", limit=1)
            if isinstance(klines, list) and len(klines) > 0:
                k = klines[-1]
                close_time_ms = int(k[6])
                key = (symbol, "1h", close_time_ms)
                if key not in sent_signals:
                    close_price = float(k[4])
                    ts = datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc)
                    msg = f"🕐 {symbol} 1H close ({ts.strftime('%Y-%m-%d %H:%M UTC')})\nClose: {close_price}"
                    print(msg)
                    broadcast(msg)
                    sent_signals.add(key)
        await asyncio.sleep(1)

# ---------------------------
# Telegram listener task
# ---------------------------
async def listen_for_users():
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    last_update_id = None
    while True:
        try:
            params = {"timeout": 10, "offset": last_update_id}
            res = requests.get(url, params=params, timeout=10).json()
            for update in res.get("result", []):
                last_update_id = update["update_id"] + 1
                if "message" not in update:
                    continue
                chat_id = update["message"]["chat"]["id"]
                if chat_id not in user_ids:
                    user_ids.add(chat_id)
                    send_telegram(chat_id, f"✅ Subscribed to EMA alerts!\nTracking: {', '.join(SYMBOLS)}")
        except Exception as e:
            print("Telegram listener error:", e)
        await asyncio.sleep(5)

# ---------------------------
# Main
# ---------------------------
async def main():
    tasks = []
    # telegram listener
    tasks.append(listen_for_users())
    # monitors for each timeframe (5m, 15m)
    for tf in TIMEFRAMES:
        tasks.append(monitor_interval(tf))
    # hourly monitor
    tasks.append(monitor_hourly())

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
