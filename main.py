import asyncio
import json
import requests
import websockets
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os

# =============================
# CONFIG
# =============================
BOT_TOKEN = os.getenv("BOT_TOKEN")  # Telegram bot token (set in Railway)
SYMBOLS = ["OPUSDT", "RENDERUSDT", "BTCUSDT"]
TIMEFRAMES = ["5m", "15m"]
HISTORICAL_CANDLES = 200

user_ids = set()  # Telegram subscribers


# =============================
# TELEGRAM HELPERS
# =============================
def send_telegram(chat_id, msg):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": msg})
    except Exception as e:
        print("Telegram send error:", e)


def broadcast(msg):
    for uid in user_ids:
        send_telegram(uid, msg)


# =============================
# EMA FUNCTION
# =============================
def get_ema(values, period):
    if len(values) < period:
        return None
    weights = np.exp(np.linspace(-1., 0., period))
    weights /= weights.sum()
    ema = np.convolve(values, weights, mode="full")[:len(values)]
    ema[:period] = ema[period]
    return np.round(ema[-1], 4)


# =============================
# AI PREDICTOR
# =============================
def predict_trend_probability(closes):
    try:
        X, y = [], []
        for i in range(30, len(closes) - 1):
            ema9 = get_ema(closes[:i], 9)
            ema26 = get_ema(closes[:i], 26)
            if not ema9 or not ema26:
                continue
            X.append([ema9, ema26, closes[i]])
            y.append(1 if closes[i + 1] > closes[i] else 0)

        if len(X) < 10:
            return None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression()
        model.fit(X_scaled, y)

        last = scaler.transform([[get_ema(closes, 9), get_ema(closes, 26), closes[-1]]])
        prob = model.predict_proba(last)[0][1]
        return round(prob * 100, 2)
    except Exception as e:
        print("AI prediction error:", e)
        return None


# =============================
# TELEGRAM LISTENER
# =============================
async def listen_for_users():
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    last_update_id = None
    while True:
        try:
            params = {"timeout": 10, "offset": last_update_id}
            res = requests.get(url, params=params).json()
            for update in res.get("result", []):
                last_update_id = update["update_id"] + 1
                chat_id = update["message"]["chat"]["id"]
                if chat_id not in user_ids:
                    user_ids.add(chat_id)
                    send_telegram(
                        chat_id,
                        "✅ Subscribed to EMA alerts with AI predictions!\n"
                        f"Tracking: {', '.join(SYMBOLS)}\n"
                        f"Timeframes: {', '.join(TIMEFRAMES)}\n"
                        "Hourly close updates are active 📊"
                    )
        except Exception as e:
            print("Telegram listener error:", e)
        await asyncio.sleep(5)


# =============================
# EMA CROSS TRACKER
# =============================
async def track_symbol(symbol, interval):
    ws_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_{interval}"

    closes = []
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={HISTORICAL_CANDLES}"
        res = requests.get(url).json()
        if isinstance(res, list):
            closes = [float(k[4]) for k in res]
            print(f"📈 {symbol} ({interval}) loaded {len(closes)} candles.")
    except Exception as e:
        print(f"Error fetching {symbol} ({interval}):", e)

    prev_ema9 = get_ema(closes, 9)
    prev_ema26 = get_ema(closes, 26)

    while True:
        try:
            async with websockets.connect(ws_url) as ws:
                print(f"📊 Tracking {symbol} ({interval}) live...")
                async for msg in ws:
                    data = json.loads(msg)
                    k = data["k"]
                    if k["x"]:  # candle closed
                        close_price = float(k["c"])
                        closes.append(close_price)
                        closes = closes[-200:]

                        ema9 = get_ema(closes, 9)
                        ema26 = get_ema(closes, 26)

                        if prev_ema9 and prev_ema26 and ema9 and ema26:
                            if prev_ema9 < prev_ema26 and ema9 >= ema26:
                                prob = predict_trend_probability(closes)
                                msg = f"📈 {symbol} ({interval}) EMA9 CROSSED ABOVE EMA26 — BUY 💰\nPrice: {close_price}"
                                if prob:
                                    msg += f"\n🤖 AI Uptrend Probability: {prob}%"
                                broadcast(msg)
                            elif prev_ema9 > prev_ema26 and ema9 <= ema26:
                                prob = predict_trend_probability(closes)
                                msg = f"📉 {symbol} ({interval}) EMA9 CROSSED BELOW EMA26 — SELL ⚠️\nPrice: {close_price}"
                                if prob:
                                    msg += f"\n🤖 AI Downtrend Probability: {100 - prob}%"
                                broadcast(msg)

                        prev_ema9, prev_ema26 = ema9, ema26
        except Exception as e:
            print(f"Error tracking {symbol} ({interval}):", e)
            await asyncio.sleep(5)


# =============================
# HOURLY CLOSE MONITOR
# =============================
async def track_hourly_closes():
    print("🕐 Starting hourly close monitor...")
    for symbol in SYMBOLS:
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=2"
            res = requests.get(url).json()
            if isinstance(res, list) and len(res) >= 2:
                prev_close = float(res[-2][4])
                msg = f"🕒 {symbol} Previous 1H close: {prev_close}"
                print(msg)
                broadcast(msg)
        except Exception as e:
            print("Initial hourly close error:", e)

    while True:
        now = datetime.utcnow()
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=5, microsecond=0)
        wait_time = (next_hour - now).total_seconds()
        await asyncio.sleep(wait_time)

        for symbol in SYMBOLS:
            try:
                url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=1"
                res = requests.get(url).json()
                if isinstance(res, list) and len(res) > 0:
                    close_price = float(res[-1][4])
                    msg = f"🕐 {symbol} 1H candle closed\nClose: {close_price}"
                    print(msg)
                    broadcast(msg)
            except Exception as e:
                print("Hourly close fetch error:", e)
        await asyncio.sleep(3600)


# =============================
# MAIN
# =============================
async def main():
    tasks = [listen_for_users(), track_hourly_closes()]
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            tasks.append(track_symbol(symbol, tf))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
