# import os
# import json
# import requests
# import asyncio
# import numpy as np
# from datetime import datetime
# from fastapi import FastAPI, Request
# import xgboost as xgb

# # ---------------- Config ----------------

# BOT_TOKEN = os.getenv("BOT_TOKEN")
# SYMBOLS = ["OPUSDT", "RENDERUSDT", "BTCUSDT"]
# TIMEFRAMES = ["5m", "15m"]

# MODEL_DIR = "models"
# os.makedirs(MODEL_DIR, exist_ok=True)
# DATA_FILE = "training_data.json"

# app = FastAPI()
# user_ids = set()
# TRADE_HISTORY = []
# TARGET_PROFIT_PERCENT = 10  # Target profit at 20x leverage
# MIN_PROBABILITY = 0.6  # Only alert if model confidence ≥60%

# # ---------------- Helpers ----------------

# def load_training_data():
#     if os.path.exists(DATA_FILE):
#         with open(DATA_FILE, "r") as f:
#             return json.load(f)
#     return []

# def save_training_data(data):
#     with open(DATA_FILE, "w") as f:
#         json.dump(data, f)

# training_data = load_training_data()

# def add_training_record(symbol, features, outcome):
#     training_data.append({"symbol": symbol, "features": features, "outcome": outcome})
#     save_training_data(training_data)

# # ---------------- Indicators ----------------

# def get_ema(values, period):
#     if len(values) < period or period < 1:
#         return None
#     weights = np.exp(np.linspace(-1., 0., period))
#     weights /= weights.sum()
#     ema = np.convolve(values, weights, mode="valid")
#     return float(np.round(ema[-1], 8))

# def get_rsi(closes, period=14):
#     if len(closes) < period + 1:
#         return None
#     deltas = np.diff(closes)
#     ups = deltas.clip(min=0)
#     downs = -deltas.clip(max=0)
#     roll_up = np.mean(ups[-period:])
#     roll_down = np.mean(downs[-period:])
#     if roll_down == 0:
#         return 100
#     rs = roll_up / roll_down
#     return round(100 - (100 / (1 + rs)), 2)

# def get_macd(closes, fast=12, slow=26, signal=9):
#     if len(closes) < slow + signal:
#         return None, None, None
#     ema_fast = get_ema(closes, fast)
#     ema_slow = get_ema(closes, slow)
#     if ema_fast is None or ema_slow is None:
#         return None, None, None
#     macd_line = ema_fast - ema_slow
#     macd_series = []
#     for i in range(slow, len(closes)):
#         fast_i = get_ema(closes[:i+1], fast)
#         slow_i = get_ema(closes[:i+1], slow)
#         if fast_i is None or slow_i is None:
#             continue
#         macd_series.append(fast_i - slow_i)
#     signal_line = get_ema(macd_series, signal) if len(macd_series) >= signal else None
#     macd_hist = macd_line - signal_line if signal_line else 0
#     return round(macd_line, 5), round(signal_line, 5) if signal_line else None, round(macd_hist, 5)

# def compute_features(closes, volumes=None):
#     ema9 = get_ema(closes, 9)
#     ema26 = get_ema(closes, 26)
#     rsi14 = get_rsi(closes, 14)
#     macd_line, macd_signal, macd_hist = get_macd(closes)
#     last_volume = volumes[-1] if volumes else 0
#     return [ema9, ema26, rsi14, macd_line, macd_signal, macd_hist, last_volume]

# def estimate_volatility(closes):
#     if len(closes) < 2:
#         return 0.001
#     returns = np.diff(closes) / closes[:-1]
#     return np.std(returns)

# # ---------------- Binance ----------------

# def fetch_klines(symbol, interval, limit=200):
#     url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
#     try:
#         r = requests.get(url, timeout=10)
#         r.raise_for_status()
#         return r.json()
#     except Exception as e:
#         print(f"[REST] Error fetching klines {symbol} {interval}: {e}")
#         return []

# # ---------------- ML ----------------

# def model_path(symbol):
#     return os.path.join(MODEL_DIR, f"{symbol}_xgb.json")

# def load_model(symbol):
#     path = model_path(symbol)
#     if os.path.exists(path):
#         model = xgb.XGBClassifier()
#         model.load_model(path)
#         return model
#     return None

# def save_model(model, symbol):
#     model.save_model(model_path(symbol))

# def train_ml_model(symbol):
#     X, y = [], []
#     for record in training_data:
#         if record["symbol"] != symbol:
#             continue
#         X.append(record["features"])
#         y.append(1 if record["outcome"] else 0)
#     if len(X) < 20:
#         return None
#     model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
#     model.fit(np.array(X), np.array(y))
#     save_model(model, symbol)
#     return model

# def predict_trend(model, closes, volumes=None):
#     feats = compute_features(closes, volumes)
#     if None in feats:
#         return None
#     return model.predict_proba([feats])[0][1] if model else None

# def predict_next_5_candles(model, closes, volumes=None):
#     preds = []
#     for i in range(5):
#         prob = predict_trend(model, closes, volumes)
#         if prob is None:
#             break
#         preds.append(prob)
#         closes.append(closes[-1] * (1 + (0.001 if prob > 0.5 else -0.001)))
#     if not preds:
#         return None
#     avg_prob = np.mean(preds)
#     trend = "Up" if avg_prob > 0.5 else "Down"
#     return trend, round(avg_prob * 100, 2)

# # ---------------- EMA Monitor ----------------

# async def monitor_ema(symbol, interval):
#     klines = fetch_klines(symbol, interval, limit=200)
#     closes = [float(k[4]) for k in klines]
#     volumes = [float(k[5]) for k in klines]

#     model = load_model(symbol) or train_ml_model(symbol)

#     prev_ema9 = get_ema(closes, 9)
#     prev_ema26 = get_ema(closes, 26)

#     while True:
#         await asyncio.sleep(5)
#         klines_new = fetch_klines(symbol, interval, limit=2)
#         if not klines_new:
#             continue

#         close_price = float(klines_new[-1][4])
#         closes.append(close_price)
#         volumes.append(float(klines_new[-1][5]))
#         ema9 = get_ema(closes, 9)
#         ema26 = get_ema(closes, 26)
#         if None in [prev_ema9, prev_ema26, ema9, ema26]:
#             prev_ema9, prev_ema26 = ema9, ema26
#             continue

#         prob = predict_trend(model, closes, volumes)
#         if prob is None:
#             prev_ema9, prev_ema26 = ema9, ema26
#             continue  # Skip until we have valid prediction

#         next_trend = predict_next_5_candles(model, closes.copy(), volumes.copy())
#         vol = estimate_volatility(closes)

#         # High-confidence, high-profit signals
#         def high_conf_signal(direction):
#             expected_move = vol * ((prob if direction == "up" else (1 - prob))) * 20 * 100  # leverage 20x
#             return prob >= MIN_PROBABILITY and expected_move >= TARGET_PROFIT_PERCENT


#         if prev_ema9 < prev_ema26 and ema9 >= ema26 and high_conf_signal("up"):
#             msg = f"📈 {symbol} ({interval}) EMA9 crossed ABOVE EMA26 — BUY 💰\nPrice: {close_price}"
#             msg += f"\n🤖 Uptrend Probability: {round(prob * 100, 2)}%"
#             msg += f"\n📊 Estimated Volatility Move: {round(vol * 100, 2)}%"
#             if next_trend:
#                 msg += f"\n🔮 Next 5 Candles: {next_trend[0]} ({next_trend[1]}%)"
#             broadcast(msg)
#             TRADE_HISTORY.append({
#                 "symbol": symbol, "interval": interval, "time": datetime.utcnow(),
#                 "price": close_price, "direction": "up", "features": compute_features(closes, volumes)
#             })

#         elif prev_ema9 > prev_ema26 and ema9 <= ema26 and high_conf_signal("down"):
#             msg = f"📉 {symbol} ({interval}) EMA9 crossed BELOW EMA26 — SELL ⚠️\nPrice: {close_price}"
#             msg += f"\n🤖 Downtrend Probability: {round((1 - prob) * 100, 2)}%"
#             msg += f"\n📊 Estimated Volatility Move: {round(vol * 100, 2)}%"
#             if next_trend:
#                 msg += f"\n🔮 Next 5 Candles: {next_trend[0]} ({next_trend[1]}%)"
#             broadcast(msg)
#             TRADE_HISTORY.append({
#                 "symbol": symbol, "interval": interval, "time": datetime.utcnow(),
#                 "price": close_price, "direction": "down", "features": compute_features(closes, volumes)
#             })

#         # Update model based on actual outcome
#         to_remove = []
#         for trade in TRADE_HISTORY:
#             interval_min = 5 if trade["interval"] == "5m" else 15
#             if (datetime.utcnow() - trade["time"]).total_seconds() >= interval_min * 60:
#                 kl_check = fetch_klines(trade["symbol"], trade["interval"], limit=2)
#                 if kl_check:
#                     next_price = float(kl_check[-1][4])
#                     actual_up = next_price > trade["price"]
#                     outcome = (actual_up and trade["direction"] == "up") or (not actual_up and trade["direction"] == "down")
#                     add_training_record(trade["symbol"], trade["features"], outcome)
#                     model = train_ml_model(trade["symbol"])
#                 to_remove.append(trade)
#         for tr in to_remove:
#             TRADE_HISTORY.remove(tr)

#         prev_ema9, prev_ema26 = ema9, ema26

# # ---------------- Telegram Webhook ----------------

# def send_telegram(chat_id, msg):
#     if not BOT_TOKEN:
#         print("❌ Missing BOT_TOKEN environment variable.")
#         return
#     url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
#     payload = {
#         "chat_id": chat_id,
#         "text": msg,
#         "parse_mode": "Markdown"
#     }
#     try:
#         r = requests.post(url, json=payload, timeout=10)
#         r.raise_for_status()
#     except Exception as e:
#         print(f"[Telegram Error]: {e}")


# def broadcast(msg):
#     for chat_id in user_ids:
#         send_telegram(chat_id, msg)

# user_ids = set()

# @app.post("/webhook/{token}")
# async def telegram_webhook(token: str, request: Request):
#     if token != BOT_TOKEN:
#         return {"ok": False}
#     data = await request.json()
#     if "message" in data:
#         chat_id = data["message"]["chat"]["id"]
#         text = data["message"].get("text", "")
#         if chat_id not in user_ids:
#             user_ids.add(chat_id)
#             send_telegram(chat_id, f"✅ Subscribed to EMA alerts!\nTracking: {', '.join(SYMBOLS)}")
#         if text.lower() == "/start":
#             send_telegram(chat_id, "👋 Welcome! EMA + AI alerts active.")
#     return {"ok": True}

# # ---------------- Startup ----------------

# @app.on_event("startup")
# async def startup_event():
#     for symbol in SYMBOLS:
#         for tf in TIMEFRAMES:
#             asyncio.create_task(monitor_ema(symbol, tf))
#     print("✅ EMA + ML Monitoring Started")

import os
import time
import json
import asyncio
import requests
import numpy as np
import joblib
from datetime import datetime, timedelta
from fastapi import FastAPI, Request
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = FastAPI()

# =================== CONFIG ===================
BOT_TOKEN = os.getenv("BOT_TOKEN")
SYMBOLS = ["BTCUSDT", "ONDOUSDT", "OPUSDT"]
API_URL = "https://api.binance.com/api/v3/klines"
MODEL_PATH = "models"
os.makedirs(MODEL_PATH, exist_ok=True)

user_ids = set()
last_trained = {}
model_cache = {}

# =================== TELEGRAM ===================
def send_telegram(chat_id, text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Telegram error: {e}")

# =================== DATA FETCH ===================
def fetch_candles(symbol, interval="5m", limit=500):
    try:
        url = f"{API_URL}?symbol={symbol}&interval={interval}&limit={limit}"
        data = requests.get(url).json()
        closes = [float(x[4]) for x in data]
        volumes = [float(x[5]) for x in data]
        return np.array(closes), np.array(volumes)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return np.array([]), np.array([])

# =================== INDICATORS ===================
def get_ema(prices, period):
    if len(prices) < period:
        return np.array([])
    return np.convolve(prices, np.ones(period)/period, mode='valid')

def get_rsi(prices, period=14):
    if len(prices) <= period:
        return np.array([])
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.convolve(gain, np.ones(period)/period, mode='valid')
    avg_loss = np.convolve(loss, np.ones(period)/period, mode='valid')
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_features(closes, volumes):
    ema_fast = get_ema(closes, 12)
    ema_slow = get_ema(closes, 26)
    rsi = get_rsi(closes)

    # Align all feature lengths
    min_len = min(len(ema_fast), len(ema_slow), len(rsi), len(volumes))
    if min_len == 0:
        raise ValueError("Not enough data for feature computation.")

    ema_fast = ema_fast[-min_len:]
    ema_slow = ema_slow[-min_len:]
    rsi = rsi[-min_len:]
    volumes = volumes[-min_len:]

    features = np.column_stack([ema_fast, ema_slow, rsi, volumes])
    return features

# =================== MODEL TRAINING ===================
def train_ml_model(symbol, closes, volumes):
    try:
        X = compute_features(closes, volumes)
        y = np.where(np.diff(closes[-len(X)-1:]) > 0, 1, 0)
        y = y[-len(X):]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, n_jobs=1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        joblib.dump(model, f"{MODEL_PATH}/{symbol}.pkl")
        model_cache[symbol] = model
        last_trained[symbol] = datetime.utcnow()
        print(f"✅ {symbol} model trained. Accuracy: {acc:.2f}")
        return model
    except Exception as e:
        print(f"Training failed for {symbol}: {e}")
        return None

def load_model(symbol):
    path = f"{MODEL_PATH}/{symbol}.pkl"
    if os.path.exists(path):
        model = joblib.load(path)
        model_cache[symbol] = model
        return model
    return None

# =================== PREDICTION ===================
def predict_signal(model, closes, volumes):
    X = compute_features(closes, volumes)
    X_latest = X[-1].reshape(1, -1)
    prob = model.predict_proba(X_latest)[0][1]
    return prob

# =================== MONITOR ===================
async def monitor_ema():
    await asyncio.sleep(10)
    while True:
        for symbol in SYMBOLS:
            closes, volumes = fetch_candles(symbol)
            if len(closes) < 50:
                continue

            model = model_cache.get(symbol) or load_model(symbol)
            if model is None or (symbol not in last_trained or datetime.utcnow() - last_trained[symbol] > timedelta(hours=4)):
                model = train_ml_model(symbol, closes, volumes)

            if model is None:
                continue

            try:
                prob = predict_signal(model, closes, volumes)
                trend = "📈 Bullish" if prob > 0.5 else "📉 Bearish"
                chance = round(prob * 100, 2)

                if chance >= 60:
                    msg = (
                        f"📊 *{symbol}*\n"
                        f"Trend: {trend}\n"
                        f"Probability: {chance}%\n"
                        f"Prediction: {'Up' if prob > 0.5 else 'Down'} in next 5 candles.\n"
                        f"⏰ {datetime.utcnow().strftime('%H:%M:%S UTC')}"
                    )
                    for uid in user_ids:
                        send_telegram(uid, msg)

            except Exception as e:
                print(f"Prediction failed for {symbol}: {e}")

            await asyncio.sleep(2)

        await asyncio.sleep(300)  # check every 5 min

# =================== TELEGRAM WEBHOOK ===================
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
            send_telegram(chat_id, "👋 Welcome! EMA + AI alerts active.")
    return {"ok": True}

# =================== STARTUP ===================
@app.on_event("startup")
async def startup_event():
    for sym in SYMBOLS:
        load_model(sym)
    asyncio.create_task(monitor_ema())
