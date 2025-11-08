# import os
# import time
# import json
# import asyncio
# import requests
# import numpy as np
# import joblib
# from datetime import datetime, timedelta
# from fastapi import FastAPI, Request
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# app = FastAPI()

# # =================== CONFIG ===================
# BOT_TOKEN = os.getenv("BOT_TOKEN")
# SYMBOLS = ["BTCUSDT", "ONDOUSDT", "OPUSDT"]
# API_URL = "https://api.binance.com/api/v3/klines"
# MODEL_PATH = "models"
# os.makedirs(MODEL_PATH, exist_ok=True)

# user_ids = set()
# last_trained = {}
# model_cache = {}

# # =================== TELEGRAM ===================
# def send_telegram(chat_id, text):
#     url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
#     payload = {"chat_id": chat_id, "text": text}
#     try:
#         requests.post(url, json=payload)
#     except Exception as e:
#         print(f"Telegram error: {e}")

# # =================== DATA FETCH ===================
# def fetch_candles(symbol, interval="5m", limit=500):
#     try:
#         url = f"{API_URL}?symbol={symbol}&interval={interval}&limit={limit}"
#         data = requests.get(url).json()
#         closes = [float(x[4]) for x in data]
#         volumes = [float(x[5]) for x in data]
#         return np.array(closes), np.array(volumes)
#     except Exception as e:
#         print(f"Error fetching data: {e}")
#         return np.array([]), np.array([])

# # =================== INDICATORS ===================
# def get_ema(prices, period):
#     if len(prices) < period:
#         return np.array([])
#     return np.convolve(prices, np.ones(period)/period, mode='valid')

# def get_rsi(prices, period=14):
#     if len(prices) <= period:
#         return np.array([])
#     deltas = np.diff(prices)
#     gain = np.where(deltas > 0, deltas, 0)
#     loss = np.where(deltas < 0, -deltas, 0)
#     avg_gain = np.convolve(gain, np.ones(period)/period, mode='valid')
#     avg_loss = np.convolve(loss, np.ones(period)/period, mode='valid')
#     rs = avg_gain / (avg_loss + 1e-10)
#     return 100 - (100 / (1 + rs))

# def compute_features(closes, volumes):
#     ema_fast = get_ema(closes, 12)
#     ema_slow = get_ema(closes, 26)
#     rsi = get_rsi(closes)

#     # Align all feature lengths
#     min_len = min(len(ema_fast), len(ema_slow), len(rsi), len(volumes))
#     if min_len == 0:
#         raise ValueError("Not enough data for feature computation.")

#     ema_fast = ema_fast[-min_len:]
#     ema_slow = ema_slow[-min_len:]
#     rsi = rsi[-min_len:]
#     volumes = volumes[-min_len:]

#     features = np.column_stack([ema_fast, ema_slow, rsi, volumes])
#     return features

# # =================== MODEL TRAINING ===================
# def train_ml_model(symbol, closes, volumes):
#     try:
#         X = compute_features(closes, volumes)
#         y = np.where(np.diff(closes[-len(X)-1:]) > 0, 1, 0)
#         y = y[-len(X):]

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#         model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, n_jobs=1)
#         model.fit(X_train, y_train)
#         preds = model.predict(X_test)
#         acc = accuracy_score(y_test, preds)

#         joblib.dump(model, f"{MODEL_PATH}/{symbol}.pkl")
#         model_cache[symbol] = model
#         last_trained[symbol] = datetime.utcnow()
#         print(f"✅ {symbol} model trained. Accuracy: {acc:.2f}")
#         return model
#     except Exception as e:
#         print(f"Training failed for {symbol}: {e}")
#         return None

# def load_model(symbol):
#     path = f"{MODEL_PATH}/{symbol}.pkl"
#     if os.path.exists(path):
#         model = joblib.load(path)
#         model_cache[symbol] = model
#         return model
#     return None

# # =================== PREDICTION ===================
# def predict_signal(model, closes, volumes):
#     X = compute_features(closes, volumes)
#     X_latest = X[-1].reshape(1, -1)
#     prob = model.predict_proba(X_latest)[0][1]
#     return prob

# # =================== MONITOR ===================
# async def monitor_ema():
#     await asyncio.sleep(10)
#     while True:
#         for symbol in SYMBOLS:
#             closes, volumes = fetch_candles(symbol)
#             if len(closes) < 50:
#                 continue

#             model = model_cache.get(symbol) or load_model(symbol)
#             if model is None or (symbol not in last_trained or datetime.utcnow() - last_trained[symbol] > timedelta(hours=4)):
#                 model = train_ml_model(symbol, closes, volumes)

#             if model is None:
#                 continue

#             try:
#                 prob = predict_signal(model, closes, volumes)
#                 trend = "📈 Bullish" if prob > 0.5 else "📉 Bearish"
#                 chance = round(prob * 100, 2)

#                 if chance >= 60:
#                     msg = (
#                         f"📊 *{symbol}*\n"
#                         f"Trend: {trend}\n"
#                         f"Probability: {chance}%\n"
#                         f"Prediction: {'Up' if prob > 0.5 else 'Down'} in next 5 candles.\n"
#                         f"⏰ {datetime.utcnow().strftime('%H:%M:%S UTC')}"
#                     )
#                     for uid in user_ids:
#                         send_telegram(uid, msg)

#             except Exception as e:
#                 print(f"Prediction failed for {symbol}: {e}")

#             await asyncio.sleep(2)

#         await asyncio.sleep(300)  # check every 5 min

# # =================== TELEGRAM WEBHOOK ===================
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

# # =================== STARTUP ===================
# @app.on_event("startup")
# async def startup_event():
#     for sym in SYMBOLS:
#         load_model(sym)
#     asyncio.create_task(monitor_ema())

# test

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

PRED_LOG_PATH = "prediction_log.json"
FEED_LOG_PATH = "feedback_log.json"

user_ids = set()
last_trained = {}
model_cache = {}

# =================== HELPERS ===================
def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

prediction_log = load_json(PRED_LOG_PATH)
feedback_log = load_json(FEED_LOG_PATH)

# =================== TELEGRAM ===================
def send_telegram(chat_id, text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
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

    min_len = min(len(ema_fast), len(ema_slow), len(rsi), len(volumes))
    if min_len == 0:
        raise ValueError("Not enough data for feature computation.")

    ema_fast = ema_fast[-min_len:]
    ema_slow = ema_slow[-min_len:]
    rsi = rsi[-min_len:]
    volumes = volumes[-min_len:]

    return np.column_stack([ema_fast, ema_slow, rsi, volumes])

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

# =================== FEEDBACK SYSTEM ===================
# def log_prediction(symbol, direction, price, prob):
#     prediction_log.append({
#         "symbol": symbol,
#         "timestamp": datetime.utcnow().isoformat(),
#         "direction": direction,
#         "price": price,
#         "prob": prob,
#         "checked": False
#     })
#     save_json(PRED_LOG_PATH, prediction_log)

# def evaluate_feedback():
#     now = datetime.utcnow()
#     new_feedbacks = 0

#     for pred in prediction_log:
#         if pred["checked"]:
#             continue

#         pred_time = datetime.fromisoformat(pred["timestamp"])
#         if (now - pred_time) < timedelta(minutes=15):  # wait 3 candles (5m each)
#             continue

#         closes, _ = fetch_candles(pred["symbol"], limit=5)
#         if len(closes) == 0:
#             continue

#         current_price = closes[-1]
#         change_pct = ((current_price - pred["price"]) / pred["price"]) * 100

#         correct = (
#             (pred["direction"] == "bullish" and change_pct >= 10) or
#             (pred["direction"] == "bearish" and change_pct <= -10)
#         )

#         feedback_log.append({
#             "symbol": pred["symbol"],
#             "timestamp": pred["timestamp"],
#             "direction": pred["direction"],
#             "price_change_pct": round(change_pct, 2),
#             "outcome": "correct" if correct else "wrong"
#         })

#         pred["checked"] = True
#         new_feedbacks += 1

#     if new_feedbacks > 0:
#         save_json(FEED_LOG_PATH, feedback_log)
#         save_json(PRED_LOG_PATH, prediction_log)
#         print(f"✅ Evaluated {new_feedbacks} past predictions.")
def evaluate_feedback():
    now = datetime.utcnow()
    new_feedbacks = 0

    for pred in prediction_log:
        if pred.get("checked"):
            continue

        pred_time = datetime.fromisoformat(pred["timestamp"])
        if (now - pred_time) < timedelta(minutes=15):  # wait 3 candles (5m each)
            continue

        closes, _ = fetch_candles(pred["symbol"], limit=5)
        if len(closes) == 0:
            continue

        current_price = closes[-1]
        change_pct = ((current_price - pred["price"]) / pred["price"]) * 100

        # Determine if signal was correct
        correct = (
            (pred["direction"] == "bullish" and change_pct >= 10) or
            (pred["direction"] == "bearish" and change_pct <= -10)
        )

        outcome = "correct" if correct else "wrong"
        reason = ""

        if correct and pred["direction"] == "bullish":
            reason = f"✅ Bullish call was correct — price rose {change_pct:.2f}% in 15m."
        elif correct and pred["direction"] == "bearish":
            reason = f"✅ Bearish call was correct — price dropped {abs(change_pct):.2f}% in 15m."
        elif not correct and pred["direction"] == "bullish":
            reason = f"❌ Bullish call was wrong — price actually fell {abs(change_pct):.2f}%."
        elif not correct and pred["direction"] == "bearish":
            reason = f"❌ Bearish call was wrong — price actually rose {abs(change_pct):.2f}%."

        # Record feedback
        feedback_log.append({
            "symbol": pred["symbol"],
            "timestamp": pred["timestamp"],
            "direction": pred["direction"],
            "price_change_pct": round(change_pct, 2),
            "outcome": outcome,
            "reason": reason
        })

        # Send result message to users
        msg = (
            f"📈 *Signal Review*\n"
            f"Symbol: *{pred['symbol']}*\n"
            f"Direction: {pred['direction'].capitalize()}\n"
            f"Result: {'✅ Correct' if correct else '❌ Wrong'}\n"
            f"{reason}\n"
            f"⏰ Checked at {datetime.utcnow().strftime('%H:%M:%S UTC')}"
        )
        for uid in user_ids:
            send_telegram(uid, msg)

        pred["checked"] = True
        new_feedbacks += 1

    if new_feedbacks > 0:
        save_json(FEED_LOG_PATH, feedback_log)
        save_json(PRED_LOG_PATH, prediction_log)
        print(f"✅ Evaluated {new_feedbacks} past predictions and sent results.")


def recent_accuracy(symbol, window=10):
    data = [f for f in feedback_log if f["symbol"] == symbol][-window:]
    if not data:
        return 0.5
    correct = sum(1 for f in data if f["outcome"] == "correct")
    return correct / len(data)

def retrain_with_feedback(symbol):
    acc = recent_accuracy(symbol)
    if acc < 0.4:
        print(f"⚠️ Skipping retrain for {symbol}: poor performance ({acc:.2f})")
        return

    closes, volumes = fetch_candles(symbol)
    if len(closes) > 50:
        train_ml_model(symbol, closes, volumes)

# =================== MONITOR ===================
async def monitor_learning():
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

            prob = predict_signal(model, closes, volumes)
            price = closes[-1]
            acc = recent_accuracy(symbol)

            # Smart alert logic
            if prob > 0.7 and acc > 0.6:
                direction = "bullish"
                msg = (
                    f"📊 *{symbol}*\n"
                    f"Trend: 📈 Bullish\n"
                    f"Probability: {prob*100:.2f}%\n"
                    f"Recent accuracy: {acc*100:.2f}%\n"
                    f"⏰ {datetime.utcnow().strftime('%H:%M:%S UTC')}"
                )
                for uid in user_ids:
                    send_telegram(uid, msg)
                log_prediction(symbol, direction, price, prob)

            elif prob < 0.3 and acc > 0.6:
                direction = "bearish"
                msg = (
                    f"📊 *{symbol}*\n"
                    f"Trend: 📉 Bearish\n"
                    f"Probability: {(1-prob)*100:.2f}%\n"
                    f"Recent accuracy: {acc*100:.2f}%\n"
                    f"⏰ {datetime.utcnow().strftime('%H:%M:%S UTC')}"
                )
                for uid in user_ids:
                    send_telegram(uid, msg)
                log_prediction(symbol, direction, price, prob)

            await asyncio.sleep(2)

        evaluate_feedback()

        # Retrain after every 20 feedback records
        if len(feedback_log) % 20 == 0 and len(feedback_log) > 0:
            for sym in SYMBOLS:
                retrain_with_feedback(sym)

        await asyncio.sleep(300)

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
            send_telegram(chat_id, f"✅ Subscribed to AI EMA alerts!\nTracking: {', '.join(SYMBOLS)}")
        if text.lower() == "/start":
            send_telegram(chat_id, "👋 Welcome! AI learning alerts active.")
    return {"ok": True}

# =================== STARTUP ===================
@app.on_event("startup")
async def startup_event():
    for sym in SYMBOLS:
        load_model(sym)
    asyncio.create_task(monitor_learning())
