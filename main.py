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

# import os
# import json
# import requests
# import asyncio
# import numpy as np
# from datetime import datetime, timedelta, timezone
# from fastapi import FastAPI, Request
# import xgboost as xgb
# from sklearn.preprocessing import StandardScaler

# # ---------------- Config ----------------
# BOT_TOKEN = os.getenv("BOT_TOKEN")
# SYMBOLS = ["OPUSDT", "RENDERUSDT", "BTCUSDT"]
# TIMEFRAMES = ["5m", "15m"]

# MODEL_DIR = "models"
# os.makedirs(MODEL_DIR, exist_ok=True)

# app = FastAPI()
# user_ids = set()
# sent_hourly = set()
# open_signals = {}  # track open trades for accuracy eval

# # ---------------- Telegram ----------------
# def send_telegram(chat_id, msg):
#     if not BOT_TOKEN:
#         print("BOT_TOKEN not set — would send:", msg)
#         return
#     try:
#         url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
#         requests.post(url, data={"chat_id": chat_id, "text": msg}, timeout=5)
#     except Exception as e:
#         print("Telegram send error:", e)

# def broadcast(msg):
#     for uid in list(user_ids):
#         send_telegram(uid, msg)

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
#     if len(macd_series) < signal:
#         signal_line = None
#     else:
#         signal_line = get_ema(macd_series, signal)
#     macd_hist = macd_line - signal_line if signal_line is not None else 0
#     return round(macd_line, 5), round(signal_line, 5) if signal_line else None, round(macd_hist, 5)

# def compute_features(closes, volumes=None):
#     ema9 = get_ema(closes, 9)
#     ema26 = get_ema(closes, 26)
#     rsi14 = get_rsi(closes, 14)
#     macd_line, macd_signal, macd_hist = get_macd(closes)
#     last_volume = volumes[-1] if volumes else 0
#     return [ema9, ema26, rsi14, macd_line, macd_signal, macd_hist, last_volume]

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

# # ---------------- Telegram Webhook ----------------
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
#             send_telegram(chat_id, "👋 Welcome! EMA + AI alerts are active.")
#     return {"ok": True}

# # ---------------- ML Model ----------------
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

# def train_ml_model(symbol, closes, volumes=None):
#     X, y = [], []
#     for i in range(50, len(closes)-1):  # use more candles for better accuracy
#         feats = compute_features(closes[:i], volumes[:i] if volumes else None)
#         if None in feats:
#             continue
#         X.append(feats)
#         y.append(1 if closes[i+1] > closes[i] else 0)
#     if len(X) < 40:
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

# # ---------------- EMA Monitor ----------------
# async def monitor_ema(symbol, interval):
#     klines = fetch_klines(symbol, interval, limit=300)
#     closes = [float(k[4]) for k in klines]
#     volumes = [float(k[5]) for k in klines]
#     model = load_model(symbol)
#     if not model:
#         model = train_ml_model(symbol, closes, volumes)

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

#         if interval == "15m":
#             print(f"[LOG] {symbol} 15m checked — Prob: {round(prob*100,2) if prob else '?'}%")
#         else:
#             if prev_ema9 < prev_ema26 and ema9 >= ema26:
#                 if prob and prob * 100 >= 60:
#                     msg = f"📈 {symbol} ({interval}) EMA9 crossed ABOVE EMA26 — BUY 💰\nPrice: {close_price}"
#                     msg += f"\n🤖 Uptrend Probability: {round(prob*100,2)}%"
#                     broadcast(msg)
#                     open_signals[symbol] = {"type": "buy", "price": close_price, "candle_count": 0, "time": datetime.utcnow()}
#             elif prev_ema9 > prev_ema26 and ema9 <= ema26:
#                 if prob and (1 - prob) * 100 >= 60:
#                     msg = f"📉 {symbol} ({interval}) EMA9 crossed BELOW EMA26 — SELL ⚠️\nPrice: {close_price}"
#                     msg += f"\n🤖 Downtrend Probability: {round((1-prob)*100,2)}%"
#                     broadcast(msg)
#                     open_signals[symbol] = {"type": "sell", "price": close_price, "candle_count": 0, "time": datetime.utcnow()}

#         # Accuracy check for open signals
#         for sym, sig in list(open_signals.items()):
#             sig["candle_count"] += 1
#             if sig["candle_count"] >= 5:
#                 new_price = closes[-1]
#                 change = (new_price - sig["price"]) / sig["price"] * 100
#                 if sig["type"] == "sell":
#                     change = -change
#                 profit = change * 20  # 20x leverage
#                 result = "✅ PROFIT" if profit > 10 else "❌ LOSS"
#                 msg = f"📊 Accuracy Check: {sym}\n{result}\nProfit: {round(profit,2)}%\nChecked after 5 candles\n🕒 {datetime.utcnow().strftime('%H:%M:%S UTC')}"
#                 broadcast(msg)
#                 print(msg)
#                 open_signals.pop(sym)
#                 model = train_ml_model(sym, closes, volumes)

#         prev_ema9, prev_ema26 = ema9, ema26

# # ---------------- Hourly Close Alerts ----------------
# async def monitor_hourly():
#     while True:
#         now = datetime.now(timezone.utc)
#         next_hour = (now.replace(minute=0, second=5, microsecond=0) + timedelta(hours=1))
#         await asyncio.sleep((next_hour - now).total_seconds())
#         for symbol in SYMBOLS:
#             klines = fetch_klines(symbol, "1h", limit=2)
#             if not klines:
#                 continue
#             last = klines[-1]
#             close_time_ms = int(last[6])
#             key = (symbol, "1h", close_time_ms)
#             if key in sent_hourly:
#                 continue
#             close_price = float(last[4])
#             ts = datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc)
#             msg = f"🕐 {symbol} 1H Close ({ts.strftime('%Y-%m-%d %H:%M UTC')})\nClose: {close_price}"
#             broadcast(msg)
#             sent_hourly.add(key)

# # ---------------- 4-Hour Retraining ----------------
# async def retrain_loop():
#     while True:
#         await asyncio.sleep(4 * 60 * 60)
#         for symbol in SYMBOLS:
#             klines = fetch_klines(symbol, "5m", limit=300)
#             closes = [float(k[4]) for k in klines]
#             volumes = [float(k[5]) for k in klines]
#             model = train_ml_model(symbol, closes, volumes)
#             if model:
#                 print(f"[RETRAIN] Model for {symbol} updated at {datetime.now()}")

# # ---------------- Startup ----------------
# @app.on_event("startup")
# async def startup_event():
#     for symbol in SYMBOLS:
#         for tf in TIMEFRAMES:
#             asyncio.create_task(monitor_ema(symbol, tf))
#     asyncio.create_task(monitor_hourly())
#     asyncio.create_task(retrain_loop())

# test2
# main.py
import os
import json
import time
import requests
import asyncio
import numpy as np
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Request
import xgboost as xgb

# ---------------- Config ----------------
BOT_TOKEN = os.getenv("BOT_TOKEN")                   # keep like this in Railway env
SYMBOLS = ["OPUSDT", "RENDERUSDT", "BTCUSDT"]
TIMEFRAMES = ["5m", "15m"]                           # we'll only BROADCAST 5m alerts; 15m logged
MODEL_DIR = "models"
DATA_FILE = "training_data.json"
os.makedirs(MODEL_DIR, exist_ok=True)

# trading/profit parameters
LEVERAGE = 20                                       # user-specified leverage assumption
TARGET_ACCOUNT_PROFIT_PCT = 10.0                    # target account profit in percent (10%)
PRICE_TARGET_PCT = TARGET_ACCOUNT_PROFIT_PCT / LEVERAGE  # required price move percent (0.5%)

# ML / training params
MIN_TRAIN_SAMPLES = 30
RETRAIN_INTERVAL_SECONDS = 4 * 3600                 # 4 hours

app = FastAPI()
user_ids = set()            # subscribers (chat_id ints)
training_data = []          # stored training records in memory
ML_MODELS = {}              # symbol -> xgb model (if trained)

# ---------------- Persistence ----------------
def load_training_data():
    global training_data
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r") as f:
                training_data = json.load(f)
        else:
            training_data = []
    except Exception as e:
        print("Failed to load training data:", e)
        training_data = []

def save_training_data():
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(training_data, f)
    except Exception as e:
        print("Failed to save training data:", e)

load_training_data()

def add_training_record(symbol, features, outcome):
    training_data.append({"symbol": symbol, "features": features, "outcome": bool(outcome), "ts": datetime.utcnow().isoformat()})
    save_training_data()

# ---------------- Binance REST ----------------
USER_AGENT = "ai-emacross-bot/1.0"
def fetch_klines(symbol: str, interval: str, limit: int = 200):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[ERR] fetch_klines {symbol} {interval}: {e}")
        return []

def closes_from_klines(klines):
    return [float(k[4]) for k in klines]

# ---------------- Indicators & Features ----------------
def get_ema(values, period):
    if len(values) < period or period < 1:
        return None
    weights = np.exp(np.linspace(-1., 0., period))
    weights /= weights.sum()
    # 'valid' to avoid earlier indexing issues; returns array length len(values)-period+1
    ema_arr = np.convolve(values, weights, mode="valid")
    if len(ema_arr) == 0:
        return None
    return float(np.round(ema_arr[-1], 8))

def get_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes)
    ups = deltas.copy()
    ups[ups < 0] = 0.0
    downs = -deltas.copy()
    downs[downs < 0] = 0.0
    roll_up = np.mean(ups[-period:]) if len(ups) >= period else np.mean(ups)
    roll_down = np.mean(downs[-period:]) if len(downs) >= period else np.mean(downs)
    if roll_down == 0:
        return 100.0
    rs = roll_up / roll_down
    return float(round(100 - (100.0 / (1 + rs)), 2))

def get_macd(closes, fast=12, slow=26, signal=9):
    if len(closes) < slow + signal:
        return None, None, None
    ema_fast = get_ema(closes, fast)
    ema_slow = get_ema(closes, slow)
    if ema_fast is None or ema_slow is None:
        return None, None, None
    macd_line = ema_fast - ema_slow
    # build macd_series (fast - slow) across history to compute signal line
    macd_series = []
    for i in range(slow, len(closes)):
        fast_i = get_ema(closes[:i+1], fast)
        slow_i = get_ema(closes[:i+1], slow)
        if fast_i is None or slow_i is None:
            continue
        macd_series.append(fast_i - slow_i)
    if len(macd_series) < signal:
        signal_line = None
    else:
        signal_line = get_ema(macd_series, signal)
    macd_hist = macd_line - signal_line if signal_line is not None else 0.0
    return (float(round(macd_line,5)), float(round(signal_line,5)) if signal_line else None, float(round(macd_hist,5)))

def compute_features(closes, volumes=None):
    ema9 = get_ema(closes, 9)
    ema26 = get_ema(closes, 26)
    rsi14 = get_rsi(closes, 14)
    macd_line, macd_signal, macd_hist = get_macd(closes)
    last_volume = float(volumes[-1]) if volumes and len(volumes) else 0.0
    # return flat array of scalar features (no arrays)
    return [ema9, ema26, rsi14, macd_line, macd_signal, macd_hist, last_volume]

# ---------------- ML: train / predict ----------------
def model_path(symbol):
    return os.path.join(MODEL_DIR, f"{symbol}_xgb.json")

def save_model(model, symbol):
    try:
        model.save_model(model_path(symbol))
    except Exception as e:
        print("save_model error:", e)

def load_model(symbol):
    path = model_path(symbol)
    if os.path.exists(path):
        try:
            m = xgb.XGBClassifier()
            m.load_model(path)
            return m
        except Exception as e:
            print("load_model error:", e)
    return None

def train_ml_model(symbol):
    # collect training records for this symbol
    X, y = [], []
    for rec in training_data:
        if rec.get("symbol") != symbol:
            continue
        feats = rec.get("features")
        outcome = rec.get("outcome")
        if not feats or any(f is None for f in feats):
            continue
        X.append(feats)
        y.append(1 if outcome else 0)
    if len(X) < MIN_TRAIN_SAMPLES:
        print(f"[ML] not enough samples to train {symbol}: {len(X)}")
        return None
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    try:
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model.fit(X, y)
        save_model(model, symbol)
        ML_MODELS[symbol] = model
        print(f"[ML] trained model for {symbol} on {len(X)} samples")
        return model
    except Exception as e:
        print("train_ml_model error:", e)
        return None

def predict_trend(model, closes, volumes=None):
    if model is None:
        return None
    feats = compute_features(closes, volumes)
    if any(f is None for f in feats):
        return None
    X = np.array([feats], dtype=float)
    try:
        prob = model.predict_proba(X)[0][1]
        return float(prob)
    except Exception as e:
        print("predict_trend error:", e)
        return None

# ---------------- Telegram helpers ----------------
def send_telegram(chat_id, text):
    if not BOT_TOKEN:
        # in dev mode print
        print(f"[TELEGRAM mock -> {chat_id}]: {text}")
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=10)
    except Exception as e:
        print("send_telegram error:", e)

def broadcast(text):
    for uid in list(user_ids):
        send_telegram(uid, text)

# ---------------- Utility: time alignment ----------------
def interval_to_minutes(interval: str):
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    raise ValueError("unsupported interval")

def seconds_until_next_close(interval_minutes: int):
    now = datetime.now(timezone.utc)
    minute = now.minute
    # next multiple of interval minutes
    next_multiple = ((minute // interval_minutes) + 1) * interval_minutes
    next_hour = now.replace(minute=0, second=0, microsecond=0)
    if next_multiple >= 60:
        next_multiple -= 60
        next_hour = next_hour + timedelta(hours=1)
    next_close = next_hour.replace(minute=next_multiple, second=5, microsecond=0)
    wait = (next_close - now).total_seconds()
    return max(wait, 1.0)

# ---------------- Cross detection & profit evaluation ----------------
SENT_SIGNALS = set()   # (symbol, interval, close_time_ms) to avoid duplicates
ACTIVE_EVAL = []       # list of dicts tracking active trades to evaluate

async def monitor_interval(interval: str):
    minutes = interval_to_minutes(interval)
    print(f"[MON] Starting monitor for {interval} (only 5m alerts broadcasted)")
    # on first observed close skip alerting (prevent immediate duplicates on deploy)
    first_seen = {symbol: True for symbol in SYMBOLS}

    while True:
        wait = seconds_until_next_close(minutes)
        print(f"[SYNC] {interval} next close in {int(wait)}s (UTC {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')})")
        await asyncio.sleep(wait)

        for symbol in SYMBOLS:
            klines = fetch_klines(symbol, interval, limit=26)
            if not isinstance(klines, list) or len(klines) < 2:
                continue
            latest = klines[-1]
            close_time_ms = int(latest[6])
            key = (symbol, interval, close_time_ms)
            if key in SENT_SIGNALS:
                continue

            # compute prev and current EMA windows
            closes = closes_from_klines(klines)
            if len(closes) < 26:
                continue
            prev = closes[:-1]
            cur = closes[:]
            prev_ema9 = get_ema(prev, 9)
            prev_ema26 = get_ema(prev, 26)
            ema9 = get_ema(cur, 9)
            ema26 = get_ema(cur, 26)
            if any(v is None for v in [prev_ema9, prev_ema26, ema9, ema26]):
                continue

            bullish = prev_ema9 < prev_ema26 and ema9 >= ema26
            bearish = prev_ema9 > prev_ema26 and ema9 <= ema26

            # skip first observed close alert on startup for each symbol/interval
            if first_seen.get(symbol, True):
                first_seen[symbol] = False
                print(f"[INIT] Skipping first-close alert for {symbol} {interval} at {datetime.utcfromtimestamp(close_time_ms/1000)}")
                # but still mark as seen, don't add to SENT_SIGNALS
                continue

            # If no cross, continue
            if not (bullish or bearish):
                continue

            # Predict using ML if available
            model = ML_MODELS.get(symbol) or load_model(symbol)
            prob = predict_trend(model, cur, None) if model else None

            # Only BROADCAST signals for 5m (user requested), 15m will be logged
            # Also require confidence >= 60% to send initial signal
            if prob is None:
                msg_line = f"[LOG] {symbol} {interval} cross detected but no model/prob available. Price: {cur[-1]}"
                print(msg_line)
                # still track evaluation if user wants; here we won't broadcast
            else:
                confidence_pct = prob * 100
                direction = "BUY" if bullish else "SELL"
                price = cur[-1]
                if interval == "5m" and confidence_pct >= 60.0:
                    # send initial signal
                    msg = f"📈 {symbol} ({interval}) EMA9 CROSSED ABOVE EMA26 — {direction}\nPrice: {price}\n🤖 Confidence: {round(confidence_pct,2)}%"
                    broadcast(msg)
                    print("[ALERT-BROADCAST]", msg)
                else:
                    # for 15m or low confidence, only log
                    print(f"[LOG] {symbol} {interval} cross {direction} at {price} (conf={round(confidence_pct,2)}%)")

            # mark sent (so we don't re-evaluate same candle signal again)
            SENT_SIGNALS.add(key)

            # Start profit evaluation for this signal (only if this was a 5m cross OR we want to track all)
            # We'll track both 5m and 15m crosses but only broadcast final result if initial was broadcast
            eval_entry = {
                "symbol": symbol,
                "interval": interval,
                "close_time_ms": close_time_ms,
                "entry_price": float(cur[-1]),
                "direction": "up" if bullish else "down",
                "start_time": datetime.utcnow().timestamp(),
                "checked_candles": 0,
                "max_candles": 5,
                "notified": (interval == "5m" and prob is not None and prob*100 >= 60.0),  # whether initial broadcast was sent
                "model_conf": round(prob*100,2) if prob is not None else None
            }
            ACTIVE_EVAL.append(eval_entry)

        # small sleep to avoid tight loop
        await asyncio.sleep(1)

# ---------------- Profit evaluation loop ----------------
async def profit_evaluator_loop():
    # wakes up often to check current price and candles
    while True:
        if not ACTIVE_EVAL:
            await asyncio.sleep(2)
            continue

        # copy list to avoid modification during iteration
        now = datetime.utcnow()
        to_remove = []
        for ev in list(ACTIVE_EVAL):
            symbol = ev["symbol"]
            interval = ev["interval"]
            entry_price = ev["entry_price"]
            direction = ev["direction"]
            checked = ev["checked_candles"]
            max_candles = ev["max_candles"]
            # fetch latest candle for the same interval
            kl = fetch_klines(symbol, interval, limit=1)
            if not kl:
                continue
            current_price = float(kl[-1][4])
            # percent move from entry (positive if price rose)
            pct_move = (current_price - entry_price) / entry_price * 100.0
            if direction == "down":
                pct_move = -pct_move  # for short, favorable move is negative price change

            # required price percent to reach targeted account profit
            required_price_pct = PRICE_TARGET_PCT  # e.g. 0.5 for 10% at 20x
            # check if reached target (price change in percent)
            if pct_move >= required_price_pct:
                # success
                profit_pct_account = pct_move * LEVERAGE  # approx account profit %
                msg = (f"✅ TRADE RESULT — {symbol} {interval} {direction.upper()}\n"
                       f"Entry: {entry_price}\nNow: {current_price}\nPrice move: {round(pct_move,4)}% -> "
                       f"Estimated account profit: {round(profit_pct_account,3)}%\n"
                       f"Model confidence: {ev.get('model_conf')}\nResult: PROFIT (hit target)")
                # broadcast only if initial broadcast was sent (only for 5m high-confidence signals)
                if ev["notified"]:
                    broadcast(msg)
                # add training record: success True
                feats = compute_features(closes_from_klines(fetch_klines(symbol, interval, limit=200)), None)
                add_training_record(symbol, feats, True)
                # retrain model for that symbol asynchronously (kick off)
                asyncio.create_task(async_train_symbol(symbol))
                to_remove.append(ev)
                continue

            # otherwise, if not reached and we've waited one candle, increment checked_candles
            # We'll use the next fully closed candle count: check kline length with limit = checked+2
            ev["checked_candles"] += 1
            if ev["checked_candles"] >= max_candles:
                # evaluate outcome after 5 candles
                profit_pct_account = pct_move * LEVERAGE
                success = pct_move >= required_price_pct
                msg = (f"🔔 TRADE EVALUATION — {symbol} {interval} {direction.upper()}\n"
                       f"Entry: {entry_price}\nNow: {current_price}\nPrice move: {round(pct_move,4)}% -> "
                       f"Estimated account profit: {round(profit_pct_account,3)}%\nModel confidence: {ev.get('model_conf')}\n"
                       f"Result: {'PROFIT' if success else 'LOSS'} (evaluated after {max_candles} candles)")
                if ev["notified"]:
                    broadcast(msg)
                # add training record with outcome
                feats = compute_features(closes_from_klines(fetch_klines(symbol, interval, limit=200)), None)
                add_training_record(symbol, feats, success)
                asyncio.create_task(async_train_symbol(symbol))
                to_remove.append(ev)

        # remove completed evaluations
        for r in to_remove:
            try:
                ACTIVE_EVAL.remove(r)
            except Exception:
                pass

        await asyncio.sleep(5)

# ---------------- Async retrain helper ----------------
async def async_train_symbol(symbol):
    # small delay to let file write settle
    await asyncio.sleep(1)
    try:
        train_ml_model(symbol)
    except Exception as e:
        print("async_train_symbol error:", e)

# ---------------- Periodic retrain (every 4h) ----------------
async def periodic_retrain_loop():
    while True:
        print("[ML] periodic retrain started")
        for s in SYMBOLS:
            try:
                train_ml_model(s)
            except Exception as e:
                print("periodic_retrain error:", e)
        await asyncio.sleep(RETRAIN_INTERVAL_SECONDS)

# ---------------- HTTP webhook endpoint ----------------
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
        if text and text.strip().lower() == "/start":
            send_telegram(chat_id, "👋 Welcome! EMA + AI alerts are active.")
    return {"ok": True}

@app.get("/")
def root():
    return {"status": "ok", "subscribers": len(user_ids), "models": list(ML_MODELS.keys())}

# ---------------- Startup tasks ----------------
@app.on_event("startup")
async def startup_event():
    # load models (if exist)
    for sym in SYMBOLS:
        m = load_model(sym)
        if m:
            ML_MODELS[sym] = m
            print(f"[ML] loaded model for {sym}")

    # start background monitors and loops
    for tf in TIMEFRAMES:
        asyncio.create_task(monitor_interval(tf))
    asyncio.create_task(profit_evaluator_loop())
    asyncio.create_task(periodic_retrain_loop())
    print("✅ EMA + ML monitoring tasks started")

# ---------------- Run notes ----------------
# - Keep BOT_TOKEN as env var in Railway; do NOT hardcode in git.
# - Webhook: register your Railway app URL /webhook/<BOT_TOKEN> using Telegram setWebhook command.
# - The code broadcasts only 5m signals when model confidence >= 60%, logs 15m crosses.
# - After each signal the code evaluates profit target = TARGET_ACCOUNT_PROFIT_PCT at LEVERAGE (price change required = TARGET/LEVERAGE).
# - After profit or after 5 candles it logs & (if initial broadcasted) sends the evaluation message and stores a labeled training sample, then retrains the model for that symbol.
