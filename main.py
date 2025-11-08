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

# test2 working but training when in positive
# main.py
# import os
# import json
# import requests
# import asyncio
# import numpy as np
# from datetime import datetime, timedelta, timezone
# from fastapi import FastAPI, Request
# import xgboost as xgb

# # ---------------- Config ----------------
# BOT_TOKEN = os.getenv("BOT_TOKEN")
# SYMBOLS = ["OPUSDT", "RENDERUSDT", "BTCUSDT"]
# TIMEFRAMES = ["5m"]
# MODEL_DIR = "models"
# os.makedirs(MODEL_DIR, exist_ok=True)

# LEVERAGE = 20
# TARGET_ACCOUNT_PROFIT_PCT = 10.0  # 10% account profit at 20x
# MAX_CANDLES_TO_CHECK = 5

# app = FastAPI()
# user_ids = set()
# sent_hourly = set()
# active_signals = []  # store {symbol, interval, direction, entry_price, time, checked_candles, model_conf}

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
#     for i in range(30, len(closes)-1):
#         feats = compute_features(closes[:i], volumes[:i] if volumes else None)
#         if None in feats:
#             continue
#         X.append(feats)
#         y.append(1 if closes[i+1] > closes[i] else 0)
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

# # ---------------- EMA Monitor ----------------
# async def monitor_ema(symbol, interval):
#     klines = fetch_klines(symbol, interval, limit=200)
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
#         now_utc = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

#         if prev_ema9 < prev_ema26 and ema9 >= ema26:
#             if prob and prob * 100 >= 60:
#                 msg = (
#                     f"📈 {symbol} ({interval}) EMA9 crossed ABOVE EMA26 — BUY 💰\n"
#                     f"Price: {close_price}\n🤖 Uptrend Probability: {round(prob*100,2)}%\n🕒 {now_utc}"
#                 )
#                 broadcast(msg)
#                 active_signals.append({
#                     "symbol": symbol,
#                     "interval": interval,
#                     "direction": "up",
#                     "entry_price": close_price,
#                     "signal_time": now_utc,
#                     "model_conf": round(prob*100, 2),
#                     "checked_candles": 0
#                 })

#         elif prev_ema9 > prev_ema26 and ema9 <= ema26:
#             if prob and (1 - prob) * 100 >= 60:
#                 msg = (
#                     f"📉 {symbol} ({interval}) EMA9 crossed BELOW EMA26 — SELL ⚠️\n"
#                     f"Price: {close_price}\n🤖 Downtrend Probability: {round((1-prob)*100,2)}%\n🕒 {now_utc}"
#                 )
#                 broadcast(msg)
#                 active_signals.append({
#                     "symbol": symbol,
#                     "interval": interval,
#                     "direction": "down",
#                     "entry_price": close_price,
#                     "signal_time": now_utc,
#                     "model_conf": round((1-prob)*100, 2),
#                     "checked_candles": 0
#                 })

#         prev_ema9, prev_ema26 = ema9, ema26

# # ---------------- Profit / Accuracy Evaluation ----------------
# async def profit_evaluator_loop():
#     while True:
#         await asyncio.sleep(60)  # check every minute
#         to_remove = []
#         for ev in list(active_signals):
#             symbol = ev["symbol"]
#             interval = ev["interval"]
#             direction = ev["direction"]
#             entry_price = ev["entry_price"]
#             ev["checked_candles"] += 1

#             klines = fetch_klines(symbol, interval, limit=2)
#             if not klines:
#                 continue
#             current_price = float(klines[-1][4])

#             # compute price change
#             if direction == "up":
#                 decimal_move = (current_price - entry_price) / entry_price
#             else:
#                 decimal_move = (entry_price - current_price) / entry_price

#             account_profit_pct = decimal_move * LEVERAGE * 100

#             profit_hit = account_profit_pct >= TARGET_ACCOUNT_PROFIT_PCT
#             is_final = ev["checked_candles"] >= MAX_CANDLES_TO_CHECK

#             if profit_hit or is_final:
#                 status = "✅ PROFIT" if profit_hit else "❌ LOSS"
#                 msg = (
#                     f"📊 Accuracy Check: {symbol} ({interval})\n"
#                     f"{status}\n"
#                     f"Entry: {entry_price} | Now: {current_price}\n"
#                     f"Profit: {round(account_profit_pct, 2)}%\n"
#                     f"Checked after {ev['checked_candles']} candles\n"
#                     f"Signal Time: {ev['signal_time']}\n"
#                     f"🧠 Model Confidence: {ev['model_conf']}%"
#                 )
#                 broadcast(msg)

#                 # retrain with label
#                 klines_all = fetch_klines(symbol, "5m", limit=400)
#                 closes = [float(k[4]) for k in klines_all]
#                 volumes = [float(k[5]) for k in klines_all]
#                 model = train_ml_model(symbol, closes, volumes)
#                 if model:
#                     print(f"[RETRAIN] Model updated for {symbol}")
#                 to_remove.append(ev)

#         for ev in to_remove:
#             active_signals.remove(ev)

# # ---------------- Startup ----------------
# @app.on_event("startup")
# async def startup_event():
#     for symbol in SYMBOLS:
#         for tf in TIMEFRAMES:
#             asyncio.create_task(monitor_ema(symbol, tf))
#     asyncio.create_task(profit_evaluator_loop())

# # ---------------- Run ----------------
# # (if running locally: uvicorn main:app --host 0.0.0.0 --port 8080)

# train when actual loss
# main.py
import os
import json
import requests
import asyncio
import numpy as np
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Request
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# ---------------- Config ----------------
BOT_TOKEN = os.getenv("BOT_TOKEN")  # keep like this
SYMBOLS = ["OPUSDT", "RENDERUSDT", "BTCUSDT"]
TIMEFRAMES = ["5m"]                # only 5m signaling (15m can be enabled if needed)
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# training / runtime params
HISTORICAL_TRAIN_LIMIT = 1000      # use up to 1000 candles when training
TRAIN_WINDOW = 200                 # window length for features
FUTURE_LOOKAHEAD = 5               # label looks this many candles ahead
MIN_TRAIN_SAMPLES = 60
ML_REFRESH_HOURS = 4
LEVERAGE = 20
TARGET_ACCOUNT_PROFIT_PCT = 10.0   # percent on account (10% = ~0.5% price move @ 20x)
CONF_THRESHOLD_PCT = 60.0          # only send signals when model confidence >= this

MODEL_META_FILE = "model_meta.json"
DATA_FILE = "training_data.json"   # optional persisted labeled records (augment retrain)

app = FastAPI()
user_ids = set()
active_signals = []  # pending signals awaiting accuracy checks

# ---------------- Helpers ----------------
def fetch_klines(symbol: str, interval: str, limit: int = 200):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[REST] Error fetching klines {symbol} {interval}: {e}")
        return []

def closes_from_klines(klines):
    return [float(k[4]) for k in klines]

def highs_from_klines(klines):
    return [float(k[2]) for k in klines]

def lows_from_klines(klines):
    return [float(k[3]) for k in klines]

def volumes_from_klines(klines):
    return [float(k[5]) for k in klines]

def now_utc_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

# ---------------- Indicators & Features ----------------
def sma(arr, period):
    arr = np.array(arr)
    if len(arr) < period: return None
    return float(np.mean(arr[-period:]))

def std(arr, period):
    arr = np.array(arr)
    if len(arr) < period: return None
    return float(np.std(arr[-period:], ddof=0))

def get_ema(values, period):
    values = np.array(values, dtype=float)
    if len(values) < period or period < 1:
        return None
    weights = np.exp(np.linspace(-1., 0., period))
    weights /= weights.sum()
    ema_full = np.convolve(values, weights, mode="valid")
    return float(np.round(ema_full[-1], 8))

def bollinger_bands(closes, period=20, mult=2):
    s = sma(closes, period)
    sd = std(closes, period)
    if s is None or sd is None:
        return None, None, None
    upper = s + mult * sd
    lower = s - mult * sd
    return float(upper), float(s), float(lower)

def vwma(prices, volumes, period=20):
    if len(prices) < period or len(volumes) < period:
        return None
    p = np.array(prices[-period:], dtype=float)
    v = np.array(volumes[-period:], dtype=float)
    if v.sum() == 0:
        return float(np.mean(p))
    return float((p * v).sum() / v.sum())

def prev_pct_change(closes):
    if len(closes) < 2:
        return None
    return float((closes[-1] - closes[-2]) / closes[-2])

def atr(highs, lows, closes, period=14):
    if len(highs) < period + 1:
        return None
    highs = np.array(highs)
    lows = np.array(lows)
    closes = np.array(closes)
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        trs.append(tr)
    if len(trs) < period:
        return None
    return float(np.mean(trs[-period:]))

def compute_features_for_window(klines_window):
    """
    klines_window: list of kline rows (open,time,high,low,close,volume,...)
    returns feature vector or None if insufficient data
    """
    closes = closes_from_klines(klines_window)
    highs = highs_from_klines(klines_window)
    lows = lows_from_klines(klines_window)
    volumes = volumes_from_klines(klines_window)

    # core indicators
    ema9 = get_ema(closes, 9)
    ema26 = get_ema(closes, 26)
    rsi14 = None
    if len(closes) >= 15:
        deltas = np.diff(closes)
        ups = deltas.clip(min=0)
        downs = -deltas.clip(max=0)
        roll_up = np.mean(ups[-14:])
        roll_down = np.mean(downs[-14:])
        rsi14 = 100 if roll_down == 0 else float(round(100 - (100 / (1 + (roll_up / roll_down))), 2))

    macd_line, macd_signal, macd_hist = (None, None, None)
    if len(closes) >= 26 + 9:
        macd_line = get_ema(closes, 12) - get_ema(closes, 26)
        # for signal line compute EMA of macd series:
        macd_series = []
        for i in range(26, len(closes)):
            f = get_ema(closes[:i+1], 12)
            s = get_ema(closes[:i+1], 26)
            if f is not None and s is not None:
                macd_series.append(f - s)
        if len(macd_series) >= 9:
            macd_signal = get_ema(macd_series, 9)
            macd_hist = macd_line - macd_signal

    # bollinger
    bb_upper, bb_mid, bb_lower = (None, None, None)
    if len(closes) >= 20:
        bb_upper, bb_mid, bb_lower = bollinger_bands(closes, 20, 2)

    # VWMA
    vwma20 = vwma(closes, volumes, 20) if len(closes) >= 20 else None

    # previous candle percent change and ATR
    prev_change = prev_pct_change(closes)
    atr14 = atr(highs, lows, closes, 14)

    last_vol = volumes[-1] if volumes else 0.0
    last_close = closes[-1] if closes else None

    feats = [
        ema9, ema26,
        bb_upper, bb_mid, bb_lower,
        vwma20,
        rsi14,
        macd_line, macd_signal, macd_hist,
        prev_change,
        atr14,
        last_vol,
        last_close
    ]
    # return None if any required core features missing (ema9/ema26/last_close)
    if ema9 is None or ema26 is None or last_close is None:
        return None
    return feats

# ---------------- Persisted training data helpers ----------------
def load_training_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_training_record(record):
    data = load_training_data()
    data.append(record)
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print("Error saving training record:", e)

# ---------------- Model plumbing ----------------
def model_path(symbol):
    return os.path.join(MODEL_DIR, f"{symbol}_xgb.json")

def load_model(symbol):
    p = model_path(symbol)
    if os.path.exists(p):
        try:
            model = xgb.XGBClassifier()
            model.load_model(p)
            return model
        except Exception as e:
            print("Failed to load model for", symbol, e)
    return None

def save_model(model, symbol):
    try:
        model.save_model(model_path(symbol))
    except Exception as e:
        print("Failed to save model for", symbol, e)

def build_training_set_from_klines(klines, lookahead=FUTURE_LOOKAHEAD):
    """
    Build X,y using sliding windows from klines list.
    klines assumed oldest->newest
    """
    X = []
    y = []
    n = len(klines)
    for i in range(30, n - lookahead):
        window_start = max(0, i - TRAIN_WINDOW + 1)
        window = klines[window_start: i + 1]  # includes current close at position -1
        feats = compute_features_for_window(window)
        if feats is None:
            continue
        close_now = float(window[-1][4])
        future_close = float(klines[i + lookahead][4])
        label = 1 if future_close > close_now else 0
        X.append(feats)
        y.append(label)
    if len(X) == 0:
        return None, None
    return np.array(X), np.array(y)

def balance_samples(X, y):
    """
    Simple undersample/upsample to balance classes.
    """
    try:
        ones = X[y == 1]
        zeros = X[y == 0]
        n1 = len(ones)
        n0 = len(zeros)
        if n1 == 0 or n0 == 0:
            return X, y
        if n1 > n0:
            ones_down = resample(ones, replace=False, n_samples=n0, random_state=42)
            X_bal = np.vstack([ones_down, zeros])
            y_bal = np.array([1]*len(ones_down) + [0]*len(zeros))
        elif n0 > n1:
            zeros_down = resample(zeros, replace=False, n_samples=n1, random_state=42)
            X_bal = np.vstack([ones, zeros_down])
            y_bal = np.array([1]*len(ones) + [0]*len(zeros_down))
        else:
            X_bal, y_bal = X, y
        # shuffle
        idx = np.random.permutation(len(y_bal))
        return X_bal[idx], y_bal[idx]
    except Exception as e:
        print("balance_samples error:", e)
        return X, y

def train_model_for_symbol(symbol):
    # fetch long historical data
    klines = fetch_klines(symbol, "5m", limit=HISTORICAL_TRAIN_LIMIT)
    if not klines or len(klines) < 100:
        print(f"[ML] insufficient klines for {symbol}: {len(klines)}")
        return None
    X, y = build_training_set_from_klines(klines, lookahead=FUTURE_LOOKAHEAD)
    if X is None or len(y) < MIN_TRAIN_SAMPLES:
        print(f"[ML] not enough training samples for {symbol}: {0 if y is None else len(y)}")
        return None

    X_bal, y_bal = balance_samples(X, y)
    # scale features
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_bal)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100)
    model.fit(Xs, y_bal)
    # save model and scaler together (we'll save scaler as npy)
    save_model(model, symbol)
    # also save scaler params
    meta = {"scaler_mean": scaler.mean_.tolist(), "scaler_var": scaler.var_.tolist(), "trained_at": now_utc_str()}
    try:
        with open(os.path.join(MODEL_DIR, f"{symbol}_meta.json"), "w") as f:
            json.dump(meta, f)
    except Exception as e:
        print("Failed to save meta:", e)
    print(f"[ML] trained & saved model for {symbol} (samples={len(y_bal)})")
    return model

def load_scaler_for_symbol(symbol):
    p = os.path.join(MODEL_DIR, f"{symbol}_meta.json")
    if not os.path.exists(p):
        return None
    try:
        m = json.load(open(p, "r"))
        mean = np.array(m["scaler_mean"])
        var = np.array(m["scaler_var"])
        scaler = StandardScaler()
        scaler.mean_ = mean
        scaler.var_ = var
        scaler.scale_ = np.sqrt(var)
        return scaler
    except Exception as e:
        print("load_scaler error:", e)
        return None

def predict_probability(model, scaler, klines_window):
    feats = compute_features_for_window(klines_window)
    if feats is None:
        return None
    X = np.array(feats).reshape(1, -1)
    if scaler is not None:
        try:
            Xs = (X - scaler.mean_) / scaler.scale_
        except Exception:
            Xs = scaler.transform(X)
    else:
        Xs = X
    try:
        prob = float(model.predict_proba(Xs)[0][1])
        return prob
    except Exception as e:
        print("predict_probability error:", e)
        return None

# ---------------- Telegram helpers ----------------
def send_telegram(chat_id, msg):
    if not BOT_TOKEN:
        print("[TEST-TG]", msg)
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": msg}, timeout=10)
    except Exception as e:
        print("Telegram send error:", e)

def broadcast(msg):
    for uid in list(user_ids):
        send_telegram(uid, msg)

# ---------------- EMA monitor & signaling ----------------
async def monitor_symbol_interval(symbol, interval):
    print(f"[MON] starting {symbol} {interval}")
    # load model+scaler if available (trained at startup)
    model = load_model(symbol)
    scaler = load_scaler_for_symbol(symbol)

    # initial klines for state
    klines = fetch_klines(symbol, interval, limit=TRAIN_WINDOW + 50)
    if not klines:
        klines = []
    prev_ema9 = None
    prev_ema26 = None
    if klines:
        prev_ema9 = get_ema(closes_from_klines(klines), 9)
        prev_ema26 = get_ema(closes_from_klines(klines), 26)

    # SYNCHRONIZE to real close time:
    def interval_minutes(itv):
        if itv.endswith("m"): return int(itv[:-1])
        if itv.endswith("h"): return int(itv[:-1]) * 60
        return 1
    minutes = interval_minutes(interval)

    while True:
        # wait until next Binance candle close (small offset for safety)
        now = datetime.now(timezone.utc)
        minute = now.minute
        next_multiple = ((minute // minutes) + 1) * minutes
        next_hour = now.replace(minute=0, second=0, microsecond=0)
        if next_multiple >= 60:
            next_multiple -= 60
            next_hour = next_hour + timedelta(hours=1)
        next_close = next_hour.replace(minute=next_multiple, second=5, microsecond=0)
        wait = (next_close - now).total_seconds()
        if wait < 0:
            wait = minutes * 60 - (now.minute % minutes) * 60
        await asyncio.sleep(max(wait, 1))

        kl = fetch_klines(symbol, interval, limit=TRAIN_WINDOW + 5)
        if not kl or len(kl) < 26:
            continue
        # update model periodically if missing
        if model is None:
            model = train_model_for_symbol(symbol)
            scaler = load_scaler_for_symbol(symbol)

        # detect cross using prev and current EMA built from closes
        closes = closes_from_klines(kl)
        ema9 = get_ema(closes, 9)
        ema26 = get_ema(closes, 26)
        if prev_ema9 is None or prev_ema26 is None:
            prev_ema9, prev_ema26 = ema9, ema26
            continue

        now_ts = now_utc_str()
        # bullish cross
        if prev_ema9 < prev_ema26 and ema9 >= ema26:
            # compute prob using model on last TRAIN_WINDOW candles
            window = kl[-TRAIN_WINDOW:] if len(kl) >= TRAIN_WINDOW else kl
            prob = None
            if model is not None:
                prob = predict_probability(model, scaler, window)
            conf_pct = None if prob is None else round(prob * 100, 2)
            if conf_pct is not None and conf_pct >= CONF_THRESHOLD_PCT:
                entry_price = float(window[-1][4])
                msg = (
                    f"📈 {symbol} {interval} — EMA9 crossed ABOVE EMA26 — BUY\n"
                    f"Price: {entry_price}\n🤖 Model Up Probability: {conf_pct}%\n🕒 {now_ts}"
                )
                broadcast(msg)
                active_signals.append({
                    "symbol": symbol,
                    "interval": interval,
                    "direction": "up",
                    "entry_price": entry_price,
                    "signal_time": now_ts,
                    "model_conf": conf_pct,
                    "checked": 0
                })

        # bearish cross
        if prev_ema9 > prev_ema26 and ema9 <= ema26:
            window = kl[-TRAIN_WINDOW:] if len(kl) >= TRAIN_WINDOW else kl
            prob = None
            if model is not None:
                prob = predict_probability(model, scaler, window)
            conf_pct = None if prob is None else round((1 - prob) * 100, 2)
            if conf_pct is not None and conf_pct >= CONF_THRESHOLD_PCT:
                entry_price = float(window[-1][4])
                msg = (
                    f"📉 {symbol} {interval} — EMA9 crossed BELOW EMA26 — SELL\n"
                    f"Price: {entry_price}\n🤖 Model Down Probability: {conf_pct}%\n🕒 {now_ts}"
                )
                broadcast(msg)
                active_signals.append({
                    "symbol": symbol,
                    "interval": interval,
                    "direction": "down",
                    "entry_price": entry_price,
                    "signal_time": now_ts,
                    "model_conf": conf_pct,
                    "checked": 0
                })

        prev_ema9, prev_ema26 = ema9, ema26

# ---------------- Profit/accuracy evaluator ----------------
async def evaluator_loop():
    """
    Checks active_signals frequently, computes profit (account % using LEVERAGE),
    and finalizes after either profit hit or max lookahead candles.
    """
    MAX_CHECK_CANDLES = FUTURE_LOOKAHEAD
    CHECK_INTERVAL_SECONDS = 30  # how often to check between closes (we also evaluate on closed candles)
    while True:
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)
        to_remove = []
        for s in list(active_signals):
            symbol = s["symbol"]
            interval = s["interval"]
            direction = s["direction"]
            entry = s["entry_price"]
            s["checked"] += 1

            # fetch latest candle
            kl = fetch_klines(symbol, interval, limit=2)
            if not kl:
                continue
            now_price = float(kl[-1][4])

            # price move favorable for up vs down
            if direction == "up":
                decimal_move = (now_price - entry) / entry
            else:
                decimal_move = (entry - now_price) / entry
            account_profit_pct = decimal_move * LEVERAGE * 100.0

            profit_hit = account_profit_pct >= TARGET_ACCOUNT_PROFIT_PCT
            final_check = s["checked"] >= MAX_CHECK_CANDLES

            if profit_hit or final_check:
                status = "✅ PROFIT" if profit_hit else "❌ LOSS"
                profit_str = f"{round(account_profit_pct, 2)}%"
                msg = (
                    f"📊 Accuracy Check — {symbol} {interval}\n"
                    f"{status}\n"
                    f"Signal Time: {s['signal_time']}\n"
                    f"Direction: {direction}\n"
                    f"Entry: {entry} | Now: {now_price}\n"
                    f"Profit (account % with {LEVERAGE}x): {profit_str}\n"
                    f"Checked Candles: {s['checked']}\n"
                    f"Model Conf: {s.get('model_conf')}%\n"
                    f"Checked At: {now_utc_str()}"
                )
                broadcast(msg)

                # create training label (outcome = profit_hit)
                # For training, we can store the feature snapshot at entry (use 5m klines)
                kl_all = fetch_klines(symbol, "5m", limit=TRAIN_WINDOW + 20)
                if kl_all:
                    window = kl_all[-TRAIN_WINDOW:] if len(kl_all) >= TRAIN_WINDOW else kl_all
                    feats = compute_features_for_window(window)
                    if feats is not None:
                        rec = {"symbol": symbol, "features": feats, "outcome": 1 if profit_hit else 0}
                        save_training_record(rec)
                # trigger immediate retrain for that symbol (asynchronously)
                asyncio.create_task(async_retrain_symbol(symbol))
                to_remove.append(s)
        for r in to_remove:
            if r in active_signals:
                active_signals.remove(r)

async def async_retrain_symbol(symbol):
    # run training in background
    try:
        model = train_model_for_symbol(symbol)
        if model:
            print(f"[ML] retrained model for {symbol} at {now_utc_str()}")
    except Exception as e:
        print("async_retrain_symbol error:", e)

# ---------------- ML refresher (periodic full retrain) ----------------
async def ml_refresher_loop():
    # initial training on startup
    for sym in SYMBOLS:
        asyncio.create_task(async_retrain_symbol(sym))
    # then periodically every ML_REFRESH_HOURS
    while True:
        await asyncio.sleep(ML_REFRESH_HOURS * 3600)
        for sym in SYMBOLS:
            asyncio.create_task(async_retrain_symbol(sym))

# ---------------- Webhook endpoint ----------------
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
            send_telegram(chat_id, f"✅ Subscribed to EMA+AI alerts!\nTracking: {', '.join(SYMBOLS)}")
        if text.strip().lower() == "/start":
            send_telegram(chat_id, "👋 Welcome! EMA + AI alerts are active.")
    return {"ok": True}

# ---------------- Root health ----------------
@app.get("/")
def root():
    models = {}
    for s in SYMBOLS:
        meta_path = os.path.join(MODEL_DIR, f"{s}_meta.json")
        models[s] = None
        if os.path.exists(meta_path):
            try:
                mj = json.load(open(meta_path))
                models[s] = mj.get("trained_at")
            except Exception:
                models[s] = "unknown"
    return {"status": "ok", "subscribers": len(user_ids), "models": models, "active_signals": len(active_signals)}

# ---------------- Startup tasks ----------------
def start_background_tasks():
    loop = asyncio.get_event_loop()
    # monitors
    for s in SYMBOLS:
        for tf in TIMEFRAMES:
            loop.create_task(monitor_symbol_interval(s, tf))
    # evaluator
    loop.create_task(evaluator_loop())
    # periodic ML refresher
    loop.create_task(ml_refresher_loop())

@app.on_event("startup")
async def startup_event():
    # start background tasks
    start_background_tasks()
    print("✅ EMA+AI service started at", now_utc_str())

# ---------------- If running directly for local debug ---------------
# uvicorn main:app --host 0.0.0.0 --port 8080

