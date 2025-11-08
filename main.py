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
from datetime import datetime, timezone
from fastapi import FastAPI, Request
import xgboost as xgb
from collections import deque

# ---------------- CONFIG ----------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
SYMBOLS = ["OPUSDT", "RENDERUSDT", "ONDOUSDT"]
TIMEFRAMES = ["5m", "15m"]  # frequent signals
MODEL_DIR = "models"
HISTORY_FILE = os.path.join(MODEL_DIR, "signal_history.json")
os.makedirs(MODEL_DIR, exist_ok=True)

LEVERAGE = 20
TARGET_ACCOUNT_PROFIT_PCT = 10.0  # target profit at 20x
MAX_CANDLES_TO_CHECK = 5
MAX_HISTORY = 10  # last 10 results per symbol

app = FastAPI()
user_ids = set()
active_signals = []  # active signals with tracking
signal_history = {sym: deque(maxlen=MAX_HISTORY) for sym in SYMBOLS}


# ---------------- UTILITIES ----------------
def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def save_signal_history():
    data = {s: list(dq) for s, dq in signal_history.items()}
    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f)


def load_signal_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                data = json.load(f)
            for s, vals in data.items():
                if s in signal_history:
                    signal_history[s].extend(vals)
            print(f"[LOAD] Signal history loaded: {data}")
        except Exception as e:
            print("[WARN] Failed to load signal history:", e)


# ---------------- TELEGRAM ----------------
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


# ---------------- INDICATORS ----------------
def get_ema(values, period):
    if len(values) < period or period < 1:
        return None
    weights = np.exp(np.linspace(-1., 0., period))
    weights /= weights.sum()
    ema = np.convolve(values, weights, mode="valid")
    return float(np.round(ema[-1], 8))


def get_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes)
    ups = deltas.clip(min=0)
    downs = -deltas.clip(max=0)
    roll_up = np.mean(ups[-period:])
    roll_down = np.mean(downs[-period:])
    if roll_down == 0:
        return 100
    rs = roll_up / roll_down
    return round(100 - (100 / (1 + rs)), 2)


def get_macd(closes, fast=12, slow=26, signal=9):
    if len(closes) < slow + signal:
        return None, None, None
    ema_fast = get_ema(closes, fast)
    ema_slow = get_ema(closes, slow)
    if ema_fast is None or ema_slow is None:
        return None, None, None
    macd_line = ema_fast - ema_slow
    macd_series = []
    for i in range(slow, len(closes)):
        fast_i = get_ema(closes[:i + 1], fast)
        slow_i = get_ema(closes[:i + 1], slow)
        if fast_i is None or slow_i is None:
            continue
        macd_series.append(fast_i - slow_i)
    if len(macd_series) < signal:
        signal_line = None
    else:
        signal_line = get_ema(macd_series, signal)
    macd_hist = macd_line - signal_line if signal_line is not None else 0
    return round(macd_line, 5), round(signal_line, 5) if signal_line else None, round(macd_hist, 5)


def compute_features(closes, volumes=None):
    ema9 = get_ema(closes, 9)
    ema26 = get_ema(closes, 26)
    rsi14 = get_rsi(closes, 14)
    macd_line, macd_signal, macd_hist = get_macd(closes)
    last_volume = volumes[-1] if volumes else 0
    return [ema9, ema26, rsi14, macd_line, macd_signal, macd_hist, last_volume]


# ---------------- BINANCE ----------------
def fetch_klines(symbol, interval, limit=200):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[REST] Error fetching klines {symbol} {interval}: {e}")
        return []


# ---------------- TELEGRAM WEBHOOK ----------------
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


# ---------------- MACHINE LEARNING ----------------
def model_path(symbol):
    return os.path.join(MODEL_DIR, f"{symbol}_xgb.json")


def load_model(symbol):
    path = model_path(symbol)
    if os.path.exists(path):
        model = xgb.XGBClassifier()
        model.load_model(path)
        return model
    return None


def save_model(model, symbol):
    model.save_model(model_path(symbol))


def train_ml_model(symbol, closes, volumes=None):
    X, y = [], []
    for i in range(30, len(closes) - 1):
        feats = compute_features(closes[:i], volumes[:i] if volumes else None)
        if None in feats:
            continue
        X.append(feats)
        y.append(1 if closes[i + 1] > closes[i] else 0)
    if len(X) < 20:
        return None
    model = xgb.XGBClassifier(eval_metric='logloss')
    model.fit(np.array(X), np.array(y))
    save_model(model, symbol)
    print(f"[ML] trained & saved model for {symbol} (samples={len(X)})")
    return model


def predict_trend(model, closes, volumes=None):
    feats = compute_features(closes, volumes)
    if None in feats:
        return None
    return model.predict_proba([feats])[0][1] if model else None


# ---------------- EMA MONITOR ----------------
async def monitor_ema(symbol, interval):
    print(f"[MON] starting {symbol} {interval}")
    klines = fetch_klines(symbol, interval, limit=200)
    closes = [float(k[4]) for k in klines]
    volumes = [float(k[5]) for k in klines]

    model = load_model(symbol)
    if not model:
        model = train_ml_model(symbol, closes, volumes)

    prev_ema9 = get_ema(closes, 9)
    prev_ema26 = get_ema(closes, 26)

    while True:
        await asyncio.sleep(5)
        klines_new = fetch_klines(symbol, interval, limit=2)
        if not klines_new:
            continue
        close_price = float(klines_new[-1][4])
        closes.append(close_price)
        volumes.append(float(klines_new[-1][5]))
        ema9 = get_ema(closes, 9)
        ema26 = get_ema(closes, 26)
        if None in [prev_ema9, prev_ema26, ema9, ema26]:
            prev_ema9, prev_ema26 = ema9, ema26
            continue

        prob = predict_trend(model, closes, volumes)
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        print(f"[{symbol} {interval}] EMA9={ema9:.4f} EMA26={ema26:.4f} prob={prob}")

        avg_profit = np.mean(signal_history[symbol]) if signal_history[symbol] else 0
        can_send = (not signal_history[symbol]) or (avg_profit >= 0)

        # CROSS UP
        if prev_ema9 < prev_ema26 and ema9 >= ema26 and can_send:
            if prob and prob * 100 >= 55:
                msg = (
                    f"📈 {symbol} ({interval}) EMA9 crossed ABOVE EMA26 — BUY 💰\n"
                    f"Price: {close_price}\n🤖 Uptrend Probability: {round(prob * 100, 2)}%\n🕒 {now_str}"
                )
                broadcast(msg)
                active_signals.append({
                    "symbol": symbol,
                    "interval": interval,
                    "direction": "up",
                    "entry_price": close_price,
                    "signal_time": now_str,
                    "model_conf": round(prob * 100, 2),
                    "checked_candles": 0,
                    "max_profit": 0.0,
                    "profit": 0.0
                })

        # CROSS DOWN
        elif prev_ema9 > prev_ema26 and ema9 <= ema26 and can_send:
            if prob and (1 - prob) * 100 >= 55:
                msg = (
                    f"📉 {symbol} ({interval}) EMA9 crossed BELOW EMA26 — SELL ⚠️\n"
                    f"Price: {close_price}\n🤖 Downtrend Probability: {round((1 - prob) * 100, 2)}%\n🕒 {now_str}"
                )
                broadcast(msg)
                active_signals.append({
                    "symbol": symbol,
                    "interval": interval,
                    "direction": "down",
                    "entry_price": close_price,
                    "signal_time": now_str,
                    "model_conf": round((1 - prob) * 100, 2),
                    "checked_candles": 0,
                    "max_profit": 0.0,
                    "profit": 0.0
                })

        prev_ema9, prev_ema26 = ema9, ema26


# ---------------- PROFIT EVALUATOR ----------------
async def profit_evaluator_loop():
    while True:
        await asyncio.sleep(60)
        to_remove = []
        for ev in list(active_signals):
            symbol = ev["symbol"]
            interval = ev["interval"]
            direction = ev["direction"]
            entry_price = ev["entry_price"]
            ev["checked_candles"] += 1

            klines = fetch_klines(symbol, interval, limit=2)
            if not klines:
                continue
            current_price = float(klines[-1][4])

            if direction == "up":
                decimal_move = (current_price - entry_price) / entry_price
            else:
                decimal_move = (entry_price - current_price) / entry_price

            account_profit_pct = decimal_move * LEVERAGE * 100
            ev["profit"] = account_profit_pct
            ev["max_profit"] = max(ev["max_profit"], account_profit_pct)

            profit_hit = ev["max_profit"] >= TARGET_ACCOUNT_PROFIT_PCT
            is_final = ev["checked_candles"] >= MAX_CANDLES_TO_CHECK

            if profit_hit or is_final:
                status = "✅ PROFIT" if ev["max_profit"] >= 0 else "❌ LOSS"
                msg = (
                    f"📊 Accuracy Check: {symbol} ({interval})\n"
                    f"{status}\n"
                    f"Entry: {entry_price} | Now: {current_price}\n"
                    f"Max Profit: {round(ev['max_profit'], 2)}%\n"
                    f"Checked after {ev['checked_candles']} candles\n"
                    f"Signal Time: {ev['signal_time']}\n"
                    f"🧠 Model Confidence: {ev['model_conf']}%"
                )
                broadcast(msg)
                signal_history[symbol].append(ev["max_profit"])
                save_signal_history()

                if ev["max_profit"] < 0:
                    klines_all = fetch_klines(symbol, "5m", limit=400)
                    closes = [float(k[4]) for k in klines_all]
                    volumes = [float(k[5]) for k in klines_all]
                    model = train_ml_model(symbol, closes, volumes)
                    if model:
                        print(f"[RETRAIN] Model updated for {symbol} (loss correction)")
                to_remove.append(ev)

        for ev in to_remove:
            active_signals.remove(ev)


# ---------------- STARTUP ----------------
@app.on_event("startup")
async def startup_event():
    load_signal_history()
    print(f"✅ EMA+AI service started at {now_utc()}")
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            asyncio.create_task(monitor_ema(symbol, tf))
    asyncio.create_task(profit_evaluator_loop())


# ---------------- RUN ----------------
# Run with: uvicorn main:app --host 0.0.0.0 --port 8080
