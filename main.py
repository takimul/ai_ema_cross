# import os
# import json
# import requests
# import asyncio
# import numpy as np
# from datetime import datetime, timedelta, timezone
# from fastapi import FastAPI, Request
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler

# BOT_TOKEN = os.getenv("BOT_TOKEN")
# SYMBOLS = ["OPUSDT", "RENDERUSDT", "BTCUSDT"]
# TIMEFRAMES = ["5m", "15m"]
# HISTORICAL_CANDLES = 300
# PRED_LOOKAHEAD = 5

# app = FastAPI()
# user_ids = set()

# # ---------------- Telegram Helpers ----------------
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

# # ---------------- EMA / AI ----------------
# def get_ema(values, period):
#     if len(values) < period:
#         return None
#     weights = np.exp(np.linspace(-1., 0., period))
#     weights /= weights.sum()
#     ema = np.convolve(values, weights, mode="full")[:len(values)]
#     ema[:period] = ema[period]
#     return float(np.round(ema[-1], 8))

# def predict_trend_probability(closes):
#     try:
#         if len(closes) < 50:
#             return None
#         X, y = [], []
#         for i in range(30, len(closes) - PRED_LOOKAHEAD):
#             ema9 = get_ema(closes[:i], 9)
#             ema26 = get_ema(closes[:i], 26)
#             if ema9 is None or ema26 is None:
#                 continue
#             X.append([ema9, ema26, closes[i]])
#             future = closes[i + PRED_LOOKAHEAD]
#             y.append(1 if future > closes[i] else 0)
#         if len(X) < 20:
#             return None
#         scaler = StandardScaler()
#         Xs = scaler.fit_transform(X)
#         model = LogisticRegression(max_iter=500)
#         model.fit(Xs, y)
#         last_feat = scaler.transform([[get_ema(closes, 9), get_ema(closes, 26), closes[-1]]])
#         prob = model.predict_proba(last_feat)[0][1]
#         return round(prob * 100, 2)
#     except Exception as e:
#         print("AI prediction error:", e)
#         return None

# # ---------------- Binance Helper ----------------
# def fetch_klines(symbol: str, interval: str, limit: int = 100):
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

# # ---------------- EMA Monitor Task ----------------
# async def monitor_ema(symbol, interval):
#     closes = [float(k[4]) for k in fetch_klines(symbol, interval, limit=120)]
#     prev_ema9 = get_ema(closes, 9)
#     prev_ema26 = get_ema(closes, 26)

#     while True:
#         await asyncio.sleep(5)  # poll every 5 sec (optional)
#         klines = fetch_klines(symbol, interval, limit=2)
#         if not klines:
#             continue
#         close_price = float(klines[-1][4])
#         ema9 = get_ema(closes + [close_price], 9)
#         ema26 = get_ema(closes + [close_price], 26)
#         if prev_ema9 and prev_ema26 and ema9 and ema26:
#             if prev_ema9 < prev_ema26 and ema9 >= ema26:
#                 prob = predict_trend_probability(closes + [close_price])
#                 msg = f"📈 {symbol} ({interval}) EMA9 CROSSED ABOVE EMA26 — BUY 💰\nPrice: {close_price}"
#                 if prob: msg += f"\n🤖 Uptrend Probability: {prob}%"
#                 broadcast(msg)
#             elif prev_ema9 > prev_ema26 and ema9 <= ema26:
#                 prob = predict_trend_probability(closes + [close_price])
#                 msg = f"📉 {symbol} ({interval}) EMA9 CROSSED BELOW EMA26 — SELL ⚠️\nPrice: {close_price}"
#                 if prob: msg += f"\n🤖 Downtrend Probability: {100-prob}%"
#                 broadcast(msg)
#         prev_ema9, prev_ema26 = ema9, ema26

# # ---------------- Startup ----------------
# @app.on_event("startup")
# async def startup_event():
#     for symbol in SYMBOLS:
#         for tf in TIMEFRAMES:
#             asyncio.create_task(monitor_ema(symbol, tf))

# working but too many signal
# main.py
# import os
# import time
# import math
# import json
# import requests
# import asyncio
# import threading
# import numpy as np
# from datetime import datetime, timedelta, timezone
# from fastapi import FastAPI, Request
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import cross_val_score
# from typing import List, Tuple, Optional

# # -------- CONFIG ----------
# BOT_TOKEN = os.getenv("BOT_TOKEN")  # set in Railway env
# SYMBOLS = ["OPUSDT", "RENDERUSDT", "BTCUSDT"]
# TIMEFRAMES = ["5m", "15m"]   # EMA cross detection intervals
# HISTORICAL_CANDLES = 500     # fetch for training / safety
# TRAIN_WINDOW = 200           # use last 200 candles for model training
# FUTURE_LOOKAHEAD = 5         # label uses price FUTURE_LOOKAHEAD candles ahead
# MIN_TRAIN_SAMPLES = 40
# ML_REFRESH_HOURS = 4         # retrain per-symbol every 4 hours

# USER_AGENT = "ai-emacross-bot/1.0"

# # runtime storage
# user_ids = set()
# sent_signals = set()   # (symbol, interval, close_time_ms) to avoid dupes
# ML_MODELS = {}         # symbol -> (model, scaler, score, trained_at_timestamp)
# FIRST_RUN = {}         # (symbol, interval) -> True/False

# app = FastAPI()


# # ---------------- Utilities ----------------
# def send_telegram(chat_id: int, text: str):
#     if not BOT_TOKEN:
#         print("[WARN] BOT_TOKEN not set; telegram message not sent:", text)
#         return
#     try:
#         url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
#         requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=10)
#     except Exception as e:
#         print("Telegram send error:", e)


# def broadcast(text: str):
#     for uid in list(user_ids):
#         send_telegram(uid, text)


# def fetch_klines(symbol: str, interval: str, limit: int = 200) -> List:
#     """Fetch klines from Binance REST API (returns raw list)."""
#     url = f"https://api.binance.com/api/v3/klines"
#     params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
#     headers = {"User-Agent": USER_AGENT}
#     try:
#         r = requests.get(url, params=params, headers=headers, timeout=10)
#         r.raise_for_status()
#         return r.json()
#     except Exception as e:
#         print(f"[ERR] fetch_klines {symbol} {interval}: {e}")
#         return []


# def closes_from_klines(klines):
#     return [float(k[4]) for k in klines]


# def get_ema(values: List[float], period: int):
#     if len(values) < period:
#         return None
#     weights = np.exp(np.linspace(-1., 0., period))
#     weights /= weights.sum()
#     ema = np.convolve(values, weights, mode="full")[:len(values)]
#     ema[:period] = ema[period]
#     return float(np.round(ema[-1], 8))


# def interval_to_minutes(interval: str) -> int:
#     if interval.endswith("m"):
#         return int(interval[:-1])
#     if interval.endswith("h"):
#         return int(interval[:-1]) * 60
#     raise ValueError("unsupported interval")


# def seconds_until_next_close(interval_minutes: int) -> float:
#     now = datetime.now(timezone.utc)
#     minute = now.minute
#     next_multiple = ((minute // interval_minutes) + 1) * interval_minutes
#     next_hour = now.replace(minute=0, second=0, microsecond=0)
#     if next_multiple >= 60:
#         next_multiple -= 60
#         next_hour = next_hour + timedelta(hours=1)
#     next_close = next_hour.replace(minute=next_multiple, second=5, microsecond=0)
#     wait = (next_close - now).total_seconds()
#     return max(wait, 1.0)


# # ---------------- ML feature building & training ----------------
# def build_features_and_labels(closes: List[float],
#                               lookback: int = TRAIN_WINDOW,
#                               lookahead: int = FUTURE_LOOKAHEAD) -> Tuple[np.ndarray, np.ndarray]:
#     n = len(closes)
#     X, y = [], []
#     for i in range(30, n - lookahead):
#         start = max(0, i - lookback + 1)
#         window = closes[start:i + 1]
#         if len(window) < 30:
#             continue
#         ema9 = get_ema(window, 9)
#         ema26 = get_ema(window, 26)
#         if ema9 is None or ema26 is None:
#             continue
#         close_now = window[-1]
#         momentum = close_now - ema9
#         ret_1 = (window[-1] - window[-2]) / window[-2] if len(window) >= 2 and window[-2] != 0 else 0.0
#         ratio = ema9 / ema26 if ema26 != 0 else 0.0
#         rets = []
#         for a, b in zip(window[1:], window[:-1]):
#             rets.append((a - b) / b if b != 0 else 0.0)
#         vol = np.std(rets) if rets else 0.0
#         X.append([ema9, ema26, momentum, ret_1, ratio, vol, close_now])
#         future_price = closes[i + lookahead]
#         y.append(1 if future_price > close_now else 0)
#     return np.array(X), np.array(y)


# def train_and_select_best(X: np.ndarray, y: np.ndarray):
#     if len(X) < MIN_TRAIN_SAMPLES:
#         return None, None, None
#     scaler = StandardScaler()
#     Xs = scaler.fit_transform(X)

#     lr = LogisticRegression(max_iter=500)
#     try:
#         lr_score = cross_val_score(lr, Xs, y, cv=3, scoring="accuracy").mean()
#     except Exception:
#         lr_score = 0.0

#     rf = RandomForestClassifier(n_estimators=100)
#     try:
#         rf_score = cross_val_score(rf, Xs, y, cv=3, scoring="accuracy").mean()
#     except Exception:
#         rf_score = 0.0

#     if rf_score >= lr_score:
#         rf.fit(Xs, y)
#         return rf, scaler, rf_score
#     else:
#         lr.fit(Xs, y)
#         return lr, scaler, lr_score


# def predict_probability_from_model(model, scaler, recent_window: List[float]) -> Optional[float]:
#     try:
#         X, _ = build_features_and_labels(recent_window, lookback=len(recent_window), lookahead=FUTURE_LOOKAHEAD)
#         if len(X) == 0:
#             ema9 = get_ema(recent_window, 9)
#             ema26 = get_ema(recent_window, 26)
#             momentum = recent_window[-1] - ema9 if ema9 else 0.0
#             ret_1 = (recent_window[-1] - recent_window[-2]) / recent_window[-2] if len(recent_window) >= 2 and recent_window[-2] != 0 else 0.0
#             ratio = ema9 / ema26 if ema26 else 0.0
#             rets = []
#             for a, b in zip(recent_window[1:], recent_window[:-1]):
#                 rets.append((a - b) / b if b != 0 else 0.0)
#             vol = np.std(rets) if rets else 0.0
#             feat = np.array([[ema9, ema26, momentum, ret_1, ratio, vol, recent_window[-1]]])
#             Xs = scaler.transform(feat)
#             prob = model.predict_proba(Xs)[0][1]
#             return round(prob * 100, 2)
#         X_last = X[-1].reshape(1, -1)
#         Xs = scaler.transform(X_last)
#         prob = model.predict_proba(Xs)[0][1]
#         return round(prob * 100, 2)
#     except Exception as e:
#         print("predict_probability_from_model error:", e)
#         return None


# # ---------------- Crossover detection & alert ----------------
# # def detect_and_alert_cross(symbol: str, interval: str, close_time_ms: int):
# #     try:
# #         klines = fetch_klines(symbol, interval, limit=HISTORICAL_CANDLES)
# #         if not klines:
# #             return
# #         closes = closes_from_klines(klines)
# #         idx = None
# #         for i, k in enumerate(klines):
# #             if int(k[6]) == close_time_ms:
# #                 idx = i
# #                 break
# #         if idx is None:
# #             idx = len(closes) - 1
# #         if idx < 1:
# #             return

# #         prev = closes[:idx]
# #         cur = closes[:idx + 1]
# #         if len(prev) < 26 or len(cur) < 26:
# #             return

# #         prev_ema9 = get_ema(prev, 9)
# #         prev_ema26 = get_ema(prev, 26)
# #         ema9 = get_ema(cur, 9)
# #         ema26 = get_ema(cur, 26)
# #         if prev_ema9 is None or prev_ema26 is None or ema9 is None or ema26 is None:
# #             return

# #         bullish = (prev_ema9 < prev_ema26) and (ema9 >= ema26)
# #         bearish = (prev_ema9 > prev_ema26) and (ema9 <= ema26)
# #         if not bullish and not bearish:
# #             return

# #         key = (symbol, interval, close_time_ms)
# #         if key in sent_signals:
# #             return

# #         # Use trained model if available
# #         model_info = ML_MODELS.get(symbol)
# #         prob = None
# #         if model_info is not None:
# #             model, scaler, score, trained_at = model_info
# #             prob = predict_probability_from_model(model, scaler, cur[-TRAIN_WINDOW:] if len(cur) >= TRAIN_WINDOW else cur)

# #         price = cur[-1]
# #         if bullish:
# #             msg = f"📈 {symbol} ({interval}) EMA9 CROSSED ABOVE EMA26 — BUY\nPrice: {price}"
# #             if prob is not None:
# #                 msg += f"\n🤖 AI Up Probability: {prob}%  (model score: {round(score,3)})"
# #             broadcast(msg)
# #             sent_signals.add(key)
# #             print("[ALERT]", msg)
# #         else:
# #             msg = f"📉 {symbol} ({interval}) EMA9 CROSSED BELOW EMA26 — SELL\nPrice: {price}"
# #             if prob is not None:
# #                 msg += f"\n🤖 AI Down Probability: {round(100-prob,2)}%  (model score: {round(score,3)})"
# #             broadcast(msg)
# #             sent_signals.add(key)
# #             print("[ALERT]", msg)

# #     except Exception as e:
# #         print("detect_and_alert_cross error:", e)

# # for fewr msg
# def detect_and_alert_cross(symbol: str, interval: str, close_time_ms: int):
#     try:
#         klines = fetch_klines(symbol, interval, limit=HISTORICAL_CANDLES)
#         if not klines:
#             return
#         closes = closes_from_klines(klines)
#         idx = None
#         for i, k in enumerate(klines):
#             if int(k[6]) == close_time_ms:
#                 idx = i
#                 break
#         if idx is None:
#             idx = len(closes) - 1
#         if idx < 1:
#             return

#         prev = closes[:idx]
#         cur = closes[:idx + 1]
#         if len(prev) < 26 or len(cur) < 26:
#             return

#         prev_ema9 = get_ema(prev, 9)
#         prev_ema26 = get_ema(prev, 26)
#         ema9 = get_ema(cur, 9)
#         ema26 = get_ema(cur, 26)
#         if prev_ema9 is None or prev_ema26 is None or ema9 is None or ema26 is None:
#             return

#         bullish = (prev_ema9 < prev_ema26) and (ema9 >= ema26)
#         bearish = (prev_ema9 > prev_ema26) and (ema9 <= ema26)
#         if not bullish and not bearish:
#             return

#         key = (symbol, interval, close_time_ms)
#         if key in sent_signals:
#             return

#         model_info = ML_MODELS.get(symbol)
#         prob = None
#         if model_info is not None:
#             model, scaler, score, trained_at = model_info
#             prob = predict_probability_from_model(
#                 model, scaler, cur[-TRAIN_WINDOW:] if len(cur) >= TRAIN_WINDOW else cur
#             )

#         price = cur[-1]

#         # Only send if probability confidence > 60%
#         if prob is not None and prob < 60:
#             print(f"[SKIP] {symbol} {interval} signal ignored (low confidence {prob}%)")
#             return

#         # Create message
#         if bullish:
#             msg = (
#                 f"📈 {symbol} ({interval}) EMA9 crossed above EMA26 — **BUY**\n"
#                 f"💰 Price: {price}\n"
#             )
#             if prob is not None:
#                 msg += f"🤖 AI Confidence (Up): {prob}%  | Model score: {round(score,3)}"
#         else:
#             msg = (
#                 f"📉 {symbol} ({interval}) EMA9 crossed below EMA26 — **SELL**\n"
#                 f"💰 Price: {price}\n"
#             )
#             if prob is not None:
#                 msg += f"🤖 AI Confidence (Down): {round(100-prob,2)}%  | Model score: {round(score,3)}"

#         broadcast(msg)
#         sent_signals.add(key)
#         print("[ALERT]", msg)

#     except Exception as e:
#         print("detect_and_alert_cross error:", e)

# # ---------------- Monitors (no startup backfill for cross signals) ----------------
# async def monitor_interval(interval: str):
#     minutes = interval_to_minutes(interval)
#     print(f"[MON] monitor {interval} starting (sync to real Binance closes every {minutes} min)")

#     # initialize first-run flag per (symbol, interval)
#     for symbol in SYMBOLS:
#         FIRST_RUN[(symbol, interval)] = True

#     while True:
#         wait = seconds_until_next_close(minutes)
#         print(f"[SYNC] {interval} next close in {int(wait)}s (UTC now {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')})")
#         await asyncio.sleep(wait)

#         for symbol in SYMBOLS:
#             klines = fetch_klines(symbol, interval, limit=26)
#             if isinstance(klines, list) and len(klines) >= 2:
#                 latest = klines[-1]
#                 close_time_ms = int(latest[6])

#                 # If FIRST_RUN: set flag false and skip alerting for this first observed close
#                 if FIRST_RUN.get((symbol, interval), True):
#                     FIRST_RUN[(symbol, interval)] = False
#                     print(f"[INIT] Skipping first close alert for {symbol} {interval} at {close_time_ms}")
#                     continue

#                 # Normal operation: detect and alert for the closed candle
#                 detect_and_alert_cross(symbol, interval, close_time_ms)

#         await asyncio.sleep(1)


# # ---------------- Hourly summary (immediate on startup + aligned to Binance time) ----------------
# async def hourly_summaries():
#     print("[HOUR] hourly summary task starting")
#     # immediate last closed 1H summary on startup
#     for symbol in SYMBOLS:
#         klines = fetch_klines(symbol, "1h", limit=2)
#         if isinstance(klines, list) and len(klines) >= 2:
#             last = klines[-2]  # previous fully closed hour
#             close_time_ms = int(last[6])
#             key = (symbol, "1h", close_time_ms)
#             if key not in sent_signals:
#                 o, h, l, c = map(float, [last[1], last[2], last[3], last[4]])
#                 ts = datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc)
#                 msg = f"🕒 Hourly Summary — {symbol}\nOpen: {o}\nHigh: {h}\nLow: {l}\nClose: {c}\nChange: {round((c-o)/o*100,2)}%"
#                 broadcast(msg)
#                 sent_signals.add(key)

#     # then sync to each top-of-hour close
#     while True:
#         now = datetime.utcnow().replace(tzinfo=timezone.utc)
#         next_hour = (now + timedelta(hours=1)).replace(minute=0, second=5, microsecond=0)
#         wait = (next_hour - now).total_seconds()
#         print(f"[HOUR] waiting {int(wait)}s until next 1h close")
#         await asyncio.sleep(wait)

#         for symbol in SYMBOLS:
#             klines = fetch_klines(symbol, "1h", limit=1)
#             if klines:
#                 k = klines[-1]
#                 close_time_ms = int(k[6])
#                 key = (symbol, "1h", close_time_ms)
#                 if key in sent_signals:
#                     continue
#                 o, h, l, c = map(float, [k[1], k[2], k[3], k[4]])
#                 ts = datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc)
#                 msg = f"🕒 Hourly Summary — {symbol}\nOpen: {o}\nHigh: {h}\nLow: {l}\nClose: {c}\nChange: {round((c-o)/o*100,2)}%"
#                 broadcast(msg)
#                 sent_signals.add(key)
#         await asyncio.sleep(1)


# # ---------------- ML refresher (per-symbol every 4 hours) ----------------
# async def ml_refresher_loop():
#     print("[ML] ML refresher starting")
#     async def train_for_symbol(symbol):
#         try:
#             klines = fetch_klines(symbol, "5m", limit=HISTORICAL_CANDLES)
#             closes = closes_from_klines(klines)
#             if len(closes) < 60:
#                 print(f"[ML] not enough data to train for {symbol}")
#                 return
#             X, y = build_features_and_labels(closes, lookback=TRAIN_WINDOW, lookahead=FUTURE_LOOKAHEAD)
#             if len(X) < MIN_TRAIN_SAMPLES:
#                 print(f"[ML] insufficient train samples for {symbol}: {len(X)}")
#                 return
#             model, scaler, score = train_and_select_best(X, y)
#             if model is None:
#                 print(f"[ML] training returned no model for {symbol}")
#                 return
#             ML_MODELS[symbol] = (model, scaler, score, int(time.time()))
#             print(f"[ML] trained model for {symbol} (score={round(score,3)})")
#         except Exception as e:
#             print("ml train error:", e)

#     # run once immediately on startup
#     for symbol in SYMBOLS:
#         await train_for_symbol(symbol)

#     # then run every ML_REFRESH_HOURS hours (no alignment needed)
#     while True:
#         await asyncio.sleep(ML_REFRESH_HOURS * 3600)
#         for symbol in SYMBOLS:
#             await train_for_symbol(symbol)


# # ---------------- Webhook endpoint ----------------
# @app.post("/webhook/{token}")
# async def webhook(token: str, request: Request):
#     if token != BOT_TOKEN:
#         return {"ok": False}
#     data = await request.json()
#     if "message" in data:
#         chat_id = data["message"]["chat"]["id"]
#         text = data["message"].get("text", "")
#         if chat_id not in user_ids:
#             user_ids.add(chat_id)
#             send_telegram(chat_id, f"✅ Subscribed to EMA alerts!\nTracking: {', '.join(SYMBOLS)}")
#         if text and text.strip().lower() == "/start":
#             send_telegram(chat_id, "👋 Welcome! EMA + AI alerts are active.")
#     return {"ok": True}


# # health root
# @app.get("/")
# def root():
#     return {"status": "ok", "subscribers": len(user_ids), "models": {s: (ML_MODELS.get(s)[2] if ML_MODELS.get(s) else None) for s in SYMBOLS}}


# # ---------------- background loop runner ----------------
# def start_background_loop():
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     tasks = []
#     # monitors per interval
#     for tf in TIMEFRAMES:
#         tasks.append(loop.create_task(monitor_interval(tf)))
#     # hourly summary
#     tasks.append(loop.create_task(hourly_summaries()))
#     # ml refresher
#     tasks.append(loop.create_task(ml_refresher_loop()))
#     try:
#         loop.run_forever()
#     except Exception as e:
#         print("background loop error:", e)


# def start_in_thread():
#     t = threading.Thread(target=start_background_loop, daemon=True)
#     t.start()


# # start background tasks on import/startup
# start_in_thread()
# new for next 5 candles prediction
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

# app = FastAPI()
# user_ids = set()
# sent_hourly = set()

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
#     if roll_down == 0: return 100
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
#     return round(macd_line,5), round(signal_line,5) if signal_line else None, round(macd_hist,5)

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
#         if None in feats: continue
#         X.append(feats)
#         y.append(1 if closes[i+1] > closes[i] else 0)
#     if len(X) < 20: return None
#     model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
#     model.fit(np.array(X), np.array(y))
#     save_model(model, symbol)
#     return model

# def predict_trend(model, closes, volumes=None):
#     feats = compute_features(closes, volumes)
#     if None in feats: return None
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
#         next_trend = predict_next_5_candles(model, closes.copy(), volumes.copy())

#         if prev_ema9 < prev_ema26 and ema9 >= ema26:
#             if prob and prob * 100 >= 60:
#                 msg = f"📈 {symbol} ({interval}) EMA9 crossed ABOVE EMA26 — BUY 💰\nPrice: {close_price}"
#                 msg += f"\n🤖 Uptrend Probability: {round(prob*100,2)}%"
#                 if next_trend:
#                     msg += f"\n🔮 Next 5 Candles Prediction: {next_trend[0]} ({next_trend[1]}%)"
#                 broadcast(msg)

#         elif prev_ema9 > prev_ema26 and ema9 <= ema26:
#             if prob and (1 - prob) * 100 >= 60:
#                 msg = f"📉 {symbol} ({interval}) EMA9 crossed BELOW EMA26 — SELL ⚠️\nPrice: {close_price}"
#                 msg += f"\n🤖 Downtrend Probability: {round((1-prob)*100,2)}%"
#                 if next_trend:
#                     msg += f"\n🔮 Next 5 Candles Prediction: {next_trend[0]} ({next_trend[1]}%)"
#                 broadcast(msg)

#         prev_ema9, prev_ema26 = ema9, ema26

# # ---------------- 4-Hour Retraining ----------------
# async def retrain_loop():
#     while True:
#         await asyncio.sleep(4*60*60)
#         for symbol in SYMBOLS:
#             klines = fetch_klines(symbol, "5m", limit=200)
#             closes = [float(k[4]) for k in klines]
#             volumes = [float(k[5]) for k in klines]
#             model = train_ml_model(symbol, closes, volumes)
#             if model:
#                 print(f"[RETRAIN] Model for {symbol} updated at {datetime.now()}")

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

# # ---------------- Startup ----------------
# @app.on_event("startup")
# async def startup_event():
#     for symbol in SYMBOLS:
#         for tf in TIMEFRAMES:
#             asyncio.create_task(monitor_ema(symbol, tf))
#     asyncio.create_task(retrain_loop())
#     print("✅ EMA + ML Monitoring Started")

# for only 10% profit
import os
import json
import requests
import asyncio
import numpy as np
from datetime import datetime
from fastapi import FastAPI, Request
import xgboost as xgb

# ---------------- Config ----------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
SYMBOLS = ["OPUSDT", "RENDERUSDT", "BTCUSDT"]
TIMEFRAMES = ["5m", "15m"]

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
DATA_FILE = "training_data.json"

app = FastAPI()
user_ids = set()
TRADE_HISTORY = []
TARGET_PROFIT_PERCENT = 0.5  # 0.5% ~10% at 20x leverage

# ---------------- Helpers ----------------
def load_training_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_training_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

training_data = load_training_data()

def add_training_record(symbol, features, outcome):
    training_data.append({"symbol": symbol, "features": features, "outcome": outcome})
    save_training_data(training_data)

# ---------------- Indicators ----------------
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
    if roll_down == 0: return 100
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
        fast_i = get_ema(closes[:i+1], fast)
        slow_i = get_ema(closes[:i+1], slow)
        if fast_i is None or slow_i is None:
            continue
        macd_series.append(fast_i - slow_i)
    signal_line = get_ema(macd_series, signal) if len(macd_series) >= signal else None
    macd_hist = macd_line - signal_line if signal_line else 0
    return round(macd_line,5), round(signal_line,5) if signal_line else None, round(macd_hist,5)

def compute_features(closes, volumes=None):
    ema9 = get_ema(closes, 9)
    ema26 = get_ema(closes, 26)
    rsi14 = get_rsi(closes, 14)
    macd_line, macd_signal, macd_hist = get_macd(closes)
    last_volume = volumes[-1] if volumes else 0
    return [ema9, ema26, rsi14, macd_line, macd_signal, macd_hist, last_volume]

# ---------------- Binance ----------------
def fetch_klines(symbol, interval, limit=200):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[REST] Error fetching klines {symbol} {interval}: {e}")
        return []

# ---------------- ML ----------------
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

def train_ml_model(symbol):
    X, y = [], []
    for record in training_data:
        if record["symbol"] != symbol:
            continue
        X.append(record["features"])
        y.append(1 if record["outcome"] else 0)
    if len(X) < 20: 
        return None
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(np.array(X), np.array(y))
    save_model(model, symbol)
    return model

def predict_trend(model, closes, volumes=None):
    feats = compute_features(closes, volumes)
    if None in feats: return None
    return model.predict_proba([feats])[0][1] if model else None

def predict_next_5_candles(model, closes, volumes=None):
    preds = []
    for i in range(5):
        prob = predict_trend(model, closes, volumes)
        if prob is None: break
        preds.append(prob)
        closes.append(closes[-1] * (1 + (0.001 if prob > 0.5 else -0.001)))
    if not preds: return None
    avg_prob = np.mean(preds)
    trend = "Up" if avg_prob > 0.5 else "Down"
    return trend, round(avg_prob*100,2)

# ---------------- EMA Monitor ----------------
async def monitor_ema(symbol, interval):
    klines = fetch_klines(symbol, interval, limit=200)
    closes = [float(k[4]) for k in klines]
    volumes = [float(k[5]) for k in klines]

    model = load_model(symbol) or train_ml_model(symbol)

    prev_ema9 = get_ema(closes, 9)
    prev_ema26 = get_ema(closes, 26)

    while True:
        await asyncio.sleep(5)
        klines_new = fetch_klines(symbol, interval, limit=2)
        if not klines_new: continue

        close_price = float(klines_new[-1][4])
        closes.append(close_price)
        volumes.append(float(klines_new[-1][5]))
        ema9 = get_ema(closes, 9)
        ema26 = get_ema(closes, 26)
        if None in [prev_ema9, prev_ema26, ema9, ema26]:
            prev_ema9, prev_ema26 = ema9, ema26
            continue

        prob = predict_trend(model, closes, volumes)
        next_trend = predict_next_5_candles(model, closes.copy(), volumes.copy())

        # High-confidence signals
        if prev_ema9 < prev_ema26 and ema9 >= ema26 and prob and prob*100>=60:
            expected_move = (0.001*prob*100)
            if expected_move >= TARGET_PROFIT_PERCENT:
                msg = f"📈 {symbol} ({interval}) EMA9 crossed ABOVE EMA26 — BUY 💰\nPrice: {close_price}"
                msg += f"\n🤖 Uptrend Probability: {round(prob*100,2)}%"
                if next_trend:
                    msg += f"\n🔮 Next 5 Candles: {next_trend[0]} ({next_trend[1]}%)"
                broadcast(msg)
                TRADE_HISTORY.append({"symbol":symbol,"interval":interval,"time":datetime.utcnow(),"price":close_price,"direction":"up","features":compute_features(closes,volumes)})

        elif prev_ema9 > prev_ema26 and ema9 <= ema26 and prob and (1-prob)*100>=60:
            expected_move = (0.001*(1-prob)*100)
            if expected_move >= TARGET_PROFIT_PERCENT:
                msg = f"📉 {symbol} ({interval}) EMA9 crossed BELOW EMA26 — SELL ⚠️\nPrice: {close_price}"
                msg += f"\n🤖 Downtrend Probability: {round((1-prob)*100,2)}%"
                if next_trend:
                    msg += f"\n🔮 Next 5 Candles: {next_trend[0]} ({next_trend[1]}%)"
                broadcast(msg)
                TRADE_HISTORY.append({"symbol":symbol,"interval":interval,"time":datetime.utcnow(),"price":close_price,"direction":"down","features":compute_features(closes,volumes)})

        # Update model based on outcome
        to_remove = []
        for trade in TRADE_HISTORY:
            interval_min = 5 if trade["interval"]=="5m" else 15
            if (datetime.utcnow()-trade["time"]).total_seconds() >= interval_min*60:
                kl_check = fetch_klines(trade["symbol"], trade["interval"], limit=2)
                if kl_check:
                    next_price = float(kl_check[-1][4])
                    actual_up = next_price > trade["price"]
                    outcome = (actual_up and trade["direction"]=="up") or (not actual_up and trade["direction"]=="down")
                    add_training_record(trade["symbol"], trade["features"], outcome)
                    model = train_ml_model(trade["symbol"])
                to_remove.append(trade)
        for tr in to_remove:
            TRADE_HISTORY.remove(tr)

        prev_ema9, prev_ema26 = ema9, ema26

# ---------------- Telegram Webhook ----------------
def send_telegram(chat_id, msg):
    if BOT_TOKEN:
        print(f"[Telegram would send to {chat_id}]: {msg}")

user_ids = set()
@app.post("/webhook/{token}")
async def telegram_webhook(token: str, request: Request):
    if token != BOT_TOKEN:
        return {"ok": False}
    data = await request.json()
    if "message" in data:
        chat_id = data["message"]["chat"]["id"]
        text = data["message"].get("text","")
        if chat_id not in user_ids:
            user_ids.add(chat_id)
            send_telegram(chat_id, f"✅ Subscribed to EMA alerts!\nTracking: {', '.join(SYMBOLS)}")
        if text.lower()=="/start":
            send_telegram(chat_id,"👋 Welcome! EMA + AI alerts active.")
    return {"ok": True}

# ---------------- Startup ----------------
@app.on_event("startup")
async def startup_event():
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            asyncio.create_task(monitor_ema(symbol, tf))
    print("✅ EMA + ML Monitoring Started")
