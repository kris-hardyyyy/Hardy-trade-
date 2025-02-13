import os
import time
import logging
import sqlite3
import traceback
import requests
import numpy as np
import tensorflow as tf
import xgboost as xgb
import re
from collections import deque
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential
from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackContext

# ================= SETUP LOGGING =================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ================= CONFIGURATION =================
class Config:
    MODEL_WEIGHTS = {"lstm": 0.6, "xgb": 0.4}
    MAX_HISTORY_DAYS = 90
    REQUEST_TIMEOUT = 30
    OCR_CONFIG = r'--oem 3 --psm 6'
    ALLOWED_SYMBOL_PATTERN = r"^[A-Za-z]{3,10}$"
    MODEL_INPUT_SIZE = 60
    RATE_LIMITS = {"free": (5, 300), "premium": (100, 60)}  # (requests, time window)

cfg = Config()

# ================= SECURITY: HIDE TOKEN =================
TELEGRAM_TOKEN = os.getenv("7277532789:AAGS5v9K6if3ZrLen8fa2ABRovn25Sazpk8")
if not TELEGRAM_TOKEN:
    logger.critical("Missing TELEGRAM_BOT_TOKEN environment variable")
    exit(1)

# ================= DATABASE SETUP (PERSISTENT RATE LIMITING) =================
class Database:
    def __init__(self, db_file="bot_data.db"):
        self.conn = sqlite3.connect(db_file, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS rate_limits (
                                user_id INTEGER PRIMARY KEY,
                                last_requests TEXT)""")
        self.conn.commit()

    def get_requests(self, user_id):
        self.cursor.execute("SELECT last_requests FROM rate_limits WHERE user_id=?", (user_id,))
        row = self.cursor.fetchone()
        return deque(map(float, row[0].split(","))) if row else deque()

    def update_requests(self, user_id, request_times):
        request_str = ",".join(map(str, request_times))
        self.cursor.execute("REPLACE INTO rate_limits (user_id, last_requests) VALUES (?, ?)", (user_id, request_str))
        self.conn.commit()

db = Database()

# ================= RATE LIMITER =================
class RateLimiter:
    def __init__(self):
        self.db = db

    def check_limit(self, user_id):
        max_requests, window = cfg.RATE_LIMITS["free"]
        now = time.time()
        requests = self.db.get_requests(user_id)

        while requests and (now - requests[0] > window):
            requests.popleft()

        if len(requests) >= max_requests:
            return False

        requests.append(now)
        self.db.update_requests(user_id, requests)
        return True

limiter = RateLimiter()

# ================= MODEL LOADER =================
class ModelLoader:
    @staticmethod
    def load_models():
        try:
            lstm_model = tf.keras.models.load_model("lstm_model.h5")
            xgb_model = xgb.Booster()
            xgb_model.load_model("xgb_model.json")
            
            # Validate models
            test_data = np.random.randn(cfg.MODEL_INPUT_SIZE, 1)
            lstm_model.predict(test_data.reshape(1, cfg.MODEL_INPUT_SIZE, 1))
            xgb_model.predict(xgb.DMatrix(test_data.reshape(1, -1)))

            return lstm_model, xgb_model
        except Exception as e:
            logger.critical(f"Model loading failed: {str(e)}")
            exit(1)

lstm_model, xgb_model = ModelLoader.load_models()

# ================= API CLIENT (CoinGecko) =================
class CoinGeckoClient:
    BASE_URL = "https://api.coingecko.com/api/v3"

    @lru_cache(maxsize=100)
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    def get_coin_id(self, symbol):
        try:
            response = requests.get(f"{self.BASE_URL}/coins/list", timeout=cfg.REQUEST_TIMEOUT)
            response.raise_for_status()
            for coin in response.json():
                if coin["symbol"].lower() == symbol.lower():
                    return coin["id"]
            raise ValueError(f"Symbol {symbol} not found")
        except requests.RequestException as e:
            logger.error(f"API Error: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    def get_historical_prices(self, symbol, days):
        try:
            coin_id = self.get_coin_id(symbol)
            response = requests.get(f"{self.BASE_URL}/coins/{coin_id}/market_chart",
                                    params={"vs_currency": "usd", "days": days},
                                    timeout=cfg.REQUEST_TIMEOUT)
            response.raise_for_status()
            return [p[1] for p in response.json().get('prices', [])]
        except requests.RequestException as e:
            logger.error(f"Price data fetch failed: {str(e)}")
            raise

gecko_client = CoinGeckoClient()

# ================= PREDICTION ENGINE =================
class PredictionEngine:
    def __init__(self, lstm_model, xgb_model):
        self.lstm_model = lstm_model
        self.xgb_model = xgb_model

    def predict(self, prices):
        if len(prices) < cfg.MODEL_INPUT_SIZE:
            raise ValueError("Not enough data for prediction")

        sequence = np.array(prices[-cfg.MODEL_INPUT_SIZE:])
        lstm_input = sequence.reshape(1, cfg.MODEL_INPUT_SIZE, 1)
        xgb_input = xgb.DMatrix(sequence.reshape(1, -1))

        lstm_pred = self.lstm_model.predict(lstm_input)[0][0]
        xgb_pred = self.xgb_model.predict(xgb_input)[0]

        return {
            "lstm": lstm_pred,
            "xgb": xgb_pred,
            "combined": (cfg.MODEL_WEIGHTS["lstm"] * lstm_pred + cfg.MODEL_WEIGHTS["xgb"] * xgb_pred)
        }

prediction_engine = PredictionEngine(lstm_model, xgb_model)

# ================= TELEGRAM COMMAND HANDLER =================
async def handle_prediction(update: Update, context: CallbackContext):
    user = update.effective_user

    if not limiter.check_limit(user.id):
        await update.message.reply_text("⏳ Please wait before making another request")
        return

    if not context.args or len(context.args) != 1:
        await update.message.reply_text("🔍 Usage: /predict <CRYPTO_SYMBOL>")
        return

    symbol = context.args[0].strip().upper()
    if not re.match(cfg.ALLOWED_SYMBOL_PATTERN, symbol):
        await update.message.reply_text("❌ Invalid cryptocurrency symbol")
        return

    try:
        prices = gecko_client.get_historical_prices(symbol, cfg.MAX_HISTORY_DAYS)
        prediction = prediction_engine.predict(prices)

        response_msg = (f"📈 **{symbol} Price Prediction**\n"
                        f"• LSTM Model: ${prediction['lstm']:.2f}\n"
                        f"• XGBoost Model: ${prediction['xgb']:.2f}\n"
                        f"• Combined Prediction: ${prediction['combined']:.2f}")

        await update.message.reply_markdown(response_msg)

    except Exception:
        logger.error(traceback.format_exc())
        await update.message.reply_text("⚠️ Could not generate prediction. Try again later.")

# ================= BOT SETUP =================
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("predict", handle_prediction))

    logger.info("🚀 Crypto AI Bot Running...")
    app.run_polling()

if __name__ == "__main__":
    main()
