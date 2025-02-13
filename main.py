import time
import datetime
import requests
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import logging
from collections import defaultdict
from functools import lru_cache
from joblib import load
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackContext

# ✅ Enable Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ✅ Load AI Models & Pre-trained Scaler
MODEL_VERSION = "1.2"
lstm_model = load_model("lstm_model.h5")
xgb_model = load("xgb_model.pkl")
scaler = load("scaler.pkl")  # Load pre-trained scaler

# ✅ Database Setup
conn = sqlite3.connect("bot_data.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        chat_id INTEGER, 
        symbol TEXT, 
        condition TEXT, 
        price REAL, 
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
conn.commit()

# ✅ Rate Limiter
RATE_LIMIT = 5
class RateLimiter:
    def __init__(self):
        self.user_commands = defaultdict(list)

    def check_rate_limit(self, update: Update):
        user_id = update.effective_user.id
        now = time.time()
        self.user_commands[user_id] = [t for t in self.user_commands[user_id] if now - t < 60]
        if len(self.user_commands[user_id]) >= RATE_LIMIT:
            update.message.reply_text("⚠️ Rate limit exceeded. Please wait 1 minute.")
            raise DispatcherHandlerStop()
        self.user_commands[user_id].append(now)

rate_limiter = RateLimiter()

# ✅ Fetch Historical Crypto Prices
def get_historical_prices(symbol, days=60):
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart?vs_currency=usd&days={days}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return [x[1] for x in data.get("prices", [])][-60:]  # Last 60 points
    except requests.RequestException as e:
        logger.error(f"Historical Data Error: {e}")
        return None

# ✅ Fetch Real-Time Price (Cached)
@lru_cache(maxsize=32)
def get_real_time_price(symbol):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=usd"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return float(data.get(symbol, {}).get("usd", None))
    except Exception as e:
        logger.error(f"Price API Error: {e}")
        return None

# ✅ Generate Price Charts
def generate_chart(symbol):
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart?vs_currency=usd&days=60"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        timestamps = [x[0] for x in data["prices"]]
        prices = [x[1] for x in data["prices"]]
        dates = [datetime.datetime.fromtimestamp(ts / 1000) for ts in timestamps]

        plt.figure(figsize=(12, 6))
        plt.plot(dates, prices, label="Price", color='blue')
        plt.title(f"{symbol.upper()} Price Chart 📈")
        plt.xlabel("Date")
        plt.ylabel("Price (USD) 💰")
        plt.legend()
        plt.gcf().autofmt_xdate()

        filename = f"chart_{symbol}_{int(time.time())}.png"
        plt.savefig(filename)
        plt.close()
        return filename
    except requests.exceptions.RequestException as e:
        logger.error(f"Chart Error: {e}")
        return None

# ✅ AI-Based Prediction System
def predict_next_price(symbol):
    historical = get_historical_prices(symbol)
    if historical is None or len(historical) < 60:
        return None

    scaled_data = scaler.transform(np.array(historical).reshape(-1, 1))
    lstm_input = scaled_data.reshape(1, 60, 1)
    xgb_input = scaled_data.flatten().reshape(1, -1)

    lstm_pred = lstm_model.predict(lstm_input)
    xgb_pred = xgb_model.predict(xgb_input)

    lstm_price = scaler.inverse_transform(lstm_pred)[0][0]
    xgb_price = scaler.inverse_transform(xgb_pred.reshape(-1, 1))[0][0]

    final_pred = (lstm_price + xgb_price) / 2

    return {
        "final": final_pred,
        "confidence": {
            "lstm": f"±{np.std(lstm_pred):.2f}",
            "xgb": f"±{np.std(xgb_pred):.2f}"
        }
    }

# ✅ Set Alert
def set_alert(chat_id, symbol, condition, price):
    c.execute("INSERT INTO alerts (chat_id, symbol, condition, price) VALUES (?, ?, ?, ?)",
              (chat_id, symbol, condition, price))
    conn.commit()
    return "✅ Alert set successfully!"

# ✅ Telegram Bot Commands
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("👋 Welcome! Use /price <crypto> to get real-time prices or /predict <crypto> for AI predictions!")

async def price(update: Update, context: CallbackContext):
    symbol = context.args[0].lower()
    price = get_real_time_price(symbol)
    if price:
        await update.message.reply_text(f"💰 {symbol.upper()} Price: ${price}")
    else:
        await update.message.reply_text("⚠️ Invalid symbol!")

async def predict(update: Update, context: CallbackContext):
    symbol = context.args[0].lower()
    result = predict_next_price(symbol)
    if result:
        await update.message.reply_text(f"📈 Predicted Price: ${result['final']:.2f}\n🔹 Confidence: LSTM {result['confidence']['lstm']}, XGB {result['confidence']['xgb']}")
    else:
        await update.message.reply_text("⚠️ Unable to fetch historical data!")

# ✅ Async Telegram Bot
app = ApplicationBuilder().token("7277532789:AAGS5v9K6if3ZrLen8fa2ABRovn25Sazpk8").build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("price", price))
app.add_handler(CommandHandler("predict", predict))

logger.info("🚀 Bot is running...")
app.run_polling()
