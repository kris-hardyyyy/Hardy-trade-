import os
import numpy as np
import requests
import pickle
import joblib
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackContext

# === Load Pre-trained Models === #
LSTM_MODEL_PATH = "lstm_model.h5"
XGB_MODEL_PATH = "xgb_model.pkl"
SCALER_PATH = "scaler.pkl"

# Ensure all models exist before proceeding
if not os.path.exists(LSTM_MODEL_PATH):
    raise FileNotFoundError(f"❌ Missing {LSTM_MODEL_PATH}")

if not os.path.exists(XGB_MODEL_PATH):
    raise FileNotFoundError(f"❌ Missing {XGB_MODEL_PATH}")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"❌ Missing {SCALER_PATH}")

# Load LSTM Model
lstm_model = load_model(LSTM_MODEL_PATH)

# Load XGBoost Model
with open(XGB_MODEL_PATH, "rb") as f:
    xgb_model = pickle.load(f)

# Load Scaler
scaler = joblib.load(SCALER_PATH)

# === Fetch Historical Prices === #
def get_historical_prices(symbol, days=60):
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart?vs_currency=usd&days={days}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        prices = [item[1] for item in data.get("prices", [])]
        return prices if len(prices) == days else None
    except Exception as e:
        print(f"⚠️ Error fetching historical prices: {e}")
        return None

# === Predict Next Price === #
def predict_next_price(symbol):
    historical = get_historical_prices(symbol)
    
    if not historical or len(historical) < 60:
        return "❌ Not enough historical data available."

    # Scale Data
    historical_array = np.array(historical).reshape(-1, 1)
    scaled_data = scaler.transform(historical_array)

    # Prepare LSTM Input
    lstm_input = scaled_data.reshape(1, 60, 1)

    # Prepare XGB Input
    xgb_input = scaled_data.flatten().reshape(1, -1)

    # Predictions
    lstm_pred = lstm_model.predict(lstm_input)
    xgb_pred = xgb_model.predict(xgb_input)

    # Inverse Scale
    lstm_price = scaler.inverse_transform(lstm_pred)[0][0]
    xgb_price = scaler.inverse_transform(xgb_pred.reshape(-1, 1))[0][0]

    return f"📊 Predicted Price:\n🔹 LSTM: ${lstm_price:.2f}\n🔹 XGBoost: ${xgb_price:.2f}"

# === Telegram Bot Commands === #
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("🤖 Welcome! Use /predict <coin_symbol> to get predictions.")

async def predict(update: Update, context: CallbackContext) -> None:
    if len(context.args) == 0:
        await update.message.reply_text("⚠️ Please provide a coin symbol. Example: /predict bitcoin")
        return
    
    symbol = context.args[0].lower()
    result = predict_next_price(symbol)
    await update.message.reply_text(result)

# === Run Telegram Bot === #
TOKEN = "7277532789:AAGS5v9K6if3ZrLen8fa2ABRovn25Sazpk8"

app = ApplicationBuilder().token(TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("predict", predict))

print("🤖 Bot is running...")
app.run_polling()
