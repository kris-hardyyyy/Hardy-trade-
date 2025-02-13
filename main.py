import numpy as np
import pickle
import tensorflow as tf
import requests
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from joblib import load
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext

# 🔹 **Load Models & Scaler**
LSTM_MODEL_PATH = "/mnt/data/lstm_model.h5"
XGB_MODEL_PATH = "/mnt/data/xgb_model.pkl"
SCALER_PATH = "/mnt/data/scaler.pkl"

# Load LSTM Model
lstm_model = load_model(LSTM_MODEL_PATH)

# Load XGBoost Model
with open(XGB_MODEL_PATH, "rb") as f:
    xgb_model = pickle.load(f)

# Load Scaler
scaler = load(SCALER_PATH)

# 🔹 **Function to Get Historical Prices**
def get_historical_prices(symbol, days=60):
    """
    Fetches the last `days` of price data for a given crypto symbol.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart?vs_currency=usd&days={days}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return [price[1] for price in data["prices"]]
    except Exception as e:
        print(f"⚠️ Error fetching historical prices: {e}")
        return None

# 🔹 **Function to Preprocess Data**
def preprocess_data(historical_prices):
    """
    Prepares historical price data for the LSTM and XGBoost models.
    """
    if len(historical_prices) < 60:
        return None, None  # Need at least 60 data points

    # Normalize prices
    scaled_data = scaler.transform(np.array(historical_prices).reshape(-1, 1))

    # LSTM expects 3D input: (samples, timesteps, features)
    lstm_input = np.array([scaled_data[-60:]]).reshape(1, 60, 1)

    # XGBoost expects 2D input: (samples, features)
    xgb_input = scaled_data[-60:].flatten().reshape(1, -1)

    return lstm_input, xgb_input

# 🔹 **Function to Predict Next Price**
def predict_next_price(symbol, historical_prices):
    """
    Predicts the next price using both LSTM and XGBoost models.
    """
    lstm_input, xgb_input = preprocess_data(historical_prices)
    if lstm_input is None or xgb_input is None:
        return "⚠️ Not enough historical data (60+ data points required)."

    # LSTM Prediction
    lstm_pred = lstm_model.predict(lstm_input)
    lstm_price = scaler.inverse_transform(lstm_pred)[0][0]

    # XGBoost Prediction
    xgb_pred = xgb_model.predict(xgb_input)
    xgb_price = scaler.inverse_transform(xgb_pred.reshape(-1, 1))[0][0]

    # Combine results
    final_pred = (lstm_price + xgb_price) / 2

    return {
        "lstm_price": round(lstm_price, 2),
        "xgb_price": round(xgb_price, 2),
        "final_pred": round(final_pred, 2),
    }

# 🔹 **Telegram Bot Command for Prediction**
def predict_command(update: Update, context: CallbackContext):
    """
    Telegram bot command to predict the next crypto price.
    Usage: /predict bitcoin
    """
    if len(context.args) < 1:
        update.message.reply_text("⚠️ Please provide a symbol! Example: /predict bitcoin")
        return

    symbol = context.args[0].lower()
    historical_prices = get_historical_prices(symbol)

    if historical_prices is None:
        update.message.reply_text("⚠️ Failed to fetch historical data. Try again later.")
        return

    prediction = predict_next_price(symbol, historical_prices)

    if isinstance(prediction, str):
        update.message.reply_text(prediction)
    else:
        response = (
            f"📊 *Price Prediction for {symbol.upper()}* 📊\n"
            f"🔹 *LSTM Model:* ${prediction['lstm_price']}\n"
            f"🔹 *XGBoost Model:* ${prediction['xgb_price']}\n"
            f"✨ *Final Prediction:* ${prediction['final_pred']}\n"
        )
        update.message.reply_text(response, parse_mode="Markdown")

# 🔹 **Setup Telegram Bot**
def main():
    """
    Main function to run the Telegram bot.
    """
    TOKEN = "7277532789:AAGS5v9K6if3ZrLen8fa2ABRovn25Sazpk8"  # 🔴 Replace with your bot token
    updater = Updater(TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    # Add command handlers
    dispatcher.add_handler(CommandHandler("predict", predict_command))

    # Start the bot
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
