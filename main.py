import os
import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
from joblib import load
from apscheduler.schedulers.background import BackgroundScheduler

# 🎯 Load AI Models & Scaler
LSTM_MODEL_PATH = "lstm_model.h5"
XGB_MODEL_PATH = "xgb_model.pkl"
SCALER_PATH = "scaler.pkl"

lstm_model = load_model(LSTM_MODEL_PATH)
xgb_model = load(XGB_MODEL_PATH)
scaler = load(SCALER_PATH)

# 🔥 Get Real-Time Crypto Price
def get_real_time_price(symbol="BTCUSDT"):
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    response = requests.get(url)
    data = response.json()
    return float(data["price"])

# 📈 Predict Next Price Using LSTM + XGBoost
def predict_next_price(symbol="BTCUSDT"):
    price = get_real_time_price(symbol)

    # Scale & Reshape
    scaled_price = scaler.transform(np.array(price).reshape(-1, 1))
    X_input = np.reshape(scaled_price, (1, 1, 1))  # For LSTM

    # Get Predictions
    lstm_pred = lstm_model.predict(X_input)
    xgb_pred = xgb_model.predict(scaled_price.reshape(1, -1))

    # Reverse Scaling
    lstm_price = scaler.inverse_transform(lstm_pred)[0][0]
    xgb_price = scaler.inverse_transform(xgb_pred.reshape(-1, 1))[0][0]

    # Final Prediction (Averaging)
    final_pred = (lstm_price + xgb_price) / 2
    return final_pred

# 📊 Generate Crypto Chart & Upload
def generate_chart(symbol="BTCUSDT"):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=50"
    data = requests.get(url).json()

    # Extract Prices
    timestamps = [x[0] for x in data]
    prices = [float(x[4]) for x in data]  # Closing Prices

    # Plot Chart
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, prices, label="Closing Price", color="blue")
    plt.title(f"{symbol} Price Trend 📈")
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid()
    plt.savefig("chart.png")

    return "chart.png"

# 🔔 Price Alert System
alerts = {}

def check_alerts():
    for chat_id, alert_data in alerts.items():
        symbol, condition, price = alert_data
        current_price = get_real_time_price(symbol)

        if (condition == "above" and current_price >= price) or (condition == "below" and current_price <= price):
            bot.send_message(chat_id=chat_id, text=f"🔔 Alert: {symbol} is now {condition} ${price}!")
            del alerts[chat_id]

scheduler = BackgroundScheduler()
scheduler.add_job(check_alerts, "interval", minutes=1)
scheduler.start()

# 🤖 Telegram Bot Token
TOKEN = "7277532789:AAGS5v9K6if3ZrLen8fa2ABRovn25Sazpk8"

# 🏆 Telegram Bot Commands
def start(update: Update, context: CallbackContext):
    update.message.reply_text("👋 Welcome to the Crypto AI Bot!\nUse /predict [symbol] to see future prices!")

def predict_command(update: Update, context: CallbackContext):
    try:
        symbol = context.args[0].upper() if context.args else "BTCUSDT"
        predicted_price = predict_next_price(symbol)
        
        message = f"📊 *AI-Predicted Price for {symbol}:* \n💰 ${predicted_price:.2f}"
        update.message.reply_text(message, parse_mode="Markdown")
    
    except Exception as e:
        update.message.reply_text("⚠️ Error predicting price. Please try again.")
        print(f"Error: {e}")

def alert_command(update: Update, context: CallbackContext):
    try:
        symbol = context.args[0].upper()
        condition = context.args[1].lower()
        price = float(context.args[2])

        if condition not in ["above", "below"]:
            update.message.reply_text("⚠️ Invalid condition. Use 'above' or 'below'.")
            return

        chat_id = update.message.chat_id
        alerts[chat_id] = (symbol, condition, price)

        update.message.reply_text(f"✅ Alert set for {symbol}: Notify when {condition} ${price}")
    
    except Exception:
        update.message.reply_text("⚠️ Usage: /alert [symbol] [above/below] [price]")

def chart_command(update: Update, context: CallbackContext):
    try:
        symbol = context.args[0].upper() if context.args else "BTCUSDT"
        chart_path = generate_chart(symbol)
        update.message.reply_photo(photo=open(chart_path, "rb"))
    
    except Exception as e:
        update.message.reply_text("⚠️ Error generating chart.")
        print(f"Error: {e}")

def help_command(update: Update, context: CallbackContext):
    help_text = "📖 *Bot Commands:* \n"
    help_text += "✅ /start - Welcome message\n"
    help_text += "✅ /predict [symbol] - Predict crypto price (e.g., /predict BTCUSDT)\n"
    help_text += "✅ /alert [symbol] [above/below] [price] - Set price alerts\n"
    help_text += "✅ /chart [symbol] - Get price chart (e.g., /chart BTCUSDT)\n"
    help_text += "✅ /help - Show this help message"
    
    update.message.reply_text(help_text, parse_mode="Markdown")

# 🚀 Start Telegram Bot
def main():
    global bot

    updater = Updater(TOKEN, use_context=True)
    bot = updater.bot
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("predict", predict_command))
    dispatcher.add_handler(CommandHandler("alert", alert_command))
    dispatcher.add_handler(CommandHandler("chart", chart_command))
    dispatcher.add_handler(CommandHandler("help", help_command))

    print("🚀 Bot is running...")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
