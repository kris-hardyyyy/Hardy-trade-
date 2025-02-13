import os
import time
import logging
import traceback
import requests
import numpy as np
import tensorflow as tf
import xgboost as xgb
import re
import cv2
from collections import deque, defaultdict
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential
from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackContext

# ================= SUPPRESSING UNNECESSARY LOGS =================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress most TensorFlow warnings
tf.get_logger().setLevel('ERROR')  # Hide unnecessary logs

# ================= INITIALIZATION =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ================= CONFIGURATION =================
class Config:
    MODEL_WEIGHTS = {"lstm": 0.6, "xgb": 0.4}
    MAX_HISTORY_DAYS = 90
    REQUEST_TIMEOUT = 30
    RATE_LIMITS = {"free": (5, 300), "premium": (100, 60)}
    OCR_CONFIG = r'--oem 3 --psm 6'
    ALLOWED_SYMBOL_PATTERN = r"^[A-Za-z]{3,10}$"
    MODEL_INPUT_SIZE = 60

cfg = Config()

# ================= SECURITY SETUP =================
TELEGRAM_TOKEN = '7277532789:AAGS5v9K6if3ZrLen8fa2ABRovn25Sazpk8'  # Hardcoded bot token
if not TELEGRAM_TOKEN:
    logger.critical("Missing TELEGRAM_BOT_TOKEN environment variable")
    exit(1)

# ================= MODEL MANAGEMENT =================
class ModelLoader:
    @staticmethod
    def load_models():
        """Load and validate machine learning models"""
        try:
            lstm_model = tf.keras.models.load_model("lstm_model.h5")
            xgb_model = xgb.Booster()
            xgb_model.load_model("xgb_model.json")
            
            # Validate model compatibility
            test_data = np.random.randn(cfg.MODEL_INPUT_SIZE)
            lstm_model.predict(test_data.reshape(1, cfg.MODEL_INPUT_SIZE, 1))
            xgb_model.predict(xgb.DMatrix(test_data.reshape(1, -1)))
            
            return lstm_model, xgb_model
            
        except Exception as e:
            logger.critical(f"Model loading failed: {str(e)}")
            exit(1)

lstm_model, xgb_model = ModelLoader.load_models()

# ================= API CLIENT =================
class CoinGeckoClient:
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    @lru_cache(maxsize=100)
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    def get_coin_id(self, symbol: str) -> str:
        """Resolve cryptocurrency symbol to CoinGecko ID"""
        endpoint = f"{self.BASE_URL}/coins/list"
        try:
            response = requests.get(endpoint, timeout=cfg.REQUEST_TIMEOUT)
            response.raise_for_status()
            for coin in response.json():
                if coin['symbol'].lower() == symbol.lower():
                    return coin['id']
            raise ValueError(f"Symbol {symbol} not found")
        except requests.RequestException as e:
            logger.error(f"API Error: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    def get_historical_prices(self, symbol: str, days: int) -> list:
        """Retrieve historical price data"""
        coin_id = self.get_coin_id(symbol)
        endpoint = f"{self.BASE_URL}/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": days}
        
        try:
            response = requests.get(endpoint, params=params, timeout=cfg.REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            return [p[1] for p in data.get('prices', [])]
        except requests.RequestException as e:
            logger.error(f"Price data fetch failed: {str(e)}")
            raise

gecko_client = CoinGeckoClient()

# ================= SECURITY & VALIDATION =================
class InputValidator:
    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        return bool(re.match(cfg.ALLOWED_SYMBOL_PATTERN, symbol))
    
    @staticmethod
    def validate_prices(prices: list) -> bool:
        if len(prices) < cfg.MODEL_INPUT_SIZE:
            return False
        prices = np.array(prices)
        if np.any(prices <= 0):
            return False
        daily_returns = np.diff(prices) / prices[:-1]
        if np.mean(np.abs(daily_returns)) > 0.5:  # 50% average daily change
            return False
        return True

# ================= RATE LIMITING =================
class RateLimiter:
    def __init__(self):
        self.users = defaultdict(lambda: {"tier": "free", "requests": deque()})
    
    def check_limit(self, user_id: int) -> bool:
        """Enforce rate limits based on user tier"""
        tier = self.users[user_id]["tier"]
        max_req, window = cfg.RATE_LIMITS[tier]
        
        now = time.time()
        requests = self.users[user_id]["requests"]
        
        # Remove expired requests
        while requests and (now - requests[0] > window):
            requests.popleft()
            
        if len(requests) >= max_req:
            return False
            
        requests.append(now)
        return True

limiter = RateLimiter()

# ================= PREDICTION ENGINE =================
class PredictionEngine:
    def __init__(self, lstm_model, xgb_model):
        self.lstm_model = lstm_model
        self.xgb_model = xgb_model
    
    def predict(self, prices: list) -> dict:
        """Generate predictions from cleaned price data"""
        if not InputValidator.validate_prices(prices):
            raise ValueError("Invalid price data")
            
        # Prepare model inputs
        sequence = np.array(prices[-cfg.MODEL_INPUT_SIZE:])
        lstm_input = sequence.reshape(1, cfg.MODEL_INPUT_SIZE, 1)
        xgb_input = xgb.DMatrix(sequence.reshape(1, -1))
        
        # Generate predictions
        lstm_pred = self.lstm_model.predict(lstm_input)[0][0]
        xgb_pred = self.xgb_model.predict(xgb_input)[0]
        
        return {
            "lstm": lstm_pred,
            "xgb": xgb_pred,
            "combined": (cfg.MODEL_WEIGHTS["lstm"] * lstm_pred +
                        cfg.MODEL_WEIGHTS["xgb"] * xgb_pred)
        }

prediction_engine = PredictionEngine(lstm_model, xgb_model)

# ================= TELEGRAM HANDLERS =================
async def handle_prediction(update: Update, context: CallbackContext):
    """Process prediction requests from users"""
    user = update.effective_user
    try:
        # Rate limiting check
        if not limiter.check_limit(user.id):
            await update.message.reply_text("⏳ Please wait before making another request")
            return

        # Input validation
        if not context.args or len(context.args) != 1:
            await update.message.reply_text("🔍 Usage: /predict <CRYPTO_SYMBOL>")
            return
            
        symbol = context.args[0].strip().upper()
        if not InputValidator.validate_symbol(symbol):
            await update.message.reply_text("❌ Invalid cryptocurrency symbol")
            return

        # Fetch data and generate prediction
        prices = gecko_client.get_historical_prices(symbol, cfg.MAX_HISTORY_DAYS)
        prediction = prediction_engine.predict(prices)
        
        # Format response
        response_msg = (
            f"📈 **{symbol} Price Prediction**\n"
            f"• LSTM Model: ${prediction['lstm']:.2f}\n"
            f"• XGBoost Model: ${prediction['xgb']:.2f}\n"
            f"• Combined Prediction: ${prediction['combined']:.2f}"
        )
        await update.message.reply_markdown(response_msg)

    except Exception as e:
        logger.error(f"Prediction failed for {user.id}: {traceback.format_exc()}")
        await update.message.reply_text("⚠️ Could not generate prediction. Please try again later.")

# ================= APPLICATION SETUP =================
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("predict", handle_prediction))
    
    logger.info("🚀 CryptoSage Prediction Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
