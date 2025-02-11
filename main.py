import logging
import ccxt
import talib
import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    CallbackContext,
    CallbackQueryHandler,
    filters,
)
from datetime import datetime, timedelta
import threading
import time
import urllib.parse
import hashlib
import hmac
import base64
import requests
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import os
import tempfile

# ============================
# CONFIGURATION (Public Mode)
# ============================
EXCHANGE = ccxt.binance({"enableRateLimit": True})
SUPPORTED_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "TRUMP/USDT", "SHIB/USDT"]

# ============================
# PROFESSIONAL ANALYSIS ENGINE
# ============================
class HedgeFundGradeAnalyzer:
    def __init__(self):
        self.timeframes = ["1h", "4h", "1d"]
        self.risk_model = self._load_risk_model()

    def _load_risk_model(self):
        return None

    def full_analysis(self, symbol: str) -> dict:
        analysis = {}
        for tf in self.timeframes:
            df = self._fetch_ohlcv(symbol, tf)
            analysis[tf] = {
                "trend": self._determine_trend(df),
                "key_levels": self._calculate_key_levels(df),
                "momentum_indicators": self._calculate_momentum(df),
                "volume_profile": self._analyze_volume(df),
            }
        analysis["consensus"] = self._generate_consensus(analysis)
        return analysis

    def _fetch_ohlcv(self, symbol: str, timeframe: str) -> pd.DataFrame:
        if symbol not in SUPPORTED_SYMBOLS:
            raise ValueError(f"Unsupported symbol: {symbol}")
        data = EXCHANGE.fetch_ohlcv(symbol, timeframe, limit=500)
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def _calculate_key_levels(self, df: pd.DataFrame) -> dict:
        support = df["low"].rolling(50).min().iloc[-1]
        resistance = df["high"].rolling(50).max().iloc[-1]
        pivot = (df["high"].iloc[-1] + df["low"].iloc[-1] + df["close"].iloc[-1]) / 3
        return {"support": support, "resistance": resistance, "pivot": pivot}

    def _determine_trend(self, df: pd.DataFrame) -> str:
        if len(df) < 50:
            return "😐 Neutral"
        sma_short = df["close"].rolling(window=20).mean().iloc[-1]
        sma_long = df["close"].rolling(window=50).mean().iloc[-1]
        return "🚀 Bullish" if sma_short > sma_long else "🐻 Bearish"

    def _calculate_momentum(self, df: pd.DataFrame) -> dict:
        close = df["close"]
        rsi = talib.RSI(close, timeperiod=14).iloc[-1] if len(close) > 14 else 50
        macd, signal, hist = talib.MACD(close)
        macd_value = macd.iloc[-1] if not macd.empty else 0
        return {"rsi": rsi, "macd": macd_value}

    def _analyze_volume(self, df: pd.DataFrame) -> dict:
        volume = df["volume"]
        recent_avg = volume.iloc[-10:].mean()
        overall_avg = volume.mean()
        volume_spike = recent_avg > overall_avg * 1.5
        volume_trend = "📈 Increasing" if recent_avg > overall_avg else "📉 Decreasing"
        return {"volume_spike": volume_spike, "volume_trend": volume_trend}

    def _generate_consensus(self, analysis: dict) -> dict:
        support = sum(analysis[tf]["key_levels"]["support"] for tf in self.timeframes) / len(self.timeframes)
        resistance = sum(analysis[tf]["key_levels"]["resistance"] for tf in self.timeframes) / len(self.timeframes)
        pivot = sum(analysis[tf]["key_levels"]["pivot"] for tf in self.timeframes) / len(self.timeframes)
        rsi = sum(analysis[tf]["momentum_indicators"]["rsi"] for tf in self.timeframes) / len(self.timeframes)
        macd = sum(analysis[tf]["momentum_indicators"]["macd"] for tf in self.timeframes) / len(self.timeframes)
        volume_spike = any(analysis[tf]["volume_profile"]["volume_spike"] for tf in self.timeframes)
        volume_trend = analysis["1h"]["volume_profile"]["volume_trend"]
        trends = [analysis[tf]["trend"] for tf in self.timeframes]
        consensus_trend = max(set(trends), key=trends.count)
        return {
            "support": support,
            "resistance": resistance,
            "pivot": pivot,
            "rsi": rsi,
            "macd": macd,
            "volume_spike": volume_spike,
            "volume_trend": volume_trend,
            "trend": consensus_trend,
        }

# ============================
# PORTFOLIO MANAGEMENT (Simulation Only)
# ============================
class PortfolioManager:
    def __init__(self):
        self.portfolio = {}
        self.risk_params = {"max_drawdown": 0.1, "daily_loss_limit": 0.05}

    def add_position(self, symbol: str, entry_price: float, size: float):
        self.portfolio[symbol] = {
            "entry": entry_price,
            "size": size,
            "stop_loss": self._calculate_stop_loss(symbol),
            "take_profit": self._calculate_take_profit(symbol),
        }

    def _calculate_stop_loss(self, symbol: str) -> float:
        analyzer = HedgeFundGradeAnalyzer()
        analysis = analyzer.full_analysis(symbol)
        return analysis["1h"]["key_levels"]["support"] * 0.98

    def _calculate_take_profit(self, symbol: str) -> float:
        analyzer = HedgeFundGradeAnalyzer()
        analysis = analyzer.full_analysis(symbol)
        return analysis["1h"]["key_levels"]["resistance"] * 1.02

# ============================
# TELEGRAM HANDLERS (Public Data Only)
# ============================
async def start_command(update: Update, context: CallbackContext):
    welcome_text = (
        "👋 Welcome to *Hardly trade analyst 💎📈*!\n\n"
        "This bot provides professional-grade market analysis using public Binance data. "
        "You can:\n"
        "• Use /analyze [symbol] to get a detailed technical analysis report (e.g. /analyze BTC).\n"
        "• Use /portfolio to view your simulated portfolio (for demonstration purposes).\n"
        "• Use /alert [symbol] [price] [above/below] to set a price alert.\n\n"
        "Enjoy your analysis and happy trading! 🚀"
    )
    await update.message.reply_markdown_v2(welcome_text)

async def analyze_command(update: Update, context: CallbackContext):
    symbol = context.args[0].upper() + "/USDT" if context.args else "BTC/USDT"
    bot_instance = HardlyTradeAnalystBot()
    try:
        report = bot_instance.generate_report(symbol)
        await update.message.reply_markdown_v2(report)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Oops! An error occurred: {str(e)}")

async def portfolio_command(update: Update, context: CallbackContext):
    bot_instance = HardlyTradeAnalystBot()
    positions = bot_instance.portfolio.portfolio
    response = "📊 *Simulated Portfolio* 📊\n"
    if not positions:
        response += "\nYour portfolio is empty. Add some positions (simulation only)!"
    else:
        for sym, data in positions.items():
            current_price = EXCHANGE.fetch_ticker(sym)["last"]
            pnl = (current_price / data["entry"] - 1) * 100
            response += (
                f"\n{sym}:\n"
                f"💰 Size: {data['size']} contracts\n"
                f"📈 PnL: {pnl:.2f}%\n"
                f"🛑 SL: ${data['stop_loss']:.2f}\n"
                f"🎯 TP: ${data['take_profit']:.2f}\n"
            )
    await update.message.reply_markdown_v2(response)

async def alert_handler(update: Update, context: CallbackContext):
    args = context.args
    if len(args) != 3:
        await update.message.reply_text("❗ Usage: /alert [SYMBOL] [PRICE] [ABOVE/BELOW]")
        return
    symbol = args[0].upper() + "/USDT"
    price = float(args[1])
    condition = args[2].lower()
    key = f"{symbol}_{datetime.now().timestamp()}"
    context.bot_data[key] = {
        "chat_id": update.effective_chat.id,
        "symbol": symbol,
        "price": price,
        "condition": condition,
    }
    threading.Thread(target=check_price_alerts, args=(context,)).start()
    await update.message.reply_text(f"🔔 Alert set: {symbol} {condition.upper()} ${price:.2f} ✅")

def check_price_alerts(context: CallbackContext):
    while True:
        keys = list(context.bot_data.keys())
        for key in keys:
            alert = context.bot_data[key]
            ticker = EXCHANGE.fetch_ticker(alert["symbol"])
            current_price = ticker["last"]
            if alert["condition"] == "above" and current_price > alert["price"]:
                context.bot.send_message(
                    chat_id=alert["chat_id"],
                    text=f"🚨 {alert['symbol']} is ABOVE ${alert['price']:.2f}! (Current: ${current_price:.2f}) 🎉",
                )
                del context.bot_data[key]
            elif alert["condition"] == "below" and current_price < alert["price"]:
                context.bot.send_message(
                    chat_id=alert["chat_id"],
                    text=f"🚨 {alert['symbol']} is BELOW ${alert['price']:.2f}! (Current: ${current_price:.2f}) 😱",
                )
                del context.bot_data[key]
        time.sleep(60)

# ============================
# AI IMAGE ANALYSIS SYSTEM (Using a Trained Model)
# ============================
class TradingChartAnalyzer:
    def __init__(self):
        self.model = self._load_ai_model()
        self.classes = [
            "Head and Shoulders",
            "Double Top/Bottom",
            "Bullish/Bearish Flag",
            "Support/Resistance",
            "Triangle Pattern",
            "No Clear Pattern",
        ]

    def _load_ai_model(self):
        try:
            model = tf.keras.models.load_model("models/chart_pattern_cnn.h5")
            return model
        except Exception as e:
            logging.error(f"Model loading error: {str(e)}")
            return None

    def analyze_screenshot(self, image_path: str) -> dict:
        try:
            img = self._preprocess_image(image_path)
            if self.model is None:
                return {"error": "AI model not available 😔"}
            predictions = self.model.predict(np.array([img]))
            primary_index = np.argmax(predictions[0])
            return {
                "primary_pattern": self.classes[primary_index],
                "confidence": float(np.max(predictions[0])),
                "secondary_patterns": self._get_secondary_patterns(predictions[0]),
            }
        except Exception as e:
            logging.error(f"Image analysis failed: {str(e)}")
            return {"error": "Analysis failed 😢"}

    def _preprocess_image(self, image_path: str) -> np.array:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self._auto_crop_chart(img)
        img = cv2.resize(img, (256, 256))
        img = img / 255.0
        return img

    def _auto_crop_chart(self, img: np.array) -> np.array:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            return img[y:y+h, x:x+w]
        return img

    def _get_secondary_patterns(self, prediction_array: np.array) -> list:
        indices = np.argsort(prediction_array)[::-1]
        secondary = [self.classes[i] for i in indices[1:3]] if len(indices) >= 3 else []
        return secondary

# ============================
# UPDATED BOT CLASS
# ============================
class HardlyTradeAnalystBot:
    def __init__(self):
        self.analyzer = HedgeFundGradeAnalyzer()
        self.chart_analyzer = TradingChartAnalyzer()
        self.portfolio = PortfolioManager()
        self.alerts = {}

    def generate_report(self, symbol: str) -> str:
        analysis = self.analyzer.full_analysis(symbol)
        report = f"""
📈 *{symbol} Professional Analysis* 📉

⏰ *Timeframe Consensus*:
- Short-term (1H): {analysis['1h']['trend']}
- Medium-term (4H): {analysis['4h']['trend']}
- Long-term (1D): {analysis['1d']['trend']}

🎯 *Key Levels*:
- Support: ${analysis['consensus']['support']:,.2f}
- Resistance: ${analysis['consensus']['resistance']:,.2f}
- Pivot Point: ${analysis['consensus']['pivot']:,.2f}

💹 *Momentum*:
- RSI: {analysis['consensus']['rsi']:.1f} ({'Overbought' if analysis['consensus']['rsi'] > 70 else 'Oversold'})
- MACD: {'Bullish' if analysis['consensus']['macd'] > 0 else 'Bearish'}

📊 *Volume Analysis*:
- Recent Volume Spike: {'Yes' if analysis['consensus']['volume_spike'] else 'No'}
- Volume Trend: {analysis['consensus']['volume_trend']}

🔔 *Professional Recommendation*:
{self._generate_recommendation(analysis)}
"""
        return report

    def _generate_recommendation(self, analysis: dict) -> str:
        score = 0
        if analysis["consensus"]["trend"] == "🚀 Bullish":
            score += 2
        if analysis["consensus"]["rsi"] < 30:
            score += 1.5
        if analysis["consensus"]["volume_spike"]:
            score += 1

        if score >= 4:
            return "💪 STRONG BUY 🚀 (Multiple confluence factors)"
        elif score >= 2.5:
            return "👍 Moderate Buy (Positive market structure)"
        elif score <= 1:
            return "🤔 Neutral/Hold (Wait for confirmation)"
        return "⚖️ Consider Partial Profit Taking"

    def generate_image_report(self, analysis: dict) -> str:
        if "error" in analysis:
            return f"⚠️ Professional Analysis Error: {analysis['error']}"
        return f"""
📊 *Professional Chart Analysis* 📊

🔍 *Detected Patterns*:
- Primary Pattern: {analysis['primary_pattern']} ({analysis['confidence']:.1%} confidence)
- Secondary Indications: {', '.join(analysis['secondary_patterns'][:2])}

💡 *Institutional Interpretation*:
{self._get_pattern_interpretation(analysis['primary_pattern'])}

📈 *Recommended Action*:
{self._get_pattern_recommendation(analysis['primary_pattern'])}
"""

    def _get_pattern_interpretation(self, pattern: str) -> str:
        interpretations = {
            "Head and Shoulders": "Classic reversal pattern suggesting trend exhaustion 🔄",
            "Double Top/Bottom": "Strong reversal signal at key price levels 🚨",
            "Bullish/Bearish Flag": "Continuation pattern indicating a pause in trend ⏸️",
            "Support/Resistance": "Key price zone with significant order flow 💼",
            "Triangle Pattern": "Volatility contraction before breakout 🔺",
        }
        return interpretations.get(pattern, "No clear institutional pattern detected 🤷‍♂️")

    def _get_pattern_recommendation(self, pattern: str) -> str:
        recommendations = {
            "Head and Shoulders": "Watch for reversal signals and consider a cautious entry. 📉",
            "Double Top/Bottom": "Evaluate risk before entry—possible reversal ahead. 🔄",
            "Bullish/Bearish Flag": "Align with the prevailing trend for a continuation move. 📈",
            "Support/Resistance": "Set your orders around these key levels. 🎯",
            "Triangle Pattern": "Prepare for a breakout; consider scaling in positions. 🚀",
        }
        return recommendations.get(pattern, "No clear recommendation available. 🤔")

# ============================
# TELEGRAM HANDLER FOR IMAGE MESSAGES
# ============================
async def handle_screenshot(update: Update, context: CallbackContext):
    try:
        photo_file = await update.message.photo[-1].get_file()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
            temp_path = temp.name
        await photo_file.download(custom_path=temp_path)
        
        bot_instance = HardlyTradeAnalystBot()
        analysis = bot_instance.chart_analyzer.analyze_screenshot(temp_path)
        report = bot_instance.generate_image_report(analysis)
        await update.message.reply_markdown_v2(report)
    except Exception as e:
        await update.message.reply_text(f"🚨 Professional Analysis Failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ============================
# MAIN SETUP
# ============================
def main():
    from telegram.ext import ApplicationBuilder
    application = ApplicationBuilder().token("7277532789:AAGS5v9K6if3ZrLen8fa2ABRovn25Sazpk8").build()

    # Register command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(CommandHandler("portfolio", portfolio_command))
    application.add_handler(CommandHandler("alert", alert_handler))
    
    # Register the photo handler for image analysis
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, handle_screenshot))
    
    # Start the bot
    application.run_polling()

if __name__ == "__main__":
    main()
