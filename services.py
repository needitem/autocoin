import os
import time
import ccxt
import datetime
import numpy as np
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()
UPBIT_API_KEY = os.getenv("UPBIT_API_KEY")
UPBIT_SECRET_KEY = os.getenv("UPBIT_SECRET_KEY")

def get_upbit_candles(market: str = "KRW-BTC", count: int = 10) -> list:
    exchange = ccxt.upbit()
    exchange.apiKey = UPBIT_API_KEY
    exchange.secret = UPBIT_SECRET_KEY
    candles_raw = exchange.fetch_ohlcv(market, timeframe='1d', limit=count)
    converted = []
    for c in candles_raw:
        converted.append({
            "timestamp": c[0],
            "opening_price": c[1],
            "high_price": c[2],
            "low_price": c[3],
            "trade_price": c[4],
            "candle_acc_trade_volume": c[5]
        })
    return converted


def fetch_btc_advanced_data() -> Dict[str, Any]:
    exchange = ccxt.upbit()
    exchange.apiKey = UPBIT_API_KEY
    exchange.secret = UPBIT_SECRET_KEY
    ticker_data = exchange.fetch_ticker("BTC/KRW")
    return {
        "last_price": ticker_data.get("last"),
        "high_24h": ticker_data.get("high"),
        "low_24h": ticker_data.get("low"),
        "volume_base": ticker_data.get("baseVolume")
    }


def fetch_binance_advanced_data() -> Dict[str, Any]:
    exchange = ccxt.binance()
    ticker_data = exchange.fetch_ticker("BTC/USDT")
    return {
        "last_price": ticker_data.get("last"),
        "high_24h": ticker_data.get("high"),
        "low_24h": ticker_data.get("low"),
        "volume_base": ticker_data.get("baseVolume"),
        "volume_quote": ticker_data.get("quoteVolume")
    }


def get_upbit_usdkrw_rate() -> float:
    exchange = ccxt.upbit()
    exchange.apiKey = UPBIT_API_KEY
    exchange.secret = UPBIT_SECRET_KEY
    # "USDT/KRW" is valid on Upbit in ccxt
    ticker_data = exchange.fetch_ticker("USDT/KRW")
    return ticker_data["last"] 