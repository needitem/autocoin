import time
import datetime
import numpy as np
import threading

from typing import Dict, Any
from services import get_upbit_candles, fetch_btc_advanced_data, fetch_binance_advanced_data, get_upbit_usdkrw_rate
from ai_strategy import analyze_market_with_ai, ai_self_reflection, apply_custom_strategy, analyze_price_target_with_ai
from data_logging import fetch_latest_news, log_correlation_to_csv
from fear_greed import get_fear_greed_data

# Store price history for correlation usage
price_history_upbit = []
price_history_binance = []
volume_history_upbit = []
volume_history_binance = []

def place_order_on_upbit(decision: str, market: str = "KRW-BTC", amount_krw: float = 10000) -> Dict[str, Any]:
    import ccxt
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    UPBIT_API_KEY = os.getenv("UPBIT_API_KEY")
    UPBIT_SECRET_KEY = os.getenv("UPBIT_SECRET_KEY")
    
    exchange = ccxt.upbit()
    exchange.apiKey = UPBIT_API_KEY
    exchange.secret = UPBIT_SECRET_KEY
    ccxt_market = market.replace("KRW-", "") + "/KRW"

    side = "buy" if decision == "매수" else "sell" if decision == "매도" else ""
    if not side:
        return {"error": "No order placed, decision was 보류 or invalid."}

    order_response = exchange.create_order(
        symbol=ccxt_market,
        type='market',
        side=side,
        amount=amount_krw / 10000.0
    )

    print("Upbit 주문 (ccxt) 예시:", order_response)
    return {
        "market": market,
        "action": decision,
        "amount_krw": amount_krw,
        "timestamp": str(datetime.datetime.now()),
        "ccxt_response": order_response
    }


def save_trade_record_to_db(trade_info: dict):
    """
    Dummy DB save function for demonstration
    """
    pass


def run_auto_trading_once(market="KRW-BTC"):
    candles = get_upbit_candles(market=market, count=5)
    decision = analyze_market_with_ai(candles, 0.0, [])  # or pass real data
    if decision in ["매수", "買", "BUY"]:
        place_order_on_upbit("매수", market, 10000)
    elif decision in ["매도", "賣", "SELL"]:
        place_order_on_upbit("매도", market, 10000)
    else:
        print("AI 결정:", decision, "→ 매매 안 함")


def run_continuous_trading(market="KRW-BTC"):
    while True:
        candles = get_upbit_candles(market=market, count=5)
        ma_value = np.mean([c["trade_price"] for c in candles]) if candles else 0
        print("단순이동평균:", ma_value)

        # Use the real Fear & Greed data from fear_greed.py
        fng = get_fear_greed_data()
        news_list = fetch_latest_news()

        basic_decision = analyze_market_with_ai(candles, fng, news_list)
        print("AI 단순 판단 결과:", basic_decision)

        strategy_decision = apply_custom_strategy(candles)
        print("워뇨띠 전략 판단 결과:", strategy_decision)
        final_decision = strategy_decision

        if final_decision in ["매수", "매도"]:
            trade_info = place_order_on_upbit(final_decision, "KRW-BTC", 10000)
            save_trade_record_to_db(trade_info)
        else:
            print("보류 - 거래 없음")

        improved_strategy = ai_self_reflection("기존 전략", fng)
        print("회고 결과:", improved_strategy)

        advanced_data = fetch_btc_advanced_data()
        print("[Upbit Advanced]", advanced_data)

        # Track Upbit price and volume
        upbit_price = advanced_data['last_price']
        upbit_volume = advanced_data['volume_base']
        price_history_upbit.append(upbit_price)
        volume_history_upbit.append(upbit_volume)

        # Fetch Binance data
        binance_data = fetch_binance_advanced_data()
        print("[Binance Advanced]", binance_data)

        binance_usd_price = binance_data['last_price']
        binance_usd_volume = binance_data['volume_base']  # baseVolume in BTC or USDT, depending on your fetch
        
        # 1) Grab Upbit’s USDT/KRW rate
        upbit_usdkrw = get_upbit_usdkrw_rate()
        
        # 2) Convert Binance price in USD to KRW
        binance_krw_price = binance_usd_price * upbit_usdkrw
        
        # 3) If you wish to convert volume to KRW as well (BTC volume → USD → KRW):
        binance_krw_volume = binance_usd_volume * binance_usd_price * upbit_usdkrw
        
        # Now store these in your arrays for UI
        price_history_binance.append(binance_krw_price)
        volume_history_binance.append(binance_krw_volume)

        if len(price_history_upbit) >= 2:
            corr_up_bn_price = np.corrcoef(price_history_upbit, price_history_binance)[0,1]
            corr_up_bn_vol = 0.0
            corr_bn_price_bn_vol = 0.0
            log_correlation_to_csv(
                time.time(),
                corr_up_bn_price,
                corr_up_bn_vol,
                corr_bn_price_bn_vol
            )

        # 업비트, 바이낸스 데이터 수집
        upbit_data = fetch_btc_advanced_data()     # { "last_price": ..., "volume_base": ... }
        binance_data = fetch_binance_advanced_data()
        fng = get_fear_greed_data()
        news_list = fetch_latest_news()

        # AI에게 매수/매도가 제안 받기
        ai_suggestion = analyze_price_target_with_ai(upbit_data, binance_data, fng, news_list)
        print("AI 매수/매도가 제안:", ai_suggestion)

        # 이후 ai_suggestion을 파싱하여 주문 로직 등을 구현
        # ...

        time.sleep(60)  # 1분 주기 