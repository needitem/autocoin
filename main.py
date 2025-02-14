import time
from datetime import datetime
from fear_greed import calculate_returns_and_volatility, FearGreedIndex, get_candles
from market_analysis import analyze_chart, analyze_orderbook, get_orderbook
import os
from dotenv import load_dotenv
import pandas as pd
from investment_strategy import InvestmentStrategy

load_dotenv()

def analyze_and_recommend(symbol: str):
    """
    코인의 기술적 분석과 공포탐욕지수를 기반으로 매수/매도 가격을 추천합니다.
    """
    print(f"\n=== {symbol} 종합 분석 및 추천 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===")
    
    # 캔들 데이터 분석
    candles_data = get_candles(f"KRW-{symbol}")
    if not candles_data:
        print("데이터를 가져올 수 없습니다.")
        return
        
    # 캔들 데이터를 DataFrame으로 변환
    df = pd.DataFrame(candles_data)
    df['trade_date'] = pd.to_datetime(df['candle_date_time_kst'])
    df = df.sort_values('trade_date')
    
    # 호가 데이터 분석
    orderbook = get_orderbook(symbol)
    if not orderbook:
        print("호가 데이터를 가져올 수 없습니다.")
        return
    
    # 차트 분석
    chart_analysis = analyze_chart(df)
    
    # MA 데이터 준비
    ma_data = {
        'MA5': chart_analysis['moving_averages']['ma5'],
        'MA20': chart_analysis['moving_averages']['ma20'],
        'MA60': chart_analysis['moving_averages']['ma60']
    }
    
    # 공포탐욕지수 계산
    prices, volumes = calculate_returns_and_volatility(candles_data)
    fg_index = FearGreedIndex()
    fg_result = fg_index.calculate(prices, volumes)
    
    # 호가 분석
    order_analysis = analyze_orderbook(orderbook)
    
    current_price = float(orderbook['orderbook_units'][0]['ask_price'])
    
    # 공포탐욕지수 기반 조정
    fg_index_value = fg_result['fear_greed_index'] * 100
    
    # RSI 기반 조정
    rsi = chart_analysis['rsi']['value']
    
    # 볼린저 밴드 기반 조정
    bb_position = chart_analysis['bollinger']['position']
    bb_upper = chart_analysis['bollinger']['upper']
    bb_lower = chart_analysis['bollinger']['lower']
    
    # 투자 전략 계산
    strategy = InvestmentStrategy()
    volatility = bb_upper / bb_lower - 1
    trend_strength = "강함" if chart_analysis['moving_averages']['trend'] == "상승" else "약함"
    
    recommendation = strategy.get_strategy_recommendation(
        current_price=current_price,
        fg_index=fg_index_value,
        rsi=rsi,
        volatility=volatility * 100,
        trend_strength=trend_strength,
        ma_data=ma_data,
        total_assets=10000000  # 1천만원 기준
    )
    
    # 결과 출력
    print(f"\n현재가: {current_price:,.2f}원")
    print(f"공포탐욕지수: {fg_index_value:.1f}")
    print(f"RSI: {rsi:.1f}")
    
    print("\n[매수 전략]")
    for level in recommendation.entry_levels:
        print(f"- {level.description}")
        print(f"  가격: {level.price:,.0f}원 (투자금액: {level.ratio * recommendation.investment_amount:,.0f}원)")
    
    print("\n[매도 전략]")
    for level in recommendation.exit_levels:
        print(f"- {level.description}")
        print(f"  가격: {level.price:,.0f}원")
    
    print(f"\n손절가: {recommendation.stop_loss:,.0f}원")
    
    print(f"\n[투자 전략 요약]")
    print(f"전략 유형: {recommendation.strategy_type} (신뢰도: {recommendation.confidence_score*100:.1f}%)")
    print(f"리스크 비율: {recommendation.risk_ratio*100:.1f}%")
    print(f"추천 투자 금액: {recommendation.investment_amount:,.0f}원")
    print(f"추천 보유 기간: {recommendation.holding_period}")
    
    print("\n⚠️ 주의: 이 분석은 참고용이며, 실제 투자는 본인의 판단하에 진행하시기 바랍니다.")

if __name__ == "__main__":
    symbol = input("코인 심볼을 입력하세요 (예: BTC, XRP, ETH): ").strip().upper()
    analyze_and_recommend(symbol)
    