#!/usr/bin/env python3
"""
AutoCoin Trading Bot 실행 스크립트
"""

import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """메인 함수"""
    print("AutoCoin Trading Bot 시작...")
    print("=" * 50)
    
    try:
        # Streamlit 없이 기본 기능 테스트
        from src.api.upbit import UpbitTradingSystem
        from src.api.bithumb import BithumbAPI
        from src.core.trading import TradingManager
        from src.db.database import DatabaseManager
        
        # 거래소 선택
        print("1. 거래소 선택")
        print("   1) Upbit")
        print("   2) Bithumb")
        choice = input("   거래소를 선택하세요 (1 또는 2, 기본값: 1): ").strip() or "1"
        
        # 시스템 초기화
        print("\n2. 시스템 초기화 중...")
        if choice == "2":
            exchange_name = "bithumb"
            trading_manager = TradingManager(exchange=exchange_name, verbose=True)
        else:
            exchange_name = "upbit"
            trading_manager = TradingManager(exchange=exchange_name, verbose=True)
        
        print(f"   - 선택된 거래소: {trading_manager.get_current_exchange()}")
        
        # 마켓 목록 가져오기
        print("\n3. 마켓 목록 조회 중...")
        markets = trading_manager.get_markets()
        print(f"   - 사용 가능한 KRW 마켓 수: {len(markets)}")
        
        # 주요 마켓 표시
        print("\n4. 주요 마켓:")
        for i, market in enumerate(markets[:5]):
            print(f"   - {market['market']}: {market.get('korean_name', 'N/A')}")
        
        # BTC 현재가 조회
        print("\n5. BTC 현재가 조회 중...")
        btc_data = trading_manager.get_market_data('KRW-BTC')
        if btc_data:
            print(f"   - 현재가: {btc_data.get('trade_price', 0):,} KRW")
            print(f"   - 변동률: {btc_data.get('signed_change_rate', 0)*100:.2f}%")
            print(f"   - 24시간 거래량: {btc_data.get('acc_trade_volume_24h', 0):,.4f} BTC")
        
        # 기술적 지표 계산
        print("\n6. 기술적 지표 계산 중...")
        ohlcv = trading_manager.get_ohlcv('KRW-BTC', count=100)
        if ohlcv is not None and not ohlcv.empty:
            indicators = trading_manager.calculate_indicators(ohlcv)
            if indicators and 'rsi' in indicators:
                latest_rsi = indicators['rsi'].iloc[-1] if not indicators['rsi'].empty else 50
                print(f"   - RSI: {latest_rsi:.2f}")
            
            # 매매 신호 분석
            from src.core.strategy import Strategy
            strategy = Strategy()
            signal = strategy.analyze(ohlcv)
            print(f"   - 매매 신호: {signal}")
        
        # 거래소 전환 테스트
        print("\n7. 거래소 전환 테스트")
        other_exchange = "bithumb" if exchange_name == "upbit" else "upbit"
        print(f"   - {other_exchange}로 전환 중...")
        if trading_manager.switch_exchange(other_exchange):
            print(f"   - 성공: 현재 거래소는 {trading_manager.get_current_exchange()}입니다")
            # 원래 거래소로 복귀
            trading_manager.switch_exchange(exchange_name)
        
        print("\n8. 시스템 준비 완료!")
        print("=" * 50)
        print("\nStreamlit UI를 실행하려면 다음 명령어를 사용하세요:")
        print("streamlit run app.py")
        
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())