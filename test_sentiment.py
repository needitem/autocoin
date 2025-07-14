"""
시장 감정 분석 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.api.news import CryptoNewsAPI
import json

def test_sentiment_analysis():
    """감정 분석 테스트"""
    print("시장 감정 분석 테스트 시작...")
    
    # 뉴스 API 초기화
    news_api = CryptoNewsAPI()
    
    # 테스트 데이터
    test_markets = ['BTC-KRW', 'ETH-KRW', 'XRP-KRW']
    
    for market in test_markets:
        print(f"\n[{market}] 감정 분석:")
        
        # 가상 가격 데이터
        price_data = {
            'signed_change_rate': 0.05,  # 5% 상승
            'acc_trade_volume_24h': 1000000000,  # 10억 거래량
            'trade_price': 50000000  # 5천만원
        }
        
        try:
            # 감정 분석 실행
            result = news_api.get_market_sentiment(market, price_data)
            
            print(f"  감정: {result.get('sentiment', 'unknown')}")
            print(f"  점수: {result.get('score', 0):.3f}")
            print(f"  신뢰도: {result.get('confidence', 0):.1%}")
            print(f"  Fear & Greed Index: {result.get('fear_greed_index', 50):.0f}")
            
            # 주요 시그널 출력
            reasons = result.get('reasons', [])
            if reasons:
                print("  주요 시그널:")
                for reason in reasons[:3]:
                    print(f"    - {reason}")
            
            # 구성 요소 출력
            components = result.get('components', {})
            if components:
                print("  구성 요소:")
                for comp, score in components.items():
                    print(f"    - {comp}: {score:.3f}")
            
        except Exception as e:
            print(f"  오류 발생: {str(e)}")
    
    print("\n테스트 완료!")

if __name__ == "__main__":
    test_sentiment_analysis()