import time
from datetime import datetime
from fear_greed import calculate_fear_greed_for_symbol, get_available_markets, get_candles, calculate_returns_and_volatility, FearGreedIndex
from latest_news import get_latest_news, print_analysis, NewsDatabase, print_comprehensive_analysis, KeywordManager
from market_analysis import print_market_analysis, analyze_chart, analyze_orderbook, get_orderbook
import os
from dotenv import load_dotenv

load_dotenv()

def get_coin_symbol(market: str) -> str:
    """
    마켓 코드에서 코인 심볼을 추출합니다.
    예: KRW-BTC -> BTC
    """
    return market.split('-')[1]

def is_relevant_to_coin(text: str, coin_symbol: str) -> bool:
    """
    텍스트가 해당 코인과 관련이 있는지 판단합니다.
    """
    keyword_manager = KeywordManager()
    return keyword_manager.is_relevant_to_coin(text, coin_symbol)

def print_market_suggestion(symbol: str):
    """
    종합적인 시장 분석을 출력합니다.
    """
    print(f"\n=== {symbol} 시장 분석 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===")
    
    # 캔들 데이터 가져오기
    candles = get_candles(symbol)
    if not candles:
        print("데이터를 가져올 수 없습니다.")
        return
        
    # 호가 데이터 가져오기
    coin_symbol = get_coin_symbol(symbol)
    orderbook = get_orderbook(coin_symbol)
    if not orderbook:
        print("호가 데이터를 가져올 수 없습니다.")
        return
        
    current_price = float(orderbook['orderbook_units'][0]['ask_price'])
    
    # 공포탐욕지수 계산
    prices, volumes = calculate_returns_and_volatility(candles)
    fg_index = FearGreedIndex()
    fg_result = fg_index.calculate(prices, volumes)
    
    # 뉴스 분석 (latest_news.py의 기능 활용)
    news = get_latest_news(coin_symbol)
    if news:
        print_analysis(news, coin_symbol)
    
    # 공포탐욕지수 기반 시장 심리 분석
    print(f"\n[시장 심리 분석]")
    index = fg_result['fear_greed_index'] * 100
    if index <= 20:
        print("💡 극단적 공포 상태: 시장 바닥 형성 가능성")
    elif index <= 35:
        print("💡 공포 상태: 저점 매수 기회 모색 구간")
    elif index >= 80:
        print("💡 극단적 탐욕 상태: 시장 과열 구간")
    elif index >= 65:
        print("💡 탐욕 상태: 고점 형성 가능성")
    else:
        print("💡 중립 상태: 기술적 분석 중요")
    
    print("\n⚠️ 주의: 이 분석은 참고용이며, 실제 투자는 더 다양한 지표와 함께 종합적으로 판단하시기 바랍니다.")

def main():
    try:
        # 뉴스 데이터베이스 초기화
        news_db = NewsDatabase()
        keyword_manager = KeywordManager()
        
        # 코인 목록 설정
        coins = ["BTC", "ETH", "XRP", "SOL", "ADA"]  # 예시 코인 목록
        
        while True:
            for coin in coins:
                try:
                    print(f"\n{'='*30} {coin} 뉴스 분석 {'='*30}")
                    
                    # 뉴스 수집
                    news_items = news_db.get_latest_news(coin)
                    if not news_items:
                        print(f"{coin}에 대한 뉴스를 찾을 수 없습니다.")
                        continue
                    
                    # AI 기반 종합 분석 실행
                    print_comprehensive_analysis(news_items, coin)
                    
                    # 각 뉴스 텍스트에 대한 키워드 분석
                    for item in news_items[:5]:  # 상위 5개 뉴스만 분석
                        text = f"{item['title']} {item.get('content', '')}"
                        keywords = keyword_manager.analyze_keywords_with_ai(text, coin)
                        
                        if keywords:
                            print(f"\n📝 뉴스 키워드 분석 결과 ({item['title'][:50]}...):")
                            for category, words in keywords.items():
                                if words:
                                    print(f"\n{category.upper()} 키워드:")
                                    for word_info in words:
                                        print(f"- {word_info['keyword']}")
                                        print(f"  중요도: {word_info['importance']}")
                                        print(f"  관련성: {word_info['relevance']}")
                                        print(f"  모니터링: {word_info['monitoring']}")
                    
                except Exception as e:
                    print(f"{coin} 분석 중 오류 발생: {str(e)}")
                    continue
                
                # 분석 간격 (5분)
                time.sleep(300)
            
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
    except Exception as e:
        print(f"프로그램 실행 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
