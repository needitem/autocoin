import time
from datetime import datetime
from fear_greed import calculate_fear_greed_for_symbol, get_available_markets, get_candles, calculate_returns_and_volatility, FearGreedIndex
from latest_news import get_latest_news, print_analysis, NewsDatabase, print_comprehensive_analysis, KeywordManager
from market_analysis import print_market_analysis, analyze_chart, analyze_orderbook, get_orderbook
import os
from dotenv import load_dotenv

load_dotenv()

# 코인별 키워드 정의
COIN_KEYWORDS = {
    "BTC": {
        "names": ["비트코인", "bitcoin", "btc", "비트"],
        "tech": ["사토시", "satoshi", "채굴", "반감기", "비트코인 코어"],
        "related": ["마이크로스트레티지", "grayscale", "gbtc", "비트코인 etf", "채굴업체", "antpool", "f2pool"]
    },
    "ETH": {
        "names": ["이더리움", "ethereum", "eth", "이더"],
        "tech": ["비탈릭", "vitalik", "가스비", "스마트컨트랙트", "이더리움 2.0", "pos", "지분증명", "샤딩"],
        "related": ["메타마스크", "인피니티", "폴리곤", "레이어2", "옵티미즘", "아비트럼"]
    },
    "XRP": {
        "names": ["리플", "ripple", "xrp"],
        "tech": ["송금", "remittance", "odl", "on-demand liquidity"],
        "related": ["리플랩스", "ripple labs", "브래드 갈링하우스", "sec 소송", "swift"]
    },
    "SOL": {
        "names": ["솔라나", "solana", "sol"],
        "tech": ["proof of history", "poh", "걸프스트림", "sealevel"],
        "related": ["serum", "세럼", "ftx", "매직에덴", "솔라나 생태계"]
    },
    "ADA": {
        "names": ["카르다노", "cardano", "ada"],
        "tech": ["우로보로스", "하스켈", "하드포크", "바실", "알론조"],
        "related": ["호스킨슨", "hoskinson", "iohk", "에이다", "하이드라"]
    },
    "DOGE": {
        "names": ["도지", "dogecoin", "doge"],
        "tech": ["스크립트", "라이트코인", "머지마이닝"],
        "related": ["머스크", "musk", "밈코인", "meme", "일론", "테슬라"]
    },
    "DOT": {
        "names": ["폴카닷", "polkadot", "dot"],
        "tech": ["파라체인", "크로스체인", "기판", "substrate"],
        "related": ["가빈우드", "gavin wood", "kusama", "쿠사마", "웹3재단"]
    },
    "MATIC": {
        "names": ["폴리곤", "polygon", "matic"],
        "tech": ["레이어2", "layer2", "사이드체인", "플라즈마"],
        "related": ["폴리곤 pos", "zk롤업", "나이트폴", "hermez"]
    },
    "LINK": {
        "names": ["체인링크", "chainlink", "link"],
        "tech": ["오라클", "oracle", "노드운영", "keepers"],
        "related": ["스마트컨트랙트", "디파이", "sergey nazarov", "link marines"]
    },
    "UNI": {
        "names": ["유니스왑", "uniswap", "uni"],
        "tech": ["덱스", "dex", "amm", "자동마켓메이커"],
        "related": ["hayden adams", "v2", "v3", "유동성풀", "lp토큰"]
    }
}

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
    text = text.lower()
    coin_info = COIN_KEYWORDS.get(coin_symbol)
    if not coin_info:
        return False
        
    # 코인 이름 관련 키워드 매칭
    name_match = any(keyword in text for keyword in coin_info['names'])
    if not name_match:
        return False
        
    # 기술/생태계 관련 키워드 매칭
    tech_match = any(keyword in text for keyword in coin_info['tech'])
    related_match = any(keyword in text for keyword in coin_info['related'])
    
    # 코인 이름이 있고, 기술이나 생태계 관련 내용이 있으면 관련성 높음
    return name_match and (tech_match or related_match)

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
