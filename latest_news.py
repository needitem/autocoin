import requests
from dotenv import load_dotenv
import os
from textblob import TextBlob
import re
import time
from datetime import datetime, timezone, timedelta
import json
from typing import Dict, List, Set, Tuple
from collections import Counter
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from database import Database
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_news.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NewsAnalyzer:
    def __init__(self):
        """뉴스 분석기 초기화"""
        self.db = Database()
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.initialize_ai()
        
    def initialize_ai(self):
        """AI 모델 초기화"""
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.model = genai.GenerativeModel('gemini-pro')
            logger.info("AI 모델 초기화 완료")
        except Exception as e:
            logger.error(f"AI 모델 초기화 실패: {str(e)}")
            self.model = None

    def get_recent_news(self, symbol: str, hours: int = 4) -> List[Dict]:
        """최근 뉴스 가져오기"""
        try:
            # 시간 범위 설정
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours)
            
            # NewsAPI 요청
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'({symbol} OR cryptocurrency OR crypto) AND ({self._get_symbol_keywords(symbol)})',
                'from': start_time.isoformat(),
                'to': end_time.isoformat(),
                'language': 'en',
                'sortBy': 'publishedAt',
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            news_data = response.json()
            
            if news_data['status'] != 'ok':
                logger.error(f"뉴스 API 오류: {news_data.get('message', '알 수 없는 오류')}")
                return []
                
            # 뉴스 데이터 처리
            processed_news = []
            for article in news_data.get('articles', []):
                news_item = {
                    'coin_symbol': symbol,
                    'title': article['title'],
                    'content': article.get('description', ''),
                    'source': article['source']['name'],
                    'url': article['url'],
                    'published_at': article['publishedAt']
                }
                
                # 감성 분석 및 호재/악재 판단
                sentiment_results = self.analyze_sentiment(f"{news_item['title']} {news_item['content']}")
                news_item.update(sentiment_results)
                
                # 중요도 점수 계산
                news_item['importance_score'] = self.calculate_importance_score(news_item)
                
                processed_news.append(news_item)
                
            return processed_news
            
        except Exception as e:
            logger.error(f"뉴스 가져오기 실패: {str(e)}")
            return []

    def _get_symbol_keywords(self, symbol: str) -> str:
        """심볼 관련 키워드 생성"""
        keywords = {
            'BTC': 'bitcoin OR btc OR "비트코인"',
            'ETH': 'ethereum OR eth OR "이더리움"',
            'XRP': 'ripple OR xrp OR "리플"',
            'SOL': 'solana OR sol',
            'ADA': 'cardano OR ada',
            'DOGE': 'dogecoin OR doge',
            'DOT': 'polkadot OR dot',
            'MATIC': 'polygon OR matic',
            'LINK': 'chainlink OR link',
            'UNI': 'uniswap OR uni'
        }
        return keywords.get(symbol, symbol)

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """감성 분석 및 호재/악재 판단"""
        try:
            # 기본 감성 분석
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # AI 기반 상세 분석
            if self.model:
                response = self.model.generate_content(
                    f"""다음 암호화폐 관련 뉴스의 호재/악재 여부를 분석해주세요:
                    
                    {text}
                    
                    다음 항목들을 0부터 1 사이의 숫자로 평가해주세요:
                    1. 호재/악재 점수 (1: 매우 긍정적, 0: 매우 부정적)
                    2. 시장 영향도 (1: 매우 큰 영향, 0: 영향 없음)
                    3. 신뢰도 (1: 매우 신뢰할 만함, 0: 신뢰하기 어려움)
                    4. 시급성 (1: 매우 시급, 0: 시급하지 않음)
                    
                    또한, 이 뉴스가 호재인지 악재인지 판단하고, 그 이유를 간단히 설명해주세요.
                    """
                )
                
                # AI 응답 파싱
                ai_scores = self._parse_ai_response(response.text)
            else:
                ai_scores = {
                    'sentiment_score': (polarity + 1) / 2,  # -1~1 범위를 0~1로 변환
                    'market_impact': 0.5,
                    'reliability': 0.5,
                    'urgency': 0.5,
                    'is_positive': polarity > 0,
                    'analysis': "AI 분석을 사용할 수 없습니다."
                }
            
            return ai_scores
            
        except Exception as e:
            logger.error(f"감성 분석 실패: {str(e)}")
            return {
                'sentiment_score': 0.5,
                'market_impact': 0.5,
                'reliability': 0.5,
                'urgency': 0.5,
                'is_positive': None,
                'analysis': "분석 실패"
            }

    def _parse_ai_response(self, response: str) -> Dict:
        """AI 응답 파싱"""
        try:
            scores = {
                'sentiment_score': 0.5,
                'market_impact': 0.5,
                'reliability': 0.5,
                'urgency': 0.5,
                'is_positive': None,
                'analysis': ""
            }
            
            # 점수 파싱
            lines = response.split('\n')
            for line in lines:
                if ':' in line:
                    key, value = line.split(':')
                    try:
                        value = float(re.search(r'[\d.]+', value.strip()).group())
                        if '호재/악재' in key:
                            scores['sentiment_score'] = value
                            scores['is_positive'] = value > 0.5
                        elif '시장 영향도' in key:
                            scores['market_impact'] = value
                        elif '신뢰도' in key:
                            scores['reliability'] = value
                        elif '시급성' in key:
                            scores['urgency'] = value
                    except:
                        continue
                        
            # 분석 내용 파싱
            analysis_start = response.find("호재인지 악재인지")
            if analysis_start != -1:
                scores['analysis'] = response[analysis_start:].strip()
                
            return scores
            
        except Exception as e:
            logger.error(f"AI 응답 파싱 실패: {str(e)}")
            return {
                'sentiment_score': 0.5,
                'market_impact': 0.5,
                'reliability': 0.5,
                'urgency': 0.5,
                'is_positive': None,
                'analysis': "파싱 실패"
            }

    def calculate_importance_score(self, news_item: Dict) -> float:
        """뉴스 중요도 점수 계산"""
        try:
            weights = {
                'sentiment_score': 0.3,
                'market_impact': 0.3,
                'reliability': 0.2,
                'urgency': 0.2
            }
            
            # 시간 가중치 (최근 뉴스일수록 높은 가중치)
            published_time = datetime.fromisoformat(news_item['published_at'].replace('Z', '+00:00'))
            time_diff = datetime.now(timezone.utc) - published_time
            time_weight = max(0.5, 1 - (time_diff.total_seconds() / (4 * 3600)))  # 4시간 기준
            
            # 점수 계산
            base_score = sum(
                news_item.get(key, 0.5) * weight 
                for key, weight in weights.items()
            )
            
            return base_score * time_weight
            
        except Exception as e:
            logger.error(f"중요도 점수 계산 실패: {str(e)}")
            return 0.5

def print_news_analysis(symbol: str):
    """뉴스 분석 결과 출력"""
    analyzer = NewsAnalyzer()
    news_items = analyzer.get_recent_news(symbol)
    
    if not news_items:
        print(f"\n{symbol}에 대한 최근 뉴스를 찾을 수 없습니다.")
        return
        
    print(f"\n=== {symbol} 뉴스 분석 결과 ===")
    print(f"분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"분석된 뉴스 수: {len(news_items)}")
    
    # 호재/악재 분류
    positive_news = [item for item in news_items if item.get('is_positive')]
    negative_news = [item for item in news_items if item.get('is_positive') is False]
    
    # 호재 출력
    if positive_news:
        print("\n[호재 뉴스]")
        for item in sorted(positive_news, key=lambda x: x['importance_score'], reverse=True):
            print(f"\n📈 {item['title']}")
            print(f"출처: {item['source']}")
            print(f"시간: {datetime.fromisoformat(item['published_at'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')}")
            print(f"중요도: {item['importance_score']:.2f}")
            print(f"시장 영향도: {item['market_impact']:.2f}")
            print(f"분석: {item['analysis']}")
            
    # 악재 출력
    if negative_news:
        print("\n[악재 뉴스]")
        for item in sorted(negative_news, key=lambda x: x['importance_score'], reverse=True):
            print(f"\n📉 {item['title']}")
            print(f"출처: {item['source']}")
            print(f"시간: {datetime.fromisoformat(item['published_at'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')}")
            print(f"중요도: {item['importance_score']:.2f}")
            print(f"시장 영향도: {item['market_impact']:.2f}")
            print(f"분석: {item['analysis']}")
            
    # 종합 분석
    total_sentiment = sum(item['sentiment_score'] for item in news_items) / len(news_items)
    total_impact = sum(item['market_impact'] for item in news_items) / len(news_items)
    
    print("\n[종합 분석]")
    print(f"전체 뉴스 감성 지수: {total_sentiment:.2f}")
    print(f"평균 시장 영향도: {total_impact:.2f}")
    
    if total_sentiment > 0.6:
        print("💹 전반적으로 긍정적인 뉴스가 우세합니다.")
    elif total_sentiment < 0.4:
        print("📉 전반적으로 부정적인 뉴스가 우세합니다.")
    else:
        print("📊 긍정적/부정적 뉴스가 균형을 이루고 있습니다.")
        
    if total_impact > 0.7:
        print("⚠️ 시장에 큰 영향을 미칠 수 있는 뉴스들이 많습니다.")
    
    print("\n⚠️ 주의: 이 분석은 참고용이며, 실제 투자는 더 다양한 지표와 함께 종합적으로 판단하시기 바랍니다.")

if __name__ == "__main__":
    symbol = input("코인 심볼을 입력하세요 (예: BTC, ETH, XRP): ").strip().upper()
    print_news_analysis(symbol)
