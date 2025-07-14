"""
향상된 시장 감정 분석 모듈
소셜 미디어, 뉴스, 기술적 지표를 종합한 고급 감정 분석
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from collections import Counter
import re

logger = logging.getLogger(__name__)

class SentimentLevel(Enum):
    EXTREME_GREED = "극도의 탐욕"
    GREED = "탐욕"
    NEUTRAL = "중립"
    FEAR = "공포"
    EXTREME_FEAR = "극도의 공포"

@dataclass
class MarketSentiment:
    """시장 감정 분석 결과"""
    overall_score: float  # -1.0 (극도의 공포) ~ 1.0 (극도의 탐욕)
    sentiment_level: SentimentLevel
    confidence: float  # 0.0 ~ 1.0
    components: Dict[str, float]  # 각 요소별 점수
    signals: List[str]  # 주요 시그널
    social_metrics: Dict[str, any]  # 소셜 미디어 지표
    market_indicators: Dict[str, float]  # 시장 지표
    timestamp: datetime
    
class EnhancedSentimentAnalyzer:
    """향상된 시장 감정 분석기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 가중치 설정
        self.weights = {
            'price_momentum': 0.15,
            'volume_analysis': 0.10,
            'volatility': 0.10,
            'news_sentiment': 0.20,
            'social_sentiment': 0.15,
            'fear_greed_index': 0.10,
            'market_dominance': 0.05,
            'institutional_flow': 0.10,
            'technical_indicators': 0.05
        }
        
        # 소셜 미디어 키워드
        self.bullish_keywords = {
            'strong': ['moon', 'bullish', 'pump', 'buy', 'long', 'breakout', 'rally', 'surge', 'explosive'],
            'medium': ['up', 'green', 'positive', 'growth', 'rising', 'gains', 'profit'],
            'weak': ['hope', 'maybe', 'possible', 'could', 'might']
        }
        
        self.bearish_keywords = {
            'strong': ['crash', 'dump', 'sell', 'short', 'bearish', 'collapse', 'plunge', 'panic'],
            'medium': ['down', 'red', 'negative', 'falling', 'decline', 'loss', 'drop'],
            'weak': ['concern', 'worry', 'uncertain', 'risky']
        }
        
    def analyze_comprehensive_sentiment(self, 
                                      market: str,
                                      price_data: Dict,
                                      historical_data: Optional[any] = None,
                                      news_items: Optional[List[Dict]] = None) -> MarketSentiment:
        """종합적인 시장 감정 분석"""
        try:
            components = {}
            signals = []
            
            # 1. 가격 모멘텀 분석
            price_momentum = self._analyze_price_momentum(price_data, historical_data)
            components['price_momentum'] = price_momentum['score']
            if price_momentum['signal']:
                signals.append(price_momentum['signal'])
            
            # 2. 거래량 분석
            volume_analysis = self._analyze_volume_patterns(price_data, historical_data)
            components['volume_analysis'] = volume_analysis['score']
            if volume_analysis['signal']:
                signals.append(volume_analysis['signal'])
            
            # 3. 변동성 분석
            volatility = self._analyze_volatility(price_data, historical_data)
            components['volatility'] = volatility['score']
            if volatility['signal']:
                signals.append(volatility['signal'])
            
            # 4. 뉴스 감정 분석
            if news_items:
                news_sentiment = self._analyze_news_sentiment_advanced(news_items)
                components['news_sentiment'] = news_sentiment['score']
                if news_sentiment['signal']:
                    signals.append(news_sentiment['signal'])
            else:
                components['news_sentiment'] = 0.0
            
            # 5. 소셜 미디어 감정 (시뮬레이션)
            social_sentiment = self._analyze_social_sentiment(market)
            components['social_sentiment'] = social_sentiment['score']
            if social_sentiment['signal']:
                signals.append(social_sentiment['signal'])
            
            # 6. Fear & Greed Index 계산
            fear_greed = self._calculate_fear_greed_index(components)
            components['fear_greed_index'] = fear_greed['score']
            
            # 7. 시장 지배력 분석
            market_dominance = self._analyze_market_dominance(market, price_data)
            components['market_dominance'] = market_dominance['score']
            
            # 8. 기관 자금 흐름 (시뮬레이션)
            institutional_flow = self._analyze_institutional_flow(price_data, historical_data)
            components['institutional_flow'] = institutional_flow['score']
            if institutional_flow['signal']:
                signals.append(institutional_flow['signal'])
            
            # 9. 기술적 지표 종합
            technical_indicators = self._analyze_technical_sentiment(historical_data)
            components['technical_indicators'] = technical_indicators['score']
            
            # 전체 점수 계산
            overall_score = self._calculate_weighted_score(components)
            
            # 감정 레벨 결정
            sentiment_level = self._determine_sentiment_level(overall_score)
            
            # 신뢰도 계산
            confidence = self._calculate_confidence(components, len(news_items or []))
            
            # 시장 지표 수집
            market_indicators = self._collect_market_indicators(
                price_data, historical_data, components
            )
            
            # 소셜 지표 수집
            social_metrics = social_sentiment.get('metrics', {})
            
            return MarketSentiment(
                overall_score=overall_score,
                sentiment_level=sentiment_level,
                confidence=confidence,
                components=components,
                signals=signals[:10],  # 상위 10개 시그널
                social_metrics=social_metrics,
                market_indicators=market_indicators,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"종합 감정 분석 오류: {str(e)}")
            return self._get_default_sentiment()
    
    def _analyze_price_momentum(self, price_data: Dict, historical_data: Optional[any]) -> Dict:
        """가격 모멘텀 분석"""
        try:
            price_change_24h = float(price_data.get('signed_change_rate', 0)) * 100
            score = 0.0
            signal = None
            
            # 24시간 변동률 기반
            if price_change_24h > 15:
                score = 0.9
                signal = "강력한 상승 모멘텀"
            elif price_change_24h > 7:
                score = 0.6
                signal = "상승 모멘텀"
            elif price_change_24h > 2:
                score = 0.3
                signal = "약한 상승세"
            elif price_change_24h < -15:
                score = -0.9
                signal = "강력한 하락 모멘텀"
            elif price_change_24h < -7:
                score = -0.6
                signal = "하락 모멘텀"
            elif price_change_24h < -2:
                score = -0.3
                signal = "약한 하락세"
            else:
                score = 0.0
                signal = "횡보 중"
            
            # 역사적 데이터가 있으면 추가 분석
            if historical_data is not None and len(historical_data) > 50:
                # 이동평균 대비 위치
                ma20 = historical_data['close'].rolling(20).mean().iloc[-1]
                ma50 = historical_data['close'].rolling(50).mean().iloc[-1]
                current_price = historical_data['close'].iloc[-1]
                
                if current_price > ma20 > ma50:
                    score += 0.1
                elif current_price < ma20 < ma50:
                    score -= 0.1
            
            return {
                'score': max(-1.0, min(1.0, score)),
                'signal': signal
            }
            
        except Exception as e:
            self.logger.error(f"가격 모멘텀 분석 오류: {str(e)}")
            return {'score': 0.0, 'signal': None}
    
    def _analyze_volume_patterns(self, price_data: Dict, historical_data: Optional[any]) -> Dict:
        """거래량 패턴 분석"""
        try:
            current_volume = float(price_data.get('acc_trade_volume_24h', 0))
            score = 0.0
            signal = None
            
            if historical_data is not None and len(historical_data) > 7:
                avg_volume = historical_data['volume'].mean()
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                # 거래량 급증
                if volume_ratio > 3:
                    score = 0.8
                    signal = "📊 거래량 폭발적 증가"
                elif volume_ratio > 2:
                    score = 0.5
                    signal = "📈 거래량 크게 증가"
                elif volume_ratio > 1.5:
                    score = 0.3
                    signal = "↗️ 거래량 증가"
                elif volume_ratio < 0.5:
                    score = -0.5
                    signal = "📉 거래량 크게 감소"
                elif volume_ratio < 0.7:
                    score = -0.3
                    signal = "↘️ 거래량 감소"
                else:
                    score = 0.0
                
                # 가격-거래량 상관관계
                price_change = price_data.get('signed_change_rate', 0)
                if price_change > 0 and volume_ratio > 1.5:
                    score += 0.2  # 가격 상승 + 거래량 증가 = 긍정적
                elif price_change < 0 and volume_ratio > 1.5:
                    score -= 0.2  # 가격 하락 + 거래량 증가 = 부정적
            
            return {
                'score': max(-1.0, min(1.0, score)),
                'signal': signal
            }
            
        except Exception as e:
            self.logger.error(f"거래량 패턴 분석 오류: {str(e)}")
            return {'score': 0.0, 'signal': None}
    
    def _analyze_volatility(self, price_data: Dict, historical_data: Optional[any]) -> Dict:
        """변동성 분석"""
        try:
            score = 0.0
            signal = None
            
            if historical_data is not None and len(historical_data) > 20:
                # 실현 변동성 계산
                returns = historical_data['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(1440)  # 일일 변동성
                
                # 변동성에 따른 감정 점수
                if volatility > 0.1:  # 10% 이상
                    score = -0.7
                    signal = "⚡ 극도로 높은 변동성 (위험)"
                elif volatility > 0.07:
                    score = -0.4
                    signal = "⚠️ 높은 변동성"
                elif volatility > 0.04:
                    score = -0.2
                    signal = "📊 보통 변동성"
                elif volatility > 0.02:
                    score = 0.2
                    signal = "✅ 안정적인 변동성"
                else:
                    score = 0.0
                    signal = "😴 매우 낮은 변동성"
                
                # 변동성 추세
                recent_vol = returns[-10:].std() * np.sqrt(1440)
                older_vol = returns[-20:-10].std() * np.sqrt(1440)
                
                if recent_vol > older_vol * 1.5:
                    score -= 0.2
                    signal += " (증가 추세)"
                elif recent_vol < older_vol * 0.7:
                    score += 0.1
                    signal += " (감소 추세)"
            
            return {
                'score': max(-1.0, min(1.0, score)),
                'signal': signal
            }
            
        except Exception as e:
            self.logger.error(f"변동성 분석 오류: {str(e)}")
            return {'score': 0.0, 'signal': None}
    
    def _analyze_news_sentiment_advanced(self, news_items: List[Dict]) -> Dict:
        """고급 뉴스 감정 분석"""
        try:
            if not news_items:
                return {'score': 0.0, 'signal': None}
            
            total_score = 0.0
            keyword_counts = Counter()
            
            for news in news_items:
                text = f"{news.get('title', '')} {news.get('summary', '')}".lower()
                news_score = 0.0
                
                # 긍정적 키워드 분석
                for strength, keywords in self.bullish_keywords.items():
                    for keyword in keywords:
                        if keyword in text:
                            keyword_counts[f"+{keyword}"] += 1
                            if strength == 'strong':
                                news_score += 0.5
                            elif strength == 'medium':
                                news_score += 0.3
                            else:
                                news_score += 0.1
                
                # 부정적 키워드 분석
                for strength, keywords in self.bearish_keywords.items():
                    for keyword in keywords:
                        if keyword in text:
                            keyword_counts[f"-{keyword}"] += 1
                            if strength == 'strong':
                                news_score -= 0.5
                            elif strength == 'medium':
                                news_score -= 0.3
                            else:
                                news_score -= 0.1
                
                # 뉴스 신선도 가중치
                published_at = news.get('published_at', '')
                if published_at:
                    try:
                        pub_time = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                        hours_old = (datetime.now() - pub_time.replace(tzinfo=None)).total_seconds() / 3600
                        
                        if hours_old < 1:
                            news_score *= 1.5  # 1시간 이내 뉴스는 가중치 증가
                        elif hours_old < 6:
                            news_score *= 1.2
                        elif hours_old > 24:
                            news_score *= 0.5  # 24시간 이상은 가중치 감소
                    except:
                        pass
                
                total_score += news_score
            
            # 평균 점수 계산
            avg_score = total_score / len(news_items) if news_items else 0
            
            # 시그널 생성
            signal = None
            top_keywords = keyword_counts.most_common(3)
            if top_keywords:
                keyword_str = ", ".join([f"{k}({v})" for k, v in top_keywords])
                
                if avg_score > 0.3:
                    signal = f"📰 긍정적 뉴스 우세: {keyword_str}"
                elif avg_score < -0.3:
                    signal = f"📰 부정적 뉴스 우세: {keyword_str}"
                else:
                    signal = f"📰 혼재된 뉴스: {keyword_str}"
            
            return {
                'score': max(-1.0, min(1.0, avg_score)),
                'signal': signal
            }
            
        except Exception as e:
            self.logger.error(f"뉴스 감정 분석 오류: {str(e)}")
            return {'score': 0.0, 'signal': None}
    
    def _analyze_social_sentiment(self, market: str) -> Dict:
        """소셜 미디어 감정 분석 (시뮬레이션)"""
        try:
            # 실제 구현시에는 Twitter API, Reddit API 등을 사용
            # 여기서는 시뮬레이션
            
            import random
            
            # 시뮬레이션 데이터
            twitter_mentions = random.randint(1000, 50000)
            reddit_posts = random.randint(50, 500)
            positive_ratio = random.uniform(0.3, 0.7)
            
            # 감정 점수 계산
            score = (positive_ratio - 0.5) * 2  # -1 ~ 1 범위로 정규화
            
            # 멘션 수에 따른 가중치
            if twitter_mentions > 20000:
                score *= 1.2
            elif twitter_mentions < 5000:
                score *= 0.8
            
            signal = None
            if score > 0.5:
                signal = f"🐦 소셜 미디어 매우 긍정적 (멘션: {twitter_mentions:,})"
            elif score > 0.2:
                signal = f"👍 소셜 미디어 긍정적"
            elif score < -0.5:
                signal = f"👎 소셜 미디어 매우 부정적"
            elif score < -0.2:
                signal = f"😟 소셜 미디어 부정적"
            
            return {
                'score': max(-1.0, min(1.0, score)),
                'signal': signal,
                'metrics': {
                    'twitter_mentions': twitter_mentions,
                    'reddit_posts': reddit_posts,
                    'positive_ratio': positive_ratio,
                    'engagement_rate': random.uniform(0.02, 0.15)
                }
            }
            
        except Exception as e:
            self.logger.error(f"소셜 감정 분석 오류: {str(e)}")
            return {'score': 0.0, 'signal': None, 'metrics': {}}
    
    def _calculate_fear_greed_index(self, components: Dict[str, float]) -> Dict:
        """Fear & Greed Index 계산"""
        try:
            # 주요 구성 요소들의 평균
            key_components = ['price_momentum', 'volume_analysis', 'volatility', 'social_sentiment']
            scores = [components.get(comp, 0.0) for comp in key_components]
            
            # 변동성은 반대로 (높은 변동성 = 공포)
            if 'volatility' in components:
                volatility_idx = key_components.index('volatility')
                scores[volatility_idx] = -scores[volatility_idx]
            
            fgi_score = np.mean(scores)
            
            return {
                'score': fgi_score,
                'signal': None
            }
            
        except Exception as e:
            self.logger.error(f"FGI 계산 오류: {str(e)}")
            return {'score': 0.0, 'signal': None}
    
    def _analyze_market_dominance(self, market: str, price_data: Dict) -> Dict:
        """시장 지배력 분석"""
        try:
            # 비트코인 도미넌스 시뮬레이션 (실제로는 API 필요)
            import random
            btc_dominance = random.uniform(40, 70)  # 40-70% 범위
            
            score = 0.0
            if market == 'BTC-KRW':
                # 비트코인 도미넌스가 높으면 긍정적
                if btc_dominance > 60:
                    score = 0.5
                elif btc_dominance > 50:
                    score = 0.2
                else:
                    score = -0.2
            else:
                # 알트코인은 반대
                if btc_dominance < 45:
                    score = 0.5
                elif btc_dominance < 55:
                    score = 0.2
                else:
                    score = -0.2
            
            return {
                'score': score,
                'signal': None,
                'btc_dominance': btc_dominance
            }
            
        except Exception as e:
            self.logger.error(f"시장 지배력 분석 오류: {str(e)}")
            return {'score': 0.0, 'signal': None}
    
    def _analyze_institutional_flow(self, price_data: Dict, historical_data: Optional[any]) -> Dict:
        """기관 자금 흐름 분석"""
        try:
            score = 0.0
            signal = None
            
            # 대량 거래 감지 (시뮬레이션)
            import random
            large_buy_orders = random.randint(0, 20)
            large_sell_orders = random.randint(0, 20)
            
            net_flow = large_buy_orders - large_sell_orders
            
            if net_flow > 10:
                score = 0.8
                signal = f"🏦 기관 대량 매수 감지 (+{net_flow})"
            elif net_flow > 5:
                score = 0.5
                signal = f"💼 기관 매수 우세"
            elif net_flow < -10:
                score = -0.8
                signal = f"🏦 기관 대량 매도 감지 ({net_flow})"
            elif net_flow < -5:
                score = -0.5
                signal = f"💼 기관 매도 우세"
            else:
                score = 0.0
            
            return {
                'score': score,
                'signal': signal
            }
            
        except Exception as e:
            self.logger.error(f"기관 자금 흐름 분석 오류: {str(e)}")
            return {'score': 0.0, 'signal': None}
    
    def _analyze_technical_sentiment(self, historical_data: Optional[any]) -> Dict:
        """기술적 지표 기반 감정 분석"""
        try:
            if historical_data is None or len(historical_data) < 50:
                return {'score': 0.0, 'signal': None}
            
            score = 0.0
            
            # RSI
            closes = historical_data['close'].values
            rsi = self._calculate_rsi(closes)
            if rsi:
                if rsi > 80:
                    score -= 0.5  # 과매수 = 부정적
                elif rsi > 70:
                    score -= 0.3
                elif rsi < 20:
                    score += 0.5  # 과매도 = 긍정적 (반등 기대)
                elif rsi < 30:
                    score += 0.3
            
            # MACD
            macd_score = self._calculate_macd_sentiment(closes)
            score += macd_score * 0.3
            
            return {
                'score': max(-1.0, min(1.0, score)),
                'signal': None
            }
            
        except Exception as e:
            self.logger.error(f"기술적 감정 분석 오류: {str(e)}")
            return {'score': 0.0, 'signal': None}
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> Optional[float]:
        """RSI 계산"""
        try:
            if len(prices) < period + 1:
                return None
            
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception:
            return None
    
    def _calculate_macd_sentiment(self, prices: np.ndarray) -> float:
        """MACD 기반 감정 점수"""
        try:
            if len(prices) < 26:
                return 0.0
            
            # 간단한 EMA 계산
            ema12 = self._calculate_ema(prices, 12)
            ema26 = self._calculate_ema(prices, 26)
            
            if ema12 and ema26:
                macd = ema12 - ema26
                signal = macd * 0.9  # 간소화된 시그널선
                
                if macd > signal:
                    return 0.5  # 상승 신호
                else:
                    return -0.5  # 하락 신호
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_ema(self, data: np.ndarray, period: int) -> Optional[float]:
        """EMA 계산"""
        try:
            if len(data) < period:
                return None
            
            multiplier = 2 / (period + 1)
            ema = np.mean(data[:period])
            
            for price in data[period:]:
                ema = (price * multiplier) + (ema * (1 - multiplier))
            
            return ema
            
        except Exception:
            return None
    
    def _calculate_weighted_score(self, components: Dict[str, float]) -> float:
        """가중 평균 점수 계산"""
        try:
            total_score = 0.0
            total_weight = 0.0
            
            for component, score in components.items():
                if component in self.weights:
                    weight = self.weights[component]
                    total_score += score * weight
                    total_weight += weight
            
            if total_weight > 0:
                return total_score / total_weight
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"가중 점수 계산 오류: {str(e)}")
            return 0.0
    
    def _determine_sentiment_level(self, score: float) -> SentimentLevel:
        """감정 레벨 결정"""
        if score >= 0.7:
            return SentimentLevel.EXTREME_GREED
        elif score >= 0.3:
            return SentimentLevel.GREED
        elif score >= -0.3:
            return SentimentLevel.NEUTRAL
        elif score >= -0.7:
            return SentimentLevel.FEAR
        else:
            return SentimentLevel.EXTREME_FEAR
    
    def _calculate_confidence(self, components: Dict[str, float], news_count: int) -> float:
        """분석 신뢰도 계산"""
        try:
            confidence = 0.5  # 기본 신뢰도
            
            # 데이터 소스 다양성
            active_components = sum(1 for score in components.values() if abs(score) > 0.1)
            confidence += (active_components / len(components)) * 0.3
            
            # 뉴스 데이터 충분성
            if news_count > 30:
                confidence += 0.1
            elif news_count > 10:
                confidence += 0.05
            
            # 컴포넌트 간 일치도
            scores = list(components.values())
            if len(scores) > 1:
                std_dev = np.std(scores)
                if std_dev < 0.3:  # 일치도 높음
                    confidence += 0.1
            
            return min(1.0, confidence)
            
        except Exception:
            return 0.5
    
    def _collect_market_indicators(self, price_data: Dict, 
                                 historical_data: Optional[any],
                                 components: Dict[str, float]) -> Dict[str, float]:
        """시장 지표 수집"""
        try:
            indicators = {
                'price_change_24h': float(price_data.get('signed_change_rate', 0)) * 100,
                'volume_24h': float(price_data.get('acc_trade_volume_24h', 0)),
                'fear_greed_index': (components.get('fear_greed_index', 0) + 1) * 50,  # 0-100 범위로 변환
            }
            
            if historical_data is not None and len(historical_data) > 0:
                indicators['volatility'] = historical_data['close'].pct_change().std() * 100
                indicators['volume_avg_7d'] = historical_data['volume'][-168:].mean() if len(historical_data) > 168 else 0
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"시장 지표 수집 오류: {str(e)}")
            return {}
    
    def _get_default_sentiment(self) -> MarketSentiment:
        """기본 감정 분석 결과"""
        return MarketSentiment(
            overall_score=0.0,
            sentiment_level=SentimentLevel.NEUTRAL,
            confidence=0.0,
            components={},
            signals=[],
            social_metrics={},
            market_indicators={},
            timestamp=datetime.now()
        )
    
    def get_sentiment_summary(self, sentiment: MarketSentiment) -> str:
        """감정 분석 요약"""
        try:
            summary = f"시장 감정: {sentiment.sentiment_level.value}\n"
            summary += f"종합 점수: {sentiment.overall_score:.2f} (신뢰도: {sentiment.confidence:.1%})\n"
            
            if sentiment.signals:
                summary += "\n주요 시그널:\n"
                for signal in sentiment.signals[:5]:
                    summary += f"• {signal}\n"
            
            return summary
            
        except Exception:
            return "감정 분석 요약 생성 실패"