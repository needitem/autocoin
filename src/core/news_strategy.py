"""
뉴스 기반 거래 전략 엔진
"""

import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class SignalStrength(Enum):
    STRONG_BUY = "강한 매수"
    BUY = "매수"
    WEAK_BUY = "약한 매수"
    HOLD = "보유"
    WEAK_SELL = "약한 매도"
    SELL = "매도"
    STRONG_SELL = "강한 매도"

@dataclass
class NewsSignal:
    signal: SignalStrength
    confidence: float  # 0.0 ~ 1.0
    reason: str
    news_count: int
    positive_score: float
    negative_score: float
    volatility_impact: float

class NewsBasedStrategy:
    def __init__(self):
        """뉴스 기반 전략 초기화"""
        self.positive_keywords = {
            'strong': ['adoption', 'bullish', 'surge', 'rally', 'breakthrough', 'partnership', 'approval', 'pump', 'moon', 'breakout'],
            'medium': ['growth', 'increase', 'positive', 'optimistic', 'upgrade', 'support', 'recovery', 'bounce', 'gains'],
            'weak': ['stable', 'steady', 'maintain', 'continue', 'holding', 'consolidation']
        }
        
        self.negative_keywords = {
            'strong': ['crash', 'dump', 'bearish', 'collapse', 'plunge', 'panic', 'regulation', 'ban', 'hack', 'scam'],
            'medium': ['decline', 'drop', 'fall', 'correction', 'weakness', 'concern', 'uncertainty', 'volatility'],
            'weak': ['caution', 'risk', 'pressure', 'challenge', 'mixed', 'neutral']
        }
        
        self.volume_keywords = {
            'high': ['massive', 'huge', 'significant', 'major', 'substantial', 'record', 'unprecedented'],
            'medium': ['notable', 'considerable', 'important', 'strong', 'solid'],
            'low': ['minor', 'small', 'slight', 'limited', 'modest']
        }
        
        self.institutional_keywords = [
            'institutional', 'banks', 'etf', 'fund', 'investment', 'corporate', 'treasury',
            'blackrock', 'grayscale', 'microstrategy', 'tesla', 'square', 'paypal'
        ]
        
        self.regulatory_keywords = [
            'regulation', 'regulatory', 'sec', 'government', 'policy', 'law', 'legal',
            'compliance', 'license', 'approval', 'ban', 'restriction'
        ]
        
        self.technical_keywords = [
            'blockchain', 'network', 'upgrade', 'fork', 'consensus', 'mining', 'staking',
            'defi', 'smart contract', 'protocol', 'layer', 'scaling'
        ]
    
    def analyze_news_sentiment(self, news_items: List[Dict]) -> Dict:
        """뉴스 리스트에서 감정 분석"""
        try:
            if not news_items:
                return self._get_neutral_sentiment()
            
            total_score = 0
            positive_count = 0
            negative_count = 0
            institutional_mentions = 0
            regulatory_mentions = 0
            technical_mentions = 0
            volume_impact = 0
            
            analyzed_news = []
            
            for news in news_items:
                title = news.get('title', '').lower()
                summary = news.get('summary', '').lower()
                text = f"{title} {summary}"
                
                # 개별 뉴스 분석
                news_score = self._analyze_single_news(text)
                total_score += news_score['score']
                
                if news_score['score'] > 0.1:
                    positive_count += 1
                elif news_score['score'] < -0.1:
                    negative_count += 1
                
                # 카테고리 분석
                if any(keyword in text for keyword in self.institutional_keywords):
                    institutional_mentions += 1
                    total_score += 0.3  # 기관 투자 가중치
                
                if any(keyword in text for keyword in self.regulatory_keywords):
                    regulatory_mentions += 1
                    # 규제 뉴스는 맥락에 따라 다름
                    if any(word in text for word in ['approval', 'license', 'legal']):
                        total_score += 0.2
                    else:
                        total_score -= 0.2
                
                if any(keyword in text for keyword in self.technical_keywords):
                    technical_mentions += 1
                    total_score += 0.1  # 기술 발전 가중치
                
                # 볼륨 임팩트 분석
                volume_impact += self._analyze_volume_impact(text)
                
                analyzed_news.append({
                    'title': news.get('title', ''),
                    'score': news_score['score'],
                    'keywords': news_score['keywords'],
                    'source': news.get('source', '')
                })
            
            # 전체 감정 계산
            average_score = total_score / len(news_items) if news_items else 0
            
            return {
                'overall_score': average_score,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': len(news_items) - positive_count - negative_count,
                'institutional_mentions': institutional_mentions,
                'regulatory_mentions': regulatory_mentions,
                'technical_mentions': technical_mentions,
                'volume_impact': volume_impact / len(news_items) if news_items else 0,
                'analyzed_news': analyzed_news,
                'total_news': len(news_items)
            }
            
        except Exception as e:
            logger.error(f"뉴스 감정 분석 오류: {str(e)}")
            return self._get_neutral_sentiment()
    
    def _analyze_single_news(self, text: str) -> Dict:
        """개별 뉴스 분석"""
        score = 0
        keywords_found = []
        
        # 긍정적 키워드 분석
        for strength, keywords in self.positive_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    if strength == 'strong':
                        score += 0.5
                    elif strength == 'medium':
                        score += 0.3
                    else:  # weak
                        score += 0.1
                    keywords_found.append(f"+{keyword}")
        
        # 부정적 키워드 분석
        for strength, keywords in self.negative_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    if strength == 'strong':
                        score -= 0.5
                    elif strength == 'medium':
                        score -= 0.3
                    else:  # weak
                        score -= 0.1
                    keywords_found.append(f"-{keyword}")
        
        return {
            'score': max(-1.0, min(1.0, score)),  # -1.0 ~ 1.0 범위로 제한
            'keywords': keywords_found
        }
    
    def _analyze_volume_impact(self, text: str) -> float:
        """볼륨 임팩트 분석"""
        impact = 0
        
        for level, keywords in self.volume_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    if level == 'high':
                        impact += 0.3
                    elif level == 'medium':
                        impact += 0.2
                    else:  # low
                        impact += 0.1
        
        return min(1.0, impact)
    
    def _get_neutral_sentiment(self) -> Dict:
        """중립 감정 반환"""
        return {
            'overall_score': 0.0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'institutional_mentions': 0,
            'regulatory_mentions': 0,
            'technical_mentions': 0,
            'volume_impact': 0.0,
            'analyzed_news': [],
            'total_news': 0
        }
    
    def generate_trading_signal(self, news_sentiment: Dict, current_price: float, 
                               price_change: float, volume_change: float = 0) -> NewsSignal:
        """뉴스 기반 거래 신호 생성"""
        try:
            score = news_sentiment['overall_score']
            positive_count = news_sentiment['positive_count']
            negative_count = news_sentiment['negative_count']
            total_news = news_sentiment['total_news']
            
            # 기본 신호 강도 계산
            if score > 0.4:
                base_signal = SignalStrength.STRONG_BUY
            elif score > 0.2:
                base_signal = SignalStrength.BUY
            elif score > 0.05:
                base_signal = SignalStrength.WEAK_BUY
            elif score < -0.4:
                base_signal = SignalStrength.STRONG_SELL
            elif score < -0.2:
                base_signal = SignalStrength.SELL
            elif score < -0.05:
                base_signal = SignalStrength.WEAK_SELL
            else:
                base_signal = SignalStrength.HOLD
            
            # 신뢰도 계산
            confidence = self._calculate_confidence(news_sentiment, total_news)
            
            # 가격 변동과 뉴스 일치성 확인
            signal_adjustment = self._adjust_signal_with_price(base_signal, price_change, score)
            
            # 기관 투자 가중치
            if news_sentiment['institutional_mentions'] > 0:
                confidence += 0.1
            
            # 규제 뉴스 가중치
            if news_sentiment['regulatory_mentions'] > 0:
                confidence += 0.05
            
            # 신호 설명 생성
            reason = self._generate_signal_reason(news_sentiment, signal_adjustment, price_change)
            
            return NewsSignal(
                signal=signal_adjustment,
                confidence=min(1.0, confidence),
                reason=reason,
                news_count=total_news,
                positive_score=positive_count / total_news if total_news > 0 else 0,
                negative_score=negative_count / total_news if total_news > 0 else 0,
                volatility_impact=news_sentiment['volume_impact']
            )
            
        except Exception as e:
            logger.error(f"거래 신호 생성 오류: {str(e)}")
            return NewsSignal(
                signal=SignalStrength.HOLD,
                confidence=0.0,
                reason="신호 생성 오류",
                news_count=0,
                positive_score=0.0,
                negative_score=0.0,
                volatility_impact=0.0
            )
    
    def _calculate_confidence(self, news_sentiment: Dict, total_news: int) -> float:
        """신뢰도 계산"""
        base_confidence = 0.3
        
        # 뉴스 개수에 따른 신뢰도
        if total_news > 20:
            base_confidence += 0.3
        elif total_news > 10:
            base_confidence += 0.2
        elif total_news > 5:
            base_confidence += 0.1
        
        # 감정 일치도
        positive_ratio = news_sentiment['positive_count'] / total_news if total_news > 0 else 0
        negative_ratio = news_sentiment['negative_count'] / total_news if total_news > 0 else 0
        
        if positive_ratio > 0.7 or negative_ratio > 0.7:
            base_confidence += 0.2
        elif positive_ratio > 0.5 or negative_ratio > 0.5:
            base_confidence += 0.1
        
        return base_confidence
    
    def _adjust_signal_with_price(self, base_signal: SignalStrength, price_change: float, news_score: float) -> SignalStrength:
        """가격 변동과 뉴스 일치성으로 신호 조정"""
        # 뉴스와 가격이 일치하는 경우 신호 강화
        if (news_score > 0 and price_change > 0) or (news_score < 0 and price_change < 0):
            return base_signal  # 일치하면 그대로
        
        # 뉴스와 가격이 반대인 경우 신호 약화
        if base_signal == SignalStrength.STRONG_BUY:
            return SignalStrength.BUY
        elif base_signal == SignalStrength.BUY:
            return SignalStrength.WEAK_BUY
        elif base_signal == SignalStrength.STRONG_SELL:
            return SignalStrength.SELL
        elif base_signal == SignalStrength.SELL:
            return SignalStrength.WEAK_SELL
        
        return base_signal
    
    def _generate_signal_reason(self, news_sentiment: Dict, signal: SignalStrength, price_change: float) -> str:
        """신호 이유 생성"""
        reasons = []
        
        # 뉴스 감정 기반 이유
        if news_sentiment['positive_count'] > news_sentiment['negative_count']:
            reasons.append(f"긍정적 뉴스 {news_sentiment['positive_count']}개")
        elif news_sentiment['negative_count'] > news_sentiment['positive_count']:
            reasons.append(f"부정적 뉴스 {news_sentiment['negative_count']}개")
        
        # 기관 투자 언급
        if news_sentiment['institutional_mentions'] > 0:
            reasons.append(f"기관 투자 관련 뉴스 {news_sentiment['institutional_mentions']}개")
        
        # 규제 관련
        if news_sentiment['regulatory_mentions'] > 0:
            reasons.append(f"규제 관련 뉴스 {news_sentiment['regulatory_mentions']}개")
        
        # 기술 발전
        if news_sentiment['technical_mentions'] > 0:
            reasons.append(f"기술 발전 관련 뉴스 {news_sentiment['technical_mentions']}개")
        
        # 가격 변동 연관성
        if abs(price_change) > 5:
            if price_change > 0:
                reasons.append(f"강한 상승세 ({price_change:.1f}%)")
            else:
                reasons.append(f"강한 하락세 ({price_change:.1f}%)")
        
        return " | ".join(reasons) if reasons else "뉴스 분석 결과"
    
    def get_risk_assessment(self, news_signal: NewsSignal) -> Dict:
        """위험도 평가"""
        risk_level = "중간"
        risk_factors = []
        
        # 신뢰도 기반 위험도
        if news_signal.confidence < 0.3:
            risk_level = "높음"
            risk_factors.append("낮은 신뢰도")
        elif news_signal.confidence > 0.7:
            risk_level = "낮음"
        
        # 뉴스 개수 기반
        if news_signal.news_count < 5:
            risk_factors.append("적은 뉴스 개수")
        
        # 변동성 기반
        if news_signal.volatility_impact > 0.7:
            risk_level = "높음"
            risk_factors.append("높은 변동성 예상")
        
        return {
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommended_position_size': self._get_position_size(news_signal.confidence, risk_level)
        }
    
    def _get_position_size(self, confidence: float, risk_level: str) -> str:
        """포지션 크기 추천"""
        if risk_level == "높음":
            return "소량 (5-10%)"
        elif risk_level == "낮음" and confidence > 0.7:
            return "중간 (20-30%)"
        else:
            return "보통 (10-20%)"