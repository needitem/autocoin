"""
AI 기반 포트폴리오 투자 전략 엔진
전체 자금을 여러 코인에 분배하고 매매 타이밍을 결정하는 시스템
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    CONSERVATIVE = "보수적"  # 안전 자산 위주
    MODERATE = "중간"       # 균형잡힌 배분
    AGGRESSIVE = "공격적"   # 고수익 추구

class ActionType(Enum):
    BUY = "매수"
    SELL = "매도"
    HOLD = "보유"
    REBALANCE = "리밸런싱"

@dataclass
class CoinAnalysis:
    """개별 코인 분석 결과"""
    symbol: str
    current_price: float
    sentiment_score: float
    technical_score: float
    volume_score: float
    momentum_score: float
    risk_score: float
    overall_score: float
    confidence: float
    target_allocation: float  # 권장 포트폴리오 비중
    current_allocation: float  # 현재 보유 비중

@dataclass
class PortfolioRecommendation:
    """포트폴리오 추천"""
    timestamp: datetime
    total_capital: float
    recommendations: List[Dict]
    risk_level: RiskLevel
    expected_return: float
    max_drawdown: float
    sharpe_ratio: float
    rebalancing_needed: bool

class AIPortfolioStrategy:
    """AI 포트폴리오 전략 엔진"""
    
    def __init__(self, risk_level: RiskLevel = RiskLevel.MODERATE):
        self.risk_level = risk_level
        self.logger = logging.getLogger(__name__)
        
        # 주요 코인 목록 (시가총액 상위 - Upbit 지원 코인만)
        self.major_coins = {
            'KRW-BTC': {'name': '비트코인', 'tier': 1, 'max_allocation': 0.4},
            'KRW-ETH': {'name': '이더리움', 'tier': 1, 'max_allocation': 0.3},
            'KRW-XRP': {'name': '리플', 'tier': 2, 'max_allocation': 0.15},
            'KRW-ADA': {'name': '카르다노', 'tier': 2, 'max_allocation': 0.15},
            'KRW-SOL': {'name': '솔라나', 'tier': 2, 'max_allocation': 0.15},
            'KRW-AVAX': {'name': '아발란체', 'tier': 3, 'max_allocation': 0.1},
            'KRW-DOT': {'name': '폴카닷', 'tier': 3, 'max_allocation': 0.1},
            'KRW-ALGO': {'name': '알고랜드', 'tier': 3, 'max_allocation': 0.1},
            'KRW-LINK': {'name': '체인링크', 'tier': 3, 'max_allocation': 0.1},
            'KRW-UNI': {'name': '유니스왑', 'tier': 3, 'max_allocation': 0.1}
        }
        
        # 리스크 레벨별 설정
        self.risk_configs = {
            RiskLevel.CONSERVATIVE: {
                'tier1_weight': 0.8,  # BTC, ETH 비중
                'tier2_weight': 0.15,
                'tier3_weight': 0.05,
                'max_single_allocation': 0.4,
                'rebalance_threshold': 0.05,
                'stop_loss': 0.15
            },
            RiskLevel.MODERATE: {
                'tier1_weight': 0.65,
                'tier2_weight': 0.25,
                'tier3_weight': 0.1,
                'max_single_allocation': 0.35,
                'rebalance_threshold': 0.08,
                'stop_loss': 0.2
            },
            RiskLevel.AGGRESSIVE: {
                'tier1_weight': 0.5,
                'tier2_weight': 0.3,
                'tier3_weight': 0.2,
                'max_single_allocation': 0.3,
                'rebalance_threshold': 0.1,
                'stop_loss': 0.25
            }
        }
    
    async def analyze_all_coins(self, exchange_api, news_api, current_portfolio: Dict = None) -> List[CoinAnalysis]:
        """모든 주요 코인 분석"""
        coin_analyses = []
        
        for symbol, info in self.major_coins.items():
            try:
                analysis = await self._analyze_single_coin(symbol, info, exchange_api, news_api)
                if analysis:
                    coin_analyses.append(analysis)
            except Exception as e:
                self.logger.error(f"코인 {symbol} 분석 실패: {str(e)}")
                continue
        
        # 현재 포트폴리오 비중 업데이트
        if current_portfolio:
            self._update_current_allocations(coin_analyses, current_portfolio)
        
        return coin_analyses
    
    async def _analyze_single_coin(self, symbol: str, info: Dict, exchange_api, news_api) -> CoinAnalysis:
        """개별 코인 종합 분석"""
        try:
            # 1. 현재 가격 및 기본 데이터
            ticker = exchange_api.get_ticker(symbol)
            current_price = float(ticker.get('trade_price', 0))
            
            # 2. 기술적 분석
            technical_score = await self._calculate_technical_score(symbol, exchange_api)
            
            # 3. 감정 분석
            sentiment_data = news_api.get_market_sentiment(symbol)
            sentiment_score = sentiment_data.get('score', 0.5) * 2 - 1  # -1~1 범위로 변환
            
            # 4. 거래량 분석
            volume_score = self._calculate_volume_score(ticker)
            
            # 5. 모멘텀 분석
            momentum_score = await self._calculate_momentum_score(symbol, exchange_api)
            
            # 6. 리스크 분석
            risk_score = await self._calculate_risk_score(symbol, exchange_api)
            
            # 7. 종합 점수 계산
            overall_score = self._calculate_overall_score(
                technical_score, sentiment_score, volume_score, momentum_score, risk_score
            )
            
            # 8. 신뢰도 계산
            confidence = sentiment_data.get('confidence', 0.5)
            
            return CoinAnalysis(
                symbol=symbol,
                current_price=current_price,
                sentiment_score=sentiment_score,
                technical_score=technical_score,
                volume_score=volume_score,
                momentum_score=momentum_score,
                risk_score=risk_score,
                overall_score=overall_score,
                confidence=confidence,
                target_allocation=0,  # 나중에 계산
                current_allocation=0   # 나중에 업데이트
            )
            
        except Exception as e:
            self.logger.error(f"코인 {symbol} 분석 중 오류: {str(e)}")
            return None
    
    async def _calculate_technical_score(self, symbol: str, exchange_api) -> float:
        """기술적 분석 점수 계산"""
        try:
            # 다양한 시간대 데이터 수집
            data_1h = exchange_api.fetch_ohlcv(symbol, interval='minute60', count=100)
            data_4h = exchange_api.fetch_ohlcv(symbol, interval='minute240', count=100)
            data_1d = exchange_api.fetch_ohlcv(symbol, interval='day', count=50)
            
            scores = []
            
            # RSI 분석
            for data in [data_1h, data_4h, data_1d]:
                if data is not None and len(data) > 20:
                    rsi = self._calculate_rsi(data['close'].values)
                    if rsi:
                        if 30 <= rsi <= 70:
                            scores.append(0.5)  # 중립
                        elif rsi < 30:
                            scores.append(0.8)  # 과매도 (매수 기회)
                        else:
                            scores.append(0.2)  # 과매수 (매도 신호)
            
            # 이동평균 분석
            if data_1d is not None and len(data_1d) > 50:
                closes = data_1d['close'].values
                ma20 = np.mean(closes[-20:])
                ma50 = np.mean(closes[-50:])
                current_price = closes[-1]
                
                if current_price > ma20 > ma50:
                    scores.append(0.8)  # 상승 트렌드
                elif current_price < ma20 < ma50:
                    scores.append(0.2)  # 하락 트렌드
                else:
                    scores.append(0.5)  # 중립
            
            return np.mean(scores) if scores else 0.5
            
        except Exception as e:
            self.logger.error(f"기술적 분석 오류 {symbol}: {str(e)}")
            return 0.5
    
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
        except:
            return None
    
    def _calculate_volume_score(self, ticker: Dict) -> float:
        """거래량 점수 계산"""
        try:
            volume_24h = float(ticker.get('acc_trade_volume_24h', 0))
            price = float(ticker.get('trade_price', 1))
            
            # 거래대금 기준 (억원 단위)
            trade_value = volume_24h * price / 100000000
            
            if trade_value > 1000:  # 1000억 이상
                return 0.9
            elif trade_value > 500:  # 500억 이상
                return 0.7
            elif trade_value > 100:  # 100억 이상
                return 0.5
            elif trade_value > 50:   # 50억 이상
                return 0.3
            else:
                return 0.1
                
        except Exception as e:
            self.logger.error(f"거래량 점수 계산 오류: {str(e)}")
            return 0.3
    
    async def _calculate_momentum_score(self, symbol: str, exchange_api) -> float:
        """모멘텀 점수 계산"""
        try:
            data = exchange_api.fetch_ohlcv(symbol, interval='day', count=30)
            if data is None or len(data) < 10:
                return 0.5
            
            closes = data['close'].values
            
            # 다양한 기간 수익률
            returns_1d = (closes[-1] - closes[-2]) / closes[-2] if len(closes) > 1 else 0
            returns_7d = (closes[-1] - closes[-8]) / closes[-8] if len(closes) > 7 else 0
            returns_30d = (closes[-1] - closes[-30]) / closes[-30] if len(closes) > 29 else 0
            
            # 가중 평균 (최근일수록 높은 가중치)
            momentum = (returns_1d * 0.5 + returns_7d * 0.3 + returns_30d * 0.2)
            
            # -1~1 범위를 0~1로 변환
            return max(0, min(1, (momentum + 1) / 2))
            
        except Exception as e:
            self.logger.error(f"모멘텀 점수 계산 오류 {symbol}: {str(e)}")
            return 0.5
    
    async def _calculate_risk_score(self, symbol: str, exchange_api) -> float:
        """리스크 점수 계산 (낮을수록 안전)"""
        try:
            data = exchange_api.fetch_ohlcv(symbol, interval='day', count=30)
            if data is None or len(data) < 20:
                return 0.5
            
            closes = data['close'].values
            returns = np.diff(closes) / closes[:-1]
            
            # 변동성 계산 (표준편차)
            volatility = np.std(returns) * np.sqrt(365)  # 연환산
            
            # 최대 낙폭 (Maximum Drawdown)
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(np.min(drawdown))
            
            # 리스크 점수 (0~1, 낮을수록 안전)
            risk_score = min(1, (volatility + max_drawdown) / 2)
            
            return risk_score
            
        except Exception as e:
            self.logger.error(f"리스크 점수 계산 오류 {symbol}: {str(e)}")
            return 0.5
    
    def _calculate_overall_score(self, technical: float, sentiment: float, volume: float, 
                               momentum: float, risk: float) -> float:
        """종합 점수 계산"""
        try:
            # 가중치 설정
            weights = {
                'technical': 0.25,
                'sentiment': 0.20,
                'volume': 0.15,
                'momentum': 0.25,
                'risk': 0.15  # 리스크는 반대로 적용 (낮을수록 좋음)
            }
            
            # 리스크는 반대로 적용
            adjusted_risk = 1 - risk
            
            overall = (
                technical * weights['technical'] +
                sentiment * weights['sentiment'] +
                volume * weights['volume'] +
                momentum * weights['momentum'] +
                adjusted_risk * weights['risk']
            )
            
            return max(0, min(1, overall))
            
        except Exception as e:
            self.logger.error(f"종합 점수 계산 오류: {str(e)}")
            return 0.5
    
    def generate_portfolio_allocation(self, coin_analyses: List[CoinAnalysis]) -> List[CoinAnalysis]:
        """포트폴리오 배분 생성"""
        try:
            config = self.risk_configs[self.risk_level]
            
            # Tier별로 분류
            tier1_coins = []
            tier2_coins = []
            tier3_coins = []
            
            for analysis in coin_analyses:
                tier = self.major_coins[analysis.symbol]['tier']
                if tier == 1:
                    tier1_coins.append(analysis)
                elif tier == 2:
                    tier2_coins.append(analysis)
                else:
                    tier3_coins.append(analysis)
            
            # 각 Tier별로 점수 기준 정렬
            tier1_coins.sort(key=lambda x: x.overall_score, reverse=True)
            tier2_coins.sort(key=lambda x: x.overall_score, reverse=True)
            tier3_coins.sort(key=lambda x: x.overall_score, reverse=True)
            
            # Tier별 배분
            self._allocate_tier(tier1_coins, config['tier1_weight'])
            self._allocate_tier(tier2_coins, config['tier2_weight'])
            self._allocate_tier(tier3_coins, config['tier3_weight'])
            
            # 최대 개별 비중 제한 적용
            max_allocation = config['max_single_allocation']
            for analysis in coin_analyses:
                if analysis.target_allocation > max_allocation:
                    excess = analysis.target_allocation - max_allocation
                    analysis.target_allocation = max_allocation
                    # 초과분을 다른 코인들에 재분배
                    self._redistribute_excess(coin_analyses, excess, analysis.symbol)
            
            return coin_analyses
            
        except Exception as e:
            self.logger.error(f"포트폴리오 배분 생성 오류: {str(e)}")
            return coin_analyses
    
    def _allocate_tier(self, tier_coins: List[CoinAnalysis], tier_weight: float):
        """Tier별 자금 배분"""
        if not tier_coins:
            return
        
        # 점수 기준으로 가중치 계산
        total_score = sum(coin.overall_score for coin in tier_coins)
        
        if total_score > 0:
            for coin in tier_coins:
                coin.target_allocation = (coin.overall_score / total_score) * tier_weight
        else:
            # 모든 점수가 0인 경우 균등 배분
            equal_weight = tier_weight / len(tier_coins)
            for coin in tier_coins:
                coin.target_allocation = equal_weight
    
    def _redistribute_excess(self, coin_analyses: List[CoinAnalysis], excess: float, exclude_symbol: str):
        """초과 배분을 다른 코인들에 재분배"""
        eligible_coins = [c for c in coin_analyses if c.symbol != exclude_symbol and c.target_allocation > 0]
        
        if eligible_coins:
            total_score = sum(coin.overall_score for coin in eligible_coins)
            if total_score > 0:
                for coin in eligible_coins:
                    additional = (coin.overall_score / total_score) * excess
                    coin.target_allocation += additional
    
    def _update_current_allocations(self, coin_analyses: List[CoinAnalysis], current_portfolio: Dict):
        """현재 포트폴리오 비중 업데이트"""
        total_value = sum(current_portfolio.values())
        
        if total_value > 0:
            for analysis in coin_analyses:
                current_value = current_portfolio.get(analysis.symbol, 0)
                analysis.current_allocation = current_value / total_value
    
    def generate_trading_recommendations(self, coin_analyses: List[CoinAnalysis], 
                                       total_capital: float) -> List[Dict]:
        """구체적인 매매 추천 생성"""
        recommendations = []
        config = self.risk_configs[self.risk_level]
        
        for analysis in coin_analyses:
            try:
                # 현재 비중과 목표 비중 차이
                allocation_diff = analysis.target_allocation - analysis.current_allocation
                
                # 리밸런싱 임계값 확인
                if abs(allocation_diff) > config['rebalance_threshold']:
                    target_value = analysis.target_allocation * total_capital
                    current_value = analysis.current_allocation * total_capital
                    trade_amount = target_value - current_value
                    
                    if trade_amount > 0:
                        action = ActionType.BUY
                        quantity = trade_amount / analysis.current_price
                        
                        # 매수 타이밍 최적화
                        buy_timing = self._calculate_buy_timing(analysis)
                        target_price = analysis.current_price * (1 - buy_timing['discount'])
                        
                    else:
                        action = ActionType.SELL
                        quantity = abs(trade_amount) / analysis.current_price
                        
                        # 매도 타이밍 최적화
                        sell_timing = self._calculate_sell_timing(analysis)
                        target_price = analysis.current_price * (1 + sell_timing['premium'])
                    
                    recommendation = {
                        'symbol': analysis.symbol,
                        'name': self.major_coins[analysis.symbol]['name'],
                        'action': action.value,
                        'current_price': analysis.current_price,
                        'target_price': target_price,
                        'quantity': quantity,
                        'trade_amount': abs(trade_amount),
                        'target_allocation': analysis.target_allocation * 100,
                        'current_allocation': analysis.current_allocation * 100,
                        'confidence': analysis.confidence,
                        'overall_score': analysis.overall_score,
                        'reason': self._generate_recommendation_reason(analysis, action),
                        'stop_loss': analysis.current_price * (1 - config['stop_loss']),
                        'take_profit': self._calculate_take_profit(analysis)
                    }
                    
                    recommendations.append(recommendation)
                
                else:
                    # 유지 추천
                    recommendations.append({
                        'symbol': analysis.symbol,
                        'name': self.major_coins[analysis.symbol]['name'],
                        'action': ActionType.HOLD.value,
                        'current_price': analysis.current_price,
                        'target_allocation': analysis.target_allocation * 100,
                        'current_allocation': analysis.current_allocation * 100,
                        'reason': '목표 비중 달성으로 현재 포지션 유지',
                        'confidence': analysis.confidence
                    })
            
            except Exception as e:
                self.logger.error(f"추천 생성 오류 {analysis.symbol}: {str(e)}")
                continue
        
        return recommendations
    
    def _calculate_buy_timing(self, analysis: CoinAnalysis) -> Dict:
        """매수 타이밍 및 가격 계산"""
        # 기술적 지표와 감정 점수를 기반으로 할인율 계산
        if analysis.technical_score < 0.4 and analysis.sentiment_score < 0.3:
            # 기술적으로 과매도 + 부정적 감정 = 큰 할인 기대
            discount = 0.03  # 3% 할인 목표
            urgency = "낮음"
        elif analysis.overall_score > 0.7:
            # 전체적으로 좋은 점수 = 빠른 매수
            discount = 0.01  # 1% 할인만 기대
            urgency = "높음"
        else:
            discount = 0.02  # 2% 할인 목표
            urgency = "보통"
        
        return {
            'discount': discount,
            'urgency': urgency,
            'timing': '시장가 매수' if urgency == "높음" else '지정가 매수'
        }
    
    def _calculate_sell_timing(self, analysis: CoinAnalysis) -> Dict:
        """매도 타이밍 및 가격 계산"""
        if analysis.technical_score > 0.6 and analysis.sentiment_score > 0.6:
            # 과매수 + 긍정적 감정 = 프리미엄 기대
            premium = 0.02  # 2% 프리미엄 목표
            urgency = "낮음"
        elif analysis.overall_score < 0.3:
            # 전체적으로 나쁜 점수 = 빠른 매도
            premium = 0.005  # 0.5% 프리미엄만 기대
            urgency = "높음"
        else:
            premium = 0.015  # 1.5% 프리미엄 목표
            urgency = "보통"
        
        return {
            'premium': premium,
            'urgency': urgency,
            'timing': '시장가 매도' if urgency == "높음" else '지정가 매도'
        }
    
    def _calculate_take_profit(self, analysis: CoinAnalysis) -> float:
        """익절가 계산"""
        base_profit = 0.15  # 기본 15% 익절
        
        # 코인 티어에 따른 조정
        tier = self.major_coins[analysis.symbol]['tier']
        if tier == 1:
            profit_target = base_profit * 0.8  # 안전 자산은 낮은 익절
        elif tier == 2:
            profit_target = base_profit
        else:
            profit_target = base_profit * 1.3  # 위험 자산은 높은 익절
        
        # 전체 점수에 따른 조정
        if analysis.overall_score > 0.8:
            profit_target *= 1.2
        elif analysis.overall_score < 0.4:
            profit_target *= 0.8
        
        return analysis.current_price * (1 + profit_target)
    
    def _generate_recommendation_reason(self, analysis: CoinAnalysis, action: ActionType) -> str:
        """추천 이유 생성"""
        reasons = []
        
        if action == ActionType.BUY:
            if analysis.overall_score > 0.7:
                reasons.append("종합 점수 우수")
            if analysis.sentiment_score > 0.6:
                reasons.append("긍정적 시장 감정")
            if analysis.technical_score > 0.6:
                reasons.append("기술적 매수 신호")
            if analysis.momentum_score > 0.6:
                reasons.append("상승 모멘텀")
        
        elif action == ActionType.SELL:
            if analysis.overall_score < 0.4:
                reasons.append("종합 점수 부진")
            if analysis.sentiment_score < 0.4:
                reasons.append("부정적 시장 감정")
            if analysis.technical_score < 0.4:
                reasons.append("기술적 매도 신호")
            if analysis.risk_score > 0.7:
                reasons.append("높은 리스크 수준")
        
        return " | ".join(reasons) if reasons else "포트폴리오 리밸런싱"