from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime
from enum import Enum

@dataclass
class InvestmentLevel:
    price: float
    ratio: float  # 전체 투자금 대비 비율
    description: str

@dataclass
class StrategyRecommendation:
    entry_levels: List[InvestmentLevel]  # 매수 진입 레벨
    exit_levels: List[InvestmentLevel]   # 매도 청산 레벨
    stop_loss: float                     # 손절가
    risk_ratio: float                    # 리스크 비율
    investment_amount: float             # 추천 투자 금액 (총 자산 대비)
    holding_period: str                  # 추천 보유 기간
    strategy_type: str                   # 전략 유형
    confidence_score: float              # 전략 신뢰도

class TradingStrategy(Enum):
    SCALPING = "스캘핑 (초단타)"
    DAYTRADING = "데이트레이딩 (단타)"
    SWING = "스윙 트레이딩 (중기)"
    POSITION = "포지션 트레이딩 (장기)"

class InvestmentStrategy:
    def __init__(self, strategy_type: TradingStrategy = TradingStrategy.SWING):
        self.strategy_type = strategy_type
        
        # 전략별 설정
        self.strategy_configs = {
            TradingStrategy.SCALPING: {
                'profit_targets': [1.005, 1.01, 1.015],  # 0.5%, 1%, 1.5%
                'loss_limits': [0.995, 0.99, 0.985],     # -0.5%, -1%, -1.5%
                'holding_period': "1시간 이내",
                'max_investment_ratio': 0.3,             # 총 자산의 30%
                'position_sizes': [0.4, 0.3, 0.3]        # 분할 매수 비율
            },
            TradingStrategy.DAYTRADING: {
                'profit_targets': [1.01, 1.02, 1.03],    # 1%, 2%, 3%
                'loss_limits': [0.99, 0.98, 0.97],       # -1%, -2%, -3%
                'holding_period': "1일 이내",
                'max_investment_ratio': 0.4,             # 총 자산의 40%
                'position_sizes': [0.5, 0.3, 0.2]
            },
            TradingStrategy.SWING: {
                'profit_targets': [1.03, 1.05, 1.08],    # 3%, 5%, 8%
                'loss_limits': [0.97, 0.95, 0.93],       # -3%, -5%, -7%
                'holding_period': "1-7일",
                'max_investment_ratio': 0.5,             # 총 자산의 50%
                'position_sizes': [0.4, 0.3, 0.3]
            },
            TradingStrategy.POSITION: {
                'profit_targets': [1.05, 1.10, 1.15],    # 5%, 10%, 15%
                'loss_limits': [0.95, 0.92, 0.90],       # -5%, -8%, -10%
                'holding_period': "7일 이상",
                'max_investment_ratio': 0.6,             # 총 자산의 60%
                'position_sizes': [0.3, 0.3, 0.4]
            }
        }
        
        self.config = self.strategy_configs[strategy_type]
        self.MAX_INVESTMENT_RATIO = self.config['max_investment_ratio']
        self.MIN_HOLDING_PERIOD = "1일"
        self.MAX_HOLDING_PERIOD = "30일"
        
    def calculate_risk_ratio(self, fg_index: float, rsi: float, volatility: float) -> float:
        """리스크 비율 계산"""
        # 공포탐욕지수 기반 리스크 조정
        if fg_index <= 20:
            fg_risk = 0.8  # 낮은 리스크
        elif fg_index >= 80:
            fg_risk = 0.2  # 높은 리스크
        else:
            fg_risk = 0.5  # 중간 리스크
            
        # RSI 기반 리스크 조정
        if rsi <= 30:
            rsi_risk = 0.8
        elif rsi >= 70:
            rsi_risk = 0.2
        else:
            rsi_risk = 0.5
            
        # 변동성 기반 리스크 조정 (변동성이 높을수록 리스크 높음)
        vol_risk = 1 - min(volatility / 100, 1)
        
        # 종합 리스크 점수 계산
        return (fg_risk * 0.4 + rsi_risk * 0.3 + vol_risk * 0.3)
    
    def calculate_investment_amount(self, total_assets: float, risk_ratio: float) -> float:
        """투자 금액 계산"""
        base_ratio = self.MAX_INVESTMENT_RATIO * risk_ratio
        return total_assets * base_ratio
    
    def get_holding_period(self, fg_index: float, trend_strength: str) -> str:
        """보유 기간 추천"""
        if fg_index <= 20 or fg_index >= 80:
            return "1-3일 (단기 트레이딩)"
        elif trend_strength == "강함":
            return "7-14일 (스윙 트레이딩)"
        else:
            return "14-30일 (포지션 트레이딩)"
    
    def calculate_entry_levels(self, current_price: float, fg_index: float, 
                             rsi: float, support_levels: List[float], 
                             ma_data: dict,
                             orderbook_analysis: dict = None) -> List[InvestmentLevel]:
        """매수 진입 레벨 계산"""
        levels = []
        config = self.strategy_configs[self.strategy_type]
        position_sizes = config['position_sizes']
        
        # 호가 분석 데이터 활용
        if orderbook_analysis:
            buy_pressure = orderbook_analysis.get('buy_pressure', 1.0)
            sell_pressure = orderbook_analysis.get('sell_pressure', 1.0)
            support_prices = orderbook_analysis.get('support_levels', [])
            resistance_prices = orderbook_analysis.get('resistance_levels', [])
            
            # 매수/매도 세력 비율에 따른 가격 조정
            pressure_ratio = buy_pressure / sell_pressure if sell_pressure > 0 else 1.0
            
            # 매수세가 강할 때는 약간 높은 가격에 매수
            if pressure_ratio > 1.5:
                base_adjustment = 1.005  # +0.5%
            # 매도세가 강할 때는 더 낮은 가격에 매수
            elif pressure_ratio < 0.67:  # 1/1.5
                base_adjustment = 0.99   # -1%
            else:
                base_adjustment = 1.0
        else:
            base_adjustment = 1.0
            support_prices = []
        
        # 시장 상황 분석
        market_condition = self._analyze_market_condition(fg_index, rsi, ma_data)
        
        # 기본 매수 가격 설정
        for i, ratio in enumerate(position_sizes):
            # 지지선 찾기
            nearest_support = self._find_nearest_support(current_price, support_prices)
            
            # 이동평균선 기반 가격
            ma_price = self._calculate_ma_based_price(current_price, ma_data)
            
            # 기본 매수가 설정 (지지선, 이동평균선, 현재가 고려)
            if nearest_support > 0:
                base_price = max(nearest_support, ma_price)
            else:
                base_price = ma_price
            
            # 시장 상황별 매수가 조정
            if market_condition == "강세":
                if i == 0:  # 1차 매수
                    target_price = current_price * 0.995  # -0.5%
                elif i == 1:  # 2차 매수
                    target_price = current_price * 0.99   # -1%
                else:  # 3차 매수
                    target_price = current_price * 0.985  # -1.5%
            elif market_condition == "약세":
                if i == 0:
                    target_price = current_price * 0.99   # -1%
                elif i == 1:
                    target_price = current_price * 0.98   # -2%
                else:
                    target_price = current_price * 0.97   # -3%
            else:  # 중립
                if i == 0:
                    target_price = current_price * 0.993  # -0.7%
                elif i == 1:
                    target_price = current_price * 0.985  # -1.5%
                else:
                    target_price = current_price * 0.975  # -2.5%
            
            # 최종 매수가 (지지선, 이동평균선, 호가 상황 반영)
            final_price = min(base_price, target_price) * base_adjustment
            
            # 매수 전략 설명 생성
            conditions = []
            if market_condition != "중립":
                conditions.append(f"{market_condition} 시장")
            if orderbook_analysis:
                if pressure_ratio > 1.5:
                    conditions.append("매수세 강함")
                elif pressure_ratio < 0.67:
                    conditions.append("매도세 강함")
            if nearest_support > 0 and nearest_support > final_price * 0.99:
                conditions.append("지지선 근처")
            
            description = f"{i+1}차 매수 포인트 (투자금의 {ratio*100:.0f}%)"
            if conditions:
                description += f" - {', '.join(conditions)}"
            
            levels.append(InvestmentLevel(final_price, ratio, description))
        
        return levels

    def _analyze_market_condition(self, fg_index: float, rsi: float, ma_data: dict) -> str:
        """시장 상황 분석"""
        # MA 데이터 분석
        ma5 = ma_data.get('MA5', 0)
        ma20 = ma_data.get('MA20', 0)
        ma60 = ma_data.get('MA60', 0)
        
        # 이동평균선 배열 확인
        ma_trend = "상승" if ma5 > ma20 > ma60 else "하락" if ma5 < ma20 < ma60 else "횡보"
        
        # 종합 분석
        if ma_trend == "상승" and rsi > 50 and fg_index > 50:
            return "강세"
        elif ma_trend == "하락" and rsi < 50 and fg_index < 50:
            return "약세"
        else:
            return "중립"

    def _find_nearest_support(self, target_price: float, support_levels: List[float]) -> float:
        """가장 가까운 지지선 찾기"""
        if not support_levels:
            return target_price
        
        # 타겟 가격보다 낮은 지지선 중 가장 가까운 것 선택
        lower_supports = [p for p in support_levels if p < target_price]
        if lower_supports:
            return max(lower_supports)
        return target_price

    def _calculate_ma_based_price(self, current_price: float, ma_data: dict) -> float:
        """이동평균선 기반 매수가 계산"""
        ma5 = ma_data.get('MA5', current_price)
        ma20 = ma_data.get('MA20', current_price)
        ma60 = ma_data.get('MA60', current_price)
        
        # 이동평균선 배열 확인
        if current_price < ma5 < ma20:  # 하락 추세
            return min(ma5, current_price * 1.01)  # 5일선 또는 1% 위
        elif ma5 < current_price < ma20:  # 5일선 위, 20일선 아래
            return min(ma20, current_price * 1.02)  # 20일선 또는 2% 위
        elif ma20 < current_price < ma60:  # 20일선 위, 60일선 아래
            return min(ma60, current_price * 1.03)  # 60일선 또는 3% 위
        else:
            return current_price

    def calculate_exit_levels(self, current_price: float, 
                            fg_index: float,
                            resistance_levels: List[float] = None) -> List[InvestmentLevel]:
        """매도 청산 레벨 계산"""
        levels = []
        config = self.strategy_configs[self.strategy_type]
        
        # 시장 상황에 따른 매도 가격 조정
        if fg_index >= 80:  # 극단적 탐욕
            profit_targets = [x * 0.995 for x in config['profit_targets']]  # 더 낮은 가격에 매도
        elif fg_index >= 60:  # 탐욕
            profit_targets = config['profit_targets']
        else:
            profit_targets = [x * 1.005 for x in config['profit_targets']]  # 더 높은 가격에 매도
        
        position_sizes = config['position_sizes']
        
        for i, (target, ratio) in enumerate(zip(profit_targets, position_sizes)):
            price = current_price * target
            
            if self.strategy_type == TradingStrategy.SCALPING:
                description = f"단타 매도 {i+1}차 (익절 {(target-1)*100:.1f}%)"
            elif self.strategy_type == TradingStrategy.DAYTRADING:
                description = f"데이트레이딩 매도 {i+1}차 (일간 {(target-1)*100:.1f}%)"
            else:
                description = f"{i+1}차 매도 포인트 (보유량의 {ratio*100}%)"
            
            levels.append(InvestmentLevel(price, ratio, description))
        
        return levels
    
    def get_strategy_recommendation(self, 
                                  current_price: float,
                                  fg_index: float,
                                  rsi: float,
                                  volatility: float,
                                  trend_strength: str,
                                  ma_data: dict,
                                  orderbook_analysis: dict = None,  # 호가 분석 결과 추가
                                  total_assets: float = 10000000
                                  ) -> StrategyRecommendation:
        """종합 투자 전략 추천"""
        
        # 리스크 비율 계산
        risk_ratio = self.calculate_risk_ratio(fg_index, rsi, volatility)
        
        # 투자 금액 계산
        base_amount = self.calculate_investment_amount(total_assets, risk_ratio)
        
        # 시장 상황별 투자 금액 조정
        if fg_index <= 20 and rsi <= 30:  # 매우 강한 매수 신호
            investment_amount = base_amount * 1.2  # 20% 증액
        elif fg_index >= 80 or rsi >= 70:  # 매우 강한 매도 신호
            investment_amount = base_amount * 0.8  # 20% 감액
        else:
            investment_amount = base_amount
        
        # 최대 투자 한도 체크
        investment_amount = min(investment_amount, total_assets * self.MAX_INVESTMENT_RATIO)
        
        # 보유 기간 추천
        holding_period = self.get_holding_period(fg_index, trend_strength)
        
        # 호가 분석 데이터에서 지지/저항 레벨 추출
        support_levels = []
        resistance_levels = []
        if orderbook_analysis:
            support_levels = orderbook_analysis.get('support_levels', [])
            resistance_levels = orderbook_analysis.get('resistance_levels', [])
        
        # 매수/매도 레벨 계산
        entry_levels = self.calculate_entry_levels(
            current_price=current_price,
            fg_index=fg_index,
            rsi=rsi,
            support_levels=support_levels,
            ma_data=ma_data,
            orderbook_analysis=orderbook_analysis  # 호가 분석 결과 전달
        )
        
        exit_levels = self.calculate_exit_levels(
            current_price=current_price,
            fg_index=fg_index,
            resistance_levels=resistance_levels
        )
        
        # 손절가 설정
        lowest_entry = min(level.price for level in entry_levels)
        ma_support = min(ma_data.get('MA5', current_price),
                        ma_data.get('MA20', current_price))
        stop_loss = max(lowest_entry * 0.97,  # 최저 매수가의 97%
                       ma_support * 0.98)     # 주요 이평선의 98%
        
        # 전략 유형 및 신뢰도 결정
        strategy_type, confidence_score = self._determine_strategy_type(
            fg_index, rsi, trend_strength, volatility
        )
        
        return StrategyRecommendation(
            entry_levels=entry_levels,
            exit_levels=exit_levels,
            stop_loss=stop_loss,
            risk_ratio=risk_ratio,
            investment_amount=investment_amount,
            holding_period=holding_period,
            strategy_type=strategy_type,
            confidence_score=confidence_score
        )

    def _determine_strategy_type(self, fg_index: float, rsi: float, 
                               trend_strength: str, volatility: float) -> Tuple[str, float]:
        """전략 유형과 신뢰도 결정"""
        if fg_index <= 20 and rsi <= 30:
            return "적극적 분할 매수 전략", 0.9
        elif fg_index <= 40 and rsi <= 40:
            return "단계적 매수 전략", 0.8
        elif fg_index >= 80 and rsi >= 70:
            return "적극적 이익실현 전략", 0.9
        elif fg_index >= 60 and rsi >= 60:
            return "단계적 이익실현 전략", 0.8
        elif trend_strength == "강함" and volatility < 30:
            return "추세 추종 전략", 0.7
        else:
            return "중립적 관망 전략", 0.6

    def recommend_strategy_type(self, fg_index: float, rsi: float, 
                              volatility: float, trend_strength: str) -> TradingStrategy:
        """시장 상황에 따른 최적 전략 추천"""
        
        # 변동성이 매우 높은 경우 (20% 이상)
        if volatility > 20:
            if fg_index <= 30 or fg_index >= 70:  # 극단적 상황
                return TradingStrategy.SCALPING  # 변동성이 높고 극단적일 때는 스캘핑
            else:
                return TradingStrategy.DAYTRADING  # 변동성이 높을 때는 단타
        
        # 추세가 강한 경우
        if trend_strength == "강함":
            if 40 <= fg_index <= 60:  # 중립적 상황
                return TradingStrategy.SWING  # 강한 추세 + 중립적 = 스윙
            elif volatility < 10:  # 낮은 변동성
                return TradingStrategy.POSITION  # 강한 추세 + 낮은 변동성 = 포지션
        
        # RSI 기반 추천
        if rsi <= 30 or rsi >= 70:  # 과매수/과매도
            if volatility > 15:
                return TradingStrategy.DAYTRADING
            else:
                return TradingStrategy.SWING
        
        # 공포탐욕지수 기반 추천
        if fg_index <= 20 or fg_index >= 80:  # 극단적 공포/탐욕
            return TradingStrategy.SCALPING if volatility > 15 else TradingStrategy.DAYTRADING
        elif fg_index <= 40 or fg_index >= 60:  # 공포/탐욕
            return TradingStrategy.SWING
        
        # 기본값: 안정적인 시장
        return TradingStrategy.POSITION

    def get_strategy_description(self, fg_index: float, rsi: float, 
                               volatility: float, trend_strength: str) -> str:
        """전략 추천 이유 설명"""
        recommended = self.recommend_strategy_type(fg_index, rsi, volatility, trend_strength)
        
        reasons = []
        if volatility > 20:
            reasons.append(f"높은 변동성 ({volatility:.1f}%)")
        elif volatility < 10:
            reasons.append(f"낮은 변동성 ({volatility:.1f}%)")
        
        if fg_index <= 20:
            reasons.append("극단적 공포 상태")
        elif fg_index >= 80:
            reasons.append("극단적 탐욕 상태")
        
        if rsi <= 30:
            reasons.append("과매도 구간")
        elif rsi >= 70:
            reasons.append("과매수 구간")
        
        if trend_strength == "강함":
            reasons.append("강한 추세")
        
        strategy_descriptions = {
            TradingStrategy.SCALPING: "빠른 진입/청산으로 변동성 활용",
            TradingStrategy.DAYTRADING: "일간 변동성을 이용한 단기 트레이딩",
            TradingStrategy.SWING: "추세를 활용한 중기 포지션",
            TradingStrategy.POSITION: "안정적인 장기 포지션"
        }
        
        return f"""
💡 추천 전략: {recommended.value}
- 추천 이유: {', '.join(reasons)}
- 전략 특징: {strategy_descriptions[recommended]}
- 현재 선택: {self.strategy_type.value}
    """

def format_strategy_message(strategy: StrategyRecommendation, total_assets: float) -> str:
    """전략 메시지 포맷팅"""
    message = [
        f"💡 투자 전략 추천 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})",
        f"\n전략 유형: {strategy.strategy_type} (신뢰도: {strategy.confidence_score*100:.1f}%)",
        f"리스크 비율: {strategy.risk_ratio*100:.1f}%",
        f"추천 투자 금액: {strategy.investment_amount:,.0f}원 (총자산의 {strategy.investment_amount/total_assets*100:.1f}%)",
        f"추천 보유 기간: {strategy.holding_period}",
        "\n매수 전략:",
    ]
    
    for level in strategy.entry_levels:
        message.append(f"- {level.description}")
        message.append(f"  가격: {level.price:,.0f}원 (투자금액: {level.ratio * strategy.investment_amount:,.0f}원)")
    
    message.append("\n매도 전략:")
    for level in strategy.exit_levels:
        message.append(f"- {level.description}")
        message.append(f"  가격: {level.price:,.0f}원")
    
    message.append(f"\n손절가: {strategy.stop_loss:,.0f}원")
    
    message.append("\n⚠️ 주의사항:")
    message.append("- 투자는 본인의 판단과 책임하에 진행하세요.")
    message.append("- 시장 상황에 따라 전략을 유연하게 조정하세요.")
    message.append("- 설정된 손절가를 반드시 준수하세요.")
    
    return "\n".join(message) 