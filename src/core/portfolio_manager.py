"""
포트폴리오 매니저 - AI 전략을 실행하고 리스크를 관리하는 시스템
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import asyncio

from src.core.ai_portfolio_strategy import AIPortfolioStrategy, RiskLevel, ActionType

logger = logging.getLogger(__name__)

@dataclass
class Portfolio:
    """포트폴리오 현황"""
    total_capital: float
    available_cash: float
    positions: Dict[str, Dict]  # {symbol: {quantity, avg_price, value}}
    last_updated: datetime

@dataclass
class RiskMetrics:
    """리스크 지표"""
    portfolio_value: float
    max_drawdown: float
    var_95: float  # 95% Value at Risk
    sharpe_ratio: float
    volatility: float
    beta: float

class PortfolioManager:
    """포트폴리오 매니저"""
    
    def __init__(self, initial_capital: float, risk_level: RiskLevel = RiskLevel.MODERATE):
        self.initial_capital = initial_capital
        self.portfolio = Portfolio(
            total_capital=initial_capital,
            available_cash=initial_capital,
            positions={},
            last_updated=datetime.now()
        )
        
        self.strategy = AIPortfolioStrategy(risk_level)
        self.trade_history = []
        self.performance_history = []
        self.logger = logging.getLogger(__name__)
        
        # 리스크 관리 설정
        self.risk_settings = {
            'max_position_size': 0.4,  # 단일 포지션 최대 40%
            'max_daily_loss': 0.05,    # 일일 최대 손실 5%
            'portfolio_stop_loss': 0.2, # 포트폴리오 전체 20% 손실시 중단
            'rebalance_threshold': 0.05, # 5% 이상 차이시 리밸런싱
        }
    
    async def analyze_and_recommend(self, exchange_api, news_api) -> Dict:
        """전체 포트폴리오 분석 및 추천"""
        try:
            self.logger.info("포트폴리오 전략 분석 시작...")
            
            # 1. 현재 포트폴리오 가치 계산
            current_portfolio_value = await self._calculate_portfolio_value(exchange_api)
            
            # 2. 모든 주요 코인 분석
            coin_analyses = await self.strategy.analyze_all_coins(
                exchange_api, news_api, self._get_current_positions_value()
            )
            
            # 3. 포트폴리오 배분 계산
            coin_analyses = self.strategy.generate_portfolio_allocation(coin_analyses)
            
            # 4. 구체적인 매매 추천 생성
            recommendations = self.strategy.generate_trading_recommendations(
                coin_analyses, current_portfolio_value
            )
            
            # 5. 리스크 지표 계산
            risk_metrics = await self._calculate_risk_metrics(exchange_api)
            
            # 6. 포트폴리오 성과 분석
            performance = self._calculate_performance()
            
            result = {
                'timestamp': datetime.now(),
                'portfolio_value': current_portfolio_value,
                'available_cash': self.portfolio.available_cash,
                'total_return': (current_portfolio_value - self.initial_capital) / self.initial_capital * 100,
                'recommendations': recommendations,
                'risk_metrics': risk_metrics.__dict__ if risk_metrics else {},
                'performance': performance,
                'coin_analyses': [
                    {
                        'symbol': coin.symbol,
                        'current_price': coin.current_price,
                        'overall_score': coin.overall_score,
                        'target_allocation': coin.target_allocation * 100,
                        'current_allocation': coin.current_allocation * 100,
                        'confidence': coin.confidence
                    }
                    for coin in coin_analyses
                ],
                'rebalancing_needed': any(r['action'] != '보유' for r in recommendations)
            }
            
            self.logger.info(f"포트폴리오 분석 완료: {len(recommendations)}개 추천")
            return result
            
        except Exception as e:
            self.logger.error(f"포트폴리오 분석 오류: {str(e)}")
            return {'error': str(e)}
    
    async def _calculate_portfolio_value(self, exchange_api) -> float:
        """현재 포트폴리오 총 가치 계산"""
        try:
            total_value = self.portfolio.available_cash
            
            for symbol, position in self.portfolio.positions.items():
                try:
                    ticker = exchange_api.get_ticker(symbol)
                    current_price = float(ticker.get('trade_price', 0))
                    position_value = position['quantity'] * current_price
                    total_value += position_value
                    
                    # 포지션 정보 업데이트
                    position['current_price'] = current_price
                    position['value'] = position_value
                    position['unrealized_pnl'] = position_value - (position['quantity'] * position['avg_price'])
                    
                except Exception as e:
                    self.logger.error(f"포지션 {symbol} 가치 계산 오류: {str(e)}")
                    continue
            
            self.portfolio.total_capital = total_value
            self.portfolio.last_updated = datetime.now()
            
            return total_value
            
        except Exception as e:
            self.logger.error(f"포트폴리오 가치 계산 오류: {str(e)}")
            return self.portfolio.total_capital
    
    def _get_current_positions_value(self) -> Dict[str, float]:
        """현재 포지션별 가치 반환"""
        positions_value = {}
        for symbol, position in self.portfolio.positions.items():
            positions_value[symbol] = position.get('value', 0)
        return positions_value
    
    async def _calculate_risk_metrics(self, exchange_api) -> Optional[RiskMetrics]:
        """리스크 지표 계산"""
        try:
            if not self.performance_history or len(self.performance_history) < 30:
                return None
            
            # 최근 30일 수익률 데이터
            returns = [p['daily_return'] for p in self.performance_history[-30:]]
            
            if not returns:
                return None
            
            returns_array = np.array(returns)
            
            # 최대 낙폭 (Maximum Drawdown)
            cumulative_returns = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(np.min(drawdown))
            
            # VaR 95% (Value at Risk)
            var_95 = np.percentile(returns_array, 5)
            
            # 샤프 비율 (무위험 수익률 0%로 가정)
            sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
            
            # 변동성 (연환산)
            volatility = np.std(returns_array) * np.sqrt(252)
            
            # 베타 (비트코인 대비, 임시로 1.0 설정)
            beta = 1.0
            
            return RiskMetrics(
                portfolio_value=self.portfolio.total_capital,
                max_drawdown=max_drawdown,
                var_95=var_95,
                sharpe_ratio=sharpe_ratio,
                volatility=volatility,
                beta=beta
            )
            
        except Exception as e:
            self.logger.error(f"리스크 지표 계산 오류: {str(e)}")
            return None
    
    def _calculate_performance(self) -> Dict:
        """포트폴리오 성과 계산"""
        try:
            current_value = self.portfolio.total_capital
            total_return = (current_value - self.initial_capital) / self.initial_capital * 100
            
            performance = {
                'total_return_pct': total_return,
                'total_return_amount': current_value - self.initial_capital,
                'initial_capital': self.initial_capital,
                'current_value': current_value
            }
            
            if self.performance_history:
                # 일별 수익률
                recent_performance = self.performance_history[-7:] if len(self.performance_history) >= 7 else self.performance_history
                daily_returns = [p['daily_return'] for p in recent_performance]
                
                performance.update({
                    'avg_daily_return': np.mean(daily_returns) if daily_returns else 0,
                    'volatility': np.std(daily_returns) if len(daily_returns) > 1 else 0,
                    'win_rate': len([r for r in daily_returns if r > 0]) / len(daily_returns) if daily_returns else 0,
                    'best_day': max(daily_returns) if daily_returns else 0,
                    'worst_day': min(daily_returns) if daily_returns else 0
                })
            
            return performance
            
        except Exception as e:
            self.logger.error(f"성과 계산 오류: {str(e)}")
            return {'total_return_pct': 0}
    
    def execute_recommendation(self, recommendation: Dict, exchange_api) -> Dict:
        """추천 실행 (시뮬레이션)"""
        try:
            symbol = recommendation['symbol']
            action = recommendation['action']
            quantity = recommendation.get('quantity', 0)
            target_price = recommendation.get('target_price', recommendation['current_price'])
            
            result = {
                'symbol': symbol,
                'action': action,
                'success': False,
                'message': ''
            }
            
            if action == '매수':
                success = self._execute_buy(symbol, quantity, target_price)
                result['success'] = success
                result['message'] = f"매수 {'성공' if success else '실패'}: {quantity:.4f}개"
                
            elif action == '매도':
                success = self._execute_sell(symbol, quantity, target_price)
                result['success'] = success
                result['message'] = f"매도 {'성공' if success else '실패'}: {quantity:.4f}개"
                
            elif action == '보유':
                result['success'] = True
                result['message'] = "포지션 유지"
            
            # 거래 기록 저장
            if result['success'] and action in ['매수', '매도']:
                self._record_trade(symbol, action, quantity, target_price)
            
            return result
            
        except Exception as e:
            self.logger.error(f"추천 실행 오류: {str(e)}")
            return {'success': False, 'message': str(e)}
    
    def _execute_buy(self, symbol: str, quantity: float, price: float) -> bool:
        """매수 실행"""
        try:
            total_cost = quantity * price
            
            # 현금 부족 확인
            if total_cost > self.portfolio.available_cash:
                self.logger.warning(f"현금 부족: 필요 {total_cost:,.0f}원, 보유 {self.portfolio.available_cash:,.0f}원")
                return False
            
            # 포지션 업데이트
            if symbol in self.portfolio.positions:
                # 기존 포지션에 추가
                existing = self.portfolio.positions[symbol]
                total_quantity = existing['quantity'] + quantity
                total_cost_basis = existing['quantity'] * existing['avg_price'] + total_cost
                new_avg_price = total_cost_basis / total_quantity
                
                self.portfolio.positions[symbol].update({
                    'quantity': total_quantity,
                    'avg_price': new_avg_price
                })
            else:
                # 새 포지션 생성
                self.portfolio.positions[symbol] = {
                    'quantity': quantity,
                    'avg_price': price,
                    'value': total_cost,
                    'current_price': price,
                    'unrealized_pnl': 0
                }
            
            # 현금 차감
            self.portfolio.available_cash -= total_cost
            
            self.logger.info(f"매수 완료: {symbol} {quantity:.4f}개 @ {price:,.0f}원")
            return True
            
        except Exception as e:
            self.logger.error(f"매수 실행 오류: {str(e)}")
            return False
    
    def _execute_sell(self, symbol: str, quantity: float, price: float) -> bool:
        """매도 실행"""
        try:
            # 포지션 확인
            if symbol not in self.portfolio.positions:
                self.logger.warning(f"보유하지 않은 코인: {symbol}")
                return False
            
            position = self.portfolio.positions[symbol]
            
            # 수량 부족 확인
            if quantity > position['quantity']:
                self.logger.warning(f"수량 부족: 보유 {position['quantity']:.4f}개, 매도 시도 {quantity:.4f}개")
                return False
            
            # 매도 금액 계산
            sell_amount = quantity * price
            
            # 포지션 업데이트
            remaining_quantity = position['quantity'] - quantity
            
            if remaining_quantity > 0.0001:  # 극소량 잔량 무시
                position['quantity'] = remaining_quantity
            else:
                # 포지션 완전 청산
                del self.portfolio.positions[symbol]
            
            # 현금 증가
            self.portfolio.available_cash += sell_amount
            
            self.logger.info(f"매도 완료: {symbol} {quantity:.4f}개 @ {price:,.0f}원")
            return True
            
        except Exception as e:
            self.logger.error(f"매도 실행 오류: {str(e)}")
            return False
    
    def _record_trade(self, symbol: str, action: str, quantity: float, price: float):
        """거래 기록"""
        trade_record = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'amount': quantity * price,
            'portfolio_value': self.portfolio.total_capital
        }
        
        self.trade_history.append(trade_record)
        
        # 최근 100개 거래만 유지
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]
    
    def update_daily_performance(self):
        """일일 성과 업데이트"""
        try:
            current_value = self.portfolio.total_capital
            
            if self.performance_history:
                previous_value = self.performance_history[-1]['portfolio_value']
                daily_return = (current_value - previous_value) / previous_value
            else:
                daily_return = (current_value - self.initial_capital) / self.initial_capital
            
            performance_record = {
                'date': datetime.now().date(),
                'portfolio_value': current_value,
                'daily_return': daily_return,
                'total_return': (current_value - self.initial_capital) / self.initial_capital,
                'available_cash': self.portfolio.available_cash,
                'positions_count': len(self.portfolio.positions)
            }
            
            self.performance_history.append(performance_record)
            
            # 최근 365일만 유지
            if len(self.performance_history) > 365:
                self.performance_history = self.performance_history[-365:]
                
        except Exception as e:
            self.logger.error(f"일일 성과 업데이트 오류: {str(e)}")
    
    def check_risk_limits(self) -> List[str]:
        """리스크 한계 확인"""
        warnings = []
        
        try:
            current_value = self.portfolio.total_capital
            total_loss = (self.initial_capital - current_value) / self.initial_capital
            
            # 포트폴리오 전체 손실 한계
            if total_loss > self.risk_settings['portfolio_stop_loss']:
                warnings.append(f"포트폴리오 손실 한계 초과: {total_loss:.1%}")
            
            # 개별 포지션 크기 확인
            for symbol, position in self.portfolio.positions.items():
                position_ratio = position.get('value', 0) / current_value
                if position_ratio > self.risk_settings['max_position_size']:
                    warnings.append(f"{symbol} 포지션 크기 초과: {position_ratio:.1%}")
            
            # 일일 손실 확인 (성과 기록이 있는 경우)
            if self.performance_history and len(self.performance_history) > 1:
                daily_return = self.performance_history[-1]['daily_return']
                if daily_return < -self.risk_settings['max_daily_loss']:
                    warnings.append(f"일일 손실 한계 초과: {daily_return:.1%}")
            
        except Exception as e:
            self.logger.error(f"리스크 한계 확인 오류: {str(e)}")
            warnings.append("리스크 확인 중 오류 발생")
        
        return warnings
    
    def get_portfolio_summary(self) -> Dict:
        """포트폴리오 요약"""
        try:
            total_value = self.portfolio.total_capital
            positions_summary = []
            
            for symbol, position in self.portfolio.positions.items():
                position_value = position.get('value', 0)
                allocation = position_value / total_value if total_value > 0 else 0
                
                positions_summary.append({
                    'symbol': symbol,
                    'quantity': position['quantity'],
                    'avg_price': position['avg_price'],
                    'current_price': position.get('current_price', position['avg_price']),
                    'value': position_value,
                    'allocation': allocation * 100,
                    'unrealized_pnl': position.get('unrealized_pnl', 0)
                })
            
            return {
                'total_value': total_value,
                'available_cash': self.portfolio.available_cash,
                'cash_ratio': self.portfolio.available_cash / total_value * 100 if total_value > 0 else 100,
                'positions': positions_summary,
                'positions_count': len(self.portfolio.positions),
                'last_updated': self.portfolio.last_updated,
                'total_return': (total_value - self.initial_capital) / self.initial_capital * 100
            }
            
        except Exception as e:
            self.logger.error(f"포트폴리오 요약 생성 오류: {str(e)}")
            return {'error': str(e)}