import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from technical_analysis import TechnicalAnalyzer

logger = logging.getLogger(__name__)

class PricePredictor:
    def __init__(self):
        self.technical_analyzer = TechnicalAnalyzer()
        self.prediction_windows = {
            'short': 7,
            'medium': 14,
            'long': 30
        }
    
    def predict_price_movement(self, df: pd.DataFrame, pattern_type: str, current_price: float) -> Dict:
        """패턴 기반 가격 움직임 예측"""
        try:
            price_changes = []
            window = 20
            
            for i in range(len(df) - window):
                segment = df.iloc[i:i+window]
                future_idx = i + window + self.prediction_windows['medium']
                
                if future_idx < len(df):
                    future_price = df['close'].iloc[future_idx]
                    change = (future_price / segment['close'].iloc[-1] - 1) * 100
                    price_changes.append(change)
            
            if price_changes:
                avg_change = np.mean(price_changes)
                std_change = np.std(price_changes)
                
                predicted_low = current_price * (1 + (avg_change - std_change)/100)
                predicted_high = current_price * (1 + (avg_change + std_change)/100)
                most_likely = current_price * (1 + avg_change/100)
                
                return {
                    'predicted_low': predicted_low,
                    'predicted_high': predicted_high,
                    'most_likely': most_likely,
                    'confidence': min(len(price_changes) / 10, 1.0)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"가격 예측 중 오류 발생: {e}")
            return None
    
    def generate_predictions(self, df: pd.DataFrame, patterns: List[Dict]) -> List[Dict]:
        """모든 발견된 패턴에 대한 가격 예측 생성"""
        try:
            predictions = []
            current_price = df['close'].iloc[-1]
            
            for pattern in patterns:
                target_price = pattern.get('target')
                if not target_price:
                    continue
                
                prediction = self.predict_price_movement(
                    df, 
                    pattern['name'].lower().replace(' ', '_'),
                    current_price
                )
                
                if prediction:
                    predictions.append({
                        'pattern': pattern['name'],
                        'type': pattern['pattern_type'],
                        'target_price': target_price,
                        'predicted_range': {
                            'low': prediction['predicted_low'],
                            'high': prediction['predicted_high'],
                            'most_likely': prediction['most_likely']
                        },
                        'confidence': prediction['confidence'],
                        'timeframe': f"{self.prediction_windows['medium']}일",
                        'key_levels': self._get_key_levels(df, current_price)
                    })
            
            return predictions
            
        except Exception as e:
            logger.error(f"예측 생성 중 오류 발생: {e}")
            return []
    
    def _get_key_levels(self, df: pd.DataFrame, current_price: float) -> Dict[str, List[float]]:
        """주요 가격 레벨 찾기"""
        try:
            # 볼린저 밴드 기반 레벨
            upper, middle, lower = self.technical_analyzer.calculate_bollinger_bands(df)
            
            # 이동평균선 기반 레벨
            ma_data = self.technical_analyzer.calculate_moving_averages(df)
            
            support_levels = [
                lower.iloc[-1],
                ma_data['MA20'].iloc[-1],
                ma_data['MA60'].iloc[-1]
            ]
            
            resistance_levels = [
                upper.iloc[-1],
                ma_data['MA5'].iloc[-1] * 1.02,
                middle.iloc[-1] * 1.05
            ]
            
            # 레벨 필터링
            support_levels = [level for level in support_levels if level < current_price]
            resistance_levels = [level for level in resistance_levels if level > current_price]
            
            return {
                'support': sorted(support_levels, reverse=True)[:3],
                'resistance': sorted(resistance_levels)[:3]
            }
            
        except Exception as e:
            logger.error(f"주요 레벨 계산 중 오류 발생: {e}")
            return {'support': [], 'resistance': []} 