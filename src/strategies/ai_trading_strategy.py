"""
AI Trading Strategy Module

This module implements AI-based trading strategies using machine learning
models and technical analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

from src.analysis.technical_analyzer import TechnicalAnalyzer
from src.analysis.market_analyzer import MarketAnalyzer
from src.exchange.exchange_api import UpbitAPI

class AITradingStrategy:
    """Class for implementing AI-based trading strategies."""
    
    def __init__(self) -> None:
        """Initialize the AITradingStrategy."""
        self.logger = logging.getLogger(__name__)
        self.technical_analyzer = TechnicalAnalyzer()
        self.market_analyzer = MarketAnalyzer()
        self.upbit = UpbitAPI()
        self.model_path = 'models/trading_model.joblib'
        self.scaler_path = 'models/feature_scaler.joblib'
        
        # Initialize or load ML model
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize or load the machine learning model."""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
            else:
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                self.scaler = StandardScaler()
        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            self.model = None
            self.scaler = None
    
    def analyze_market(self, market: str) -> Dict[str, Any]:
        """
        Analyze market using AI and generate trading signals.
        
        Args:
            market (str): Market symbol (e.g., 'KRW-BTC')
            
        Returns:
            Dict[str, Any]: Analysis results and trading signals
        """
        try:
            # Fetch market data
            candles = self.upbit.get_candles_minutes(market, unit=15)
            market_index = self.upbit.get_market_index(market)
            orderbook = self.upbit.get_orderbook(market)
            
            if not candles:
                return self._get_empty_result()
            
            # Prepare data for analysis
            df = pd.DataFrame(candles)
            
            # Get technical and market analysis
            market_data = {
                'symbol': market,
                'timeframe': '15m',
                'ohlcv': candles
            }
            
            technical_analysis = self.technical_analyzer.analyze_technical_indicators(market_data)
            market_analysis = self.market_analyzer.analyze_market(market_data)
            
            # Extract features for ML model
            features = self._extract_features(df, technical_analysis, market_analysis)
            
            # Generate AI predictions
            signals = self._generate_signals(features, market_index, orderbook)
            
            return {
                'market': market,
                'timestamp': datetime.now().isoformat(),
                'signals': signals,
                'analysis': {
                    'technical': technical_analysis,
                    'market': market_analysis
                },
                'confidence': signals.get('confidence', 0.0),
                'risk_level': signals.get('risk_level', 'UNKNOWN')
            }
            
        except Exception as e:
            self.logger.error(f"Error in market analysis: {str(e)}")
            return self._get_empty_result()
    
    def _extract_features(self,
                         df: pd.DataFrame,
                         technical_analysis: Dict[str, Any],
                         market_analysis: Dict[str, Any]) -> np.ndarray:
        """Extract features for the ML model."""
        try:
            features = []
            
            # Technical indicators
            indicators = technical_analysis.get('indicators', {})
            features.extend([
                indicators.get('RSI', {}).get('value', 50),
                indicators.get('MACD', {}).get('value', 0),
                indicators.get('BB', {}).get('width', 0)
            ])
            
            # Market conditions
            market_conditions = market_analysis.get('market_conditions', {})
            features.extend([
                market_conditions.get('momentum', 0),
                market_conditions.get('price_change_24h', 0)
            ])
            
            # Volatility
            volatility = market_analysis.get('volatility', {})
            features.extend([
                volatility.get('daily_volatility', 0),
                volatility.get('atr_percent', 0)
            ])
            
            # Volume
            volume = market_analysis.get('volume_analysis', {})
            features.extend([
                volume.get('volume_change_percent', 0),
                volume.get('price_up_volume_ratio', 1.0)
            ])
            
            features = np.array(features).reshape(1, -1)
            
            # Scale features if scaler exists
            if self.scaler is not None:
                features = self.scaler.transform(features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return np.zeros((1, 9))  # Return zero features in case of error
    
    def _generate_signals(self,
                         features: np.ndarray,
                         market_index: Dict[str, Any],
                         orderbook: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate trading signals using the ML model."""
        try:
            if self.model is None:
                return self._generate_fallback_signals(market_index, orderbook)
            
            # Get model prediction
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            # Map prediction to action
            action_map = {
                0: 'SELL',
                1: 'HOLD',
                2: 'BUY'
            }
            
            action = action_map.get(prediction, 'HOLD')
            confidence = float(max(probabilities))
            
            # Calculate risk level
            risk_level = self._calculate_risk_level(
                confidence,
                market_index.get('volatility_24h', 0),
                market_index.get('volume_change_24h', 0)
            )
            
            # Generate detailed signals
            return {
                'action': action,
                'confidence': confidence,
                'risk_level': risk_level,
                'probabilities': {
                    'sell': float(probabilities[0]),
                    'hold': float(probabilities[1]),
                    'buy': float(probabilities[2])
                },
                'market_conditions': {
                    'volatility': market_index.get('volatility_24h', 0),
                    'volume_change': market_index.get('volume_change_24h', 0),
                    'price_change': market_index.get('price_change_24h', 0)
                },
                'orderbook_analysis': self._analyze_orderbook(orderbook)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return self._generate_fallback_signals(market_index, orderbook)
    
    def _generate_fallback_signals(self,
                                 market_index: Dict[str, Any],
                                 orderbook: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate fallback signals when ML model is unavailable."""
        try:
            # Use simple rules for fallback
            volatility = market_index.get('volatility_24h', 0)
            volume_change = market_index.get('volume_change_24h', 0)
            price_change = market_index.get('price_change_24h', 0)
            
            # Determine action based on simple rules
            if price_change > 2 and volume_change > 20:
                action = 'BUY'
                confidence = 0.6
            elif price_change < -2 and volume_change > 20:
                action = 'SELL'
                confidence = 0.6
            else:
                action = 'HOLD'
                confidence = 0.5
            
            risk_level = self._calculate_risk_level(
                confidence,
                volatility,
                volume_change
            )
            
            return {
                'action': action,
                'confidence': confidence,
                'risk_level': risk_level,
                'market_conditions': {
                    'volatility': volatility,
                    'volume_change': volume_change,
                    'price_change': price_change
                },
                'orderbook_analysis': self._analyze_orderbook(orderbook)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating fallback signals: {str(e)}")
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'risk_level': 'HIGH'
            }
    
    def _calculate_risk_level(self,
                            confidence: float,
                            volatility: float,
                            volume_change: float) -> str:
        """Calculate risk level based on market conditions."""
        try:
            risk_score = 0
            
            # Add risk based on confidence
            if confidence < 0.6:
                risk_score += 2
            elif confidence < 0.8:
                risk_score += 1
            
            # Add risk based on volatility
            if volatility > 5:
                risk_score += 2
            elif volatility > 3:
                risk_score += 1
            
            # Add risk based on volume change
            if abs(volume_change) > 50:
                risk_score += 2
            elif abs(volume_change) > 30:
                risk_score += 1
            
            # Determine risk level
            if risk_score >= 4:
                return 'HIGH'
            elif risk_score >= 2:
                return 'MEDIUM'
            else:
                return 'LOW'
            
        except Exception as e:
            self.logger.error(f"Error calculating risk level: {str(e)}")
            return 'HIGH'
    
    def _analyze_orderbook(self, orderbook: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze orderbook for market depth and pressure."""
        try:
            if not orderbook:
                return {}
            
            first_order = orderbook[0]
            
            total_bid_size = first_order.get('total_bid_size', 0)
            total_ask_size = first_order.get('total_ask_size', 0)
            
            # Calculate buy/sell pressure
            if total_ask_size > 0:
                buy_pressure = total_bid_size / total_ask_size
            else:
                buy_pressure = 1.0
            
            return {
                'buy_pressure': float(buy_pressure),
                'total_bid_size': float(total_bid_size),
                'total_ask_size': float(total_ask_size),
                'is_buy_pressure_high': bool(buy_pressure > 1.2),
                'is_sell_pressure_high': bool(buy_pressure < 0.8)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing orderbook: {str(e)}")
            return {}
    
    def _get_empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'market': '',
            'timestamp': datetime.now().isoformat(),
            'signals': {
                'action': 'HOLD',
                'confidence': 0.0,
                'risk_level': 'UNKNOWN'
            },
            'analysis': {
                'technical': {},
                'market': {}
            }
        }
    
    def train_model(self, training_data: List[Dict[str, Any]]) -> None:
        """
        Train the ML model with historical data.
        
        Args:
            training_data (List[Dict[str, Any]]): Historical trading data with labels
        """
        try:
            if not training_data:
                self.logger.warning("No training data provided")
                return
            
            # Prepare training data
            X = []  # Features
            y = []  # Labels
            
            for data in training_data:
                # Extract features
                features = self._extract_features(
                    pd.DataFrame(data['candles']),
                    data['technical_analysis'],
                    data['market_analysis']
                )
                
                X.append(features.flatten())
                y.append(data['label'])  # 0: SELL, 1: HOLD, 2: BUY
            
            X = np.array(X)
            y = np.array(y)
            
            # Fit scaler
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            
            # Save model and scaler
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            self.logger.info("Model training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}") 