"""
Fear & Greed Analysis Module

This module calculates and analyzes the Fear & Greed Index for cryptocurrency markets,
providing insights into market sentiment and potential trading opportunities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import requests

class FearGreedAnalyzer:
    """Class for analyzing Fear & Greed Index in cryptocurrency markets."""
    
    def __init__(self) -> None:
        """Initialize the FearGreedAnalyzer with default settings."""
        self.logger = logging.getLogger(__name__)
        self.metrics_weight = {
            'market_momentum': 0.25,
            'volatility': 0.25,
            'trading_volume': 0.25,
            'social_sentiment': 0.25
        }

    def calculate_fear_greed_index(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the Fear & Greed Index based on various market metrics.
        
        Args:
            market_data (Dict[str, Any]): Market data containing OHLCV and other metrics
            
        Returns:
            Dict[str, Any]: Fear & Greed Index analysis results
        """
        try:
            # Convert OHLCV data to DataFrame
            df = pd.DataFrame(market_data['ohlcv'])
            
            # Calculate individual components
            momentum = self._calculate_market_momentum(df)
            volatility = self._calculate_volatility(df)
            volume = self._analyze_volume(df)
            sentiment = self._analyze_social_sentiment(market_data['symbol'])
            
            # Calculate weighted index
            index_value = (
                momentum * self.metrics_weight['market_momentum'] +
                volatility * self.metrics_weight['volatility'] +
                volume * self.metrics_weight['trading_volume'] +
                sentiment * self.metrics_weight['social_sentiment']
            )
            
            # Determine market sentiment category
            sentiment_category = self._get_sentiment_category(index_value)
            
            return {
                'index_value': float(index_value),
                'sentiment_category': sentiment_category,
                'components': {
                    'market_momentum': float(momentum),
                    'volatility': float(volatility),
                    'trading_volume': float(volume),
                    'social_sentiment': float(sentiment)
                },
                'metadata': {
                    'symbol': market_data['symbol'],
                    'calculation_time': datetime.now().isoformat(),
                    'data_timeframe': market_data.get('timeframe', 'unknown')
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Fear & Greed Index: {str(e)}")
            raise Exception(f"Failed to calculate Fear & Greed Index: {str(e)}")

    def _calculate_market_momentum(self, df: pd.DataFrame) -> float:
        """
        Calculate market momentum component.
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            float: Market momentum score (0-100)
        """
        try:
            # Calculate price changes
            df['returns'] = df['close'].pct_change()
            
            # Calculate moving averages
            ma_7 = df['close'].rolling(window=7).mean()
            ma_30 = df['close'].rolling(window=30).mean()
            
            # Calculate momentum indicators
            roc = (df['close'] - df['close'].shift(7)) / df['close'].shift(7) * 100
            ma_trend = (ma_7 - ma_30) / ma_30 * 100
            
            # Combine indicators
            momentum = (roc.iloc[-1] + ma_trend.iloc[-1]) / 2
            
            # Normalize to 0-100 scale
            normalized = 50 + (momentum * 5)  # Adjust multiplier as needed
            return float(np.clip(normalized, 0, 100))
            
        except Exception as e:
            self.logger.error(f"Error calculating market momentum: {str(e)}")
            return 50.0  # Return neutral value on error

    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """
        Calculate volatility component.
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            float: Volatility score (0-100)
        """
        try:
            # Calculate daily volatility
            returns = df['close'].pct_change()
            current_vol = returns.std() * np.sqrt(365)
            
            # Calculate historical volatility (30-day rolling)
            hist_vol = returns.rolling(window=30).std() * np.sqrt(365)
            avg_hist_vol = hist_vol.mean()
            
            # Compare current volatility to historical
            vol_ratio = current_vol / avg_hist_vol if avg_hist_vol != 0 else 1
            
            # Convert to 0-100 scale (higher volatility = more fear)
            normalized = 100 - (50 * vol_ratio)  # Adjust multiplier as needed
            return float(np.clip(normalized, 0, 100))
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {str(e)}")
            return 50.0  # Return neutral value on error

    def _analyze_volume(self, df: pd.DataFrame) -> float:
        """
        Analyze trading volume component.
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            float: Volume analysis score (0-100)
        """
        try:
            # Calculate volume metrics
            current_vol = df['volume'].iloc[-1]
            avg_vol = df['volume'].rolling(window=30).mean()
            vol_ratio = current_vol / avg_vol.iloc[-1] if avg_vol.iloc[-1] != 0 else 1
            
            # Calculate volume trend
            vol_change = df['volume'].pct_change()
            vol_trend = vol_change.rolling(window=7).mean().iloc[-1]
            
            # Combine metrics
            volume_score = (vol_ratio + (1 + vol_trend)) / 2
            
            # Normalize to 0-100 scale
            normalized = 50 * volume_score
            return float(np.clip(normalized, 0, 100))
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume: {str(e)}")
            return 50.0  # Return neutral value on error

    def _analyze_social_sentiment(self, symbol: str) -> float:
        """
        Analyze social media sentiment component.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            float: Social sentiment score (0-100)
        """
        try:
            # This is a placeholder for social sentiment analysis
            # In a real implementation, you would:
            # 1. Fetch social media data (Twitter, Reddit, etc.)
            # 2. Analyze sentiment using NLP
            # 3. Calculate weighted sentiment score
            
            # For now, return a neutral value
            return 50.0
            
        except Exception as e:
            self.logger.error(f"Error analyzing social sentiment: {str(e)}")
            return 50.0  # Return neutral value on error

    def _get_sentiment_category(self, index_value: float) -> str:
        """
        Get sentiment category based on index value.
        
        Args:
            index_value (float): Fear & Greed Index value
            
        Returns:
            str: Sentiment category
        """
        if index_value >= 90:
            return "Extreme Greed"
        elif index_value >= 75:
            return "Greed"
        elif index_value >= 60:
            return "Moderate Greed"
        elif index_value >= 40:
            return "Neutral"
        elif index_value >= 25:
            return "Moderate Fear"
        elif index_value >= 10:
            return "Fear"
        else:
            return "Extreme Fear"

    def get_historical_sentiment(self,
                               days: int = 30,
                               symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get historical sentiment data.
        
        Args:
            days (int): Number of days of historical data
            symbol (Optional[str]): Trading pair symbol
            
        Returns:
            Dict[str, Any]: Historical sentiment data
        """
        try:
            # This is a placeholder for historical sentiment data
            # In a real implementation, you would:
            # 1. Fetch historical market data
            # 2. Calculate Fear & Greed Index for each day
            # 3. Return the historical values
            
            dates = pd.date_range(end=datetime.now(), periods=days)
            values = np.random.normal(50, 15, days)  # Random values for demonstration
            
            sentiment_history = []
            for date, value in zip(dates, values):
                sentiment_history.append({
                    'date': date.isoformat(),
                    'value': float(np.clip(value, 0, 100)),
                    'category': self._get_sentiment_category(float(np.clip(value, 0, 100)))
                })
            
            return {
                'symbol': symbol,
                'history': sentiment_history,
                'metadata': {
                    'days': days,
                    'start_date': dates[0].isoformat(),
                    'end_date': dates[-1].isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting historical sentiment: {str(e)}")
            raise Exception(f"Failed to get historical sentiment: {str(e)}")

    def get_market_sentiment_summary(self) -> Dict[str, Any]:
        """
        Get overall market sentiment summary.
        
        Returns:
            Dict[str, Any]: Market sentiment summary
        """
        try:
            # This is a placeholder for market sentiment summary
            # In a real implementation, you would:
            # 1. Calculate Fear & Greed Index for major cryptocurrencies
            # 2. Analyze market trends and patterns
            # 3. Generate market summary and recommendations
            
            return {
                'overall_sentiment': "Neutral",
                'market_summary': "Market showing balanced fear and greed levels",
                'risk_level': "Moderate",
                'trading_recommendation': "Consider balanced position sizing",
                'metadata': {
                    'calculation_time': datetime.now().isoformat(),
                    'data_sources': ['price', 'volume', 'social']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market sentiment summary: {str(e)}")
            raise Exception(f"Failed to get market sentiment summary: {str(e)}") 