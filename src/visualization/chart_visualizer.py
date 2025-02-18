"""
Chart Visualization Module

This module handles the visualization of cryptocurrency market data
using interactive charts and technical analysis overlays.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import numpy as np

class ChartVisualizer:
    """Class for visualizing cryptocurrency market data."""
    
    def __init__(self) -> None:
        """Initialize the ChartVisualizer with default settings."""
        self.logger = logging.getLogger(__name__)
        self.default_height = 800
        self.default_width = 1200

    def plot_analysis(self,
                     df: pd.DataFrame,
                     indicators: Optional[List[str]] = None) -> go.Figure:
        """
        Plot market analysis with indicators.
        
        Args:
            df (pd.DataFrame): Market data DataFrame
            indicators (Optional[List[str]]): List of indicators to plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        try:
            # Create figure with secondary y-axis
            fig = make_subplots(rows=2, cols=1,
                              shared_xaxes=True,
                              vertical_spacing=0.03,
                              row_heights=[0.7, 0.3])

            # Add candlestick
            fig.add_trace(go.Candlestick(x=df['timestamp'],
                                       open=df['open'],
                                       high=df['high'],
                                       low=df['low'],
                                       close=df['close'],
                                       name='OHLC'),
                         row=1, col=1)

            # Add volume bars
            colors = ['red' if row['open'] > row['close'] else 'green' for _, row in df.iterrows()]
            fig.add_trace(go.Bar(x=df['timestamp'],
                               y=df['volume'],
                               name='Volume',
                               marker_color=colors),
                         row=2, col=1)

            if indicators:
                self._add_indicators(fig, df, indicators)

            # Update layout
            fig.update_layout(
                title='Market Analysis',
                yaxis_title='Price',
                yaxis2_title='Volume',
                xaxis_rangeslider_visible=False,
                height=800
            )

            return fig

        except Exception as e:
            self.logger.error(f"Error creating chart: {str(e)}")
            return self._create_error_figure(str(e))

    def _add_indicators(self,
                       fig: go.Figure,
                       df: pd.DataFrame,
                       indicators: List[str]) -> None:
        """Add technical indicators to the chart."""
        try:
            for indicator in indicators:
                if indicator == 'MA':
                    # Add Moving Averages
                    ma_20 = df['close'].rolling(window=20).mean()
                    ma_50 = df['close'].rolling(window=50).mean()
                    
                    fig.add_trace(go.Scatter(x=df['timestamp'],
                                           y=ma_20,
                                           name='MA20',
                                           line=dict(color='blue')),
                                row=1, col=1)
                    
                    fig.add_trace(go.Scatter(x=df['timestamp'],
                                           y=ma_50,
                                           name='MA50',
                                           line=dict(color='orange')),
                                row=1, col=1)

                elif indicator == 'RSI':
                    # Add RSI
                    rsi = self._calculate_rsi(df['close'])
                    fig.add_trace(go.Scatter(x=df['timestamp'],
                                           y=rsi,
                                           name='RSI',
                                           line=dict(color='purple')),
                                row=2, col=1)
                    
                    # Add RSI reference lines
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                elif indicator == 'MACD':
                    # Add MACD
                    macd_line, signal_line, macd_hist = self._calculate_macd(df['close'])
                    
                    fig.add_trace(go.Scatter(x=df['timestamp'],
                                           y=macd_line,
                                           name='MACD',
                                           line=dict(color='blue')),
                                row=2, col=1)
                    
                    fig.add_trace(go.Scatter(x=df['timestamp'],
                                           y=signal_line,
                                           name='Signal',
                                           line=dict(color='orange')),
                                row=2, col=1)
                    
                    colors = ['red' if val < 0 else 'green' for val in macd_hist]
                    fig.add_trace(go.Bar(x=df['timestamp'],
                                       y=macd_hist,
                                       name='MACD Histogram',
                                       marker_color=colors),
                                row=2, col=1)

        except Exception as e:
            self.logger.error(f"Error adding indicators: {str(e)}")

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series(index=prices.index)

    def _calculate_macd(self,
                       prices: pd.Series,
                       fast_period: int = 12,
                       slow_period: int = 26,
                       signal_period: int = 9) -> tuple:
        """Calculate MACD indicator."""
        try:
            # Calculate MACD line
            ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
            ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            
            # Calculate Signal line
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            
            # Calculate MACD histogram
            macd_hist = macd_line - signal_line
            
            return macd_line, signal_line, macd_hist
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {str(e)}")
            return pd.Series(index=prices.index), pd.Series(index=prices.index), pd.Series(index=prices.index)

    def _create_error_figure(self, error_message: str) -> go.Figure:
        """Create an error figure when chart creation fails."""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {error_message}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            title='Error in Chart Visualization',
            height=400
        )
        return fig

    def plot_comparison(self,
                       market_data_list: List[Dict[str, Any]],
                       normalize: bool = True) -> go.Figure:
        """
        Create a comparison chart for multiple assets.
        
        Args:
            market_data_list (List[Dict[str, Any]]): List of market data for different assets
            normalize (bool): Whether to normalize prices for comparison
            
        Returns:
            go.Figure: Comparison chart
        """
        try:
            if not market_data_list:
                return self._create_error_figure("No market data available for comparison")
            
            fig = go.Figure()
            
            for market_data in market_data_list:
                df = pd.DataFrame(market_data['ohlcv'])
                prices = df['close']
                
                if normalize:
                    prices = (prices - prices.min()) / (prices.max() - prices.min())
                
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=prices,
                        name=market_data['symbol'],
                        mode='lines'
                    )
                )
            
            title = "Price Comparison" if not normalize else "Normalized Price Comparison"
            
            fig.update_layout(
                height=self.default_height,
                width=self.default_width,
                title_text=title,
                showlegend=True,
                template='plotly_dark',
                yaxis_title="Price" if not normalize else "Normalized Price"
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating comparison chart: {str(e)}")
            return self._create_error_figure(str(e))

    def plot_correlation_matrix(self, market_data_list: List[Dict[str, Any]]) -> go.Figure:
        """
        Create a correlation matrix heatmap.
        
        Args:
            market_data_list (List[Dict[str, Any]]): List of market data for different assets
            
        Returns:
            go.Figure: Correlation matrix heatmap
        """
        try:
            if not market_data_list:
                return self._create_error_figure("No market data available for correlation")
            
            # Create price DataFrame
            price_data = {}
            for market_data in market_data_list:
                df = pd.DataFrame(market_data['ohlcv'])
                price_data[market_data['symbol']] = df['close']
            
            price_df = pd.DataFrame(price_data)
            
            # Calculate correlation matrix
            corr_matrix = price_df.corr()
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmin=-1,
                zmax=1
            ))
            
            fig.update_layout(
                height=self.default_height,
                width=self.default_width,
                title_text="Asset Correlation Matrix",
                template='plotly_dark'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating correlation matrix: {str(e)}")
            return self._create_error_figure(str(e)) 