"""
Crypto Trading Application Main Entry Point.

This module serves as the main entry point for the crypto trading application.
It initializes all necessary components and orchestrates the trading process.
"""

import os
import sys
from pathlib import Path
import streamlit as st
from typing import Any, Dict, List

# Add src directory to Python path
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from src.analysis.fear_greed import FearGreedAnalyzer
from src.analysis.market_analysis import MarketAnalyzer
from src.analysis.pattern_analysis import PatternAnalyzer
from src.analysis.technical_analysis import TechnicalAnalyzer
from src.data.data_fetcher import DataFetcher
from src.news.latest_news import NewsAnalyzer
from src.strategies.investment_strategy import InvestmentStrategy, TradingStrategyType
from src.strategies.trading_strategy import TradingStrategy
from src.utils.database import Database
from src.utils.error_handler import ErrorHandler
from src.utils.performance import PerformanceMonitor
from src.visualization.chart_visualizer import ChartVisualizer

# Set up page config
st.set_page_config(
    page_title="Crypto Trading Analysis",
    page_icon="📈",
    layout="wide"
)

class CryptoTradingApp:
    """Main application class that orchestrates the crypto trading process."""

    def __init__(self) -> None:
        """Initialize the crypto trading application and its components."""
        # Core utilities
        self.db = Database()
        self.error_handler = ErrorHandler()
        self.performance_monitor = PerformanceMonitor()
        
        # Analysis components
        self.data_fetcher = DataFetcher()
        self.market_analyzer = MarketAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.pattern_analyzer = PatternAnalyzer()
        self.fear_greed_analyzer = FearGreedAnalyzer()
        
        # Strategy components
        self.investment_strategy = InvestmentStrategy(TradingStrategyType.MODERATE)
        self.trading_strategy = TradingStrategy()
        
        # Visualization and news
        self.chart_visualizer = ChartVisualizer()
        self.news_analyzer = NewsAnalyzer()

    def run(self) -> None:
        """Execute the main trading process."""
        try:
            st.title("Crypto Trading Analysis")

            # Sidebar
            with st.sidebar:
                st.header("Settings")
                symbol = st.selectbox(
                    "Select Cryptocurrency",
                    ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
                    index=0
                )
                
                timeframe = st.selectbox(
                    "Select Timeframe",
                    ["1h", "4h", "1d"],
                    index=2
                )
                
                strategy_type = st.selectbox(
                    "Select Trading Strategy",
                    [strategy.name for strategy in TradingStrategyType],
                    index=1
                )
                
                self.investment_strategy = InvestmentStrategy(
                    TradingStrategyType[strategy_type]
                )

            # Main content
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Market Analysis")
                try:
                    # Get market data and analysis
                    market_data = self.data_fetcher.fetch_data(symbol, timeframe)
                    
                    # Display chart
                    fig = self.chart_visualizer.plot_analysis(
                        market_data,
                        indicators=["MA", "RSI", "MACD"]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    self.error_handler.log_error("Error in market analysis", e)
                    st.error("Failed to load market analysis. Please try again later.")

            with col2:
                st.subheader("Trading Signals")
                try:
                    # Get trading signals
                    signals = self.investment_strategy.generate_signals(market_data)
                    
                    # Display signals
                    st.write("Current Position:", signals.get("position", "NEUTRAL"))
                    st.write("Risk Level:", signals.get("risk_level", "MODERATE"))
                    st.write("Confidence Score:", f"{signals.get('confidence', 0.0):.2f}")
                    
                    if signals.get("entry_points"):
                        st.write("Entry Points:", signals["entry_points"])
                    if signals.get("exit_points"):
                        st.write("Exit Points:", signals["exit_points"])
                        
                except Exception as e:
                    self.error_handler.log_error("Error in signal generation", e)
                    st.error("Failed to generate trading signals. Please try again later.")

            # News section
            st.subheader("Latest Crypto News")
            try:
                news_data = self.news_analyzer.get_latest_news(symbol.split('/')[0])
                if news_data:
                    for article in news_data[:5]:  # Display top 5 news
                        st.write(f"**{article['title']}**")
                        st.write(f"*{article['date']}*")
                        st.write(article['summary'])
                        st.write("---")
                else:
                    st.info("No recent news available.")
                    
            except Exception as e:
                self.error_handler.log_error("Error fetching news", e)
                st.error("Failed to load news. Please try again later.")

        except Exception as e:
            self.error_handler.log_error("Application error", e)
            st.error("An unexpected error occurred. Please try again later.")

def main() -> None:
    """Initialize and run the crypto trading application."""
    app = CryptoTradingApp()
    app.run()

if __name__ == "__main__":
    main() 