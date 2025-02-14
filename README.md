# AutoCoin - AI-Powered Cryptocurrency Analysis Tool 🚀

## Overview

AutoCoin is a sophisticated cryptocurrency analysis tool that combines real-time news monitoring, market analysis, and AI-powered insights to help traders make informed decisions. The system integrates multiple data sources and uses advanced AI models for sentiment analysis and market prediction.

## Features 🌟

### 1. News Analysis System
- Real-time cryptocurrency news monitoring
- AI-powered sentiment analysis using Google's Gemini Pro
- Automatic categorization of bullish/bearish news
- Importance scoring based on multiple factors
- Support for major cryptocurrencies (BTC, ETH, XRP, etc.)

### 2. Market Analysis
- Technical indicator calculations
- Market manipulation detection
- Support and resistance level analysis
- Trend analysis and prediction
- Real-time price monitoring

### 3. Performance Optimization
- Efficient caching system
- Rate limiting for API calls
- Performance monitoring and metrics
- Asynchronous operations support

### 4. Database System
- Structured data storage for news and market data
- Historical data management
- Efficient data retrieval and backup
- Automated data cleanup

## Installation 🔧

1. Clone the repository:
```bash
git clone https://github.com/yourusername/autocoin.git
cd autocoin
```

2. Install TA-Lib (Required for technical analysis):

### Windows:
```bash
# 1. Download ta-lib-0.4.0-msvc.zip from
# http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-msvc.zip

# 2. Unzip to C:\ta-lib

# 3. Add system environment variable
# Variable name: TA_LIBRARY_PATH
# Variable value: C:\ta-lib\c\lib

# 4. Add C:\ta-lib\c\include to system PATH

# Now you can install the Python wrapper
pip install TA-Lib
```

### Linux:
```bash
# Install TA-Lib
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

# Now you can install the Python wrapper
pip install TA-Lib
```

### macOS:
```bash
# Using Homebrew
brew install ta-lib

# Now you can install the Python wrapper
pip install TA-Lib
```

3. Install other required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables in `.env` file:
```env
UPBIT_ACCESS_KEY=your_upbit_access_key
UPBIT_SECRET_KEY=your_upbit_secret_key
NEWS_API_KEY=your_newsapi_key
GEMINI_API_KEY=your_gemini_api_key
```

## Troubleshooting 🔍

### TA-Lib Installation Issues

1. Windows - "ta_libc.h not found" error:
   - Make sure you've downloaded and extracted ta-lib to C:\ta-lib
   - Verify system environment variables are set correctly
   - Try restarting your IDE/terminal after setting environment variables

2. Linux - Missing dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install build-essential
   sudo apt-get install python3-dev
   ```

3. General TA-Lib issues:
   - Make sure you have a C++ compiler installed
   - For Windows: Install Visual Studio Build Tools
   - For Linux: Install build-essential package

## Usage 📚

### News Analysis
```python
from latest_news import print_news_analysis

# Analyze news for a single coin
print_news_analysis("BTC")  # For Bitcoin
print_news_analysis("ETH")  # For Ethereum

# Example output includes:
# - Bullish/Bearish news categorization
# - News importance scores
# - Market impact analysis
# - Overall sentiment analysis
```

### Market Analysis
```python
from market_analysis import MarketAnalyzer

analyzer = MarketAnalyzer()
# Get market analysis for Bitcoin
analysis = analyzer.analyze_market("BTC")
```

## Output Examples 📊

### News Analysis Output
```
=== BTC 뉴스 분석 결과 ===
분석 시간: 2024-03-20 15:30:00
분석된 뉴스 수: 10

[호재 뉴스]
📈 Bitcoin Surges Past $65,000
출처: CoinDesk
시간: 2024-03-20 15:25
중요도: 0.85
시장 영향도: 0.90

[악재 뉴스]
📉 Regulatory Concerns Emerge
출처: Reuters
시간: 2024-03-20 15:20
중요도: 0.75
시장 영향도: 0.80
```

## Project Structure 📁

```
autocoin/
├── latest_news.py      # News analysis system
├── market_analysis.py  # Market analysis tools
├── database.py        # Database management
├── performance.py     # Performance optimization
├── requirements.txt   # Project dependencies
└── .env              # Environment variables
```

## Dependencies 📦

- Python 3.8+
- Required packages (see requirements.txt):
  - pandas
  - numpy
  - requests
  - python-dotenv
  - google-generativeai
  - textblob
  - scikit-learn
  - SQLAlchemy
  - aiosqlite
  - ta-lib

## Contributing 🤝

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer ⚠️

This tool is for informational purposes only. Cryptocurrency trading involves significant risk, and you should never trade with money you cannot afford to lose. Always do your own research and consult with financial advisors before making investment decisions.

## Support 💬

For support, please open an issue in the GitHub repository or contact the maintainers directly.

---
Made with ❤️ by Needitem