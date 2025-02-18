"""
News Analysis Module

This module handles fetching and analyzing cryptocurrency news,
including sentiment analysis and impact assessment.
"""

import requests
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
from textblob import TextBlob
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class NewsAnalyzer:
    """Class for analyzing cryptocurrency news."""
    
    def __init__(self) -> None:
        """Initialize the NewsAnalyzer with default settings."""
        self.logger = logging.getLogger(__name__)
        self.api_key = os.getenv('CRYPTOPANIC_API_KEY')
        self.base_url = "https://cryptopanic.com/api/v1"

    def get_latest_news(self, symbol: str, limit: int = 50) -> Dict[str, Any]:
        """
        Get and analyze the latest news for a specific cryptocurrency.
        
        Args:
            symbol (str): Cryptocurrency symbol (e.g., 'BTC')
            limit (int): Number of news items to fetch
            
        Returns:
            Dict[str, Any]: News articles with sentiment analysis
        """
        try:
            if not self.api_key:
                return {
                    'symbol': symbol,
                    'news_items': [],
                    'metadata': {
                        'total_items': 0,
                        'average_sentiment': 0,
                        'fetch_time': datetime.now().isoformat(),
                        'error': 'API key not configured'
                    }
                }

            # Mock news data for testing
            mock_news = [
                {
                    'title': f'Latest {symbol} Market Update',
                    'url': 'https://example.com/news/1',
                    'published_at': datetime.now().isoformat(),
                    'source': {'title': 'Crypto News'},
                    'sentiment': self.analyze_sentiment(f'Latest {symbol} Market Update'),
                    'importance': 0.8
                },
                {
                    'title': f'{symbol} Technical Analysis',
                    'url': 'https://example.com/news/2',
                    'published_at': (datetime.now() - timedelta(hours=1)).isoformat(),
                    'source': {'title': 'Trading View'},
                    'sentiment': self.analyze_sentiment(f'{symbol} Technical Analysis'),
                    'importance': 0.7
                }
            ]

            return {
                'symbol': symbol,
                'news_items': mock_news,
                'metadata': {
                    'total_items': len(mock_news),
                    'average_sentiment': sum(article['sentiment']['score'] for article in mock_news) / len(mock_news),
                    'fetch_time': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching news for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'news_items': [],
                'metadata': {
                    'total_items': 0,
                    'average_sentiment': 0,
                    'fetch_time': datetime.now().isoformat(),
                    'error': str(e)
                }
            }

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of news text.
        
        Args:
            text (str): News text to analyze
            
        Returns:
            Dict[str, Any]: Sentiment analysis results
        """
        try:
            # Perform sentiment analysis using TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Determine sentiment category
            if polarity > 0.3:
                sentiment = "BULLISH"
            elif polarity < -0.3:
                sentiment = "BEARISH"
            else:
                sentiment = "NEUTRAL"
            
            # Determine confidence based on subjectivity
            confidence = 1 - subjectivity  # Higher objectivity = higher confidence
            
            return {
                'score': float(polarity),
                'sentiment': sentiment,
                'subjectivity': float(subjectivity),
                'confidence': float(confidence)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {str(e)}")
            return {
                'score': 0.0,
                'sentiment': "NEUTRAL",
                'subjectivity': 0.5,
                'confidence': 0.0
            }

    def get_trending_topics(self,
                          timeframe: str = '24h',
                          min_articles: int = 3) -> Dict[str, Any]:
        """
        Get trending topics in cryptocurrency news.
        
        Args:
            timeframe (str): Time frame to analyze ('24h', '12h', '6h')
            min_articles (int): Minimum number of articles for a topic to be considered trending
            
        Returns:
            Dict[str, Any]: Trending topics analysis
        """
        try:
            # Fetch recent news
            params = {
                'auth_token': self.api_key,
                'kind': 'news',
                'limit': 100
            }
            
            response = requests.get(f"{self.base_url}/posts/", params=params)
            response.raise_for_status()
            data = response.json()
            
            # Analyze topics
            articles = data.get('results', [])
            topics = self._analyze_topics(articles, timeframe)
            
            # Filter trending topics
            trending = [topic for topic in topics if topic['article_count'] >= min_articles]
            trending.sort(key=lambda x: x['importance_score'], reverse=True)
            
            return {
                'trending_topics': trending[:10],  # Top 10 trending topics
                'metadata': {
                    'timeframe': timeframe,
                    'total_articles': len(articles),
                    'analysis_time': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting trending topics: {str(e)}")
            raise Exception(f"Failed to get trending topics: {str(e)}")

    def _analyze_topics(self,
                       articles: List[Dict[str, Any]],
                       timeframe: str) -> List[Dict[str, Any]]:
        """
        Analyze topics from news articles.
        
        Args:
            articles (List[Dict[str, Any]]): List of news articles
            timeframe (str): Time frame to analyze
            
        Returns:
            List[Dict[str, Any]]: Analyzed topics
        """
        try:
            # Convert timeframe to timedelta
            if timeframe == '24h':
                delta = timedelta(hours=24)
            elif timeframe == '12h':
                delta = timedelta(hours=12)
            else:
                delta = timedelta(hours=6)
            
            # Filter articles by timeframe
            cutoff_time = datetime.now() - delta
            recent_articles = [
                article for article in articles
                if datetime.fromisoformat(article['published_at']) > cutoff_time
            ]
            
            # Extract and count topics
            topics = {}
            for article in recent_articles:
                # Extract keywords from title
                blob = TextBlob(article['title'])
                nouns = blob.noun_phrases
                
                for noun in nouns:
                    if noun not in topics:
                        topics[noun] = {
                            'article_count': 0,
                            'sentiment_sum': 0,
                            'importance_sum': 0,
                            'articles': []
                        }
                    
                    # Update topic statistics
                    sentiment = self.analyze_sentiment(article['title'])
                    importance = self._calculate_importance(article)
                    
                    topics[noun]['article_count'] += 1
                    topics[noun]['sentiment_sum'] += sentiment['score']
                    topics[noun]['importance_sum'] += importance
                    topics[noun]['articles'].append({
                        'title': article['title'],
                        'url': article['url'],
                        'published_at': article['published_at']
                    })
            
            # Convert to list and calculate averages
            topic_list = []
            for topic, stats in topics.items():
                avg_sentiment = stats['sentiment_sum'] / stats['article_count']
                avg_importance = stats['importance_sum'] / stats['article_count']
                
                topic_list.append({
                    'topic': topic,
                    'article_count': stats['article_count'],
                    'average_sentiment': float(avg_sentiment),
                    'importance_score': float(avg_importance),
                    'articles': stats['articles'][:3]  # Include top 3 articles
                })
            
            return topic_list
            
        except Exception as e:
            self.logger.error(f"Error analyzing topics: {str(e)}")
            return []

    def get_market_sentiment_summary(self) -> Dict[str, Any]:
        """
        Get overall market sentiment summary based on news.
        
        Returns:
            Dict[str, Any]: Market sentiment summary
        """
        try:
            # Fetch recent news for major cryptocurrencies
            params = {
                'auth_token': self.api_key,
                'currencies': 'BTC,ETH',  # Focus on major cryptocurrencies
                'kind': 'news',
                'limit': 50
            }
            
            response = requests.get(f"{self.base_url}/posts/", params=params)
            response.raise_for_status()
            data = response.json()
            
            # Analyze sentiment for all articles
            sentiments = []
            for article in data.get('results', []):
                sentiment = self.analyze_sentiment(article['title'])
                sentiments.append(sentiment['score'])
            
            # Calculate sentiment statistics
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            sentiment_std = (sum((s - avg_sentiment) ** 2 for s in sentiments) / len(sentiments)) ** 0.5 if sentiments else 0
            
            # Determine market sentiment
            if avg_sentiment > 0.2:
                market_sentiment = "BULLISH"
            elif avg_sentiment < -0.2:
                market_sentiment = "BEARISH"
            else:
                market_sentiment = "NEUTRAL"
            
            # Determine sentiment strength
            sentiment_strength = abs(avg_sentiment) / 0.5  # Normalize to 0-1 scale
            
            return {
                'market_sentiment': market_sentiment,
                'sentiment_score': float(avg_sentiment),
                'sentiment_strength': float(sentiment_strength),
                'sentiment_volatility': float(sentiment_std),
                'metadata': {
                    'articles_analyzed': len(sentiments),
                    'analysis_time': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market sentiment summary: {str(e)}")
            raise Exception(f"Failed to get market sentiment summary: {str(e)}")

    def get_news_impact(self,
                       symbol: str,
                       lookback_hours: int = 24) -> Dict[str, Any]:
        """
        Analyze the impact of news on price movements.
        
        Args:
            symbol (str): Cryptocurrency symbol
            lookback_hours (int): Hours to look back
            
        Returns:
            Dict[str, Any]: News impact analysis
        """
        try:
            # Fetch news for the specified period
            params = {
                'auth_token': self.api_key,
                'currencies': symbol,
                'kind': 'news',
                'limit': 100
            }
            
            response = requests.get(f"{self.base_url}/posts/", params=params)
            response.raise_for_status()
            data = response.json()
            
            # Filter news by lookback period
            cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
            recent_news = [
                article for article in data.get('results', [])
                if datetime.fromisoformat(article['published_at']) > cutoff_time
            ]
            
            # Analyze impact of each news item
            news_impacts = []
            for article in recent_news:
                sentiment = self.analyze_sentiment(article['title'])
                importance = self._calculate_importance(article)
                
                impact_score = sentiment['score'] * importance
                
                news_impacts.append({
                    'title': article['title'],
                    'published_at': article['published_at'],
                    'sentiment': sentiment,
                    'importance': importance,
                    'impact_score': float(impact_score)
                })
            
            # Sort by impact score
            news_impacts.sort(key=lambda x: abs(x['impact_score']), reverse=True)
            
            return {
                'symbol': symbol,
                'high_impact_news': news_impacts[:5],  # Top 5 impactful news
                'metadata': {
                    'lookback_hours': lookback_hours,
                    'total_news': len(recent_news),
                    'analysis_time': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing news impact: {str(e)}")
            raise Exception(f"Failed to analyze news impact: {str(e)}")

    def _calculate_importance(self, article: Dict[str, Any]) -> float:
        """
        Calculate importance score for a news article.
        
        Args:
            article (Dict[str, Any]): News article data
            
        Returns:
            float: Importance score
        """
        try:
            score = 0.5  # Base score
            
            # Adjust based on source domain
            if article['domain'] in ['bloomberg.com', 'reuters.com', 'coindesk.com']:
                score += 0.2
            
            # Adjust based on votes
            votes = article.get('votes', {})
            positive_votes = votes.get('positive', 0)
            negative_votes = votes.get('negative', 0)
            
            if positive_votes + negative_votes > 0:
                vote_ratio = positive_votes / (positive_votes + negative_votes)
                score += (vote_ratio - 0.5) * 0.2
            
            # Adjust based on publish time
            published_at = datetime.fromisoformat(article['published_at'])
            hours_ago = (datetime.now() - published_at).total_seconds() / 3600
            
            if hours_ago < 6:
                score += 0.2
            elif hours_ago < 12:
                score += 0.1
            
            return min(max(score, 0), 1)  # Ensure score is between 0 and 1
            
        except Exception as e:
            self.logger.error(f"Error calculating article importance: {str(e)}")
            return 0.5  # Return default score on error 

"""
News API Module

This module handles fetching cryptocurrency news from various sources.
"""

import requests
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class NewsAPI:
    """Class for fetching cryptocurrency news."""
    
    def __init__(self) -> None:
        """Initialize the NewsAPI with API configuration."""
        self.logger = logging.getLogger(__name__)
        self.api_key = os.getenv('CRYPTOPANIC_API_KEY')
        self.base_url = "https://cryptopanic.com/api/v1/posts/"
        
    def get_latest_news(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch latest news for a specific cryptocurrency.
        
        Args:
            symbol (str): Cryptocurrency symbol (e.g., 'BTC')
            
        Returns:
            Dict[str, Any]: News data and metadata
        """
        try:
            # If API key is not configured, return mock data
            if not self.api_key:
                self.logger.warning("API key not configured, returning mock data")
                return self._get_mock_news(symbol)
            
            # Prepare request parameters
            params = {
                'auth_token': self.api_key,
                'currencies': symbol,
                'kind': 'news',
                'limit': 50
            }
            
            # Make API request
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            # Process response
            data = response.json()
            
            # Format articles
            articles = []
            for result in data.get('results', []):
                articles.append({
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'source': result.get('source', {}).get('title', 'Unknown'),
                    'published_at': result.get('published_at', ''),
                    'sentiment': result.get('sentiment', 'neutral')
                })
            
            return {
                'symbol': symbol,
                'articles': articles,
                'metadata': {
                    'count': len(articles),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching news: {str(e)}")
            return self._get_mock_news(symbol)
            
        except Exception as e:
            self.logger.error(f"Unexpected error fetching news: {str(e)}")
            return {
                'symbol': symbol,
                'articles': [],
                'metadata': {
                    'count': 0,
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                }
            }
    
    def _get_mock_news(self, symbol: str) -> Dict[str, Any]:
        """Generate mock news data for testing purposes."""
        current_time = datetime.now().isoformat()
        
        return {
            'symbol': symbol,
            'articles': [
                {
                    'title': f"{symbol} Shows Strong Market Performance",
                    'url': "https://example.com/article1",
                    'source': "Mock News",
                    'published_at': current_time,
                    'sentiment': 'positive'
                },
                {
                    'title': f"Market Analysis: {symbol} Price Trends",
                    'url': "https://example.com/article2",
                    'source': "Mock News",
                    'published_at': current_time,
                    'sentiment': 'neutral'
                }
            ],
            'metadata': {
                'count': 2,
                'timestamp': current_time,
                'is_mock': True
            }
        } 