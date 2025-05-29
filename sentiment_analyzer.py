"""
Sentiment analysis module for the trading bot
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import requests
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import yfinance as yf
import logging

class SentimentAnalyzer:
    """Sentiment analysis for market data"""
    
    def __init__(self):
        """Initialize sentiment analyzer"""
        self.logger = logging.getLogger(__name__)
        self.sia = SentimentIntensityAnalyzer()
        self._download_nltk_data()
        
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
            
    def analyze_news_sentiment(self, symbol: str, 
                             lookback_days: int = 7) -> Dict[str, float]:
        """
        Analyze news sentiment for a symbol
        
        Args:
            symbol: Stock symbol
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary with sentiment scores
        """
        # Get news data (placeholder - implement actual news API)
        news_data = self._get_news_data(symbol, lookback_days)
        
        if not news_data:
            return {
                'sentiment_score': 0.0,
                'sentiment_magnitude': 0.0,
                'news_count': 0
            }
            
        # Analyze sentiment for each news item
        sentiments = []
        for news in news_data:
            sentiment = self.sia.polarity_scores(news['text'])
            sentiments.append(sentiment['compound'])
            
        # Calculate aggregate sentiment
        sentiment_score = np.mean(sentiments)
        sentiment_magnitude = np.std(sentiments)
        
        return {
            'sentiment_score': sentiment_score,
            'sentiment_magnitude': sentiment_magnitude,
            'news_count': len(news_data)
        }
        
    def analyze_social_sentiment(self, symbol: str,
                               lookback_days: int = 7) -> Dict[str, float]:
        """
        Analyze social media sentiment for a symbol
        
        Args:
            symbol: Stock symbol
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary with sentiment scores
        """
        # Get social media data (placeholder - implement actual social media API)
        social_data = self._get_social_data(symbol, lookback_days)
        
        if not social_data:
            return {
                'social_sentiment': 0.0,
                'social_volume': 0,
                'sentiment_change': 0.0
            }
            
        # Analyze sentiment for each post
        sentiments = []
        for post in social_data:
            sentiment = self.sia.polarity_scores(post['text'])
            sentiments.append(sentiment['compound'])
            
        # Calculate social sentiment metrics
        social_sentiment = np.mean(sentiments)
        social_volume = len(social_data)
        
        # Calculate sentiment change
        if len(sentiments) > 1:
            sentiment_change = sentiments[-1] - sentiments[0]
        else:
            sentiment_change = 0.0
            
        return {
            'social_sentiment': social_sentiment,
            'social_volume': social_volume,
            'sentiment_change': sentiment_change
        }
        
    def analyze_market_sentiment(self, symbol: str,
                               lookback_days: int = 7) -> Dict[str, float]:
        """
        Analyze overall market sentiment
        
        Args:
            symbol: Stock symbol
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary with market sentiment scores
        """
        # Get price data
        price_data = yf.download(symbol, 
                               period=f'{lookback_days}d',
                               interval='1d')
        
        if price_data.empty:
            return {
                'market_sentiment': 0.0,
                'volatility': 0.0,
                'volume_sentiment': 0.0
            }
            
        # Calculate price momentum
        returns = price_data['Close'].pct_change()
        price_momentum = returns.mean()
        
        # Calculate volatility
        volatility = returns.std()
        
        # Calculate volume sentiment
        volume_change = price_data['Volume'].pct_change().mean()
        
        # Combine metrics into market sentiment
        market_sentiment = (
            price_momentum * 0.4 +
            (1 - volatility) * 0.3 +
            volume_change * 0.3
        )
        
        return {
            'market_sentiment': market_sentiment,
            'volatility': volatility,
            'volume_sentiment': volume_change
        }
        
    def get_combined_sentiment(self, symbol: str,
                             lookback_days: int = 7) -> Dict[str, float]:
        """
        Get combined sentiment analysis
        
        Args:
            symbol: Stock symbol
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary with combined sentiment scores
        """
        # Get individual sentiment scores
        news_sentiment = self.analyze_news_sentiment(symbol, lookback_days)
        social_sentiment = self.analyze_social_sentiment(symbol, lookback_days)
        market_sentiment = self.analyze_market_sentiment(symbol, lookback_days)
        
        # Combine sentiment scores
        combined_score = (
            news_sentiment['sentiment_score'] * 0.3 +
            social_sentiment['social_sentiment'] * 0.3 +
            market_sentiment['market_sentiment'] * 0.4
        )
        
        # Calculate sentiment strength
        sentiment_strength = (
            news_sentiment['sentiment_magnitude'] * 0.3 +
            abs(social_sentiment['sentiment_change']) * 0.3 +
            market_sentiment['volatility'] * 0.4
        )
        
        return {
            'combined_score': combined_score,
            'sentiment_strength': sentiment_strength,
            'news_sentiment': news_sentiment['sentiment_score'],
            'social_sentiment': social_sentiment['social_sentiment'],
            'market_sentiment': market_sentiment['market_sentiment'],
            'news_count': news_sentiment['news_count'],
            'social_volume': social_sentiment['social_volume'],
            'volatility': market_sentiment['volatility']
        }
        
    def _get_news_data(self, symbol: str, 
                      lookback_days: int) -> List[Dict]:
        """
        Get news data (placeholder - implement actual news API)
        
        Args:
            symbol: Stock symbol
            lookback_days: Number of days to look back
            
        Returns:
            List of news items
        """
        # TODO: Implement actual news API integration
        # This is a placeholder that returns sample data
        return [
            {'text': f'Positive news about {symbol}', 'date': datetime.now()},
            {'text': f'Mixed sentiment for {symbol}', 'date': datetime.now()},
            {'text': f'Negative development for {symbol}', 'date': datetime.now()}
        ]
        
    def _get_social_data(self, symbol: str,
                        lookback_days: int) -> List[Dict]:
        """
        Get social media data (placeholder - implement actual social media API)
        
        Args:
            symbol: Stock symbol
            lookback_days: Number of days to look back
            
        Returns:
            List of social media posts
        """
        # TODO: Implement actual social media API integration
        # This is a placeholder that returns sample data
        return [
            {'text': f'Bullish on {symbol}', 'date': datetime.now()},
            {'text': f'Bearish sentiment for {symbol}', 'date': datetime.now()},
            {'text': f'Neutral view on {symbol}', 'date': datetime.now()}
        ] 