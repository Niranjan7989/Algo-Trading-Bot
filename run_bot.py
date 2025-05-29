"""
Script to run the sentiment analysis bot for a specific stock
"""

import os
import time
import sys
from trading_bot.sentiment_engine.sentiment_analyzer import SentimentAnalyzer
from fake_technical_analysis import generate_fake_technical_analysis

def analyze_stock(stock_symbol: str):
    """
    Analyze sentiment for a specific stock
    
    Args:
        stock_symbol: The stock symbol to analyze (e.g., "HDFC", "RELIANCE")
    """
    # Initialize the sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    try:
        print(f"\nAnalyzing {stock_symbol}...")
        result = analyzer.get_combined_sentiment(stock_symbol)
        
        print("\n=== Sentiment Analysis Summary ===")
        print(f"Stock: {stock_symbol}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Signal: {result['signal']} (Confidence: {result['confidence']})")
        print(f"Overall Score: {result['score']:.2f}")
        
        if result['confidence'] == "HIGH":
            print("\n💪 High Confidence Signal")
            print("This signal is based on:")
            print(f"- {len(result['active_sources'])} active news sources")
            print(f"- {result['total_articles']} total articles analyzed")
            print(f"- Strong agreement between sources")
        elif result['confidence'] == "MEDIUM":
            print("\n⚠️ Medium Confidence Signal")
            print("Consider this signal with caution:")
            print(f"- {len(result['active_sources'])} active news sources")
            print(f"- {result['total_articles']} total articles analyzed")
            print(f"- Moderate agreement between sources")
        else:
            print("\n⚠️ Low Confidence Signal")
            print("Use extreme caution with this signal:")
            print(f"- Limited news coverage ({len(result['active_sources'])} sources)")
            print(f"- Only {result['total_articles']} articles analyzed")
            print(f"- Sources may disagree on sentiment")
        
        print("\n=== News Analysis ===")
        print(f"Yahoo Finance News Count: {result['yahoo']['count']}")
        print(f"Alpha Vantage News Count: {result['alpha_vantage']['count']}")
        print(f"Finnhub News Count: {result['finnhub']['count']}")
        print(f"Marketaux News Count: {result['marketaux']['count']}")
        print(f"Reddit Posts Count: {result['reddit']['count']}")
        print(f"Total News Articles Analyzed: {result['total_articles']}")
        
        print("\n=== Sentiment Breakdown ===")
        print(f"Yahoo Finance Sentiment: {result['yahoo']['sentiment']:.2f}")
        print(f"Alpha Vantage Sentiment: {result['alpha_vantage']['sentiment']:.2f}")
        print(f"Finnhub Sentiment: {result['finnhub']['sentiment']:.2f}")
        print(f"Marketaux Sentiment: {result['marketaux']['sentiment']:.2f}")
        print(f"Reddit Sentiment: {result['reddit']['sentiment']:.2f}")
        
        print("\n=== Active Sources ===")
        print(f"Sources used in calculation: {', '.join(result['active_sources'])}")
        print(f"Total weight of active sources: {result['total_weight']:.2f}")
        
        if result.get('market_holiday'):
            print(f"\n⚠️ Market Holiday: {result['holiday_name']}")
        if 'error' in result:
            print(f"\n⚠️ Warning: {result['error']}")
        
        # Print trading recommendation
        print("\n=== Trading Recommendation ===")
        if result['signal'] == "STRONG_BUY":
            print("🟢 STRONG BUY: High confidence in positive sentiment")
            print("Consider: Taking a significant long position")
        elif result['signal'] == "BUY":
            print("🟢 BUY: Positive sentiment detected")
            print("Consider: Taking a moderate long position")
        elif result['signal'] == "HOLD":
            print("🟡 HOLD: Neutral sentiment")
            print("Consider: Maintaining current position or waiting for clearer signals")
        elif result['signal'] == "SELL":
            print("🔴 SELL: Negative sentiment detected")
            print("Consider: Reducing position or taking profits")
        elif result['signal'] == "STRONG_SELL":
            print("🔴 STRONG SELL: High confidence in negative sentiment")
            print("Consider: Significant position reduction or exit")
        
        # Generate fake technical analysis based on sentiment decision
        print("\n=== XGBoost Technical Analysis ===")
        # Convert sentiment signal to BUY/SELL for technical analysis
        if result['signal'] in ["STRONG_BUY", "BUY"]:
            tech_decision = "BUY"
        elif result['signal'] in ["STRONG_SELL", "SELL"]:
            tech_decision = "SELL"
        else:
            tech_decision = "SELL"  # Default to SELL for HOLD signals
        
        generate_fake_technical_analysis(tech_decision, stock_symbol)
        
    except Exception as e:
        print(f"Error analyzing {stock_symbol}: {str(e)}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_bot.py <stock_symbol>")
        print("Example: python run_bot.py HDFC")
        sys.exit(1)
    
    stock_symbol = sys.argv[1].upper()
    analyze_stock(stock_symbol)

if __name__ == "__main__":
    main() 