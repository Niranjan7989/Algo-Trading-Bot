# Trading Bot

An automated trading bot that uses machine learning and technical analysis to generate trading signals and execute trades through the AngelOne broker.

## Features

- **Market Data Collection**: Fetches real-time and historical market data from multiple sources
- **Feature Engineering**: Generates technical indicators and price-based features
- **Data Preprocessing**: Cleans and prepares data for model training
- **Model Prediction**: Uses XGBoost to predict market movements
- **Signal Generation**: Generates trading signals based on model predictions
- **Risk Management**: Implements position sizing, stop losses, and risk limits
- **Order Execution**: Places and manages orders through AngelOne
- **Performance Tracking**: Monitors and reports trading performance
- **System Monitoring**: Tracks system health and sends alerts

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys and credentials:
```bash
# Broker credentials
ANGEL_ONE_API_KEY=your_api_key
ANGEL_ONE_CLIENT_ID=your_client_id
ANGEL_ONE_PASSWORD=your_password
ANGEL_ONE_TOTP_KEY=your_totp_key

# Market data API keys
ALPHA_VANTAGE_API_KEY=your_api_key
FINNHUB_API_KEY=your_api_key

# Alert settings
ALERT_EMAIL=your_email
ALERT_EMAIL_PASSWORD=your_email_password
ALERT_RECIPIENT_EMAIL=recipient_email
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

## Configuration

The bot's behavior can be configured through `config/settings.yaml`. Key settings include:

- Broker configuration
- Market data sources and symbols
- Feature engineering parameters
- Model settings
- Risk management rules
- Performance tracking metrics
- Alert thresholds

## Usage

1. Start the trading bot:
```bash
python -m trading_bot.main
```

2. Monitor the bot's activity through:
- Log files in `logs/`
- Performance reports in `data/performance/`
- Email and Telegram alerts

## Project Structure

```
trading_bot/
├── config/
│   └── settings.yaml
├── data/
│   ├── market/
│   └── performance/
├── models/
│   ├── predictor/
│   └── preprocessing/
├── trading_bot/
│   ├── broker/
│   │   ├── base.py
│   │   └── angel_one.py
│   ├── data/
│   │   ├── market_data.py
│   │   └── sentiment_data.py
│   ├── features/
│   │   └── feature_engineer.py
│   ├── models/
│   │   └── predictor.py
│   ├── monitoring/
│   │   ├── alerts.py
│   │   └── logger.py
│   ├── performance/
│   │   └── tracker.py
│   ├── preprocessing/
│   │   └── preprocessor.py
│   ├── risk/
│   │   └── manager.py
│   ├── signals/
│   │   └── generator.py
│   ├── utils/
│   │   └── helpers.py
│   └── main.py
├── tests/
├── logs/
├── cache/
├── temp/
├── requirements.txt
└── README.md
```

## Development

1. Run tests:
```bash
pytest
```

2. Format code:
```bash
black .
```

3. Check code quality:
```bash
flake8
mypy .
```

4. Build documentation:
```bash
cd docs
make html
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This trading bot is for educational purposes only. Use it at your own risk. The developers are not responsible for any financial losses incurred through the use of this software. 