# Robinhood_AI_Trading
# AI-Powered Stock Trading System

## Overview
This project is an AI-powered stock trading system that combines real-time market data, news sentiment, technical indicators, and machine learning to make informed trading decisions. It integrates with Robinhood for trading, uses OpenAI's GPT model for analysis, and communicates results via Slack.

## Features
- Real-time stock data retrieval and analysis
- Technical indicator calculations (Bollinger Bands, RSI, MACD, Moving Averages)
- News sentiment analysis from Google News and Alpha Vantage
- Fear and Greed Index integration
- AI-driven trading decisions using OpenAI's GPT model
- Automated trading execution (buy/sell orders)
- Slack bot for user interaction and result delivery
- Trading history tracking and analysis
- Multi-language support (English and Korean)

## Components
1. **AIStockTrading Class**: Core logic for stock analysis and trading decisions
2. **Slack Bot**: Handles user commands and delivers trading insights
3. **Data Visualization**: Streamlit dashboard for real-time trading data visualization

## Technologies Used
- Python
- Robinhood API (robin_stocks)
- OpenAI GPT
- SQLite
- Slack API
- Streamlit
- Pandas for data manipulation
- Various APIs (Google News, Alpha Vantage, Fear and Greed Index)

## Setup and Installation
1. Clone the repository
2. Install required dependencies: `pip install -r requirements.txt`
3. Set up environment variables in a `.env` file:
```
ROBINHOOD_USERNAME=your_username
ROBINHOOD_PASSWORD=your_password
ROBINHOOD_TOTP_CODE=your_totp_code
SERPAPI_API_KEY=your_serpapi_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
SLACK_BOT_TOKEN=your_slack_bot_token
SLACK_APP_TOKEN=your_slack_app_token
SLACK_WEBHOOK_URL=your_slack_webhook_url
OPENAI_API_KEY=your_openai_api_key
```
4. Run the main script: `python main.py`
5. For the dashboard, run: `streamlit run dashboard.py`

## Usage
1. Interact with the bot in Slack by mentioning it with a stock symbol:
```@YourBotName AAPL ```
2. The bot will analyze the stock and provide trading recommendations
3. View real-time trading data and history on the Streamlit dashboard

## Data Flow
1. User requests analysis for a stock via Slack
2. System fetches real-time market data, news, and calculates technical indicators
3. Data is sent to OpenAI's GPT model for analysis
4. Trading decision is made based on AI analysis
5. Decision is executed (if applicable), recorded in the database, and sent to user via Slack
6. Real-time updates are reflected on the Streamlit dashboard

## Flowchart

+----------------+     +-------------------+     +-------------+
|   User Input   |     |   Data Collection |     | AI Analysis |
| (Slack Bot)    | --> | - Stock Data      | --> | - Technical |
| - Stock Symbol |     | - News Articles   |     | - Sentiment |
|                |     | - Market Indicators|     | - GPT Model |
+----------------+     +-------------------+     +-------------+
|                                              |
|                                              |
v                                              v
+----------------+     +-------------------+     +-------------+
| User Review    |     |     Reporting     |     |   Trade     |
| - View Results | <-- | - Database Record | <-- | Execution   |
| - Monitor      |     | - Slack Notify    |     | - Buy/Sell  |
|   Performance  |     | - Dashboard Update|     | - Order     |
+----------------+     +-------------------+     +-------------+

이 순서도는 AI 주식 거래 시스템의 주요 단계를 보여줍니다:
1. **User Input**: 사용자가 Slack 봇을 통해 주식 심볼을 입력합니다.
2. **Data Collection**: 시스템이 주식 데이터, 뉴스 기사, 시장 지표 등을 수집합니다.
3. **AI Analysis**: 수집된 데이터를 바탕으로 기술적 분석, 감성 분석, GPT 모델 예측을 수행합니다.
4. **Trade Execution**: 분석 결과를 바탕으로 매수/매도 결정을 내리고 주문을 실행합니다.
5. **Reporting**: 거래 결과를 데이터베이스에 기록하고, Slack으로 알림을 보내며, 대시보드를 업데이트합니다.
6. **User Review**: 사용자가 결과를 확인하고 성과를 모니터링합니다.

## Contributing
Contributions to improve the system are welcome. Please follow these steps:
1. Fork the repository
2. Create a new branch
3. Make your changes and commit them
4. Create a pull request with a description of your changes

## License


## Disclaimer
This system is for educational purposes only. Always consult with a financial advisor before making investment decisions.