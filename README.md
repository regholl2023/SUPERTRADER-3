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
![AI Stock Trading System Flowchart](https://mermaid.ink/img/pako:eNp9U-9PwjAQ_VcufdaAQDARiYkbJkQTE6OJH_oFaXv0S9euawcCxv-9K2MwRBq-tL3e3Xv33pWL0KEkIuhC0iCQb0wMOeaCiSSVQVB4cojFQThQ8L7ELChEqngk2sC4m_A88xm14MaIKkL1aJhqN5GfY-FfmPBY3j1H5-Gu3-3eDWC5eJrtF4J6DYsxcXhDKI5hMQZx8QwCYwgizMQiWUNNGFrOTLHKcmEkSHSEaEtJqQPL5aZwT20Yn8KWgVpJ5F5yD7AqGZd59k_lHqfVH-q4Nn1XjI08BzSLHd_wJrDHJZhHtcDNCrcVIvFpyXQe50QxsEfATKA7V5aMrMQq904IXPGRFEWq9fMgSeTrGTQf-q2LVmd43uk99obQOhv0Ly6H_fEQTp-enxbrzITqt4-gkw5gMYgS4dpIlgfBpQYrpGYEF5NJeEn3RnN9dWPDcJc1QUrHREnLIl1TtdQw4nt3jQbLYPc7ZyLnPRV4jMc8Uh1-4XzZMfakVeOKUbZCyJiUDCxKNVNOt5mGmJf0ZygwXjVrRqZx0Y9jRzQRSaP8aJW20OmOMjBRSjBLZIJEBV2Iaf4JXBTvXQ)

[View larger image](https://mermaid.ink/img/pako:eNp9U-9PwjAQ_VcufdaAQDARiYkbJkQTE6OJH_oFaXv0S9euawcCxv-9K2MwRBq-tL3e3Xv33pWL0KEkIuhC0iCQb0wMOeaCiSSVQVB4cojFQThQ8L7ELChEqngk2sC4m_A88xm14MaIKkL1aJhqN5GfY-FfmPBY3j1H5-Gu3-3eDWC5eJrtF4J6DYsxcXhDKI5hMQZx8QwCYwgizMQiWUNNGFrOTLHKcmEkSHSEaEtJqQPL5aZwT20Yn8KWgVpJ5F5yD7AqGZd59k_lHqfVH-q4Nn1XjI08BzSLHd_wJrDHJZhHtcDNCrcVIvFpyXQe50QxsEfATKA7V5aMrMQq904IXPGRFEWq9fMgSeTrGTQf-q2LVmd43uk99obQOhv0Ly6H_fEQTp-enxbrzITqt4-gkw5gMYgS4dpIlgfBpQYrpGYEF5NJeEn3RnN9dWPDcJc1QUrHREnLIl1TtdQw4nt3jQbLYPc7ZyLnPRV4jMc8Uh1-4XzZMfakVeOKUbZCyJiUDCxKNVNOt5mGmJf0ZygwXjVrRqZx0Y9jRzQRSaP8aJW20OmOMjBRSjBLZIJEBV2Iaf4JXBTvXQ)

For those who want to view or edit the flowchart, here's the Mermaid code:

```mermaid
graph TD
    A[User Input] --> B[Data Collection]
    B --> C[AI Analysis]
    C --> D[Trade Execution]
    D --> E[Reporting]
    E --> F[User Review]

    subgraph A[1. User Input]
        A1[Slack Bot Interaction]
        A2[Enter Stock Symbol]
    end

    subgraph B[2. Data Collection]
        B1[Fetch Stock Data]
        B2[Gather News Articles]
        B3[Collect Market Indicators]
    end

    subgraph C[3. AI Analysis]
        C1[Perform Technical Analysis]
        C2[Conduct Sentiment Analysis]
        C3[Generate GPT Model Prediction]
    end

    subgraph D[4. Trade Execution]
        D1[Make Buy/Sell Decision]
        D2[Place Order]
    end

    subgraph E[5. Reporting]
        E1[Record in Database]
        E2[Send Slack Notification]
        E3[Update Dashboard]
    end

    subgraph F[6. User Review]
        F1[View Results]
        F2[Monitor Performance]
    end

    A1 --> A2
    B1 --> B2 --> B3
    C1 --> C2 --> C3
    D1 --> D2
    E1 --> E2 --> E3
    F1 --> F2

    classDef default fill:#f9f,stroke:#333,stroke-width:2px;
    classDef subProcess fill:#bbf,stroke:#333,stroke-width:1px;
    class A,B,C,D,E,F default;
    class A1,A2,B1,B2,B3,C1,C2,C3,D1,D2,E1,E2,E3,F1,F2 subProcess;


## Contributing
Contributions to improve the system are welcome. Please follow these steps:
1. Fork the repository
2. Create a new branch
3. Make your changes and commit them
4. Create a pull request with a description of your changes

## License


## Disclaimer
This system is for educational purposes only. Always consult with a financial advisor before making investment decisions.