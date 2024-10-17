# AI Stock Advisor

AI Stock Advisor is an advanced system that leverages artificial intelligence to analyze the stock market and suggest investment decisions. This system utilizes various data sources to perform comprehensive market analysis and uses OpenAI's GPT model to generate investment decisions.

## Key Features

- Real-time stock price and chart data collection
- Technical indicator calculation (Bollinger Bands, RSI, MACD, etc.)
- News data collection and analysis
- Integration of VIX index and Fear & Greed index
- YouTube video transcript analysis
- AI-based investment decisions using OpenAI GPT model
- SQLite databases for recording investment decisions and performance
- Sharing analysis results via Slack

## Installation

1. Install required Python packages:

```
pip install -r requirements.txt
```

2. Create a `.env` file and set the following environment variables:

```
ROBINHOOD_USERNAME=your_robinhood_username
ROBINHOOD_PASSWORD=your_robinhood_password
ROBINHOOD_TOTP_CODE=your_robinhood_totp_code
SERPAPI_API_KEY=your_serpapi_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
SLACK_BOT_TOKEN=your_slack_bot_token
SLACK_APP_TOKEN=your_slack_app_token
SLACK_WEBHOOK_URL=your_slack_webhook_url
OPENAI_API_KEY=your_openai_api_key
```

3. Store Larry Williams' trading strategy in the `strategy.txt` file.

## Usage

1. Run the main script:

```
python main.py
```

2. In Slack, mention the bot and input a stock symbol to request analysis:

```
@YourBotName AAPL
```

3. The bot will perform analysis on the stock and post results in the Slack channel.

## Key Classes and Functions

- `AIStockAdvisorSystem`: Main class for stock analysis and investment decisions
- `get_current_price()`: Retrieve current stock price
- `get_chart_data()`: Calculate chart data and technical indicators
- `get_news()`: Collect news data from various sources
- `get_vix_index()`: Retrieve VIX index
- `get_fear_and_greed_index()`: Retrieve Fear & Greed index
- `ai_stock_analysis()`: Perform AI-based analysis using OpenAI GPT
- `analyze_and_post_to_slack()`: Post analysis results to Slack

## Databases

The system uses two SQLite databases:

1. `ai_stock_analysis_records.db`: Store individual analysis records
2. `ai_stock_performance.db`: Track performance of investment decisions

## Slack Bot Configuration

The Slack bot handles the following events:

- `app_mention`: Triggered when the bot is mentioned
- `message`: General message events (for logging purposes)

## System Flowchart

Here's a simplified flowchart of how the AI Stock Advisor system works:

<antArtifact identifier="ai-stock-advisor-flowchart" type="application/vnd.ant.mermaid" title="AI Stock Advisor System Flowchart">
graph TD
    A[Start] --> B[Receive Stock Symbol]
    B --> C[Collect Data]
    C --> D[Calculate Technical Indicators]
    C --> E[Fetch News]
    C --> F[Get VIX Index]
    C --> G[Get Fear & Greed Index]
    C --> H[Get YouTube Transcript]
    D --> I[AI Analysis]
    E --> I
    F --> I
    G --> I
    H --> I
    I --> J[Generate Investment Decision]
    J --> K[Record Decision in Database]
    K --> L[Post Results to Slack]
    L --> M[End]
