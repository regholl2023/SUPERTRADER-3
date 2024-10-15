import os
import sys
import json
import logging
import sqlite3
import requests
import pandas as pd
from typing import Dict, Any
from datetime import datetime
from pydantic import BaseModel

import pyotp
import robin_stocks as r
from dotenv import load_dotenv
from openai import OpenAI
import fear_and_greed
from deep_translator import GoogleTranslator
from youtube_transcript_api import YouTubeTranscriptApi
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# 환경 변수 로드 및 로깅 설정
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 설정 클래스
class Config:
    ROBINHOOD_USERNAME = os.getenv("username")
    ROBINHOOD_PASSWORD = os.getenv("password")
    ROBINHOOD_TOTP_CODE = os.getenv("totpcode")
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
    ALPHA_VANTAGE_API_KEY = os.getenv("Alpha_Vantage_API_KEY")
    SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
    SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
    SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    INITIAL_BALANCE = 1000

# 거래 결정 모델
class TradingDecision(BaseModel):
    decision: str
    percentage: int
    reason: str

# AI Stock Trading 클래스
class AIStockTrading:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.logger = logging.getLogger(f"{symbol}_analyzer")
        self.login = self._get_login()
        self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.db_connection = self._setup_database()
        self.initial_balance = Config.INITIAL_BALANCE
        self.balance = self._get_initial_balance()
        self.shares = self._get_initial_shares()
        self.trading_value = self._calculate_trading_value()

    def _setup_database(self):
        conn = sqlite3.connect('trading_data.db')
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trading_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            timestamp DATETIME,
            decision TEXT,
            percentage INTEGER,
            reason TEXT,
            balance REAL,
            shares INTEGER,
            trading_value REAL,
            profit_rate REAL
        )
        ''')
        conn.commit()
        return conn

    def _get_login(self):
        totp = pyotp.TOTP(Config.ROBINHOOD_TOTP_CODE).now()
        self.logger.info(f"Current OTP: {totp}")
        login = r.robinhood.login(Config.ROBINHOOD_USERNAME, Config.ROBINHOOD_PASSWORD, mfa_code=totp)
        self.logger.info("Successfully logged in to Robinhood")
        return login

    def get_chart_data(self):
        self.logger.info(f"Fetching chart data for {self.symbol}")
        monthly_historicals = r.robinhood.stocks.get_stock_historicals(
            self.symbol, interval="day", span="3month", bounds="regular"
        )
        daily_historicals = r.robinhood.stocks.get_stock_historicals(
            self.symbol, interval="5minute", span="day", bounds="regular"
        )
        monthly_df = self._process_df(monthly_historicals)
        daily_df = self._process_df(daily_historicals)
        return self._add_indicators(monthly_df, daily_df)

    def _process_df(self, historicals):
        df = pd.DataFrame(historicals)
        df = df[['begins_at', 'open_price', 'close_price', 'high_price', 'low_price', 'volume']]
        df['begins_at'] = pd.to_datetime(df['begins_at'])
        for col in ['open_price', 'close_price', 'high_price', 'low_price']:
            df[col] = df[col].astype(float)
        df['volume'] = df['volume'].astype(int)
        df.columns = ['Date', 'Open', 'Close', 'High', 'Low', 'Volume']
        df.set_index('Date', inplace=True)
        return df

    def _add_indicators(self, monthly_df, daily_df):
        for df in [monthly_df, daily_df]:
            df = self._calculate_bollinger_bands(df)
            df = self._calculate_rsi(df)
            df = self._calculate_macd(df)
        monthly_df = self._calculate_moving_averages(monthly_df)
        return monthly_df, daily_df

    def _calculate_bollinger_bands(self, df: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        df['SMA'] = df['Close'].rolling(window=window).mean()
        df['STD'] = df['Close'].rolling(window=window).std()
        df['Upper_Band'] = df['SMA'] + (df['STD'] * num_std)
        df['Lower_Band'] = df['SMA'] - (df['STD'] * num_std)
        return df

    def _calculate_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df

    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        df['EMA_fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
        df['EMA_slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
        df['MACD'] = df['EMA_fast'] - df['EMA_slow']
        df['Signal_Line'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
        return df

    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        windows = [10, 20, 60]
        for window in windows:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        return df

    def get_news(self):
        return {
            "google_news": self._get_news_from_google(),
            "alpha_vantage_news": self._get_news_from_alpha_vantage()
        }

    def _get_news_from_google(self):
        self.logger.info("Fetching news from Google")
        url = "https://www.searchapi.io/api/v1/search"
        params = {
            "api_key": Config.SERPAPI_API_KEY,
            "engine": "google_news",
            "q": self.symbol,
            "num": 5
        }
        headers = {"Accept": "application/json"}
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            news_items = []
            for result in data.get('organic_results', [])[:5]:
                news_items.append({
                    'title': result['title'],
                    'date': result['date']
                })
            self.logger.info(f"Retrieved {len(news_items)} news items from Google")
            return news_items
        except Exception as e:
            self.logger.error(f"Error during Google News API request: {e}")
            return []

    def _get_news_from_alpha_vantage(self):
        self.logger.info("Fetching news from Alpha Vantage")
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={self.symbol}&apikey={Config.ALPHA_VANTAGE_API_KEY}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if "feed" not in data:
                self.logger.warning("No news data found in Alpha Vantage response")
                return []
            news_items = []
            for item in data["feed"][:10]:
                title = item.get("title", "제목 없음")
                time_published = item.get("time_published", "날짜 없음")
                if time_published != "날짜 없음":
                    dt = datetime.strptime(time_published, "%Y%m%dT%H%M%S")
                    time_published = dt.strftime("%Y-%m-%d %H:%M:%S")
                news_items.append({
                    'title': title,
                    'pubDate': time_published
                })
            self.logger.info(f"Retrieved {len(news_items)} news items from Alpha Vantage")
            return news_items
        except Exception as e:
            self.logger.error(f"Error during Alpha Vantage API request: {e}")
            return []

    def get_fear_and_greed_index(self):
        self.logger.info("Fetching Fear and Greed Index")
        fgi = fear_and_greed.get()
        return {
            "value": fgi.value,
            "description": fgi.description,
            "last_update": fgi.last_update.isoformat()
        }

    def get_youtube_transcript(self):
        video_id = 'rWl9ehSIiXc'
        self.logger.info(f"Fetching YouTube transcript for video ID: {video_id}")
        try:
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
            full_transcript = " ".join(item['text'] for item in transcript_data)
            self.logger.info(f"Retrieved transcript with {len(full_transcript)} characters")
            return full_transcript.strip()
        except Exception as e:
            self.logger.error(f"Error fetching YouTube transcript: {str(e)}")
            return f"An error occurred: {str(e)}"

    def ai_trading(self):
        monthly_df, daily_df = self.get_chart_data()
        news = self.get_news()
        youtube_transcript = self.get_youtube_transcript()
        fgi = self.get_fear_and_greed_index()

        self.logger.info("Sending request to OpenAI")
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an expert in Stock investing. Analyze the provided data including technical indicators, market data, recent news headlines, the Fear and Greed Index, YouTube video transcript, and the chart image. Tell me whether to buy, sell, or hold at the moment. Consider the following in your analysis:
                            - Technical indicators and market data
                            - Recent news headlines and their potential impact on Stock price
                            - The Fear and Greed Index and its implications
                            - Overall market sentiment
                            - Insights from the YouTube video transcript

                            Particularly important is to always refer to the trading method of 'Mark Minervini', a legendary stock investor, to assess the current situation and make trading decisions. Mark Minervini's trading method is as follows:

                            {youtube_transcript}

                            Based on this trading method, analyze the current market situation and make a judgment by synthesizing it with the provided data.

                            Respond with:
                            1. A decision (buy, sell, or hold)
                            2. If the decision is 'buy', provide a intensity expressed as a percentage ratio (1 to 100).
                            If the decision is 'sell', provide a intensity expressed as a percentage ratio (1 to 100).
                            If the decision is 'hold', set the percentage to 0.
                            3. A reason for your decision

                            Ensure that the percentage is an integer between 1 and 100 for buy/sell decisions, and exactly 0 for hold decisions.
                            Your percentage should reflect the strength of your conviction in the decision based on the analyzed data."""},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "symbol": self.symbol,
                                "monthly_data": monthly_df.to_json(),
                                "daily_data": daily_df.to_json(),
                                "fear_and_greed_index": fgi,
                                "news": news
                            })
                        }
                    ]
                }
            ],
            max_tokens=4095,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "trading_decision",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "decision": {"type": "string", "enum": ["buy", "sell", "hold"]},
                            "percentage": {"type": "integer"},
                            "reason": {"type": "string"}
                        },
                        "required": ["decision", "percentage", "reason"],
                        "additionalProperties": False
                    }
                }
            }
        )
        result = TradingDecision.model_validate_json(response.choices[0].message.content)
        self.logger.info("Received response from OpenAI")

        reason_kr = self._translate_to_korean(result.reason)

        self.logger.info(f"### AI Decision: {result.decision.upper()} ###")
        self.logger.info(f"### Percentage: {result.percentage} ###")
        self.logger.info(f"### Reason: {result.reason} ###")

        self._send_slack_message(result, reason_kr, news, fgi)

        shares_to_trade = self.calculate_shares_to_trade(result.percentage)

        if result.decision == "buy":
            self.buy(shares_to_trade)
        elif result.decision == "sell":
            shares_to_sell = min(shares_to_trade, self.shares)
            self.sell(shares_to_sell)
        else:  # hold
            self.logger.info("Holding current position")

        profit_rate = self._calculate_profit_rate()
        self.logger.info(f"Current balance: ${self.balance:.2f}")
        self.logger.info(f"Current shares: {self.shares}")
        self.logger.info(f"Current position value: ${self.trading_value:.2f}")
        self.logger.info(f"Current profit rate: {profit_rate:.2f}%")

        # Record the trading decision and current state
        self._record_trading_decision({
            'decision': result.decision,
            'percentage': result.percentage,
            'reason': result.reason
        })

        return result, reason_kr

    def get_current_price(self) -> float:
        quote = r.robinhood.stocks.get_latest_price(self.symbol)[0]
        return float(quote)

    def calculate_shares_to_trade(self, percentage: int) -> int:
        current_price = self.get_current_price()
        if percentage == 0:
            return 0
        trade_value = self.balance * (percentage / 100)
        return int(trade_value / current_price)

    def buy(self, shares: int):
        try:
            current_price = self.get_current_price()
            self.logger.info(f"Buy order placed for {shares} shares of {self.symbol}")
            self.shares += shares
            self.balance -= shares * current_price
            self.trading_value = self.shares * current_price
        except Exception as e:
            self.logger.error(f"Error placing buy order: {e}")

    def sell(self, shares: int):
        try:
            current_price = self.get_current_price()
            self.logger.info(f"Sell order placed for {shares} shares of {self.symbol}")
            self.shares -= shares
            self.balance += shares * current_price
            self.trading_value = self.shares * current_price
        except Exception as e:
            self.logger.error(f"Error placing sell order: {e}")

    def _translate_to_korean(self, text):
        self.logger.info("Translating text to Korean")
        try:
            translator = GoogleTranslator(source='auto', target='ko')
            translated = translator.translate(text)
            self.logger.info("Translation successful")
            return translated
        except Exception as e:
            self.logger.error(f"Error during translation: {e}")
            return text

    def _send_slack_message(self, result: TradingDecision, reason_kr: str, news: Dict[str, Any],
                            fgi: Dict[str, Any]):
        self.logger.info("Preparing to send Slack message")
        webhook_url = Config.SLACK_WEBHOOK_URL
        message = f"""AI Trading Decision for {self.symbol}:
        Decision: {result.decision}
        Percentage: {result.percentage}
        Reason: {result.reason}
        Reason_KO: {reason_kr}

        Recent News:
        {self._format_news(news)}

        Fear and Greed Index:
        Value: {fgi['value']}
        Description: {fgi['description']}
        Last Update: {fgi['last_update']}"""

        payload = {"text": message}
        response = requests.post(webhook_url, json=payload)

        if response.status_code != 200:
            self.logger.error(f"Failed to send Slack message. Status code: {response.status_code}")
        else:
            self.logger.info("Slack message sent successfully")

    def _format_news(self, news: Dict[str, Any]) -> str:
        formatted_news = []
        for source, items in news.items():
            for item in items[:3]:  # Limiting to top 3 news items per source
                formatted_news.append(f"- {item['title']} ({item.get('date', 'N/A')})")
        return "\n".join(formatted_news)

    def _calculate_profit_rate(self):
        total_value = self.balance + self.trading_value
        try:
            profit_rate = ((total_value - self.initial_balance) / self.initial_balance) * 100
            return round(profit_rate, 2)
        except ZeroDivisionError:
            self.logger.warning("Initial balance is zero. Setting profit rate to 0.")
            return 0

    def _get_initial_balance(self):
        cursor = self.db_connection.cursor()
        cursor.execute('SELECT balance FROM trading_records ORDER BY timestamp DESC LIMIT 1')
        result = cursor.fetchone()
        return result[0] if result else self.initial_balance

    def _get_initial_shares(self):
        cursor = self.db_connection.cursor()
        cursor.execute('SELECT shares FROM trading_records WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1',
                       (self.symbol,))
        result = cursor.fetchone()
        return result[0] if result else 0

    def _calculate_trading_value(self):
        current_price = self.get_current_price()
        return self.shares * current_price

    def _record_trading_decision(self, decision: Dict[str, Any]):
        timestamp = datetime.now().isoformat()
        profit_rate = self._calculate_profit_rate()
        cursor = self.db_connection.cursor()
        cursor.execute('''
        INSERT INTO trading_records 
        (symbol, timestamp, decision, percentage, reason, balance, shares, trading_value, profit_rate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            self.symbol,
            timestamp,
            decision['decision'],
            decision['percentage'],
            decision['reason'],
            self.balance,
            self.shares,
            self.trading_value,
            profit_rate
        ))
        self.db_connection.commit()

    def get_trading_history(self):
        cursor = self.db_connection.cursor()
        cursor.execute('SELECT * FROM trading_records WHERE symbol = ? ORDER BY timestamp DESC', (self.symbol,))
        return cursor.fetchall()

# Slack Bot 설정
app = App(token=Config.SLACK_BOT_TOKEN)

def extract_symbol(text):
    """텍스트에서 주식 심볼을 추출하고 대문자로 변환"""
    import re
    symbol_match = re.search(r'\b[A-Za-z]{1,5}\b', text)
    return symbol_match.group(0).upper() if symbol_match else None

def process_trading(symbol, say):
    """주식 거래 분석 및 결과 전송"""
    logger.info(f"{symbol} 심볼에 대한 거래 프로세스 시작")
    say(f"{symbol}에 대한 거래 분석을 처리 중입니다...")

    try:
        analyzer = AIStockTrading(symbol)
        result, reason_kr = analyzer.ai_trading()

        response = f"""Trading Decision for {symbol}:
        Decision: {result.decision}
        Percentage: {result.percentage}
        Reason: {result.reason}
        Korean Reason: {reason_kr}"""

        logger.info(f"{symbol}에 대한 거래 분석 완료")
        say(response)

        # 거래 내역 조회 및 전송
        trading_history = analyzer.get_trading_history()
        history_message = "Trading History:\n" + "\n".join(str(record) for record in trading_history[:5])
        logger.info(f"{symbol}에 대한 거래 내역 전송")
        say(history_message)
    except Exception as e:
        logger.error(f"{symbol} 처리 중 오류 발생: {str(e)}", exc_info=True)
        say(f"{symbol} 처리 중 오류가 발생했습니다. 나중에 다시 시도해 주세요.")

@app.event("app_mention")
def handle_mention(event, say):
    """앱 멘션 이벤트 처리"""
    logger.info(f"앱 멘션 이벤트 수신: {event}")
    symbol = extract_symbol(event['text'])
    if symbol:
        logger.info(f"멘션에서 추출한 심볼: {symbol}")
        process_trading(symbol, say)
    else:
        logger.warning("멘션에서 유효한 심볼을 찾지 못했습니다")
        say("유효한 주식 심볼을 입력해주세요. 예: @YourBotName AAPL 또는 @YourBotName aapl")

@app.event("message")
def handle_message(event, logger):
    """일반 메시지 이벤트 처리 (로깅 목적)"""
    logger.debug(f"메시지 이벤트 수신: {event}")

def main():
    """메인 실행 함수"""
    handler = SocketModeHandler(app, Config.SLACK_APP_TOKEN)
    logger.info("Slack 봇 시작")
    handler.start()

if __name__ == "__main__":
    main()