import os
import pyotp
import robin_stocks as r
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
import json
import requests
import fear_and_greed
from deep_translator import GoogleTranslator
from datetime import datetime
import logging
from youtube_transcript_api import YouTubeTranscriptApi
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class TradingDecision(BaseModel):
    decision: str
    percentage: int
    reason: str

class StockAnalyzer:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.logger = self._setup_logger()
        self.login = self._get_login()
        self.openai_client = OpenAI()
        self.balance = 1000
        self.shares = 0
        self.position_value = 0.0

    def _setup_logger(self):
        logger = logging.getLogger(f"{self.symbol}_analyzer")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _get_login(self):
        load_dotenv()
        username = os.getenv("username")
        password = os.getenv("password")
        totpcode = os.getenv("totpcode")
        totp = pyotp.TOTP(totpcode).now()
        self.logger.info(f"Current OTP: {totp}")
        login = r.robinhood.login(username, password, mfa_code=totp)
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

    def get_all_news_title(self, news):
        # 모든 뉴스 제목을 저장할 리스트 생성
        all_news_titles = []

        # Google News 제목 추가
        all_news_titles.extend([news_item['title'] for news_item in news['google_news']])

        # Alpha Vantage News 제목 추가 (현재 비어있지만, 데이터가 있을 경우를 대비)
        all_news_titles.extend([news_item['title'] for news_item in news['alpha_vantage_news']])

        # 번호가 붙은 뉴스 제목 생성
        numbered_news = [f"({i + 1}) {title}" for i, title in enumerate(all_news_titles)]

        # 번호가 붙은 뉴스 제목을 개행 문자로 연결
        all_news_titles = '\n'.join(numbered_news)

        return all_news_titles


    def _get_news_from_google(self):
        self.logger.info("Fetching news from Google")
        load_dotenv()
        SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
        url = "https://www.searchapi.io/api/v1/search"
        params = {
            "api_key": SERPAPI_API_KEY,
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
        load_dotenv()
        Alpha_Vantage_API_KEY = os.getenv("Alpha_Vantage_API_KEY")
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={self.symbol}&apikey={Alpha_Vantage_API_KEY}"
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

    def analyze(self):
        monthly_df, daily_df = self.get_chart_data()
        news = self.get_news()
        all_news_title = self.get_all_news_title(news)


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
        self.logger.info(f"### News: {all_news_title} ###")

        self._send_slack_message(result, reason_kr, all_news_title, fgi)

        shares_to_trade = self.calculate_shares_to_trade(result.percentage)

        if result.decision == "buy":
            self.buy(shares_to_trade)

        elif result.decision == "sell":
            shares_to_sell = min(shares_to_trade, self.shares)
            self.sell(shares_to_sell)
        else:  # hold
            self.logger.info("Holding current position")

        self.logger.info(f"Current balance: ${self.balance:.2f}")
        self.logger.info(f"Current shares: {self.shares}")
        self.logger.info(f"Current position value: ${self.position_value:.2f}")

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
            self.logger.info(f"Buy order placed for {shares} shares of {self.symbol}")
            self.shares += shares
            self.balance -= shares * self.get_current_price()
            self.position_value = self.shares * self.get_current_price()
        except Exception as e:
            self.logger.error(f"Error placing buy order: {e}")

    def sell(self, shares: int):
        try:
            self.logger.info(f"Sell order placed for {shares} shares of {self.symbol}")
            self.shares -= shares
            self.balance += shares * self.get_current_price()
            self.position_value = self.shares * self.get_current_price()
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

    def _send_slack_message(self, result: TradingDecision, reason_kr: str, all_news_title, fgi: Dict[str, Any]):
        self.logger.info("Preparing to send Slack message")
        load_dotenv()
        webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        message = f"""AI Trading Decision for {self.symbol}:
        Decision: {result.decision}
        percentage : {result.percentage}
        Reason: {result.reason}
        News Title : {all_news_title}
        Reason_KO: {reason_kr}

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

def main(symbol: str):
    analyzer = StockAnalyzer(symbol)
    result, reason_kr = analyzer.analyze()
    print(f"Decision: {result.decision}")
    print(f"Percentage: {result.percentage}")
    print(f"Reason: {result.reason}")
    print(f"Korean Reason: {reason_kr}")

if __name__ == "__main__":
    symbol = str(input("Enter stock symbol: "))
    main(symbol)