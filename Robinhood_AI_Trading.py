import os
import re
import json
import logging
import sqlite3
import requests
import pandas as pd
from typing import Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel
import yfinance as yf

import pyotp
import robin_stocks as r
from dotenv import load_dotenv
from openai import OpenAI
import fear_and_greed
from deep_translator import GoogleTranslator
from youtube_transcript_api import YouTubeTranscriptApi
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Load environment variables and set up logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration class
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

# Trading decision model
class TradingDecision(BaseModel):
    decision: str
    percentage: int
    reason: str
    expected_next_day_price: float

# AI Stock Advisor System class
class AIStockAdvisorSystem:
    def __init__(self, stock: str):
        self.stock = stock
        self.logger = logging.getLogger(f"{stock}_analyzer")
        self.login = self._get_login()
        self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.db_connection = self._setup_database('ai_stock_analysis_records.db')
        self.performance_db_connection = self._setup_database('ai_stock_performance.db')
        self._migrate_and_update_performance_data()

    # 1. Robinhood Login
    def _get_login(self):
        # Generate TOTP and log in to Robinhood
        totp = pyotp.TOTP(Config.ROBINHOOD_TOTP_CODE).now()
        self.logger.info(f"Current OTP: {totp}")
        login = r.robinhood.login(Config.ROBINHOOD_USERNAME, Config.ROBINHOOD_PASSWORD, mfa_code=totp)
        self.logger.info("Successfully logged in to Robinhood")
        return login

    # 2-1 Data Collection - Current Price
    def get_current_price(self):
        # Fetch current price from Robinhood
        self.logger.info(f"Fetching current price for {self.stock}")
        try:
            quote = r.robinhood.stocks.get_latest_price(self.stock)
            current_price = round(float(quote[0]), 2)
            self.logger.info(f"Current price for {self.stock}: ${current_price:.2f}")
            return current_price
        except Exception as e:
            self.logger.error(f"Error fetching current price: {str(e)}")
            return None

    # 2-2 Data Collection - Chart Data for 1day and 3Months
    def get_chart_data(self):
        # Fetch chart data for the stock
        self.logger.info(f"Fetching chart data for {self.stock}")
        monthly_historicals = r.robinhood.stocks.get_stock_historicals(
            self.stock, interval="day", span="3month", bounds="regular"
        )
        daily_historicals = r.robinhood.stocks.get_stock_historicals(
            self.stock, interval="5minute", span="day", bounds="regular"
        )
        monthly_df = self._process_df(monthly_historicals)
        daily_df = self._process_df(daily_historicals)
        return self._add_indicators(monthly_df, daily_df)

    def _process_df(self, historicals):
        # Process historical data into a DataFrame
        df = pd.DataFrame(historicals)
        df = df[['begins_at', 'open_price', 'close_price', 'high_price', 'low_price', 'volume']]
        df['begins_at'] = pd.to_datetime(df['begins_at'])
        for col in ['open_price', 'close_price', 'high_price', 'low_price']:
            df[col] = df[col].astype(float)
        df['volume'] = df['volume'].astype(int)
        df.columns = ['Date', 'Open', 'Close', 'High', 'Low', 'Volume']
        df.set_index('Date', inplace=True)
        return df

    # 2-3 Data Collection - Technical Indicators
    def _add_indicators(self, monthly_df, daily_df):
        # Add technical indicators to the DataFrames
        for df in [monthly_df, daily_df]:
            df = self._calculate_bollinger_bands(df)
            df = self._calculate_rsi(df)
            df = self._calculate_macd(df)
        monthly_df = self._calculate_moving_averages(monthly_df)
        return monthly_df, daily_df

    def _calculate_bollinger_bands(self, df: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        # Calculate Bollinger Bands
        df['SMA'] = df['Close'].rolling(window=window).mean()
        df['STD'] = df['Close'].rolling(window=window).std()
        df['Upper_Band'] = df['SMA'] + (df['STD'] * num_std)
        df['Lower_Band'] = df['SMA'] - (df['STD'] * num_std)
        return df

    def _calculate_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        # Calculate Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df

    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        # Calculate Moving Average Convergence Divergence (MACD)
        df['EMA_fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
        df['EMA_slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
        df['MACD'] = df['EMA_fast'] - df['EMA_slow']
        df['Signal_Line'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
        return df

    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate Moving Averages
        windows = [10, 20, 60, 120]
        for window in windows:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        return df

    # 2-4 Data Collection - VIX Index
    def get_vix_index(self):
        # Fetch VIX INDEX data
        self.logger.info("Fetching VIX INDEX data")
        try:
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="1d")
            current_vix = round(vix_data['Close'].iloc[-1], 2)
            self.logger.info(f"Current VIX INDEX: {current_vix}")
            return current_vix
        except Exception as e:
            self.logger.error(f"Error fetching VIX INDEX: {str(e)}")
            return None

    # 2-5 Data Collection - Fear & Greed Index
    def get_fear_and_greed_index(self):
        # Fetch Fear and Greed Index
        self.logger.info("Fetching Fear and Greed Index")
        fgi = fear_and_greed.get()
        return {
            "value": fgi.value,
            "description": fgi.description,
            "last_update": fgi.last_update.isoformat()
        }

    # 2-6 Data Collection - News
    def get_news(self):
        # Fetch news from multiple sources
        return {
            "google_news": self._get_news_from_google(),
            "alpha_vantage_news": self._get_news_from_alpha_vantage(),
            "robinhood_news": self._get_news_from_robinhood()
        }

    def _get_news_from_google(self):
        # Fetch news from Google
        self.logger.info("Fetching news from Google")
        url = "https://www.searchapi.io/api/v1/search"
        params = {
            "api_key": Config.SERPAPI_API_KEY,
            "engine": "google_news",
            "q": self.stock,
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

    def _get_news_from_robinhood(self):
        self.logger.info("Fetching news from Robinhood")
        try:
            news_data = r.robinhood.stocks.get_news(self.stock)
            news_items = []
            for item in news_data[:5]:  # Limit to 5 news items
                news_items.append({
                    'title': item['title'],
                    'published_at': item['published_at']
                })
            self.logger.info(f"Retrieved {len(news_items)} news items from Robinhood")
            return news_items
        except Exception as e:
            self.logger.error(f"Error fetching news from Robinhood: {str(e)}")
            return []

    def _get_news_from_alpha_vantage(self):
        # Fetch news from Alpha Vantage
        self.logger.info("Fetching news from Alpha Vantage")
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={self.stock}&apikey={Config.ALPHA_VANTAGE_API_KEY}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if "feed" not in data:
                self.logger.warning("No news data found in Alpha Vantage response")
                return []
            news_items = []
            for item in data["feed"][:10]:
                title = item.get("title", "No title")
                time_published = item.get("time_published", "No date")
                if time_published != "No date":
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

    # 2-7 Data Collection - Youtube
    def get_youtube_transcript(self):
        # Fetch YouTube video transcript
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

    def _translate_to_korean(self, text):
        # Translate text to Korean
        self.logger.info("Translating text to Korean")
        try:
            translator = GoogleTranslator(source='auto', target='ko')
            translated = translator.translate(text)
            self.logger.info("Translation successful")
            return translated
        except Exception as e:
            self.logger.error(f"Error during translation: {e}")
            return text

    # 3. Database - ai_stock_analysis_records & ai_stock_performance
    def _setup_database(self, db_name: str):
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        if db_name == 'ai_stock_analysis_records.db':
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_stock_analysis_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                Stock TEXT,
                Time DATETIME,
                Decision TEXT,
                Percentage INTEGER,
                Reason TEXT,
                CurrentPrice REAL,
                ExpectedNextDayPrice REAL,
                ExpectedPriceDifference REAL
            )
            ''')
        elif db_name == 'ai_stock_performance.db':
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock TEXT,
                date DATE,
                avg_current_price REAL,
                next_date DATE,
                avg_expected_next_day_price REAL,
                actual_next_day_price REAL,
                price_difference REAL,
                error_percentage REAL DEFAULT 0,
                count INTEGER DEFAULT 1
            )
            ''')

        conn.commit()
        self.logger.info(f"Database {db_name} setup completed")
        return conn

    def _migrate_and_update_performance_data(self):
        cursor_analysis = self.db_connection.cursor()
        cursor_performance = self.performance_db_connection.cursor()

        # 1. ai_stock_analysis_records에서 모든 데이터 가져오기
        cursor_analysis.execute("""
        SELECT Stock, DATE(Time) as Date, CurrentPrice, ExpectedNextDayPrice
        FROM ai_stock_analysis_records
        """)
        records = cursor_analysis.fetchall()

        # 2. 데이터를 DataFrame으로 변환하고 Stock과 Date로 그룹화
        df = pd.DataFrame(records, columns=['Stock', 'Date', 'CurrentPrice', 'ExpectedNextDayPrice'])
        grouped = df.groupby(['Stock', 'Date'])

        # 3. 각 그룹에 대해 평균 계산
        aggregated = grouped.agg({
            'CurrentPrice': 'mean',
            'ExpectedNextDayPrice': 'mean',
            'Stock': 'count'  # 이것은 각 그룹의 레코드 수를 계산합니다
        }).rename(columns={'Stock': 'Count'})

        # 4. 계산된 데이터를 ai_stock_performance.db에 입력 (이미 존재하는 데이터는 건너뛰기)
        for (stock, date), row in aggregated.iterrows():
            next_date = (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')

            # 해당 stock과 date에 대한 데이터가 이미 존재하는지 확인
            cursor_performance.execute("""
            SELECT COUNT(*) FROM stock_performance
            WHERE stock = ? AND date = ?
            """, (stock, date))

            if cursor_performance.fetchone()[0] == 0:
                # 데이터가 존재하지 않는 경우에만 새로운 데이터 삽입
                cursor_performance.execute("""
                INSERT INTO stock_performance
                (stock, date, next_date, avg_current_price, avg_expected_next_day_price, count)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (stock, date, next_date, row['CurrentPrice'], row['ExpectedNextDayPrice'], row['Count']))
                self.logger.info(f"Inserted new performance data for {stock} on {date}")
            else:
                self.logger.info(f"Skipped existing performance data for {stock} on {date}")

        self.performance_db_connection.commit()

        # 5. & 6. next_date 기반으로 실제 주식 종가 가져와 저장
        self._fetch_actual_stock_prices()

        self.logger.info("Performance data migration and update completed")

    def _record_trading_decision(self, decision: Dict[str, Any]):
        time_ = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        current_price = decision['CurrentPrice']
        expected_next_day_price = decision['ExpectedNextDayPrice']
        expected_price_difference = round(expected_next_day_price - current_price, 2)

        cursor = self.db_connection.cursor()
        cursor.execute('''
        INSERT INTO ai_stock_analysis_records 
        (Stock, Time, Decision, Percentage, Reason, CurrentPrice, ExpectedNextDayPrice, ExpectedPriceDifference)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            self.stock,
            time_,
            decision['Decision'],
            decision['Percentage'],
            decision['Reason'],
            current_price,
            expected_next_day_price,
            expected_price_difference
        ))
        self.db_connection.commit()

        # Update performance data
        self._update_performance_data(decision)
        self._fetch_actual_stock_prices()

    def _update_performance_data(self, decision):
        cursor_analysis = self.db_connection.cursor()
        cursor_performance = self.performance_db_connection.cursor()

        # ai_stock_analysis_records에서 가장 최근 데이터의 Time 가져오기
        cursor_analysis.execute("""
        SELECT Time FROM ai_stock_analysis_records
        WHERE Stock = ?
        ORDER BY Time DESC LIMIT 1
        """, (self.stock,))

        latest_time = cursor_analysis.fetchone()
        if not latest_time:
            self.logger.error(f"No records found for {self.stock} in ai_stock_analysis_records")
            return

        decision_date = datetime.strptime(latest_time[0], '%Y-%m-%d %H:%M:%S').date()
        next_date = decision_date + timedelta(days=1)

        current_price = float(decision['CurrentPrice'])
        expected_next_day_price = float(decision['ExpectedNextDayPrice'])

        # 기존 데이터 확인
        cursor_performance.execute("""
        SELECT avg_current_price, avg_expected_next_day_price, count
        FROM stock_performance
        WHERE stock = ? AND date = ?
        """, (self.stock, decision_date.strftime('%Y-%m-%d')))

        existing_data = cursor_performance.fetchone()

        if existing_data:
            # 기존 데이터가 있으면 평균과 카운트 업데이트
            old_avg_current, old_avg_expected, old_count = existing_data
            new_count = old_count + 1
            new_avg_current = ((old_avg_current * old_count) + current_price) / new_count
            new_avg_expected = ((old_avg_expected * old_count) + expected_next_day_price) / new_count

            cursor_performance.execute("""
            UPDATE stock_performance
            SET avg_current_price = ?,
                avg_expected_next_day_price = ?,
                count = ?
            WHERE stock = ? AND date = ?
            """, (new_avg_current, new_avg_expected, new_count, self.stock, decision_date.strftime('%Y-%m-%d')))

            self.logger.info(f"Performance data updated for {self.stock} on {decision_date}")
        else:
            # 새로운 데이터 삽입
            cursor_performance.execute("""
            INSERT INTO stock_performance 
            (stock, date, next_date, avg_current_price, avg_expected_next_day_price, count)
            VALUES (?, ?, ?, ?, ?, 1)
            """, (self.stock, decision_date.strftime('%Y-%m-%d'), next_date.strftime('%Y-%m-%d'),
                  current_price, expected_next_day_price))

            self.logger.info(f"New performance data inserted for {self.stock} on {decision_date}")

        self.performance_db_connection.commit()

        # 업데이트된 데이터 로깅
        cursor_performance.execute("""
        SELECT avg_current_price, avg_expected_next_day_price
        FROM stock_performance
        WHERE stock = ? AND date = ?
        """, (self.stock, decision_date.strftime('%Y-%m-%d')))
        updated_data = cursor_performance.fetchone()
        if updated_data:
            self.logger.info(f"Updated/Inserted values - Avg Current Price: {updated_data[0]:.2f}, Avg Expected Next Day Price: {updated_data[1]:.2f}")

        # 실제 주가 데이터 업데이트
        self._fetch_actual_stock_prices()


    def _fetch_actual_stock_prices(self):
        cursor = self.performance_db_connection.cursor()

        cursor.execute("""
        SELECT DISTINCT stock, date, next_date, avg_expected_next_day_price
        FROM stock_performance
        WHERE actual_next_day_price IS NULL
        """)
        stocks_to_update = cursor.fetchall()

        for stock, date, next_date, avg_expected_next_day_price in stocks_to_update:
            next_date = datetime.strptime(next_date, '%Y-%m-%d').date()

            if next_date > datetime.now().date():
                self.logger.info(f"Skipping future date for {stock}: {next_date}")
                continue

            try:
                ticker = yf.Ticker(stock)
                hist = ticker.history(start=next_date, end=next_date + timedelta(days=1))

                if not hist.empty:
                    actual_price = round(hist['Close'].iloc[0], 2)
                    price_difference = round(actual_price - avg_expected_next_day_price, 2)
                    error_percentage = abs(round((price_difference / actual_price) * 100, 2)) if actual_price != 0 else 0

                    cursor.execute("""
                    UPDATE stock_performance
                    SET actual_next_day_price = ?,
                        price_difference = ?,
                        error_percentage = ?
                    WHERE stock = ? AND date = ?
                    """, (actual_price, price_difference, error_percentage, stock, date))
                    self.logger.info(
                        f"Updated actual next day price for {stock} on {next_date}: {actual_price:.2f}, "
                        f"difference: {price_difference:.2f}, error percentage: {error_percentage:.2f}%")
                else:
                    self.logger.warning(f"No data available for {stock} on {next_date}")
            except Exception as e:
                self.logger.error(f"Error fetching data for {stock} on {next_date}: {str(e)}")

        self.performance_db_connection.commit()
        self.logger.info("Actual next day stock prices fetched and updated")

    def _update_error_percentage(self):
        cursor = self.performance_db_connection.cursor()

        cursor.execute('''
        UPDATE stock_performance
        SET error_percentage = CASE
            WHEN actual_next_day_price IS NOT NULL AND actual_next_day_price != 0
            THEN (price_difference / actual_next_day_price) * 100
            ELSE 0
        END
        ''')

        self.performance_db_connection.commit()
        self.logger.info("Updated error_percentage for all records")

    # 4. AI Stock Analysis
    def ai_stock_analysis(self):
        # Perform AI-based Stock analysis
        monthly_df, daily_df = self.get_chart_data()
        news = self.get_news()
        youtube_transcript = self.get_youtube_transcript()
        fgi = self.get_fear_and_greed_index()
        current_price = self.get_current_price()
        vix_index = self.get_vix_index()

        if current_price is None:
            self.logger.error("Failed to get current price. Aborting analysis.")
            return None, None

        self.logger.info("Sending request to OpenAI")
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an expert in Stock investing. Analyze the provided data including technical indicators, market data, recent news headlines, the Fear and Greed Index, YouTube video transcript, VIX INDEX, and the chart image. Tell me whether to buy, sell, or hold at the moment. Consider the following in your analysis:
                        - Technical indicators and market data
                        - Recent news headlines and their potential impact on Stock price
                        - The Fear and Greed Index and its implications
                        - VIX INDEX and its implications for market volatility
                        - Overall market sentiment
                        - Insights from the YouTube video transcript
                        - Current stock price: ${current_price}
                        - Current VIX INDEX: {vix_index}

                        Particularly important is to always refer to the trading method of 'Mark Minervini', a legendary stock investor, to assess the current situation and make trading decisions. Mark Minervini's trading method is as follows:

                        {youtube_transcript}

                        Based on this trading method, analyze the current market situation and make a judgment by synthesizing it with the provided data.

                        Additionally, predict the next day's closing price for the stock based on your analysis.

                        Respond with:
                        1. A decision (BUY, SELL, or HOLD)
                        2. If the decision is 'BUY' or 'SELL', provide an intensity expressed as a percentage ratio (1 to 100).
                           If the decision is 'HOLD', set the percentage to 0.
                        3. A reason for your decision
                        4. A prediction for the next day's closing price

                        Ensure that the percentage is an integer between 1 and 100 for buy/sell decisions, and exactly 0 for hold decisions.
                        Your percentage should reflect the strength of your conviction in the decision based on the analyzed data.
                        The next day's closing price prediction should be a float value."""},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "stock": self.stock,
                                "monthly_data": monthly_df.to_json(),
                                "daily_data": daily_df.to_json(),
                                "fear_and_greed_index": fgi,
                                "vix_index": vix_index,
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
                            "decision": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                            "percentage": {"type": "integer"},
                            "reason": {"type": "string"},
                            "expected_next_day_price": {"type": "number"},
                        },
                        "required": ["decision", "percentage", "reason","expected_next_day_price"],
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
        self.logger.info(f"### Current Price: {current_price:.2f} ###")
        self.logger.info(f"### Expected Next Day Price: {result.expected_next_day_price:.2f} ###")
        self.logger.info(f"### VIX INDEX: {vix_index} ###")

        # Record the trading decision and current state
        self._record_trading_decision({
            'Decision': result.decision,
            'Percentage': result.percentage,
            'Reason': result.reason,
            'CurrentPrice': round(current_price,2),
            'ExpectedNextDayPrice': round(result.expected_next_day_price, 2),
            'VIX_INDEX': vix_index
        })

        return result, reason_kr, news, fgi, current_price, vix_index


# Slack Bot Configuration
app = App(token=Config.SLACK_BOT_TOKEN)

def extract_stock(text):
    # Extract stock name from text and convert to uppercase
    stock_match = re.search(r'\b[A-Za-z]{1,5}\b', text)
    return stock_match.group(0).upper() if stock_match else None

def process_trading(stock, say):
    # Process stock trading analysis and send results
    logger.info(f"Starting the stock trading analysis for {stock}")
    say(f"Processing analysis for {stock}...")

    def _format_news(news: Dict[str, Any]) -> str:
        formatted_news = []
        for source, items in news.items():
            formatted_news.append(f"{source.capitalize()}:")
            for item in items[:5]:  # Limiting to top 3 news items per source
                formatted_news.append(f"- {item['title']} ({item.get('date') or item.get('published_at') or 'N/A'})")
        return "\n".join(formatted_news)

    try:
        analyzer = AIStockAdvisorSystem(stock)
        result, reason_kr, news, fgi, current_price, vix_index = analyzer.ai_stock_analysis()

        response = f"""AI Trading Decision for {stock}:
        Decision: {result.decision}
        Percentage: {result.percentage}%
        Current Price: ${current_price:.2f}
        Predicted NextDay Price: ${result.expected_next_day_price:.2f}
        VIX INDEX: {vix_index}
        Reason: {result.reason}
        Reason_KO: {reason_kr}
        Recent News:
        {_format_news(news)}

        Fear and Greed Index:
        Value: {fgi['value']:.2f}
        Description: {fgi['description']}
        Last Update: {fgi['last_update']}"""

        logger.info(f"Completed the stock trading analysis for {stock}")
        say(response)

    except Exception as e:
        logger.error(f"Error occurred while processing {stock}: {str(e)}", exc_info=True)
        say(f"An error occurred while processing {stock}. Please try again later.")


@app.event("app_mention")
def handle_mention(event, say):
    # Handle app mention events
    logger.info(f"Received app mention event: {event}")
    stock = extract_stock(event['text'])
    if stock:
        logger.info(f"Extracted stock from mention: {stock}")
        process_trading(stock, say)
    else:
        logger.warning("Could not find valid stock name in mention")
        say("Please enter a valid stock symbol. For example: @YourBotName AAPL or @YourBotName aapl")

@app.event("message")
def handle_message(event, logger):
    # Handle general message events (for logging purposes)
    logger.debug(f"Received message event: {event}")

def main():
    # Main execution function
    handler = SocketModeHandler(app, Config.SLACK_APP_TOKEN)
    logger.info("Starting AI Stock Advisor")
    handler.start()

if __name__ == "__main__":
    main()