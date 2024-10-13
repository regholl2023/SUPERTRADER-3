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
import base64
import logging
from youtube_transcript_api import YouTubeTranscriptApi
from pydantic import BaseModel, Field
from typing import List


class TradingDecision(BaseModel):
    decision: str
    reason: str

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    login = get_login()
    result, reason_kr = openAI_response(login)
    logger.info(f"Trading Decision: {result.decision}")
    logger.info(f"Reason: {result.reason}")
    logger.info(f"Reason in Korean: {reason_kr}")


def get_login():
    load_dotenv()

    username = os.getenv("username")
    password = os.getenv("password")
    totpcode = os.getenv("totpcode")
    totp = pyotp.TOTP(totpcode).now()
    logger.info(f"Current OTP: {totp}")

    login = r.robinhood.login(username, password, mfa_code=totp)
    logger.info("Successfully logged in to Robinhood")
    return login


def get_chart_data(login):
    symbol = "NVDA"

    logger.info(f"Fetching chart data for {symbol}")

    # 월간 데이터 가져오기
    monthly_historicals = r.robinhood.stocks.get_stock_historicals(
        symbol,
        interval="day",
        span="month",
        bounds="regular"
    )

    # 일간 데이터 가져오기
    daily_historicals = r.robinhood.stocks.get_stock_historicals(
        symbol,
        interval="5minute",
        span="day",
        bounds="regular"
    )

    # 데이터프레임 생성 및 처리 함수
    def process_df(historicals):
        df = pd.DataFrame(historicals)
        df = df[['begins_at', 'open_price', 'close_price', 'high_price', 'low_price', 'volume']]
        df['begins_at'] = pd.to_datetime(df['begins_at'])
        for col in ['open_price', 'close_price', 'high_price', 'low_price']:
            df[col] = df[col].astype(float)
        df['volume'] = df['volume'].astype(int)
        df.columns = ['Date', 'Open', 'Close', 'High', 'Low', 'Volume']
        df.set_index('Date', inplace=True)
        return df

    monthly_df = process_df(monthly_historicals)
    daily_df = process_df(daily_historicals)

    logger.info(f"NVIDIA 30-day chart data: {len(monthly_df)} rows")
    logger.info(f"NVIDIA daily chart data: {len(daily_df)} rows")

    return monthly_df, daily_df


def get_fear_and_greed_index():
    logger.info("Fetching Fear and Greed Index")
    fgi = fear_and_greed.get()
    return {
        "value": fgi.value,
        "description": fgi.description,
        "last_update": fgi.last_update.isoformat()
    }


def get_news_from_alpha_vantage():
    logger.info("Fetching news from Alpha Vantage")
    load_dotenv()
    Alpha_Vantage_API_KEY = os.getenv("Alpha_Vantage_API_KEY")

    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=NVDA&apikey={Alpha_Vantage_API_KEY}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if "feed" not in data:
            logger.warning("No news data found in Alpha Vantage response")
            return []

        news_items = []
        for item in data["feed"][:10]:  # 최대 10개 아이템
            title = item.get("title", "제목 없음")
            time_published = item.get("time_published", "날짜 없음")

            # 날짜 형식 변환
            if time_published != "날짜 없음":
                dt = datetime.strptime(time_published, "%Y%m%dT%H%M%S")
                time_published = dt.strftime("%Y-%m-%d %H:%M:%S")

            news_items.append({
                'title': title,
                'pubDate': time_published
            })

        logger.info(f"Retrieved {len(news_items)} news items from Alpha Vantage")
        return news_items

    except requests.RequestException as e:
        logger.error(f"Error during Alpha Vantage API request: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in get_news_from_alpha_vantage: {e}")

    return []


def get_news_from_google():
    logger.info("Fetching news from Google")
    load_dotenv()
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

    url = "https://www.searchapi.io/api/v1/search"

    params = {
        "api_key": SERPAPI_API_KEY,
        "engine": "google_news",
        "q": "nvidia",
        "num": 5
    }

    headers = {
        "Accept": "application/json"
    }
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

        logger.info(f"Retrieved {len(news_items)} news items from Google")
        return news_items

    except requests.RequestException as e:
        logger.error(f"Error during Google News API request: {e}")
    except json.JSONDecodeError:
        logger.error("JSON parsing error in Google News response")
    except KeyError as e:
        logger.error(f"Unexpected response structure from Google News: {e}")

    return []


def translate_to_korean(text):
    logger.info("Translating text to Korean")
    try:
        translator = GoogleTranslator(source='auto', target='ko')
        translated = translator.translate(text)
        logger.info("Translation successful")
        return translated
    except Exception as e:
        logger.error(f"Error during translation: {e}")
        return text  # 번역 실패 시 원본 텍스트 반환


def get_youtube_transcript():
    video_id = "TWINrTppUl4"
    logger.info(f"Fetching YouTube transcript for video ID: {video_id}")
    try:
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = " ".join(item['text'] for item in transcript_data)
        logger.info(f"Retrieved transcript with {len(full_transcript)} characters")
        return full_transcript.strip()

    except Exception as e:
        logger.error(f"Error fetching YouTube transcript: {str(e)}")
        return f"An error occurred: {str(e)}"


def send_slack_message(result, reason_kr, fgi):
    logger.info("Preparing to send Slack message")
    load_dotenv()
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    message = f"""AI Trading Decision for NVIDIA:
    Decision: {result.decision}
    Reason: {result.reason}
    Reason_KO: {reason_kr}

    Fear and Greed Index:
    Value: {fgi['value']}
    Description: {fgi['description']}
    Last Update: {fgi['last_update']}"""

    payload = {
        "text": message
    }
    response = requests.post(webhook_url, json=payload)

    if response.status_code != 200:
        logger.error(f"Failed to send Slack message. Status code: {response.status_code}")
    else:
        logger.info("Slack message sent successfully")


def openAI_response(login):
    logger.info("Initiating OpenAI response process")
    client = OpenAI()

    monthly_df, daily_df = get_chart_data(login)
    news_google = get_news_from_google()
    news_alpha_vantage = get_news_from_alpha_vantage()
    youtube_transcript = get_youtube_transcript()
    fgi = get_fear_and_greed_index()

    logger.info("Sending request to OpenAI")
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": """You are an expert in Stock investing. Analyze the provided data including technical indicators, market data, recent news headlines, the Fear and Greed Index, YouTube video transcript, and the chart image. Tell me whether to buy, sell, or hold at the moment. Consider the following in your analysis:
                - Technical indicators and market data
                - Recent news headlines and their potential impact on Stock price
                - The Fear and Greed Index and its implications
                - Overall market sentiment
                - Insights from the YouTube video transcript

                Respond with a decision (buy, sell, or hold) and a reason for your decision."""},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "monthly_data": monthly_df.to_json(),
                            "daily_data": daily_df.to_json(),
                            "fear_and_greed_index": fgi,
                            "news_google": news_google,
                            "news_alpha_vantage": news_alpha_vantage,
                            "youtube_transcript": youtube_transcript
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
                        "reason": {"type": "string"}
                    },
                    "required": ["decision", "reason"],
                    "additionalProperties": False
                }
            }
        }
    )
    # 최신 pydantic 메서드 사용
    result = TradingDecision.model_validate_json(response.choices[0].message.content)
    logger.info("Received response from OpenAI")

    # 결과 한국어로 번역
    reason_kr = translate_to_korean(result.reason)

    logger.info(f"### AI Decision: {result.decision.upper()} ###")
    logger.info(f"### Reason: {result.reason} ###")

    send_slack_message(result,reason_kr, fgi)

    return result, reason_kr


if __name__ == "__main__":
    logger.info("Starting main program")
    main()
    logger.info("Program completed")