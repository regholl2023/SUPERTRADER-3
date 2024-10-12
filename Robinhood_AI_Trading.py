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





def main():
    login = get_login()
    NVDA_monthly_df, NVDA_daily_df = get_nvidia_chart_data(login)
    NVDA_news = get_nvidia_news()
    AI_result, fgi = openAI_response(NVDA_monthly_df, NVDA_daily_df,NVDA_news)
    send_slack_message(AI_result, fgi)

def get_login():
    load_dotenv()
    username = os.getenv("username")
    password = os.getenv("password")
    totpcode = os.getenv("totpcode")
    totp = pyotp.TOTP(totpcode).now()
    print("Current OTP:", totp)

    login = r.robinhood.login(username, password, mfa_code=totp)
    return login


def get_nvidia_chart_data(login):
    symbol = "NVDA"

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

    print(f"NVIDIA 30일 차트 데이터: {len(monthly_df)} 행")
    print(f"NVIDIA 당일 차트 데이터: {len(daily_df)} 행")

    return monthly_df, daily_df

def get_fear_and_greed_index():
    fgi = fear_and_greed.get()
    return {
        "value": fgi.value,
        "description": fgi.description,
        "last_update": fgi.last_update.isoformat()
    }

def get_nvidia_news():
    # SearchApi 키
    load_dotenv()
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

    # SearchApi 엔드포인트 URL
    url = "https://www.searchapi.io/api/v1/search"

    # 검색 매개변수 설정
    params = {
        "api_key": SERPAPI_API_KEY,
        "engine": "google_news",  # Google News 엔진 사용
        "q": "nvidia",
        "num": 5  # 상위 5개 결과만 가져오기
    }

    # 헤더 설정
    headers = {
        "Accept": "application/json"
    }
    try:
        # GET 요청 보내기
        response = requests.get(url, params=params, headers=headers)

        # 응답 상태 확인
        response.raise_for_status()

        # JSON 응답 파싱
        data = response.json()

        # 뉴스 제목과 날짜만 추출
        news_items = []
        for result in data.get('organic_results', [])[:5]:
            news_items.append({
                'title': result['title'],
                'date': result['date']
            })

        return news_items

    except requests.RequestException as e:
        print(f"API 요청 중 오류 발생: {e}")
    except json.JSONDecodeError:
        print("JSON 파싱 오류")
    except KeyError as e:
        print(f"예상치 못한 응답 구조: {e}")

    return []

def translate_to_korean(text):
    try:
        translator = GoogleTranslator(source='auto', target='ko')
        return translator.translate(text)
    except Exception as e:
        print(f"번역 중 오류 발생: {e}")
        return text  # 번역 실패 시 원본 텍스트 반환


def openAI_response(monthly_df, daily_df, NVDA_news):
    client = OpenAI()

    # 공포 탐욕 지수 가져오기
    fgi = get_fear_and_greed_index()

    response = client.chat.completions.create(
      model="gpt-4o",
      messages=[
        {
          "role": "system",
          "content": "You are an expert in Stock investing. Tell me whether to buy, sell or hold at the moment based on the monthly and daily chart data provided. Response in json format.\n\nResponse Example:\n{\"decision\":\"buy\",\"reason\":\"some technical reason based on both monthly and daily data\"}"
        },
        {
          "role": "user",
          "content": json.dumps({
              "monthly_data": monthly_df.to_json(),
              "daily_data": daily_df.to_json(),
              "fear_and_greed_index": fgi,
              "nvidia_news":NVDA_news
          })
        }
      ],
      response_format={"type": "json_object"}
    )
    result = json.loads(response.choices[0].message.content)

    print(json.dumps({
              "monthly_data": monthly_df.to_json(),
              "daily_data": daily_df.to_json(),
              "fear_and_greed_index": fgi,
              "nvidia_news":NVDA_news}))

    # 결과 한국어로 번역
    result['decision_kr'] = translate_to_korean(result['decision'])
    result['reason_kr'] = translate_to_korean(result['reason'])
    return result, fgi


def send_slack_message(ai_result, fgi):
    load_dotenv()
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    message = f"""AI Trading Decision for NVIDIA:
    Decision: {ai_result['decision']}
    결정: {ai_result['decision_kr']}
    Reason: {ai_result['reason']}
    이유: {ai_result['reason_kr']}

    Fear and Greed Index:
    Value: {fgi['value']}
    Description: {fgi['description']}
    Last Update: {fgi['last_update']}"""

    payload = {
        "text": message
    }
    response = requests.post(webhook_url, json=payload)

    if response.status_code != 200:
        print(f"Failed to send Slack message. Status code: {response.status_code}")
    else:
        print("Slack message sent successfully")


if __name__ == "__main__":
    main()