import os
import pyotp
import robin_stocks as r
from dotenv import load_dotenv
import datetime
import pandas as pd
from openai import OpenAI
import json

def main():
    login = get_login()
    NVDA_df= get_nvidia_chart_data(login)
    AI_result = openAI_request(NVDA_df)
    print(AI_result)

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
    interval = "day"
    span = "month"
    bounds = "regular"

    # 현재 날짜로부터 30일 전의 날짜 계산
    end_date = datetime.datetime.now().date()
    start_date = end_date - datetime.timedelta(days=30)

    historicals = r.robinhood.stocks.get_stock_historicals(
        symbol,
        interval=interval,
        span=span,
        bounds=bounds,
        info=None
    )

    # 데이터프레임 생성
    df = pd.DataFrame(historicals)

    # 필요한 컬럼만 선택
    df = df[['begins_at', 'open_price', 'close_price', 'high_price', 'low_price', 'volume']]

    # 데이터 타입 변환
    df['begins_at'] = pd.to_datetime(df['begins_at']).dt.date
    for col in ['open_price', 'close_price', 'high_price', 'low_price']:
        df[col] = df[col].astype(float)
    df['volume'] = df['volume'].astype(int)

    # 컬럼 이름 변경
    df.columns = ['Date', 'Open', 'Close', 'High', 'Low', 'Volume']

    # 인덱스를 Date로 설정
    df.set_index('Date', inplace=True)

    print(f"NVIDIA 30일 차트 데이터 ({start_date} ~ {end_date}):")
    return df


def openAI_request(df):
    client = OpenAI()

    response = client.chat.completions.create(
      model="gpt-4o",
      messages=[
        {
          "role": "system",
          "content": [
            {
              "type": "text",
              "text": "You are an expert in Stock investing. Tell me whether to buy, sell or hold at the moment based on the chart data provided. response in json format.\n\nResponse Example:\n{\"decision\":\"buy\",\"reason\":\"some technical reason\"}\n{\"decision\":\"sell\",\"reason\":\"some technical reason\"}\n{\"decision\":\"hold\",\"reason\":\"some technical reason\"}"
            }
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": df.to_json()
            }
          ]
        }
      ],
      response_format={
        "type": "json_object"
      }
    )

    result = json.loads(response.choices[0].message.content)
    return result


if __name__ == "__main__":
    main()