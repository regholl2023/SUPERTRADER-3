import os
import pyotp
import robin_stocks as r
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
import json
import requests

def main():
    login = get_login()
    NVDA_monthly_df, NVDA_daily_df = get_nvidia_chart_data(login)

    print(NVDA_monthly_df)
    # AI_result = openAI_response(NVDA_monthly_df, NVDA_daily_df)
    # send_slack_message(AI_result)

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


def openAI_response(monthly_df, daily_df):
    client = OpenAI()

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
              "daily_data": daily_df.to_json()
          })
        }
      ],
      response_format={"type": "json_object"}
    )
    result = json.loads(response.choices[0].message.content)
    return result

def send_slack_message(ai_result):
    load_dotenv()
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    message = f"AI Trading Decision for NVIDIA:\nDecision: {ai_result['decision']}\nReason: {ai_result['reason']}"
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