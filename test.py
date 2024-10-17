# import yfinance as yf
# from datetime import datetime, timedelta
#
# ticker = yf.Ticker("ARM")
#
# next_date = "2024-10-15"
# next_date = datetime.strptime(next_date, '%Y-%m-%d').date()
#
# print("next_date : ", next_date)
# for i in range(5):
#     check_date = next_date + timedelta(days=i)
#
#     print(check_date)
#     hist = ticker.history(start=check_date, end=check_date + timedelta(days=1))
#     # print(hist)
#
#     if not hist.empty:
#         actual_price = round(hist['Close'].iloc[0], 2)
#         print("actual_price: ", actual_price)

# import sqlite3
# from datetime import datetime
#
# def update_analysis_records_date():
#     # 데이터베이스 연결
#     conn = sqlite3.connect('ai_stock_analysis_records.db')
#     cursor = conn.cursor()
#
#     try:
#         # 현재 모든 레코드의 Time 값을 가져옵니다.
#         cursor.execute("SELECT id, Time FROM ai_stock_analysis_records")
#         records = cursor.fetchall()
#
#         # 각 레코드의 날짜를 2024-10-16으로 업데이트합니다.
#         for record in records:
#             id, old_time = record
#             # 기존 시간 파싱
#             old_datetime = datetime.strptime(old_time, '%Y-%m-%d %H:%M:%S')
#             # 새로운 날짜와 기존 시간을 결합
#             new_time = datetime(2024, 10, 15, old_datetime.hour, old_datetime.minute, old_datetime.second)
#             new_time_str = new_time.strftime('%Y-%m-%d %H:%M:%S')
#
#             # 레코드 업데이트
#             cursor.execute("UPDATE ai_stock_analysis_records SET Time = ? WHERE id = ?", (new_time_str, id))
#
#         # 변경사항 커밋
#         conn.commit()
#         print(f"Successfully updated {len(records)} records.")
#
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         conn.rollback()
#
#     finally:
#         # 연결 종료
#         conn.close()
#
# if __name__ == "__main__":
#     update_analysis_records_date()

import schedule
import time
from datetime import datetime
import pytz
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv
import os

load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
ROBINHOOD_BOT_USER_ID = os.getenv("ROBINHOOD_BOT_USER_ID")

# 메시지를 보낼 Slack 채널 ID
SLACK_CHANNEL_ID = "webhook"

# 알림을 보낼 주식 목록
STOCKS = ["NVDA"]

# Slack 클라이언트 초기화
client = WebClient(token=SLACK_BOT_TOKEN)


def send_slack_message(message):
    try:
        response = client.chat_postMessage(
            channel=SLACK_CHANNEL_ID,
            text=message,
            unfurl_links=False
        )
        print(f"Message sent: {response['ts']}")
    except SlackApiError as e:
        print(f"Error sending message: {e}")


def send_stock_reminder():
    current_time = datetime.now(pytz.timezone('America/New_York')).strftime("%Y-%m-%d %H:%M:%S")
    message = f"Stock Reminder ({current_time} ET):\n"
    for stock in STOCKS:
        message += f"<@{ROBINHOOD_BOT_USER_ID}> {stock}\n"
    send_slack_message(message)
    print(f"Reminder sent at {current_time}")


def main():
    # 뉴욕 시간으로 매일 아침 8:30에 실행
    schedule.every().day.at("16:00").do(send_stock_reminder).tag('stock_reminder')

    print("Stock reminder bot is running. Press Ctrl+C to stop.")

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()