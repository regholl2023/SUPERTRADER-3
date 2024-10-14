import streamlit as st
import sqlite3
import pandas as pd
import time
from deep_translator import GoogleTranslator

# 데이터베이스 연결
@st.cache_resource
def get_connection():
    return sqlite3.connect('trading_data.db', check_same_thread=False)

conn = get_connection()

# 데이터 로드 함수
@st.cache_data(ttl=5)  # 5초마다 캐시 갱신
def load_data():
    query = "SELECT * FROM trading_records"
    df = pd.read_sql_query(query, conn)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# 번역 함수
@st.cache_data
def translate_to_korean(text):
    translator = GoogleTranslator(source='auto', target='ko')
    return translator.translate(text)

# 메인 앱
def main():
    st.title("실시간 거래 데이터 대시보드")

    # 자동 새로고침 설정
    auto_refresh = st.sidebar.checkbox("자동 새로고침 활성화", value=True)
    refresh_interval = st.sidebar.number_input("새로고침 간격 (초)", min_value=1, value=5)

    # 플레이스홀더 생성
    placeholder = st.empty()

    iteration = 0
    while True:
        with placeholder.container():
            # 데이터 로드
            df = load_data()

            # 기본 통계
            st.subheader("전체 거래 통계")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("총 거래 횟수", len(df))
            col2.metric("총 심볼 수", df['symbol'].nunique())
            col3.metric("총 잔고", f"${df['balance'].sum():,.2f}")
            col4.metric("총 주식 수", df['shares'].sum())

            # 최근 거래 내역 테이블
            st.subheader("최근 거래 내역")
            st.dataframe(df[['timestamp', 'symbol', 'decision', 'percentage', 'balance', 'shares', 'trading_value']]
                         .sort_values('timestamp', ascending=False).head(20), key=f"trading_history_{iteration}")

            # 심볼별 최근 거래 이유
            st.subheader("심볼별 최근 거래 이유")
            for symbol in df['symbol'].unique():
                # 해당 심볼의 가장 최근 데이터 가져오기
                latest_data = df[df['symbol'] == symbol].sort_values('timestamp', ascending=False).iloc[0]

                # 심볼, decision, percentage 표시
                st.write(f"**{symbol} - {latest_data['decision'].capitalize()} ({latest_data['percentage']}%)**")

                recent_reasons = df[df['symbol'] == symbol][
                    ['timestamp', 'decision', 'percentage', 'reason']].sort_values('timestamp', ascending=False).head(3)
                for idx, row in recent_reasons.iterrows():
                    reason_kr = translate_to_korean(row['reason'])
                    st.write(
                        f"- {row['timestamp']} - {row['decision'].capitalize()} ({row['percentage']}%): {reason_kr}",
                        key=f"reason_{iteration}_{symbol}_{idx}")
                st.write("")  # 심볼 간 구분

        # 자동 새로고침이 활성화되어 있으면 일정 시간 후 다시 실행
        if auto_refresh:
            time.sleep(refresh_interval)
            iteration += 1
        else:
            break

if __name__ == "__main__":
    main()