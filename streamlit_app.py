import streamlit as st
import sqlite3
import pandas as pd
import time
from deep_translator import GoogleTranslator

# Database connection
@st.cache_resource
def get_connection():
    return sqlite3.connect('ai_trading_records.db', check_same_thread=False)

conn = get_connection()

# Data loading function
@st.cache_data(ttl=5)  # Refresh cache every 5 seconds
def load_data():
    query = "SELECT * FROM ai_trading_records"
    df = pd.read_sql_query(query, conn)
    df['Time'] = pd.to_datetime(df['Time'])
    return df

# Translation function
@st.cache_data
def translate_to_korean(text):
    translator = GoogleTranslator(source='auto', target='ko')
    return translator.translate(text)

# Main app
def main():
    st.title("AI 추천 주식 대시보드")

    # Auto-refresh settings
    auto_refresh = st.sidebar.checkbox("자동 새로고침 활성화", value=True)
    refresh_interval = st.sidebar.number_input("새로고침 간격 (초)", min_value=1, value=5)

    # Create placeholder
    placeholder = st.empty()

    iteration = 0
    while True:
        with placeholder.container():
            # Load data
            df = load_data()

            # Basic query statistics
            st.subheader("전체 주식 조회 통계")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("총 주식 조회 횟수", len(df))
            col2.metric("총 조회된 주식 수", df['Stock'].nunique())

            # Recent stock query history table
            st.subheader("최근 주식 조회 내역")
            st.dataframe(df[['Time', 'Stock', 'Decision', 'Percentage']]
                         .sort_values('Time', ascending=False).head(20), key=f"trading_history_{iteration}")

            # Recent reasons for AI decisions by stock
            st.subheader("주식별 최근 상황")
            for Stock in df['Stock'].unique():
                # Get the most recent data for the stock
                latest_data = df[df['Stock'] == Stock].sort_values('Time', ascending=False).iloc[0]

                # Display Stock, Decision, Percentage
                st.write(f"**{Stock} - {latest_data['Decision'].capitalize()} ({latest_data['Percentage']}%)**")

                recent_reasons = df[df['Stock'] == Stock][
                    ['Time', 'Decision', 'Percentage', 'Reason']].sort_values('Time', ascending=False).head(3)
                for idx, row in recent_reasons.iterrows():
                    reason_kr = translate_to_korean(row['Reason'])
                    st.write(
                        f"- {row['Time']} - {row['Decision'].capitalize()} ({row['Percentage']}%): {reason_kr}",
                        key=f"Reason_{iteration}_{Stock}_{idx}")
                st.write("")  # Separator between Stocks

        # If auto-refresh is enabled, run again after a specified time
        if auto_refresh:
            time.sleep(refresh_interval)
            iteration += 1
        else:
            break

if __name__ == "__main__":
    main()