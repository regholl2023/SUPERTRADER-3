import streamlit as st
import sqlite3
import pandas as pd
import time
from deep_translator import GoogleTranslator


# Database connections
@st.cache_resource
def get_analysis_connection():
    return sqlite3.connect('ai_stock_analysis_records.db', check_same_thread=False)


@st.cache_resource
def get_performance_connection():
    return sqlite3.connect('ai_stock_performance.db', check_same_thread=False)


conn_analysis = get_analysis_connection()
conn_performance = get_performance_connection()


# Data loading functions
@st.cache_data(ttl=5)  # Refresh cache every 5 seconds
def load_analysis_data():
    query = "SELECT * FROM ai_stock_analysis_records"
    df = pd.read_sql_query(query, conn_analysis)
    df['Time'] = pd.to_datetime(df['Time'])
    return df


@st.cache_data(ttl=5)  # Refresh cache every 5 seconds
def load_performance_data():
    query = "SELECT * FROM stock_performance"
    df = pd.read_sql_query(query, conn_performance)
    df['date'] = pd.to_datetime(df['date'])
    return df


# Translation function
@st.cache_data
def translate_to_korean(text):
    translator = GoogleTranslator(source='auto', target='ko')
    return translator.translate(text)


# Performance metrics calculation
def calculate_performance_metrics(df):
    # Remove rows where actual_next_day_price is None
    df = df[df['Actual Next Day Price ($)'].notna()]

    total_predictions = len(df)
    if total_predictions > 0:
        correct_predictions = len(df[df['Error Percentage (%)'].abs() <= 5])  # Within 5% error
        accuracy = (correct_predictions / total_predictions) * 100
        avg_error = df['Error Percentage (%)'].abs().mean()
        return accuracy, avg_error
    else:
        return None, None  # 유효한 예측이 없을 경우 (None, None) 반환


# 데이터프레임 컬럼 이름 변경
def rename_performance_columns(df):
    df = df.rename(columns={
        'date': 'Date',
        'stock': 'Stock Name',
        'avg_current_price': 'Avg Current Price ($)',
        'next_date': 'Next Date',
        'avg_expected_next_day_price': 'Avg Expected Next Day Price ($)',
        'actual_next_day_price': 'Actual Next Day Price ($)',
        'price_difference': 'Price Difference ($)',
        'error_percentage': 'Error Percentage (%)'
    })
    return df


# 데이터프레임 컬럼 이름 변경
def rename_analysis_columns(df):
    df = df.rename(columns={
        'Time': 'Date & Time',
        'Stock': 'Stock Name',
        'Decision': 'AI Decision',
        'Percentage': 'Decision Confidence (%)',
        'Reason': 'Reason',
        'CurrentPrice': 'Current Price ($)',
        'ExpectedNextDayPrice': 'Expected Next Day Price ($)',
        'ExpectedPriceDifference': 'Price Difference ($)'
    })
    return df


# Main app
def main():
    st.title("AI Stock Advisor Dashboard")

    # Auto-refresh settings
    auto_refresh = st.sidebar.checkbox("자동 새로고침 활성화", value=True)
    refresh_interval = st.sidebar.number_input("새로고침 간격 (초)", min_value=1, value=5)

    # Create placeholder
    placeholder = st.empty()

    iteration = 0
    while True:
        with placeholder.container():
            # Load data
            df_analysis = load_analysis_data()
            df_analysis = rename_analysis_columns(df_analysis)

            df_performance = load_performance_data()
            df_performance = rename_performance_columns(df_performance)

            # Basic query statistics
            st.subheader("전체 주식 조회 통계")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("총 주식 조회 횟수", len(df_analysis))
            col2.metric("총 조회된 주식 수", df_analysis['Stock Name'].nunique())

            # Performance metrics
            accuracy, avg_error = calculate_performance_metrics(df_performance)
            if accuracy is not None:
                col3.metric("예측 정확도 (5% 이내)", f"{accuracy:.2f}%")
            else:
                col3.metric("예측 정확도 (5% 이내)", "데이터 없음")

            if avg_error is not None:
                col4.metric("평균 오차", f"{avg_error:.2f}%")
            else:
                col4.metric("평균 오차", "데이터 없음")

            # Recent stock query history table
            st.subheader("최근 거래 내역")
            st.dataframe(df_performance[['Date', 'Stock Name', 'Avg Current Price ($)', 'Next Date',
                                         'Avg Expected Next Day Price ($)', 'Actual Next Day Price ($)',
                                         'Price Difference ($)', 'Error Percentage (%)']]
                         .sort_values('Date', ascending=False).head(20), key=f"trading_history_{iteration}")

            # Recent stock query history table
            st.subheader("최근 주식 조회 내역")
            st.dataframe(df_analysis[['Date & Time', 'Stock Name', 'AI Decision', 'Decision Confidence (%)',
                                      'Reason', 'Current Price ($)', 'Expected Next Day Price ($)',
                                      'Price Difference ($)']]
                         .sort_values('Date & Time', ascending=False).head(20), key=f"analysis_history_{iteration}")

            # Recent reasons for AI decisions by stock
            st.subheader("주식별 최근 상황")
            for Stock in df_analysis['Stock Name'].unique():
                # Get the most recent data for the stock
                latest_data = \
                df_analysis[df_analysis['Stock Name'] == Stock].sort_values('Date & Time', ascending=False).iloc[0]

                # Display Stock, Decision, Percentage
                st.write(
                    f"**{Stock} - {latest_data['AI Decision'].capitalize()} ({latest_data['Decision Confidence (%)']}%)**")

                recent_reasons = df_analysis[df_analysis['Stock Name'] == Stock][
                    ['Date & Time', 'AI Decision', 'Decision Confidence (%)', 'Reason']].sort_values('Date & Time',
                                                                                                     ascending=False).head(
                    3)
                for idx, row in recent_reasons.iterrows():
                    reason_kr = translate_to_korean(row['Reason'])
                    st.write(
                        f"- {row['Date & Time']} - {row['AI Decision'].capitalize()} ({row['Decision Confidence (%)']}%): {reason_kr}",
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