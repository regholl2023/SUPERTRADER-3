import streamlit as st
import sqlite3
import pandas as pd
import time
from deep_translator import GoogleTranslator
import plotly.graph_objects as go

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
    total_predictions = len(df)
    correct_predictions = len(df[df['error_percentage'].abs() <= 5])  # Within 1% error
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    avg_error = df['error_percentage'].abs().mean()
    return accuracy, avg_error

# Create performance chart
def create_performance_chart(df):
    fig = go.Figure()
    for stock in df['stock'].unique():
        stock_data = df[df['stock'] == stock]
        fig.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['error_percentage'],
                                 mode='lines+markers', name=stock))
    fig.update_layout(title='Stock-wise Prediction Error Over Time',
                      xaxis_title='Date', yaxis_title='Error Percentage')
    return fig


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
            df_performance = load_performance_data()

            # Basic query statistics
            st.subheader("전체 주식 조회 통계")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("총 주식 조회 횟수", len(df_analysis))
            col2.metric("총 조회된 주식 수", df_analysis['Stock'].nunique())

            # Performance metrics
            accuracy, avg_error = calculate_performance_metrics(df_performance)
            col3.metric("예측 정확도 (1% 이내)", f"{accuracy:.2f}%")
            col4.metric("평균 오차", f"{avg_error:.2f}%")

            # Recent stock query history table
            st.subheader("AI Trading Advisor Performance")
            st.dataframe(df_performance[['date', 'stock', 'avg_current_price', 'next_date', 'avg_expected_next_day_price', 'actual_next_day_price',
                                      'price_difference', 'error_percentage']]
                         .sort_values('date', ascending=False).head(20), key=f"trading_history_{iteration}")

            # Recent stock query history table
            st.subheader("최근 주식 조회 내역")
            st.dataframe(df_analysis[['Time', 'Stock', 'Decision', 'Percentage', 'Reason', 'CurrentPrice', 'ExpectedNextDayPrice', 'ExpectedPriceDifference']]
                         .sort_values('Time', ascending=False).head(20), key=f"trading_history_{iteration}")

            # Performance chart
            st.subheader("주식별 예측 오차 추이")
            performance_chart = create_performance_chart(df_performance)
            st.plotly_chart(performance_chart, key=f"performance_chart_{iteration}")  # 수정된 부분

            # Recent reasons for AI decisions by stock
            st.subheader("주식별 최근 상황")
            for Stock in df_analysis['Stock'].unique():
                # Get the most recent data for the stock
                latest_data = df_analysis[df_analysis['Stock'] == Stock].sort_values('Time', ascending=False).iloc[0]

                # Display Stock, Decision, Percentage
                st.write(f"**{Stock} - {latest_data['Decision'].capitalize()} ({latest_data['Percentage']}%)**")

                recent_reasons = df_analysis[df_analysis['Stock'] == Stock][
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