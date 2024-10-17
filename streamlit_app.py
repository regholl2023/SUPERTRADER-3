import streamlit as st
import sqlite3
import pandas as pd
import time
from deep_translator import GoogleTranslator
from datetime import datetime, timedelta


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

def filter_data(df, stock):
    if stock != "전체 주식":
        return df[df['Stock Name'] == stock]
    return df


# 오늘의 주식 추천 함수
def get_todays_recommendations(df):
    today = datetime.now().date()
    recommendations = df[df['Date'].dt.date == today].copy()
    recommendations = recommendations[
        recommendations['Avg Expected Next Day Price ($)'] > recommendations['Avg Current Price ($)']]

    # 예상 수익과 수익률 계산
    recommendations['Expected Profit ($)'] = recommendations['Avg Expected Next Day Price ($)'] - recommendations[
        'Avg Current Price ($)']
    recommendations['Expected Return (%)'] = (recommendations['Expected Profit ($)'] / recommendations[
        'Avg Current Price ($)']) * 100

    return recommendations[['Stock Name', 'Avg Current Price ($)', 'Avg Expected Next Day Price ($)',
                            'Expected Profit ($)', 'Expected Return (%)']]


# 스타일링 함수
def style_recommendations(df):
    return df.style.format({
        'Avg Current Price ($)': '${:.2f}',
        'Avg Expected Next Day Price ($)': '${:.2f}',
        'Expected Profit ($)': '${:.2f}',
        'Expected Return (%)': '{:.2f}%'
    })

# 날짜 형식 변환 함수
def format_date(date):
    if isinstance(date, str):
        return date
    elif pd.notnull(date):
        return date.strftime('%Y-%m-%d')
    else:
        return ''


# 조건부 스타일링 함수
def style_dataframe(df):
    def highlight_positive_expectation(row):
        if row['Avg Expected Next Day Price ($)'] > row['Avg Current Price ($)'] and row['Price Difference ($)'] >= 0:
            return ['background-color: green; color: black'] * len(row)
        else:
            return [''] * len(row)

    return df.style.apply(highlight_positive_expectation, axis=1) \
        .format({
        'Date': format_date,
        'Next Date': format_date,
        'Avg Current Price ($)': '${:.2f}',
        'Avg Expected Next Day Price ($)': '${:.2f}',
        'Actual Next Day Price ($)': '${:.2f}',
        'Price Difference ($)': '${:.2f}',
        'Error Percentage (%)': '{:.2f}%'
    })


# 한국어 번역 함수
@st.cache_data
def translate_to_korean(text):
    translator = GoogleTranslator(source='auto', target='ko')
    return translator.translate(text)


# 최근 주식 조회 내역 스타일링 함수 수정
def style_recent_queries(df):
    today = datetime.now().date()

    def highlight_buy_today(row):
        if row['Date & Time'].date() == today and row['AI Decision'] == 'BUY':
            return ['background-color: green; color: black'] * len(row)
        else:
            return [''] * len(row)

    # Reason 열을 한국어로 번역
    df['Reason_KR'] = df['Reason'].apply(translate_to_korean)

    # 열 순서 변경
    columns = ['Date & Time', 'Stock Name', 'AI Decision', 'Decision Confidence (%)',
               'Current Price ($)', 'Expected Next Day Price ($)', 'Price Difference ($)', 'Reason_KR']
    df = df[columns]

    return df.style.apply(highlight_buy_today, axis=1) \
        .format({
        'Date & Time': lambda x: x.strftime('%Y-%m-%d %H:%M:%S'),
        'Decision Confidence (%)': '{:.2f}%',
        'Current Price ($)': '${:.2f}',
        'Expected Next Day Price ($)': '${:.2f}',
        'Price Difference ($)': '${:.2f}'
    })

# 오늘과 어제 날짜 계산 함수
def get_today_and_yesterday():
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    return today, yesterday

# 최근 주식 조회 내역 필터링 함수 수정
def filter_recent_queries(df, selected_stock):
    today, yesterday = get_today_and_yesterday()
    if selected_stock != "전체 주식":
        return df[(df['Date & Time'].dt.date.isin([today, yesterday])) &
                  (df['Stock Name'] == selected_stock)].sort_values('Date & Time', ascending=False)
    else:
        return df[df['Date & Time'].dt.date.isin([today, yesterday])].sort_values('Date & Time', ascending=False)



# Main app
def main():
    st.title("AI Stock Advisor Dashboard")

    # 데이터 로드
    df_analysis = load_analysis_data()
    df_analysis = rename_analysis_columns(df_analysis)
    df_performance = load_performance_data()
    df_performance = rename_performance_columns(df_performance)

    # 날짜 열 형식 변환
    df_performance['Date'] = pd.to_datetime(df_performance['Date'], errors='coerce')
    df_performance['Next Date'] = pd.to_datetime(df_performance['Next Date'], errors='coerce')

    # 주식 선택 옵션과 예측 정확도를 나란히 배치
    col1, col2 = st.columns([3, 2])

    with col1:
        stock_options = ["전체 주식"] + sorted(df_analysis['Stock Name'].unique().tolist())
        selected_stock = st.selectbox("주식 선택", stock_options)

    # 데이터 필터링
    filtered_analysis = filter_data(df_analysis, selected_stock)
    filtered_performance = filter_data(df_performance, selected_stock)

    # 예측 정확도 계산 및 표시
    with col2:
        accuracy, _ = calculate_performance_metrics(filtered_performance)
        if accuracy is not None:
            st.write("예측 정확도")
            col2_1, col2_2 = st.columns([7, 3])
            with col2_1:
                st.progress(accuracy / 100)
            with col2_2:
                st.write(f"{int(round(accuracy))}%")
        else:
            st.write("예측 정확도: 데이터 없음")

    # 오늘의 추천 주식 필터링
    todays_recommendations = get_todays_recommendations(df_performance)

    # Auto-refresh settings
    auto_refresh = st.sidebar.checkbox("자동 새로고침 활성화", value=True)
    refresh_interval = st.sidebar.number_input("새로고침 간격 (초)", min_value=1, value=5)

    # Create placeholder
    placeholder = st.empty()

    iteration = 0
    while True:
        with placeholder.container():
            # 오늘의 주식 추천
            today = datetime.now().strftime("%Y년 %m월 %d일")
            st.subheader(f"오늘의 AI 추천 주식 ({today})")
            if not todays_recommendations.empty:
                st.dataframe(style_recommendations(todays_recommendations), hide_index=True)
            else:
                st.write("오늘 추천된 주식이 없습니다.")

            # Recent stock query history table
            st.subheader("AI 주식 예측 내역")
            recent_queries = filter_recent_queries(filtered_analysis, selected_stock)
            if not recent_queries.empty:
                st.dataframe(
                    style_recent_queries(recent_queries),
                    hide_index=True,
                    key=f"analysis_history_{iteration}"
                )
            else:
                if selected_stock != "전체 주식":
                    st.write(f"{selected_stock}에 대한 최근 2일간의 주식 조회 내역이 없습니다.")
                else:
                    st.write("최근 2일간의 주식 조회 내역이 없습니다.")

            # Recent reasons for AI decisions by stock (마지막 섹션)
            st.subheader("주식별 AI 판단 내용")
            if selected_stock != "전체 주식":
                st.write(f"**{selected_stock}**")
                recent_reasons = filtered_analysis[['Date & Time', 'AI Decision', 'Decision Confidence (%)', 'Reason']].sort_values('Date & Time', ascending=False).head(5)
                for idx, row in recent_reasons.iterrows():
                    reason_kr = translate_to_korean(row['Reason'])
                    st.write(
                        f"- {row['Date & Time']} - {row['AI Decision'].capitalize()} ({row['Decision Confidence (%)']}%)"
                    )
                    st.write(f"  {reason_kr}")
                    st.write("")
            else:
                st.write("특정 주식을 선택하면 해당 주식의 최근 상황을 볼 수 있습니다.")

        # If auto-refresh is enabled, run again after a specified time
        if auto_refresh:
            time.sleep(refresh_interval)
            iteration += 1
        else:
            break

if __name__ == "__main__":
    main()