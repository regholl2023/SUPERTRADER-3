import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

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

# 메인 앱
def main():
    st.title("Live Trading Data Dashboard")

    # 자동 새로고침 설정
    auto_refresh = st.sidebar.checkbox("Enable auto-refresh", value=True)
    refresh_interval = st.sidebar.number_input("Refresh interval (seconds)", min_value=1, value=5)

    # 플레이스홀더 생성
    placeholder = st.empty()

    iteration = 0
    while True:
        with placeholder.container():
            # 데이터 로드
            df = load_data()

            # 기본 통계
            st.header("Overall Trading Statistics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Trades", len(df))
            col2.metric("Total Symbols", df['symbol'].nunique())
            col3.metric("Total Balance", f"${df['balance'].sum():,.2f}")
            col4.metric("Total Shares", df['shares'].sum())

            # 거래 결정 분포
            st.subheader("Trading Decisions Distribution")
            decision_counts = df['decision'].value_counts()
            fig_decisions = px.pie(values=decision_counts.values, names=decision_counts.index, title="Trading Decisions")
            st.plotly_chart(fig_decisions, use_container_width=True, key=f"decisions_chart_{iteration}")

            # 심볼별 거래 횟수
            st.subheader("Trades by Symbol")
            symbol_counts = df['symbol'].value_counts()
            fig_symbols = px.bar(x=symbol_counts.index, y=symbol_counts.values, title="Number of Trades by Symbol")
            fig_symbols.update_xaxes(title="Symbol")
            fig_symbols.update_yaxes(title="Number of Trades")
            st.plotly_chart(fig_symbols, use_container_width=True, key=f"symbols_chart_{iteration}")

            # 전체 잔고 변화 그래프
            st.subheader("Total Balance Over Time")
            df_balance = df.groupby('timestamp')['balance'].sum().reset_index()
            fig_balance = px.line(df_balance, x='timestamp', y='balance', title="Total Balance History")
            st.plotly_chart(fig_balance, use_container_width=True, key=f"balance_chart_{iteration}")

            # 심볼별 잔고 변화 그래프
            st.subheader("Balance Over Time by Symbol")
            fig_balance_by_symbol = go.Figure()
            for symbol in df['symbol'].unique():
                df_symbol = df[df['symbol'] == symbol]
                fig_balance_by_symbol.add_trace(go.Scatter(x=df_symbol['timestamp'], y=df_symbol['balance'],
                                                           mode='lines', name=symbol))
            fig_balance_by_symbol.update_layout(title="Balance History by Symbol", xaxis_title="Date", yaxis_title="Balance")
            st.plotly_chart(fig_balance_by_symbol, use_container_width=True, key=f"balance_by_symbol_chart_{iteration}")

            # 거래 내역 테이블
            st.subheader("Recent Trading History")
            st.dataframe(df[['timestamp', 'symbol', 'decision', 'percentage', 'balance', 'shares', 'trading_value']]
                         .sort_values('timestamp', ascending=False).head(20), key=f"trading_history_{iteration}")

            # 최근 거래 이유
            st.subheader("Recent Trading Reasons")
            recent_reasons = df[['timestamp', 'symbol', 'decision', 'reason']].sort_values('timestamp', ascending=False).head(5)
            for idx, row in recent_reasons.iterrows():
                st.write(f"**{row['timestamp']}** - {row['symbol']} {row['decision'].capitalize()}: {row['reason']}", key=f"reason_{iteration}_{idx}")

        # 자동 새로고침이 활성화되어 있으면 일정 시간 후 다시 실행
        if auto_refresh:
            time.sleep(refresh_interval)
            iteration += 1
        else:
            break

if __name__ == "__main__":
    main()