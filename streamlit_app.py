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


LANGUAGES = {'afrikaans': 'af', 'albanian': 'sq', 'amharic': 'am', 'arabic': 'ar', 'armenian': 'hy', 'assamese': 'as', 'aymara': 'ay', 'azerbaijani': 'az', 'bambara': 'bm', 'basque': 'eu', 'belarusian': 'be', 'bengali': 'bn', 'bhojpuri': 'bho', 'bosnian': 'bs', 'bulgarian': 'bg', 'catalan': 'ca', 'cebuano': 'ceb', 'chichewa': 'ny', 'chinese (simplified)': 'zh-CN', 'chinese (traditional)': 'zh-TW', 'corsican': 'co', 'croatian': 'hr', 'czech': 'cs', 'danish': 'da', 'dhivehi': 'dv', 'dogri': 'doi', 'dutch': 'nl', 'english': 'en', 'esperanto': 'eo', 'estonian': 'et', 'ewe': 'ee', 'filipino': 'tl', 'finnish': 'fi', 'french': 'fr', 'frisian': 'fy', 'galician': 'gl', 'georgian': 'ka', 'german': 'de', 'greek': 'el', 'guarani': 'gn', 'gujarati': 'gu', 'haitian creole': 'ht', 'hausa': 'ha', 'hawaiian': 'haw', 'hebrew': 'iw', 'hindi': 'hi', 'hmong': 'hmn', 'hungarian': 'hu', 'icelandic': 'is', 'igbo': 'ig', 'ilocano': 'ilo', 'indonesian': 'id', 'irish': 'ga', 'italian': 'it', 'japanese': 'ja', 'javanese': 'jw', 'kannada': 'kn', 'kazakh': 'kk', 'khmer': 'km', 'kinyarwanda': 'rw', 'konkani': 'gom', 'korean': 'ko', 'krio': 'kri', 'kurdish (kurmanji)': 'ku', 'kurdish (sorani)': 'ckb', 'kyrgyz': 'ky', 'lao': 'lo', 'latin': 'la', 'latvian': 'lv', 'lingala': 'ln', 'lithuanian': 'lt', 'luganda': 'lg', 'luxembourgish': 'lb', 'macedonian': 'mk', 'maithili': 'mai', 'malagasy': 'mg', 'malay': 'ms', 'malayalam': 'ml', 'maltese': 'mt', 'maori': 'mi', 'marathi': 'mr', 'meiteilon (manipuri)': 'mni-Mtei', 'mizo': 'lus', 'mongolian': 'mn', 'myanmar': 'my', 'nepali': 'ne', 'norwegian': 'no', 'odia (oriya)': 'or', 'oromo': 'om', 'pashto': 'ps', 'persian': 'fa', 'polish': 'pl', 'portuguese': 'pt', 'punjabi': 'pa', 'quechua': 'qu', 'romanian': 'ro', 'russian': 'ru', 'samoan': 'sm', 'sanskrit': 'sa', 'scots gaelic': 'gd', 'sepedi': 'nso', 'serbian': 'sr', 'sesotho': 'st', 'shona': 'sn', 'sindhi': 'sd', 'sinhala': 'si', 'slovak': 'sk', 'slovenian': 'sl', 'somali': 'so', 'spanish': 'es', 'sundanese': 'su', 'swahili': 'sw', 'swedish': 'sv', 'tajik': 'tg', 'tamil': 'ta', 'tatar': 'tt', 'telugu': 'te', 'thai': 'th', 'tigrinya': 'ti', 'tsonga': 'ts', 'turkish': 'tr', 'turkmen': 'tk', 'twi': 'ak', 'ukrainian': 'uk', 'urdu': 'ur', 'uyghur': 'ug', 'uzbek': 'uz', 'vietnamese': 'vi', 'welsh': 'cy', 'xhosa': 'xh', 'yiddish': 'yi', 'yoruba': 'yo', 'zulu': 'zu'}
# Translation function
@st.cache_data
def translate_text(text, target_lang):
    if target_lang != 'en':
        translator = GoogleTranslator(source='auto', target=target_lang)
        return translator.translate(text)
    else:
        return text


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
        return None, None


# Rename dataframe columns
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


# Rename dataframe columns
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
    if stock != "All Stocks":
        return df[df['Stock Name'] == stock]
    return df


# Today's stock recommendations function
def get_todays_recommendations(df):
    today = datetime.now().date()
    recommendations = df[df['Date'].dt.date == today].copy()
    recommendations = recommendations[
        recommendations['Avg Expected Next Day Price ($)'] > recommendations['Avg Current Price ($)']]

    # Calculate expected profit and rate of return
    recommendations['Expected Profit ($)'] = recommendations['Avg Expected Next Day Price ($)'] - recommendations[
        'Avg Current Price ($)']
    recommendations['Expected Return (%)'] = (recommendations['Expected Profit ($)'] / recommendations[
        'Avg Current Price ($)']) * 100

    return recommendations[['Stock Name', 'Avg Current Price ($)', 'Avg Expected Next Day Price ($)',
                            'Expected Profit ($)', 'Expected Return (%)']]


# styling functions
def style_recommendations(df):
    return df.style.format({
        'Avg Current Price ($)': '${:.2f}',
        'Avg Expected Next Day Price ($)': '${:.2f}',
        'Expected Profit ($)': '${:.2f}',
        'Expected Return (%)': '{:.2f}%'
    })

# Date format conversion function
def format_date(date):
    if isinstance(date, str):
        return date
    elif pd.notnull(date):
        return date.strftime('%Y-%m-%d')
    else:
        return ''


# Conditional styling functions
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

# Modify recent stock view history styling function
def style_recent_queries(df, target_lang):
    today = datetime.now().date()

    def highlight_buy_today(row):
        if row['Date & Time'].date() == today and row['AI Decision'] == 'BUY':
            return ['background-color: green; color: black'] * len(row)
        else:
            return [''] * len(row)

    # Translate Reason column to target language
    df['Translated_Reason'] = df['Reason'].apply(lambda x: translate_text(x, target_lang))

    # Change column order
    columns = ['Date & Time', 'Stock Name', 'AI Decision', 'Decision Confidence (%)',
               'Current Price ($)', 'Expected Next Day Price ($)', 'Price Difference ($)', 'Translated_Reason']
    df = df[columns]

    return df.style.apply(highlight_buy_today, axis=1) \
        .format({
        'Date & Time': lambda x: x.strftime('%Y-%m-%d %H:%M:%S'),
        'Decision Confidence (%)': '{:.2f}%',
        'Current Price ($)': '${:.2f}',
        'Expected Next Day Price ($)': '${:.2f}',
        'Price Difference ($)': '${:.2f}'
    })

# Today and yesterday date calculation function
def get_today_and_yesterday():
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    return today, yesterday

# Modification of recent stock inquiry history filtering function
def filter_recent_queries(df, selected_stock):
    today, yesterday = get_today_and_yesterday()
    if selected_stock != "All Stocks":
        return df[(df['Date & Time'].dt.date.isin([today, yesterday])) &
                  (df['Stock Name'] == selected_stock)].sort_values('Date & Time', ascending=False)
    else:
        return df[df['Date & Time'].dt.date.isin([today, yesterday])].sort_values('Date & Time', ascending=False)

# Main app
def main():
    st.title("AI Stock Advisor Dashboard")

    # data load
    df_analysis = load_analysis_data()
    df_analysis = rename_analysis_columns(df_analysis)
    df_performance = load_performance_data()
    df_performance = rename_performance_columns(df_performance)

    # Convert date column format
    df_performance['Date'] = pd.to_datetime(df_performance['Date'], errors='coerce')
    df_performance['Next Date'] = pd.to_datetime(df_performance['Next Date'], errors='coerce')

    # Stock picking options and forecast accuracy side by side
    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        stock_options = ["All Stocks"] + sorted(df_analysis['Stock Name'].unique().tolist())
        selected_stock = st.selectbox("Stock selection", stock_options, key="stock_selection")

    # data filtering
    filtered_analysis = filter_data(df_analysis, selected_stock)
    filtered_performance = filter_data(df_performance, selected_stock)

    # Calculate and display prediction accuracy
    with col2:
        accuracy, _ = calculate_performance_metrics(df_performance)
        if accuracy is not None:
            st.write("Prediction Accuracy")
            col2_1, col2_2 = st.columns([7, 3])
            with col2_1:
                st.progress(accuracy / 100)
            with col2_2:
                st.write(f"{int(round(accuracy))}%")
        else:
            st.write("Prediction Accuracy: no data")

    with st.sidebar:
        selected_language = st.selectbox(
            "Select Language",
            options=list(LANGUAGES.keys()),
            index=list(LANGUAGES.keys()).index('english'),
            format_func=lambda x: x.capitalize(),
            key="sidebar_language_selection"
        )

    target_lang = LANGUAGES[selected_language]

    # 오늘의 추천 주식 필터링
    todays_recommendations = get_todays_recommendations(df_performance)

    # Auto-refresh settings
    auto_refresh = st.sidebar.checkbox("Enable auto-refresh", value=True)
    refresh_interval = st.sidebar.number_input("Refresh interval (seconds)", min_value=1, value=5)

    # Create placeholder
    placeholder = st.empty()

    iteration = 0
    while True:
        with placeholder.container():
            # Today's Stock Recommendations
            today = datetime.now().strftime("%m/%d/%Y")
            st.subheader(f"Today's AI recommended stocks ({today})")
            if not todays_recommendations.empty:
                st.dataframe(style_recommendations(todays_recommendations), hide_index=True)
            else:
                st.write("There are no stocks recommended today.")

            # Recent stock query history table
            st.subheader(f"AI prediction history for {selected_stock}")
            recent_queries = filter_recent_queries(filtered_analysis, selected_stock)
            if not recent_queries.empty:
                st.dataframe(
                    style_recent_queries(recent_queries, target_lang),  # target_lang 인자 추가
                    hide_index=True,
                    key=f"analysis_history_{iteration}"
                )
            else:
                if selected_stock != "All Stocks":
                    st.write(f"There are no stock searches for {selected_stock} in the last 2 days.")
                else:
                    st.write("There is no stock inquiry history for the last 2 days.")

            # Recent reasons for AI decisions by stock
            # AI Decision Details
            st.subheader("AI Decision Details")
            if selected_stock != "All Stocks":
                st.write(f"**{selected_stock}**")
                recent_reasons = df_analysis[df_analysis['Stock Name'] == selected_stock][
                    ['Date & Time', 'AI Decision', 'Decision Confidence (%)', 'Reason']].sort_values('Date & Time',
                                                                                                     ascending=False).head(
                    5)
                for idx, row in recent_reasons.iterrows():
                    translated_reason = translate_text(row['Reason'], target_lang)
                    st.write(
                        f"- {row['Date & Time']} - {row['AI Decision'].capitalize()} ({row['Decision Confidence (%)']}%)"
                    )
                    st.write(f"  {translated_reason}")
                    st.write("")
            else:
                message = "If you select a specific stock, you can see the latest status of that stock."
                translated_message = translate_text(message, target_lang)
                st.write(translated_message)

        # If auto-refresh is enabled, run again after a specified time
        if auto_refresh:
            time.sleep(refresh_interval)
            iteration += 1
        else:
            break

if __name__ == "__main__":
    main()