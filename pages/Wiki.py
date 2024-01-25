import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import LocalAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.llms import OpenAI
import openai
import json
from datetime import datetime
import os
import QA


st.set_page_config(
    page_title="Wiki",
    page_icon="📖",  # Open book icon
    layout="wide",  # Set the layout to wide
)

#tab2, tab1 = st.tabs(['Q&A','Show Wiki'])

#with tab2:
    #st.write('')

#QA.qa_agent()
#with tab1:


st.markdown(
    """
    # User Guide for Technical Analysis and Backtesting Dashboard

    **Welcome to the Technical Analysis and Backtesting Dashboard!**

    This is a fun tool I developed to play around with technical indicators in FX. The user-friendly interface allows for a seamless experience, this user guide will walk you through using this tool for analyzing forex data and backtesting trading strategies..
    """
)

st.markdown(
    """
    ## Page 1: Technical Analysis

    1. **Input Parameters:**
       - **Currency Pairs:** Specify the currency pairs for analysis.
       - **Date Range:** Define the start and end dates for insightful analysis.
       - **Interval:** Select the desired time interval for data granularity.

    2. **Moving Averages:**
       - **Simple Moving Averages (SMA):** Customize SMA windows for trend identification.
       - **Exponential Moving Averages (EMA):** Explore exponential moving averages for advanced analysis.

    3. **Trend Indicators:**
       - **Parabolic SAR:** Enable this indicator for trend reversal insights.
       - **Average Directional Index (ADX):** Utilize ADX for understanding trend strength.

    4. **Momentum Indicators:**
       - **Relative Strength Index (RSI):** Gauge momentum with RSI.
       - **Stochastic Oscillator:** Utilize the Stochastic Oscillator for momentum analysis.

    5. **Volatility Indicators:**
       - **Bollinger Bands:** Visualize volatility through Bollinger Bands.

    6. **Oscillators:**
       - **Moving Average Convergence Divergence (MACD):** Leverage MACD for trend reversal signals.
       - **Commodity Channel Index (CCI):** Gain insights into potential overbought or oversold conditions.

    7. **Support and Resistance Indicators:**
       - **Pivot Points:** Enable pivot points for support and resistance analysis.

    8. **Display Charts:**
       - Click the "TA Analysis Dashboard" button to access insightful charts.
    """
)

st.markdown(
    """
    ## Page 2: Backtesting

    1. **Settings:**
       - **Base Currency:** Define the base currency for backtesting.
       - **Quote Currency:** Specify the quote currency for backtesting.
       - **Date Range:** Set the start and end dates for the backtest.
       - **Interval:** Choose the time interval for historical data.

    2. **Write Your Trading Strategy:**
       - Enter your custom trading strategy in the provided code editor.
       - Example strategies, such as Moving Average Crossover, RSI, and Bollinger Bands, are included for reference.

    3. **Backtest:**
       - Set the maximum balance for the backtest.
       - Adjust the balance slider to select the desired backtesting amount.
       - Click the "Backtest" button to view comprehensive results.

    4. **Backtest Results:**
       - Review the starting and ending portfolio values.
       - Analyze the performance with the visual representation provided.
       
    """
)

