import streamlit as st

import time
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

tab2, tab1 = st.tabs(['Q&A','Show Wiki'])

with tab2:
    st.write('')

QA.qa_agent()
    



with tab1:
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
           
           """)

    try:
        st.image('https://media2.giphy.com/media/Wr1Otx0OtapvaNUca7/giphy.gif')
    except: pass


    st.markdown("- Additionally the candle chart can be adjusted as such")

    try:

        st.image('https://media2.giphy.com/media'
                 '/v1.Y2lkPTc5MGI3NjExbG1ocT'
                 'RpNWM4ZHVyNzJiOHFvaTAyZTYybGQwdXc1bzFhY'
                 'mh4aG0yYiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/UItiNU'
                 '3l8xx2FRCGud/giphy.gif')
    except: pass

    st.markdown("- See the raw data tab to download raw data:")


    try:
        st.image('https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExZ'
                 'TA0Y2R2aTMwY2M1ZzloY3I0eG05dXR4NnpuYj'
                 'FlY21rYm5qNDNlaSZlcD12MV9pbnRlcm5hbF9naWZfYnl'
                 'faWQmY3Q9Zw/OSy6dHwxGrYopGcK8q/giphy.gif')

    except: pass

    st.markdown("""

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
           
        8. **Example**
           - Example case for Moving Averages and Bollinger Bands 

        """
    )

    try:
        st.image('https://media2.giphy.com/media/aSYDBxWrcysgR5cfLj/giphy.gif')
        
    except: pass

    


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
    try:
        st.image('https://media3.giphy.com/media/TzAU5TnmxP80nE80PV/giphy.gif')
    except: pass

    st.markdown(
        """
       ## Page 3: Chat Bot Integration

    Welcome to the Chat Bot Integration page! This section seamlessly integrates a chat bot into the Technical Analysis and Backtesting Dashboard, enhancing your interactive experience. The chat bot has been fine-tuned on the project's wiki, providing information and assistance related to the tool. Here's a breakdown of the functionalities:

    1. **Chat Bot Introduction:**

        - To initiate interaction with the chat bot, click the "Assistant" section. The bot is designed to answer questions and provide insights about the project.

     2. **Conversation Management:**

        - In the "Save Conversation" tab, you can download and upload conversation sessions. This feature enables you to revisit and continue conversations at a later time or to share conversations with other people.

    3. **Local Deployment:**

        -  Both the embedding model and the LLM (Language Model) are hosted locally, this was done to maximize control over the model as well as to familiarize myself with AI development methodology.

    4. **Langchain LLM Configuration:**

        - The Langchain LLM (Language Model) is configured to operate with OpenAI. The API base is set to my local machine.

    5. **Text Processing:**

        - The tool processes the text from the specified file, splitting it into chunks for efficient handling. The LocalAIEmbeddings model is used for creating embeddings.

    6. **Knowledge Base Building:**

        - The FAISS (Facebook AI Similarity Search) is employed to build a knowledge base using the generated embeddings. This knowledge base assists the chat bot in providing relevant responses.

    7. **User-Chat Interaction:**

        - User and assistant messages are displayed, allowing you to see the ongoing conversation. Each message can be deleted individually by clicking the "Delete Message" button.

    8. **User Input and Chat Bot Response:**

        - The chat bot prompts the user to ask it questions and awaits a response. Upon receiving the user's input, the chat bot processes the question, performs a similarity search, and utilizes the Langchain QA chain to formulate a response.


    Feel free to explore the capabilities of the chat bot and leverage it for additional insights into the Technical Analysis and Backtesting Dashboard!


        """
    )


