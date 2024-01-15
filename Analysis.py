import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import simple_TA_lib
today = datetime.today()
default_start_date = today.replace(day=1)
default_end_date = today + timedelta(days=1)


# Set the page configuration
st.set_page_config(
    page_title="Analysis",
    page_icon="🔍",
    layout="wide",  # Set the layout to wide
)

def ta_analysis_DB():


    with st.sidebar:
        tab1, tab2 = st.tabs(['Input Parameters', 'Technical Indicators'])

    with tab1:
        with st.expander("**Input Parameters**", expanded=True):


            currencypair1 = st.text_input('Pair 1', 'EUR')
            currencypair2 = st.text_input('Pair 2', 'USD')
            start_date = st.date_input('Start Date', default_start_date)
            end_date = st.date_input('End Date', default_end_date)
            interval_options = ['Daily', 'Weekly', 'Monthly', 'Hourly'][::-1]
            io_decoder = {'Daily': '1d', 'Weekly': '1wk', 'Monthly': '1mo', 'Hourly': '1h'}
            interval = st.selectbox('Select Interval', interval_options, index=0)
            interval = io_decoder[interval]

        if 'sma_windows' not in st.session_state:
            st.session_state.sma_windows = []
        if 'ema_spans' not in st.session_state:
            st.session_state.ema_spans = []
        if 'adx_windows' not in st.session_state:
            st.session_state.adx_windows = []
        if 'selected_smas' not in st.session_state:
            st.session_state.selected_smas = []
        if 'selected_emas' not in st.session_state:
            st.session_state.selected_emas = []
        if 'selected_adx' not in st.session_state:
            st.session_state.selected_adx = []


    # Technical Indicators:

    with tab2:

        st.write('**Moving Averages:**')
        with st.expander("SMA"):
            sma_form = st.form("sma_form")
            sma_window_input = sma_form.text_input("Enter SMA Window", key="sma_window")
            sma_add_button = sma_form.form_submit_button("Add SMA")


            if sma_add_button:
                if sma_window_input:
                    sma_window = int(sma_window_input)
                    st.session_state.sma_windows.append(sma_window)

            selected_smas = st.multiselect("Select SMAs to Display", st.session_state.sma_windows, st.session_state.sma_windows)

            if st.session_state.selected_smas != selected_smas:
                st.session_state.selected_smas = selected_smas
                st.session_state.sma_windows = selected_smas

        with st.expander("EMA"):
            ema_form = st.form("ema_form")
            ema_span_input = ema_form.text_input("Enter EMA Span", key="ema_span")
            ema_add_button = ema_form.form_submit_button("Add EMA")

            if ema_add_button:
                if ema_span_input:
                    ema_span = int(ema_span_input)
                    st.session_state.ema_spans.append(ema_span)

            selected_emas = st.multiselect("Select EMAs to Display", st.session_state.ema_spans, st.session_state.ema_spans)

            if st.session_state.selected_emas != selected_emas:
                st.session_state.selected_emas = selected_emas
                st.session_state.ema_spans = selected_emas


    # Trend Indicators:

        st.write('**Trend Indicators:**')
        with st.expander("Parabolic SAR"):
            sar_enabled = st.checkbox("Enable Parabolic SAR", value=False)

            if sar_enabled:
                acceleration = st.slider("Acceleration", min_value=0.01, max_value=0.1, value=0.02, step=0.01, key="sar_acceleration")
                maximum = st.slider("Maximum", min_value=0.1, max_value=1.0, value=0.2, step=0.1, key="sar_maximum")

        with st.expander("ADX"):
            adx_form = st.form("adx_form")
            adx_window_input = adx_form.text_input("Enter ADX Window", key="adx_window")
            adx_add_button = adx_form.form_submit_button("Add ADX")

            if adx_add_button:
                if adx_window_input:
                    adx_window = int(adx_window_input)
                    st.session_state.adx_windows.append(adx_window)

            selected_adx = st.multiselect("Select ADX to Display", st.session_state.adx_windows, st.session_state.adx_windows)

            if st.session_state.selected_adx != selected_adx:
                st.session_state.selected_adx = selected_adx
                st.session_state.adx_windows = selected_adx

        # Momentum Indicators
        st.write('**Momentum Indicators:**')
        with st.expander("RSI"):
            rsi_enabled = st.checkbox("Enable RSI", value=False)

            if rsi_enabled:
                rsi_window_input = st.text_input("RSI Window", value='14')
                try:
                    rsi_window = int(rsi_window_input)
                except ValueError:
                    st.error("Please enter a valid integer for RSI Window.")
                    st.stop()

        with st.expander("Stochastic Oscillator"):
            stochastic_enabled = st.checkbox("Enable Stochastic Oscillator", value=False)

            if stochastic_enabled:
                k_period = st.slider("K Period", min_value=1, max_value=50, value=14, step=1, key="stochastic_k_period")
                d_period = st.slider("D Period", min_value=1, max_value=50, value=3, step=1, key="stochastic_d_period")

        # Volatility Indicators:

        st.write('**Volatility Indicators:**')
        # Add Bollinger Bands parameters to the sidebar
        with st.expander("Bollinger Bands"):
            bb_enabled = st.checkbox("Enable Bollinger Bands", value=False)

            if bb_enabled:
                bb_window = st.slider("Bollinger Bands Window", min_value=5, max_value=50, value=20, step=1, key="bb_window")
                bb_std_dev = st.slider("Number of Standard Deviations", min_value=1, max_value=5, value=2, step=1,
                                       key="bb_std_dev")

        # Oscillators:
        st.write('**Oscillators:**')
        with st.expander("MACD"):
            macd_enabled = st.checkbox("Enable MACD", value=False)

            if macd_enabled:
                short_window = st.slider("Short-term Window", min_value=1, max_value=50, value=12, step=1, key="macd_short_window")
                long_window = st.slider("Long-term Window", min_value=1, max_value=50, value=26, step=1, key="macd_long_window")
                signal_window = st.slider("Signal Window", min_value=1, max_value=50, value=9, step=1, key="macd_signal_window")

        with st.expander("CCI"):
            cci_enabled = st.checkbox("Enable Commodity Channel Index (CCI)", value=False)

            if cci_enabled:
                cci_window = st.slider("CCI Window", min_value=1, max_value=50, value=20, step=1, key="cci_window")


        symbol = f'{currencypair1}{currencypair2}=X'
        forex_data, close_prices = simple_TA_lib.get_forex_data(symbol, start_date, end_date, interval)


        # Support and Resistance Indicators:
        st.write('**Support and Resistance Indicators:**')

        with st.expander("Pivot Points"):
            pivot_enabled = st.checkbox("Enable Pivot Points", value=False)

            if pivot_enabled:
                # Calculate pivot points using the new function

                pivot_data = simple_TA_lib.calculate_pivot_points(forex_data)


        fig_candlestick = go.Figure(data=[go.Candlestick(x=forex_data.index,
                                                          open=forex_data['Open'],
                                                          high=forex_data['High'],
                                                          low=forex_data['Low'],
                                                          close=forex_data['Close'])])

        # initial_x_range = [end_date - timedelta(days=3), end_date]
        # fig_candlestick.update_layout(xaxis_range=initial_x_range)


    fig_adx = go.Figure()
    fig_rsi = go.Figure()
    fig_bollinger = go.Figure()
    fig_macd = go.Figure()
    fig_cci = go.Figure()

    # Add candlestick chart to the first figure
    for window in st.session_state.selected_smas:
        sma = simple_TA_lib.calculate_sma(forex_data, window)
        fig_candlestick.add_trace(go.Scatter(x=forex_data.index, y=sma, mode='lines', name=f'SMA {window}'))

    for span in st.session_state.selected_emas:
        ema = simple_TA_lib.calculate_ema(forex_data, span)
        fig_candlestick.add_trace(go.Scatter(x=forex_data.index, y=ema, mode='lines', name=f'EMA {span}'))

    # Add ADX to the second figure
    for window in st.session_state.selected_adx:
        adx = simple_TA_lib.calculate_adx(forex_data, window)
        fig_adx.add_trace(go.Scatter(x=forex_data.index, y=adx, mode='lines', name=f'ADX {window}'))

    # Add RSI to the third figure if enabled
    if rsi_enabled:
        rsi = simple_TA_lib.calculate_rsi(forex_data, window=rsi_window)
        fig_rsi.add_trace(go.Scatter(x=forex_data.index, y=rsi, mode='lines', name=f'RSI {rsi_window}'))

    fig_candlestick.update_layout(title_text=f'{symbol} Candlestick Chart',
                                  xaxis_title='Date',
                                  yaxis_title='Price',
                                  xaxis_rangeslider_visible=True,
                                  height=600)

    fig_adx.update_layout(title_text=f'{symbol} Average Directional Index (ADX)',
                          xaxis_title='Date',
                          yaxis_title='ADX')

    fig_rsi.update_layout(title_text=f'{symbol} Relative Strength Index (RSI)',
                          xaxis_title='Date',
                          yaxis_title='RSI')

    if macd_enabled:
        macd_line, signal_line, histogram = simple_TA_lib.calculate_macd(forex_data, short_window, long_window, signal_window)
        fig_macd.add_trace(go.Scatter(x=forex_data.index, y=macd_line, mode='lines', name='MACD Line'))
        fig_macd.add_trace(go.Scatter(x=forex_data.index, y=signal_line, mode='lines', name='Signal Line'))
        fig_macd.add_trace(go.Bar(x=forex_data.index, y=histogram, name='Histogram'))

    fig_macd.update_layout(title_text=f'{symbol} MACD Indicator',
                           xaxis_title='Date',
                           yaxis_title='MACD Values',
                           xaxis_rangeslider_visible=True,
                           height=400)

    # Add Bollinger Bands to the first figure
    if bb_enabled:
        bollinger_data = simple_TA_lib.calculate_bollinger_bands(forex_data, window=bb_window, num_std_dev=bb_std_dev)
        fig_candlestick.add_trace(go.Scatter(x=forex_data.index, y=bollinger_data['UpperBand'], mode='lines', line=dict(color='red'), name='Upper Band'))
        fig_candlestick.add_trace(go.Scatter(x=forex_data.index, y=bollinger_data['SMA'], mode='lines', line=dict(color='blue'), name='SMA'))
        fig_candlestick.add_trace(go.Scatter(x=forex_data.index, y=bollinger_data['LowerBand'], mode='lines', line=dict(color='green'), name='Lower Band'))

    fig_stochastic = go.Figure()

    # Add Stochastic Oscillator to the fourth figure if enabled
    if stochastic_enabled:
        stochastic_data = simple_TA_lib.calculate_stochastic_oscillator(forex_data, k_period, d_period)
        fig_stochastic.add_trace(go.Scatter(x=forex_data.index, y=stochastic_data['%K'], mode='lines', name='%K'))
        fig_stochastic.add_trace(go.Scatter(x=forex_data.index, y=stochastic_data['%D'], mode='lines', name='%D'))

    fig_stochastic.update_layout(title_text=f'{symbol} Stochastic Oscillator',
                                 xaxis_title='Date',
                                 yaxis_title='Percentage',
                                 xaxis_rangeslider_visible=True,
                                 height=400)

    if cci_enabled:
        cci_values = simple_TA_lib.calculate_cci(forex_data, window=cci_window)
        fig_cci.add_trace(go.Scatter(x=forex_data.index, y=cci_values, mode='lines', name=f'CCI ({cci_window})'))

    fig_cci.update_layout(title_text=f'{symbol} Commodity Channel Index (CCI)',
                          xaxis_title='Date',
                          yaxis_title='CCI Values',
                          xaxis_rangeslider_visible=True,
                          height=400)


    if pivot_enabled:
        fig_candlestick.add_trace(go.Scatter(x=forex_data.index, y=pivot_data['Pivot'], mode='markers', marker=dict(color='orange', size=4), name='Pivot'))
        fig_candlestick.add_trace(go.Scatter(x=forex_data.index, y=pivot_data['R1'], mode='markers', marker=dict(color='purple', size=4), name='R1'))
        fig_candlestick.add_trace(go.Scatter(x=forex_data.index, y=pivot_data['S1'], mode='markers', marker=dict(color='brown', size=4), name='S1'))
        fig_candlestick.add_trace(go.Scatter(x=forex_data.index, y=pivot_data['R2'], mode='markers', marker=dict(color='pink', size=4), name='R2'))
        fig_candlestick.add_trace(go.Scatter(x=forex_data.index, y=pivot_data['S2'], mode='markers', marker=dict(color='gray', size=4), name='S2'))
        fig_candlestick.add_trace(go.Scatter(x=forex_data.index, y=pivot_data['R3'], mode='markers', marker=dict(color='cyan', size=4), name='R3'))
        fig_candlestick.add_trace(go.Scatter(x=forex_data.index, y=pivot_data['S3'], mode='markers', marker=dict(color='yellow', size=4), name='S3'))


    if sar_enabled:
        sar = simple_TA_lib.calculate_sar(forex_data, acceleration=acceleration, maximum=maximum)
        fig_candlestick.add_trace(go.Scatter(x=forex_data.index, y=sar,
                                             mode='markers',
                                             marker=dict(color='grey', size=2.5),


                                             name='Parabolic SAR'))

    st.title("TA Analysis Dashboard")

    st.plotly_chart(fig_candlestick, use_container_width=True)

    if len(st.session_state.adx_windows) != 0:
        st.plotly_chart(fig_adx)

    # Display RSI chart only if enabled
    if rsi_enabled:
        st.plotly_chart(fig_rsi, use_container_width=True)


    # Display the Stochastic Oscillator chart
    if stochastic_enabled:
        st.plotly_chart(fig_stochastic, use_container_width=True)

    if macd_enabled:
        st.plotly_chart(fig_macd, use_container_width=True)

    if cci_enabled:
        st.plotly_chart(fig_cci, use_container_width=True)


ta_analysis_DB()
