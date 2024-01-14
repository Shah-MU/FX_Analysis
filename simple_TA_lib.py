import yfinance as yf
import pandas as pd
import numpy as np


def get_forex_data(symbol, start_date, end_date, interval='1d'):
    forex_data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    close_prices = np.array(forex_data['Close'])
    return forex_data, close_prices




def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()

def calculate_ema(data, span):
    return data['Close'].ewm(span=span, adjust=True).mean()

def calculate_adx(data, window):
    data['High-Low'] = data['High'] - data['Low']
    data['High-PrevClose'] = np.abs(data['High'] - data['Close'].shift(1))
    data['Low-PrevClose'] = np.abs(data['Low'] - data['Close'].shift(1))

    data['TR'] = data[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    data['ATR'] = data['TR'].rolling(window=window).mean()

    data['DMplus'] = np.where((data['High'] - data['High'].shift(1)) > (data['Low'].shift(1) - data['Low']),
                              np.maximum(data['High'] - data['High'].shift(1), 0), 0)
    data['DMminus'] = np.where((data['Low'].shift(1) - data['Low']) > (data['High'] - data['High'].shift(1)),
                               np.maximum(data['Low'].shift(1) - data['Low'], 0), 0)

    data['Smoothed_DMplus'] = data['DMplus'].rolling(window=window).mean()
    data['Smoothed_DMminus'] = data['DMminus'].rolling(window=window).mean()

    data['DIplus'] = (data['Smoothed_DMplus'] / data['ATR']) * 100
    data['DIminus'] = (data['Smoothed_DMminus'] / data['ATR']) * 100

    data['DX'] = (np.abs(data['DIplus'] - data['DIminus']) / (data['DIplus'] + data['DIminus'])) * 100
    data['ADX'] = data['DX'].rolling(window=window).mean()

    return data['ADX']

def calculate_sar(data, acceleration=0.02, maximum=0.2):
    # Calculates Parabolic SAR values
    # Requires columns like 'High' and 'Low'
    # Returns a pandas Series with SAR values
    data['SAR'] = np.nan
    data['AF'] = acceleration

    trend = 1  # Initial trend assumption (1 for uptrend, -1 for downtrend)
    sar = data['Low'][0] if trend == 1 else data['High'][0]
    extreme_point = data['High'][0] if trend == 1 else data['Low'][0]

    for i in range(2, len(data)):
        prev_sar = sar
        prev_extreme_point = extreme_point

        if trend == 1:
            sar = prev_sar + acceleration * (prev_extreme_point - prev_sar)
        else:
            sar = prev_sar - acceleration * (prev_sar - prev_extreme_point)

        if trend == 1:
            extreme_point = max(data['High'][i - 1], data['High'][i - 2], prev_extreme_point)
        else:
            extreme_point = min(data['Low'][i - 1], data['Low'][i - 2], prev_extreme_point)

        if trend == 1 and data['Low'][i] < sar:
            trend = -1
            sar = prev_extreme_point
            extreme_point = data['Low'][i]
            acceleration = 0.02
        elif trend == -1 and data['High'][i] > sar:
            trend = 1
            sar = prev_extreme_point
            extreme_point = data['High'][i]
            acceleration = 0.02

        acceleration = min(acceleration + 0.02, maximum)
        data['SAR'][i] = sar

    return data['SAR']

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_stochastic_oscillator(data, k_period=14, d_period=3):
    data['Lowest_Low'] = data['Low'].rolling(window=k_period).min()
    data['Highest_High'] = data['High'].rolling(window=k_period).max()

    data['%K'] = ((data['Close'] - data['Lowest_Low']) / (data['Highest_High'] - data['Lowest_Low'])) * 100
    data['%D'] = data['%K'].rolling(window=d_period).mean()

    return data[['%K', '%D']]

def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    data['SMA'] = data['Close'].rolling(window=window).mean()
    data['UpperBand'] = data['SMA'] + num_std_dev * data['Close'].rolling(window=window).std()
    data['LowerBand'] = data['SMA'] - num_std_dev * data['Close'].rolling(window=window).std()
    return data[['SMA', 'UpperBand', 'LowerBand']]





def calculate_macd(data, short_window=12, long_window=26, signal_window=9):

    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()

    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram

def calculate_cci(data, window=20):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    mean_deviation = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    cci = (typical_price - typical_price.rolling(window=window).mean()) / (0.015 * mean_deviation)
    return cci

def calculate_pivot_points(data):
    data['Pivot'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['R1'] = 2 * data['Pivot'] - data['Low']
    data['S1'] = 2 * data['Pivot'] - data['High']
    data['R2'] = data['Pivot'] + (data['High'] - data['Low'])
    data['S2'] = data['Pivot'] - (data['High'] - data['Low'])
    data['R3'] = data['High'] + 2 * (data['Pivot'] - data['Low'])
    data['S3'] = data['Low'] - 2 * (data['High'] - data['Pivot'])
    return data[['Pivot', 'R1', 'S1', 'R2', 'S2', 'R3', 'S3']]