import streamlit as st
import backtrader as bt
import yfinance as yf
from streamlit_ace import st_ace
import matplotlib
import pandas as pd
matplotlib.use('Agg')


st.set_page_config(
    page_title="Backtesting",
    page_icon="📈",
    layout="wide",  # Set the layout to wide
)


def get_forex_data(base_currency, quote_currency, start_date, end_date, interval):
    ticker_symbol = f'{base_currency}{quote_currency}=X'
    forex_data = yf.download(ticker_symbol, start=start_date, end=end_date, interval=interval)

    if forex_data.empty:
        raise ValueError(f'No data available for {ticker_symbol} in the specified date range.')

    return forex_data


# Sidebar for user input
st.sidebar.header("Settings")

base_currency = st.sidebar.text_input("Base Currency", "USD")
quote_currency = st.sidebar.text_input("Quote Currency", "EUR")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))


# Interval selection
interval_options = ['Daily', 'Weekly', 'Monthly', 'Hourly']
io_decoder = {'Daily': '1d', 'Weekly': '1wk', 'Monthly': '1mo', 'Hourly': '1h'}
selected_interval = st.sidebar.selectbox('Select Interval', interval_options, index=0)
interval = io_decoder[selected_interval]

# Get forex data
forex_data = get_forex_data(base_currency, quote_currency, start_date, end_date, interval)


# Get forex data
forex_data = get_forex_data(base_currency, quote_currency, start_date, end_date, interval)

st.title("Forex Trading Strategy Dashboard")

class MyStrategy(bt.Strategy):
    # Enter strategy code now
    pass

default_code = """

class MyStrategy(bt.Strategy):
    # Enter strategy code now
    pass
"""



code = st_ace(default_code,language='python')
if st.checkbox('Apply Code', True):
    exec(code)



with st.expander("Example Strategy (Moving Average Crossover)"):
    st.code('''class MyStrategy(bt.Strategy):
    params = (
        ('short_period', 20),
        ('long_period', 50),
        ('position_size', 0.1),
    )

    def __init__(self):
        self.short_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.short_period)
        self.long_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.long_period)

    def next(self):
        # Check if we are already in a position
        if self.position:

            # If short_ma is less than long_ma, sell to close the position
            if self.short_ma < self.long_ma:
                self.sell()

        else:

            # If short_ma is greater than long_ma, buy
            if self.short_ma > self.long_ma:
                self.buy()

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')



    ''')
with st.expander("Example Strategy (RSI)"):
    st.code(
        '''class MyStrategy(bt.Strategy):
    params = (
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30)
    )

    def __init__(self):
        self.rsi = bt.indicators.RelativeStrengthIndex(period=self.params.rsi_period)

    def next(self):
        if not self.position:
            # Buy signal: RSI crosses below the oversold level
            if self.rsi < self.params.rsi_oversold:
                self.buy()

        else:
            # Sell signal: RSI crosses above the overbought level
            if self.rsi > self.params.rsi_overbought:
                self.sell()

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')
'''

    )
with st.expander("Example Strategy (Mean Reversion Bollinger Bands)"):
    st.code('''class MyStrategy(bt.Strategy):
    params = (
        ('bollinger_period', 20),
        ('bollinger_dev', 2),
        ('position_size', 0.1),
    )

    def __init__(self):
        self.bollinger = bt.indicators.BollingerBands(self.data.close, period=self.params.bollinger_period, devfactor=self.params.bollinger_dev)

    def next(self):
        if not self.position:
            # Buy signal: Price crosses below the lower band of Bollinger Bands
            if self.data.close < self.bollinger.lines.bot:
                self.buy()

        else:
            # Sell signal: Price crosses above the upper band of Bollinger Bands
            if self.data.close > self.bollinger.lines.top:
                self.sell()

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')
''')



balMax = st.sidebar.number_input('Enter You Maximum Balance',value=1000)
balance = st.sidebar.slider("Select Desired Balance", min_value=1, max_value=balMax, value=int(balMax/2))


def backtest():

    def run_backtest(forex_data):
        cerebro = bt.Cerebro()
        data = bt.feeds.PandasData(
            dataname=forex_data, open='Open', high='High', low='Low', close='Close', volume='Volume', openinterest=None
        )
        cerebro.adddata(data)
        cerebro.addstrategy(MyStrategy)
        cerebro.broker.set_cash(balance)
        cerebro.run()

        return cerebro


    def main():




        # Run backtest
        cerebro = run_backtest(forex_data)

        # Display results
        st.subheader("Backtest Results")
        st.write(f"Starting Portfolio Value: ${cerebro.broker.startingcash:.2f}")
        st.write(f"Ending Portfolio Value: ${cerebro.broker.getvalue():.2f}")

        # Plot the backtest results

        figure = cerebro.plot()[0][0]

        # show the plot in Streamlit
        st.pyplot(figure)

    if __name__ == "__main__":
        main()

if st.button('Backtest'):
    backtest()
