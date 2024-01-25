import streamlit as st
import backtrader as bt
import yfinance as yf
import pandas as pd
from streamlit_ace import st_ace

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
        ('long_period', 50)
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

def backtest():
    def get_forex_data(base_currency, quote_currency, start_date, end_date):
        ticker_symbol = f'{base_currency}{quote_currency}=X'
        forex_data = yf.download(ticker_symbol, start=start_date, end=end_date)

        if forex_data.empty:
            raise ValueError(f'No data available for {ticker_symbol} in the specified date range.')

        return forex_data

    def run_backtest(forex_data, short_period, long_period):
        cerebro = bt.Cerebro()
        data = bt.feeds.PandasData(
            dataname=forex_data, open='Open', high='High', low='Low', close='Close', volume='Volume', openinterest=None
        )
        cerebro.adddata(data)
        cerebro.addstrategy(MyStrategy, short_period=short_period, long_period=long_period)
        cerebro.broker.set_cash(100000)
        cerebro.run()

        return cerebro


    def main():
        st.title("Forex Trading Strategy Dashboard")

        # Sidebar for user input
        st.sidebar.header("Settings")
        base_currency = st.sidebar.text_input("Base Currency", "USD")
        quote_currency = st.sidebar.text_input("Quote Currency", "EUR")
        start_date = st.sidebar.text_input("Start Date", "2023-01-01")
        end_date = st.sidebar.text_input("End Date", "2024-01-01")
        short_period = st.sidebar.slider("Short Period", 5, 50, 20)
        long_period = st.sidebar.slider("Long Period", 10, 100, 50)

        # Get forex data
        forex_data = get_forex_data(base_currency, quote_currency, start_date, end_date)

        # Run backtest
        cerebro = run_backtest(forex_data, short_period, long_period)

        # Display results
        st.subheader("Backtest Results")
        st.write(f"Starting Portfolio Value: ${cerebro.broker.startingcash:.2f}")
        st.write(f"Ending Portfolio Value: ${cerebro.broker.getvalue():.2f}")

        # Plot the backtest results

        cerebro.plot(style="candlestick")
        st.info('Click to generate plot')



    if __name__ == "__main__":
        main()

if st.button('Backtest'):
    backtest()