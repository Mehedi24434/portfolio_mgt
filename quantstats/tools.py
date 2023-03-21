import numpy as np
import pandas as pd
# from pandas.io.data import DataReader
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
import ccxt
binance = ccxt.binance()


def preprocess_df(coin,time,limit):
    coin_ticker = binance.fetch_ohlcv(coin, time, limit)
    df = pd.DataFrame(coin_ticker, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']) 
    df['Date'] = pd.to_datetime(df['Date'], unit = 'ms' ) 
    return df

def comp(returns):
    """Calculates total compounded returns"""
    return returns.add(1).prod() - 1

def compsum(returns):
    """Calculates rolling compounded returns"""
    return returns.add(1).cumprod() - 1

    

