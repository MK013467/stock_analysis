import pandas as pd
import yahoo_fin.stock_info as si
import os

#store stocks' data into csv file
import numpy as np
class DataBuilder:

    def __init__( self, start_date , end_date , ticker ):

        self.start_date = start_date
        self.end_date = end_date
        self.ticker = ticker

    def create_data( self, window_size = 100):
        self.df = si.get_data( self.ticker, start_date = self.start_date , end_date = self.end_date)[["adjclose"]]

        for i in range( 1,window_size+1):
            self.df[f"lag {i}"] = self.df["adjclose"].shift(i)

        self.df.dropna(inplace = True)
        self.df['up'] = (self.df.shift(-1)['adjclose'] > self.df["adjclose"]).astype(int)
        self.df["week_up"] = (self.df.shift(-7)['adjclose'] > self.df["adjclose"]).astype(int)

        self.df.index = pd.to_datetime(self.df.index)


    def save_csv(self ):
        path = os.path.join("../../data", f"{self.ticker}.csv")
        self.df.to_csv(path , mode = "w", index_label = "date")

tickers = ["MSFT", "NKE", "AAPL" , "PEP", "GE" , "SBUX", "AMZN", "GOOG", "META","NFLX"]
window_size = 100
for ticker in tickers:
    builder = DataBuilder("2008-01-01", "2023-12-31",ticker)
    builder.create_data(window_size=window_size)
    builder.save_csv()
