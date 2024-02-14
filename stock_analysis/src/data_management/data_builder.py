import pandas as pd
import yahoo_fin.stock_info as si
import os

#store stocks' data into csv file

class DataBuilder:

    def __init__( self, start_date , end_date , ticker ):

        self.start_date = start_date
        self.end_date = end_date
        self.ticker = ticker

    def create_data( self ):
        self.df = si.get_data( self.ticker, start_date = self.start_date , end_date = self.end_date)["adjclose"]
        self.df.index = pd.to_datetime(self.df.index)

    def save_csv(self ):
        path = os.path.join( "../data", f"{self.ticker}.csv")
        self.df.to_csv(path , mode = "w", index_label = "date")



