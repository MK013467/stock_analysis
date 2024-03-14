import os
import pickle
import sys
import pandas as pd
from src.exception_handler import CustomException
from datetime import datetime, date
import yahoo_fin.stock_info as si
from src.exception_handler import CustomException
import warnings
# warnings.filterwarnings("ignore")

#Get company name and Return predicted Value
class PredictPipeline:
    def __init__(self, ticker):
        self.ticker = ticker
        self.modeldir_path = "./models/lgb"
        self.model_path = os.path.join(self.modeldir_path, f"{ticker}_best_model.pkl")

    def load_model(self):
        try:
            model = pickle.load(open(self.model_path, "rb"))
            return model
        except CustomException as e:
            print(e)


    def fetch_stock_data(self, window_size = 100):

        df = si.get_data(self.ticker , start_date="2020-01-01", end_date=date.today())[["adjclose"]]

        for i in range(1, window_size + 1):
            df[f"lag {i}"] = df["adjclose"].shift(i)

        df.dropna(inplace=True)
        df.index = pd.to_datetime(df.index)

        return df.iloc[:,1:]

    def predict(self):
        model = self.load_model()
        df = self.fetch_stock_data(100)
        X = df.values
        predictions = model.predict(X)
        return predictions


