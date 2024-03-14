from dataclasses import dataclass
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.logger import logging
import src.exception_handler as CustomException
from data_builder import DataBuilder
import warnings

warnings.filterwarnings("ignore")

@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join("../../data")
    train_data_path =  str = os.path.join("../../data", "train")
    test_data_path:str = os.path.join("../../data", "test")

class DataIngestion:

    def __init__( self):

        self.ingestion_config = DataIngestionConfig()

    def split_data(self,  ticker, test_size  = 0.2 ):
        logging.info("Start data ingestion method")

        try:
            raw_filepath = os.path.join(self.ingestion_config.raw_data_path, ticker+".csv")
            train_filepath = os.path.join(self.ingestion_config.train_data_path , ticker+".csv")
            test_filepath = os.path.join(self.ingestion_config.test_data_path, ticker+".csv")

            df = pd.read_csv( raw_filepath )
            split_rows = int(df.shape[0] * (1-test_size))

            train_df = df[:split_rows]
            test_df = df[split_rows:]

            train_df.to_csv(train_filepath, index = False, mode="w", header = "adjclose")
            test_df.to_csv(test_filepath, index = False , mode="w", header = "adjclose")

        except Exception as e:
            raise CustomException(e)

    # scale data using MinMaxScaler
    def scale_data(self, train_df, test_df):
        scaler = MinMaxScaler(feature_range=(0,1))

        # Scale only the 'adjclose' column

        train_scaled = scaler.fit_transform(train_df.loc[:, 'adjclose'].values.reshape(-1,1))
        test_scaled = scaler.transform(test_df.loc[:,'adjclose'].values.reshape(-1,1))

        # Reconstruct the DataFrame
        train_df.loc[:, 'adjclose'] = train_scaled.ravel()  # Use ravel() to convert the 2D array to 1D
        test_df.loc[:, 'adjclose'] = test_scaled.ravel()
        return train_df, test_df

tickers = ["MSFT", "NKE", "AAPL" , "PEP", "GE" , "SBUX", "AMZN", "GOOG", "META","NFLX"]
window_size = 100

if __name__ == "__main__":
    start_date = "2010-01-01"
    end_date = "2020-12-31"
    data_ingestion = DataIngestion()

    for ticker in tickers:
        data_builder = DataBuilder(start_date, end_date, ticker)
        data_builder.create_data()
        data_builder.save_csv()
        data_ingestion.split_data(ticker)


