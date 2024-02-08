from dataclasses import dataclass
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.logger import logging
import src.exception_handler as CustomException
from data_builder import DataBuilder


@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join("../data")
    train_data_path =  str = os.path.join("../data","train")
    test_data_path:str = os.path.join("../data", "test")

class DataIngestion:

    def __init__( self):

        self.ingestion_config = DataIngestionConfig()

    def split_data(self,  ticker, test_size  = 0.2 ):
        logging.info("Start data ingestion method")

        try:
            raw_filepath = os.path.join(self.ingestion_config.raw_data_path, ticker)
            train_filepath = os.path.join(self.ingestion_config.train_data_path , ticker)
            test_filepath = os.path.join(self.ingestion_config.test_data_path, ticker)

            df = pd.read_csv( raw_filepath )
            split_rows = int(df.shape[0] * (1-test_size))

            train_df = df[:split_rows]
            test_df = df[split_rows:]
            train_df, test_df = self.scale_data(train_df.values, test_df.values)

            train_df.to_csv(train_filepath, index = False, mode="w", header = "adjclose")
            test_df.to_csv(test_filepath, index = False , mode="w", header = "adjclose")

        except Exception as e:
            raise CustomException(e)

    # scale data using MinMaxScaler
    def scale_data(self, train_df, test_df):    # Assuming 'date' is the DataFrame index and 'adjclose' is the column to scale
        scaler = MinMaxScaler()

        # Scale only the 'adjclose' column
        train_scaled = scaler.fit_transform(train_df[['adjclose']].values)
        test_scaled = scaler.transform(test_df[['adjclose']].values)

        # Reconstruct the DataFrame
        train_df_scaled = pd.DataFrame(train_scaled, columns=['adjclose'], index=train_df.index)
        test_df_scaled = pd.DataFrame(test_scaled, columns=['adjclose'], index=test_df.index)

        return train_df_scaled, test_df_scaled

if __name__ == "__main__":
    start_date = "2011-01-01"
    end_date = "2020-12-31"
    tickers = ["MSFT", "NKE", "AAPL" , "PEP", "GE" , "SBUX"]
    data_ingestion = DataIngestion()

    for ticker in tickers:
        data_builder = DataBuilder(start_date, end_date, ticker)
        data_builder.create_data()
        data_builder.save_csv()
        data_ingestion.split_data(ticker)

        train_filepath = os.path.join(data_ingestion.ingestion_config.train_data_path, f"{ticker}.csv")
        test_filepath = os.path.join(data_ingestion.ingestion_config.test_data_path, f"{ticker}.csv")

        # Scale the train and test data
        data_ingestion.scale_data(train_filepath, test_filepath)



