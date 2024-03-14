import sys
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from dataclasses import dataclass
from tensorflow.keras.layers import Dropout, LSTM, Dense
from src.logger import logging
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_tuner import HyperModel, RandomSearch
data_path = "../../data/train/AAPL.csv"
window_size = len(pd.read_csv(data_path).columns)
class LSTMHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()
        base_units = hp.Choice("base_units", values=[32, 64])
        model.add(LSTM(units=base_units, return_sequences=True, input_shape=self.input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=base_units * 2, return_sequences=False))  # Corrected line
        model.add(Dropout(0.2))  # Assuming you want another Dropout layer here
        model.add(Dense(128))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy() , metrics=["accuracy"])
        return model

# split times series dataset into X,y ( will be used to create (X_train, y_train) , (X_test, y_test ) )
@dataclass
class ModelTrainerConfig:
    traindata_file_path  = os.path.join("../../data","train")
    testdata_file_path = os.path.join("../../data", "test")
    model_save_path = os.path.join("../../models")

class StockPredictionModel:
    def __init__(self, ticker):
        self.config = ModelTrainerConfig()
        self.ticker = ticker
    # load train, test dataset
    def load_data(self):
        train_filepath = os.path.join(self.config.traindata_file_path, self.ticker + ".csv")
        test_filepath = os.path.join(self.config.testdata_file_path, self.ticker + ".csv")
        train_df = pd.read_csv(train_filepath)
        test_df = pd.read_csv(test_filepath)

        return train_df, test_df
    # Split X,y
    def split_Xy(self, df):
        X, y = df.iloc[:, 2:-2 ], df.iloc[:, -2]
        return X, y

    def tune_and_train_model(self, X_train, y_train, X_test , y_test):
        input_shape = (X_train.shape[1], 1)  # Assuming all features are used
        hypermodel = LSTMHyperModel(input_shape=input_shape)

        tuner = RandomSearch(
            hypermodel,
            objective='accuracy',
            max_trials=10,
            executions_per_trial=1,
            directory='tuner_dir',
            project_name=self.ticker
        )

        tuner_directory = 'tuner_dir'
        #
        if not os.path.exists(tuner_directory):
            os.makedirs(tuner_directory)
        checkpoint_dir = 'tuner_dir/'
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create a ModelCheckpoint callback
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'checkpoint_{epoch}'),
            save_weights_only=True,
            save_best_only=True,
            monitor='accuracy',
            verbose=1,
        )

        tuner.search(X_train, y_train, epochs=10,  validation_data = (X_test, y_test), callbacks=[checkpoint_callback])
        self.model = tuner.get_best_models(num_models=1)[0]

    #save the best model
    def save_best_model(self):
        model_path = os.path.join(self.config.model_save_path, f'{self.ticker}_best_model.h5')
        self.model.save(model_path, save_format='tf')

    def run(self):
        train_df, test_df = self.load_data()
        X_train, y_train = self.split_Xy(train_df)
        X_test, y_test = self.split_Xy(test_df)
        X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
        self.tune_and_train_model(X_train, y_train, X_test, y_test)
        self.save_best_model()

# model = StockPredictionModel("AAPL")
# df = pd.read_csv(model.config.traindata_file_path+"/AAPL.csv")
# print(df.head())
# print(os.getcwd())
# model_trainer = StockPredictionModel("AAPL")
# print(model_trainer.config.model_save_path)
# X_train, y_train = model_trainer.split_Xy(df)
# print(X_train.columns)
if __name__ == '__main__':
    tickers = ["MSFT", "NKE" , "PEP", "GE" , "SBUX", "AMZN", "GOOG", "META","NFLX"]
    for ticker in tickers:
        model_trainer = StockPredictionModel(ticker)
        model_trainer.run()
        print(f"Model training and tuning completed for {ticker}")