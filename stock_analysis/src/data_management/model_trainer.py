import sys
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from dataclasses import dataclass
from tensorflow.keras.layers import Dropout, LSTM, Dense
from src.logger import logging
from src.utils import evaluate_model
from src.utils import plot_prediction
from src.exception_handler import CustomException
from sklearn.preprocessing import MinMaxScaler


# split times series dataset into X,y ( will be used to create (X_train, y_train) , (X_test, y_test ) )
def scale_data( train_df , test_df , scaler):
    train_df['adjclose'] =  scaler.fit_transform(train_df['adjclose'].values.reshape(-1,1))
    test_df['adjclose'] = scaler.fit_transform(test_df['adjclose'].values.reshape(-1,1))

def split_Xy( df, window_size = 60):
    X = []
    y = []

    prices = df['adjclose'].values

    for i in range(len(prices) - window_size):
        X.append(prices[i:i + window_size])
        y.append(prices[i + window_size])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    return (X, y)
@dataclass
class ModelTrainerConfig:
    traindata_file_path  = os.path.join("./data","train")
    testdata_file_path = os.path.join("./data", "test")
    model_save_path = os.path.join("./models")
class ModelTrainer:

    def __init__(self , ticker:str ):
        self.model_trainer_config = ModelTrainerConfig()
        self.ticker = ticker
        self.train_df = pd.read_csv(os.path.join(self.model_trainer_config.traindata_file_path, self.ticker + ".csv" ))
        self.test_df  = pd.read_csv(os.path.join(self.model_trainer_config.testdata_file_path, self.ticker + ".csv" ))
        self.scaler = MinMaxScaler()
        scale_data(train_df = self.train_df, test_df = self.test_df, scaler = self.scaler)


        # by defult analyze past 60days to predict today's stock price

    def build_model(self , layer_units= 100):
        model = Sequential([
            LSTM(layer_units , return_sequences = True),
            Dropout(0.2),
            LSTM(layer_units, return_sequences = True),
            Dropout(0.1),
            LSTM(layer_units),
            Dense(1)
        ])

        return model

    def save_model(self, model , model_name):

        save_path = os.path.join(self.model_trainer_config.model_save_path, f"{model_name}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path+self.ticker+".keras")
        logging.info(f"Model saved to {save_path}")

    def train_model(self, epochs, learning_rate = 1e-3):

        X_train , y_train = split_Xy( self.train_df,  window_size = 60)
        X_test , y_test = split_Xy( self.test_df,  window_size= 60 )

        callback = tf.keras.callbacks.EarlyStopping("loss", patience =3 , min_delta = 0.001)

        model = self.build_model()
        model.compile(loss='mse', optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate) , metrics=[tf.keras.metrics.R2Score()])
        model.fit(X_train , y_train , epochs = epochs, callbacks = [callback] , verbose=2)
        self.save_model(model, self.ticker)
        # return scaler for plotting data
        return model


model_trainer_config = ModelTrainerConfig()
model_train  = ModelTrainer("AAPL")
model = model_train.train_model(epochs = 20)