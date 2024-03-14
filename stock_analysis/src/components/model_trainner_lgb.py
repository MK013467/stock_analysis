import os
import pickle
from dataclasses import dataclass
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

@dataclass
class ModelMGBTrainerConfig:
    traindata_file_path = os.path.join("..","..","data","train")
    testdata_file_path = os.path.join("..","..","data","test")
    modeldir_path = os.path.join("..","..","models","lgb"
                                             )

class LGBMSTockPredictionModel:

    def __init__(self, ticker):
        self.config  = ModelMGBTrainerConfig()
        self.ticker = ticker
        self.model_save_path = os.path.join(self.config.modeldir_path, f"{self.ticker}_best_model.pkl")

    def load_data(self):
        train_filepath = os.path.join(self.config.traindata_file_path, self.ticker)+".csv"
        test_filepath = os.path.join(self.config.testdata_file_path, self.ticker)+".csv"
        train_df = pd.read_csv(train_filepath)
        test_df = pd.read_csv(test_filepath)

        return train_df, test_df

    def split_Xy(self, df):
        X, y  = df.iloc[:, 2:-2], df.iloc[:, -2]
        return X,y

    def tune_model(self, X_train, y_train ):
        parameters = {
            "boosting_type": ["gbdt", "dart"],
            "num_leaves": [31,50, 70],
            "learning_rate": [1e-3, 1e-2 , 1e-1],
            "n_estimators": [100, 200, 500],
            "min_child_sample":[20, 30 , 50]
        }

        clasifiers = lgb.LGBMClassifier()
        random_search = RandomizedSearchCV ( estimator = clasifiers, param_distributions= parameters , n_iter = 10 , scoring = "accuracy")

        random_search.fit(X_train ,y_train)

        return random_search.best_estimator_

    def save_model(self , model):
        with open(self.model_save_path , "wb") as file:
            pickle.dump(model, file)

    def run(self):

        train_df, test_df = self.load_data()
        X_train, y_train = self.split_Xy(train_df)
        best_estimator = self.tune_model(X_train, y_train)
        self.save_model(best_estimator)


if __name__ == '__main__':
    tickers = ["MSFT", "NKE", "PEP", "GE", "SBUX", "AMZN", "GOOG", "META", "NFLX"]
    for ticker in tickers:
        print(f"Training model for {ticker}")
        model_trainer = LGBMSTockPredictionModel(ticker)
        model_trainer.run()