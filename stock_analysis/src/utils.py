import matplotlib.pyplot as plt
import numpy as np
import os

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set and print the results.
    """
    test_loss, test_metric = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss}")
    print(f"Test Metric (R2 Score): {test_metric}")
    return test_loss, test_metric


def plot_prediction(y_test, predictions, ticker="Stock", save_path=None):
    """
    Plot the actual stock prices vs. predicted stock prices.

    Parameters:
    - y_test: Actual stock prices.
    - predictions: Predicted stock prices by the model.
    - ticker: Ticker symbol of the stock (for plot title).
    - save_path: If provided, saves the plot to the specified path.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual Prices', color='blue')
    plt.plot(predictions, label='Predicted Prices', linestyle='--', color='red')
    plt.title(f'{ticker} Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

