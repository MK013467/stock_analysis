import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.exception_handler import CustomException

class DataPlotter:
    def __init__(self):
        print("Initializing DataPlotter")
    def plotDataframe(self, df:pd.DataFrame, title:str, xlabel:str = "date", ylabel:str = "price"):

        if not isinstance(df, pd.DataFrame):
            raise CustomException()

        fig, ax = plt.subplots()
        df.plot(ax=ax)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.legend(loc="best")
        plt.show()

    def plotDataframes(self, dfs:list, title:str, xlabel:str="date", ylabel:str ="price"):

        n = len(dfs)
        fig, axs = plt.subplots(n, 1, figsize=(10,6*n))
        if n == 1:
            axs = [axs]

        for i , df in enumerate(dfs):
            df.plot(ax = axs[i])
            axs[i].set_title(title)
            axs[i].set_xlabel(xlabel)
            axs[i].set_ylabe(ylabel)

        plt.tight_layout
        plt.show()

df = pd.read_csv("../data/train/MSFT.csv")
DataPlotter().plotDataframe(df, title="MSFT")
