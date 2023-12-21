import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# using uci iris.data
s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(s, header=None, encoding="utf-8")

class Adaline:

    def __init__(self, learning_rate=0.01, n_iter= 50):
        self.learning_rate  = learning_rate
        self.n_iter = n_iter


    def fit(self,X , y):
        '''
        initialize and optimize the model
        :param X: size: number of samples * features
        :param y: size: number of samples*1
        :return: updated instance

        '''

        #initialize
        y = y.reshape(y.shape[0],1)
        sample_size = y.shape[0]
        self.w = np.random.RandomState(1).normal(loc = 0 , scale = 0.01, size = (X.shape[1],1))
        self.b = np.zeros((y.shape[0],1), dtype = "float32")
        self.losses = []

        for i in range(self.n_iter):
            Z  = self.net_input(X)
            loss = np.sum((y - Z)**2)/sample_size
            dZ = 2.0*(y-Z)/sample_size
            # print("X shape: ",X.shape )
            # print("dZ shape: ",dZ.shape )
            # print("y shape: ",y.shape )
            # print("W shape:", self.w.shape)
            dw = np.dot(X.T,dZ)

            # print(dw.shape)
            self.w += dw*self.learning_rate
            self.b += dZ*self.learning_rate
            self.losses.append(loss)

    def net_input(self, X):

        return np.dot(X,self.w)+self.b

    def predict(self,X):
        return np.where(self.net_input(X)>0.5, 0 , 1)



y = df.iloc[0:100 ,4 ].values
y = np.where(y == "Iris-setosa", 0,1)
X = df.iloc[0:100 , [0,2]].values


X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[: , 0].mean())/X[:,0].std()
X_std[:,1] = (X[:,1] - X[: , 1].mean())/X[:,1].std()

fig, ax = plt.subplots(nrows =1 , ncols =2 , figsize = (10,4))
ada1 = Adaline(n_iter=15, learning_rate=0.1)
ada1.fit(X_std,y)
ada2 = Adaline(n_iter=15 , learning_rate=0.0001)
