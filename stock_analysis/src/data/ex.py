import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
ex = pd.read_csv('MSFT.csv')
scaled = scaler.fit_transform(ex.loc[:,'adjclose'].values.reshape(-1,1))
print(ex.loc[:,'adjclose'].values.shape)
print(scaled.ravel().shape)
