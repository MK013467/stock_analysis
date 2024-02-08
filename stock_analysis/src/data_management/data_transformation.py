from sklearn.preprocessing import MinMaxScaler


class DataTransformation:

    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit_transform(self, df):

        return self.scaler.fit_transform(df)




