# self implementation of logistic_regression

class LogisticRegression:

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
        y = y.reshape(-1,1)
        sample_size = X.shape[0]
        self.w = np.random.RandomState(1).normal(loc = 0 , scale = 0.01, size = (X.shape[1],1))
        self.b = np.zeros((sample_size,1), dtype = "float32")
        self.losses = []

        #L(w,b) =
        for i in range(self.n_iter):
            Z  = self.net_input(X)
            output = self.sigmoid(Z)
            loss = np.sum(-y*np.log(output) - (1-y)*np.log(1-output))/sample_size
            dZ = 2.0*(y-output)/sample_size
            dw = np.dot(X.T,dZ)

            # print(dw.shape)
            self.w += dw*self.learning_rate
            self.b += dZ*self.learning_rate
            self.losses.append(loss)

    def net_input(self, X):

        return np.dot(X,self.w)+self.b.mean()

    def sigmoid(self,Z):

        return 1/(1+np.exp(-Z))

    def predict(self,X):
        input = self.net_input(X)
        output = self.sigmoid(input)
        return np.where( output>0.5, 0 , 1)
