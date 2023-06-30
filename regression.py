import numpy as np

class LinearRegression():
  def __init__(self) -> None:
    pass

  def __add_bias(self, X):
    return np.concatenate((X,  np.expand_dims(np.ones(X.T.shape[-1]), axis=1)),axis=1)
  
  def fit(self, X, y):
    X_b = self.__add_bias(X)
    self.theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    self.intercept_ = self.theta[0]
    self.coef_ = self.theta[1:]

    
  def predict(self, X):
    #print(self.__add_bias(X).shape, self.theta.shape)
    return self.__add_bias(X) @ self.theta
  

class BatchGradientDescentRegression():
    def __init__(self, n_iter, lr, seed = None) -> None:
        self.n_iter = n_iter
        self.lr = lr
        if seed is not None:
            np.random.seed = seed

    def __add_bias(self, X):
        return np.concatenate((X,  np.expand_dims(np.ones(X.T.shape[-1]), axis=1)),axis=1)
  
    def fit(self, X, y):
        X_b = self.__add_bias(X)
        self.theta = np.random.rand(X_b.shape[1], 1)
        m = len(X_b)
        for i in range(self.n_iter):
            mse = (2/m) * X_b.T @ (X_b @ self.theta - y)
            self.theta = self.theta - self.lr * mse

        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]

    def predict(self, X):
        return self.__add_bias(X) @ self.theta
    

class SGDRegression():
    def __init__(self, n_iter, lr0, lr1,eta0, seed = None, penalty="", alpha = None) -> None:
        self.n_iter = n_iter
        self.penalty = penalty
        self.lr0 = lr0
        self.lr1 = lr1
        self.eta0 = eta0
        assert not ((penalty == "") and (alpha != None)), "Alpha must be none, when penalty is empty"
        self.alpha = alpha

        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)

    def __add_bias(self, X):
        return np.c_[np.ones((X.T.shape[-1],1)), X]
  
    def learning_schedule(self, t):
        return self.lr0 / (t + self.lr1)
    
    def learning_schedule_optimal(self, t):
        #optimal_init = 1.0 / (self.eta0 * m)
        return 1 / (t + self.eta0)

    def fit(self, X, y):
        X_b = self.__add_bias(X)
        self.theta = self.rng.standard_normal((X_b.shape[1], 1))
        m = len(X_b)

        for epoch in range(self.n_iter):
            for iteration in range(m):
                sample_index =  self.rng.integers(m)
                Xi = X_b[sample_index: sample_index+1]
                yi = y[sample_index: sample_index+1]
                
                if self.penalty == "":
                    gradient = 2 * Xi.T @ (Xi @ self.theta - yi)
                elif self.penalty == "l1":
                    gradient = 2 * Xi.T @ (Xi @ self.theta - yi) + (self.alpha / m) * np.sum(self.theta)
                elif self.penalty == "l2":
                    gradient = 2 * Xi.T @ (Xi @ self.theta - yi) + (2*self.alpha) * np.sum(np.square(self.theta ))

                eta = self.learning_schedule_optimal(epoch * m + iteration + 1)

                self.theta = self.theta - eta * gradient


        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]

    def predict(self, X):
        return self.__add_bias(X) @ self.theta
    

class MiniBatchGradientDescentRegression():
    def __init__(self, n_iter, lr, batch_size, seed = None) -> None:
        self.n_iter = n_iter
        self.lr = lr
        self.batch_size = batch_size
        if seed is not None:
            np.random.seed = seed

    def __add_bias(self, X):
        return np.concatenate((X,  np.expand_dims(np.ones(X.T.shape[-1]), axis=1)),axis=1)
  
    def fit(self, X, y):
        X_b = self.__add_bias(X)
        
        m = len(X_b)
        assert m >= self.batch_size, "Wrong mini batch size"
        if self.batch_size < 1:
            mini_batch_size = m * self.batch_size
        else:
            mini_batch_size = self.batch_size
        
        self.theta = np.random.rand(X_b.shape[1], 1)
        
        for i in range(self.n_iter):
            indx_arr = np.random.randint(0, int(mini_batch_size), int(mini_batch_size))
            Xmb = X_b[indx_arr]
            mse = (2/mini_batch_size) * Xmb.T @ (Xmb @ self.theta - y[indx_arr])
            self.theta = self.theta - self.lr * mse


        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]

    def predict(self, X):
        return self.__add_bias(X) @ self.theta
    

class Ridge():
  def __init__(self, alpha) -> None:
    self.alpha = alpha

  def __add_bias(self, X):
    return np.c_[np.ones((X.T.shape[-1],1)), X]
  
  def fit(self, X, y):
    X_b = self.__add_bias(X)
    A = np.identity(X_b.shape[1])
    A[0, 0] = 0
    self.theta = np.linalg.inv(X_b.T @ X_b  + self.alpha * A) @ X_b.T @ y
    self.intercept_ = self.theta[0]
    self.coef_ = self.theta[1:]
    
  def predict(self, X):
    return self.__add_bias(X) @ self.theta