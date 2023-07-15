import numpy as np

def softmax(X):
    """ Implementing the softmax function from scratch is a little tricky. 
        When you divide exponents that can potentially be very large, you 
        run into the issue of numerical stability. To avoid this, we use 
        a normalization trick. Notice that if we multiply the top and bottom 
        of the fraction by a constant C and push it into the sum, we get the 
        following equivalent expression:"""
    

    e_x = np.exp(X - np.max(X, axis=1, keepdims=True)) 

    return e_x / np.sum(e_x, axis=1, keepdims=True)

def sigmoid(t):
    return 1/(1 + np.exp(-t))

class OneHotEncoder():
    def __init__(self) -> None:
        pass

    def transform(self, X):
        n_values = np.max(X) + 1
        return np.eye(n_values)[X]


class StandardScaler():
    def __init__(self) -> None:
        
        pass

    def fit(self, X):
        self.mean = np.mean(X)
        self.std = np.std(X,  ddof=0)
        return (self.mean, self.std)

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
