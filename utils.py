import numpy as np


def sigmoid(t):
    return 1/(1 + np.exp(-t))


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
