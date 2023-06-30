import numpy as np

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
