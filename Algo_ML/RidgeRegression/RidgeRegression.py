import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.theta = None
        
    def fit(self, X, y):
        # Add intercept term to X
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        
        # Compute theta using ridge regression formula
        A = np.eye(X.shape[1])
        A[0, 0] = 0  # Don't regularize the intercept term
        self.theta = np.linalg.inv(X.T.dot(X) + self.alpha * A).dot(X.T).dot(y)
        
    def predict(self, X):
        # Add intercept term to X
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        
        return X.dot(self.theta)