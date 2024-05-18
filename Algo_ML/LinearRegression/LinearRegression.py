import numpy as np

class LinearRegression_GradientDescent:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape 
        self.weights = np.ones(n_features) # Initialize weights to ones
        self.bias = 0

        # Gradient Descent #
        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            # Compute gradients #
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) # Derivative with respect to weights
            db = (1 / n_samples) * np.sum(y_pred - y) # Derivative with respect to bias
            # Update parameters #
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

class LinearRegression_NormalEquation:
    def __init__(self):
        self.intercept_ = None
        self.coef_ = None
        
    def fit(self, X, y):
        # Add intercept term to X
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        
        # Calculate theta using the normal equation
        theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        
    def predict(self, X):
        # Add intercept term to X
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        
        return X.dot(np.concatenate(([self.intercept_], self.coef_)))



