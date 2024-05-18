## LinearRegression (using Gradient Descent)
![image](https://github.com/SAMI-THAKUR/Machine_Learning/assets/118300788/a9763030-09b4-4b5b-818e-a79c0bf68ec5)

### Initialization (`__init__`):

```python
def __init__(self, learning_rate=0.01, n_iterations=1000):
    self.learning_rate = learning_rate
    self.n_iterations = n_iterations
    self.weights = None
    self.bias = None
```

- The `__init__` method initializes the `LinearRegression` object with two optional parameters: `learning_rate`, representing the learning rate for gradient descent (default is 0.01), and `n_iterations`, representing the number of iterations for gradient descent (default is 1000). It also initializes the `weights` and `bias` attributes to `None`.

### Fit Method (`fit`):

```python
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
```

- The `fit` method is used to train the linear regression model. It takes two arguments: `X`, representing the feature matrix, and `y`, representing the target vector.
- Inside this method:
  - It retrieves the number of samples (`n_samples`) and features (`n_features`) from the shape of `X`.
  - It initializes the `weights` attribute to an array of ones with the same number of elements as features, and sets the `bias` attribute to 0.
  - It iterates through the specified number of iterations (`n_iterations`) to perform gradient descent:
    - It calculates the predicted target values (`y_pred`) using the current weights and bias.
    - It computes the gradients (`dw` and `db`) of the loss function with respect to the weights and bias, respectively.
    - It updates the weights and bias using gradient descent to minimize the loss function.

### Predict Method (`predict`):

```python
def predict(self, X):
    return np.dot(X, self.weights) + self.bias
```

- The `predict` method is used to make predictions using the trained linear regression model. It takes one argument: `X`, representing the feature matrix of new data points.
- It computes the predicted target values by performing matrix multiplication between the feature matrix `X` and the weights, and then adding the bias term.
- Finally, it returns the predicted target values.
