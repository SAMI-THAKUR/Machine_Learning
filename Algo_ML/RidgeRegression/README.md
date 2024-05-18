## RidgeRegression
![image](https://github.com/SAMI-THAKUR/Machine_Learning/assets/118300788/7c86f3c5-340f-4e8f-af7f-84da75b5ffca)

### Initialization (`__init__`):

```python
def __init__(self, alpha=1):
    self.alpha = alpha
    self.theta = None
```

- The `__init__` method initializes the `RidgeRegression` object. It takes an optional parameter `alpha` (default is 1), which represents the regularization strength. It also initializes the `theta` attribute to `None`, which will later store the learned parameters.

### Fit Method (`fit`):

```python
def fit(self, X, y):
    # Add intercept term to X
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    # Compute theta using ridge regression formula
    A = np.eye(X.shape[1])
    A[0, 0] = 0  # Don't regularize the intercept term
    self.theta = np.linalg.inv(X.T.dot(X) + self.alpha * A).dot(X.T).dot(y)
```

- The `fit` method is used to train the ridge regression model. It takes two arguments: `X`, representing the feature matrix, and `y`, representing the target vector.
- Inside this method:
  - It adds a column of ones (representing the intercept term) to the feature matrix `X` by concatenating it along the columns.
  - It constructs the regularization matrix `A` using `np.eye` to create an identity matrix with the same number of columns as `X`. It sets the element at index `(0, 0)` to 0 to exclude regularization for the intercept term.
  - It computes the parameters (`theta`) using the ridge regression formula: \(\theta = (X^T X + \alpha A)^{-1} X^T y\), where `@` represents matrix multiplication, `T` denotes matrix transpose, and `inv` is the matrix inverse function from NumPy.

### Predict Method (`predict`):\*\*

```python
def predict(self, X):
    # Add intercept term to X
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    return X.dot(self.theta)
```

- The `predict` method is used to make predictions using the trained ridge regression model. It takes one argument: `X`, representing the feature matrix of new data points.
- Inside this method:
  - It adds a column of ones (representing the intercept term) to the feature matrix `X` by concatenating it along the columns.
  - It computes the predicted target values by performing matrix multiplication between the feature matrix `X` and the learned parameters `theta`.
- Finally, it returns the predicted target values.
