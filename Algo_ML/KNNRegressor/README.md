## KNNRegressor Class Methods

## Initialization (`__init__`):

```python
def __init__(self, k=3):
    self.k = k
```

The `__init__` method initializes the `KNNRegressor` object with an optional parameter `k`, which represents the number of neighbors to consider during regression. If `k` is not provided during object creation, it defaults to 3.

## Fit Method (`fit`):

```python
def fit(self, X, y):
    self.X_train = X
    self.y_train = y
```

The `fit` method is used to train the regressor. It takes two arguments: `X`, representing the training data (features), and `y`, representing the corresponding target values. Inside this method, the training data (`X`) and target values (`y`) are stored as attributes of the class instance (`self.X_train` and `self.y_train`, respectively).

## Euclidean Distance Method (`euclidean_distance`):

```python
def euclidean_distance(self, x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))
```

The `euclidean_distance` method calculates the Euclidean distance between two data points `x1` and `x2`. This distance metric is commonly used in KNN to measure the distance between a test sample and all training samples. It's computed as the square root of the sum of squared differences between corresponding elements of `x1` and `x2`.

## Predict Method (`predict`):

```python
def predict(self, X):
    X_array = X.values
    predictions = []
    for sample in X_array:
        distances = [self.euclidean_distance(sample, x) for x in self.X_train.values]
        nearest_neighbors = np.argsort(distances)[:self.k]
        knn_labels = [self.y_train.iloc[i] for i in nearest_neighbors]
        prediction = np.mean(knn_labels)
        predictions.append(prediction)
    return predictions
```

The `predict` method predicts the target values for a given set of test samples `X`. It iterates over each test sample and performs the following steps:

- Converts `X` to a NumPy array (`X_array`) for efficient computation.
- Calculates the Euclidean distances between the test sample and all training samples using list comprehension.
- Sorts the distances and selects the indices of the `k` nearest neighbors using `np.argsort`.
- Retrieves the target values of these nearest neighbors from the training target values (`self.y_train`).
- Calculates the average of the target values of the `k` nearest neighbors and assigns it as the predicted target value for the test sample.
- Appends the predicted target value to the `predictions` list.
  Finally, the method returns the list of predicted target values for all test samples.
