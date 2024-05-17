import numpy as np

class KNNRegressor:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def predict(self, X):
        X_array = X.values
        predictions = []
        for sample in X_array:
            distances = [self.euclidean_distance(sample,x) for x in self.X_train.values]
            nearest_neighbors = np.argsort(distances)[:self.k]
            knn_labels = [self.y_train.iloc[i] for i in nearest_neighbors]
            prediction = np.mean(knn_labels)
            predictions.append(prediction)
        return predictions
    



