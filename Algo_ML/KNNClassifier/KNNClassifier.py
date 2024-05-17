import numpy as np

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def predict(self, X_test):
        predictions = []
        for sample in X_test:
            # Calculate distances between the test sample and all training samples
            distances = [self.euclidean_distance(sample, x_train) for x_train in self.X_train]
            # Sort by distance and return indices of the first k neighbors
            k_indices = np.argsort(distances)[:self.k]
            knn_labels = [self.y_train.iloc[i] for i in k_indices]
            most_common = max(set(knn_labels), key=knn_labels.count)
            predictions.append(most_common)
        return predictions
    


