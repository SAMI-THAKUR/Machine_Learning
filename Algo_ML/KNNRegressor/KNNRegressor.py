import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

url="https://drive.google.com/file/d/1VD06DjyGegNAWdJxFqKW-BtNsSbZsbez/view?usp=drive_link"
url='https://drive.google.com/uc?id=' + url.split('/')[-2]
ad = pd.read_csv(url)

X = ad.drop('sales' , axis=1)
y = ad['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101) 

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
    
knn = KNNRegressor(k=20)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))


