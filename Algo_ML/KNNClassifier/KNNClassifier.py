import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report

url = 'https://drive.google.com/file/d/1oXrF8KHsElHXI_u-BGuijMe4F83U7dvj/view?usp=drive_link'
url = 'https://drive.google.com/uc?id='+url.split('/')[-2]
df = pd.read_csv(url)

X = df.drop('Cancer Present',axis=1)
y = df['Cancer Present']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

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
    
knn = KNNClassifier(k=20)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

