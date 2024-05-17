import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay

url="https://drive.google.com/file/d/1d-GRWB_MuFMHAL5ARiCiFGtMkMu5XhG6/view?usp=drive_link"
url='https://drive.google.com/uc?id=' + url.split('/')[-2]
df = pd.read_csv(url)

X = df.drop('species' , axis=1)
y = df['species']

from sklearn.preprocessing import LabelEncoder
# Assuming 'y' is a DataFrame column containing the labels
# Initialize LabelEncoder
label_encoder = LabelEncoder()
# Fit and transform the labels
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


class SoftmaxRegression:
    def __init__(self, lr=0.01, num_iterations=1000):
        self.lr = lr
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.num_classes = None

    def softmax(self, z):
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.num_classes = len(np.unique(y))
        self.weights = np.zeros((num_features, self.num_classes))
        self.bias = np.zeros(self.num_classes)

        # One-hot encode labels
        y_one_hot = np.eye(self.num_classes)[y]
        
        # Gradient Descent
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.softmax(linear_model)
            #print(linear_model)
            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T , (y_predicted - y_one_hot))
            db = (1 / num_samples) * np.sum(y_predicted - y_one_hot, axis=0)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.softmax(linear_model)
        return np.argmax(y_predicted, axis=1) # Indices of Max element 

soft = SoftmaxRegression(num_iterations=5000);
soft.fit(X_train, y_train)
y_pred = soft.predict(X_test)
w = soft.weights
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)