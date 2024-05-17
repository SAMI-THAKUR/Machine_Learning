import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

url = "https://drive.google.com/file/d/1VD06DjyGegNAWdJxFqKW-BtNsSbZsbez/view?usp=drive_link"
url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
ad = pd.read_csv(url)

X = ad.drop('sales' , axis=1)
y = ad['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101) 

class Linear_Regression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
    def fit(self,X,y):
        X = np.insert(X.values, 0, 1, axis=1)  # Inserting a column of ones for intercept
        beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
    def predict(self,X):
        y_pred = X.dot(self.coef_) + self.intercept_
        return y_pred

linear = Linear_Regression()
linear.fit(X_train, y_train)
predicted_y = linear.predict(X_test)
mse = mean_squared_error(predicted_y,y_test)
print(np.sqrt(mse))

