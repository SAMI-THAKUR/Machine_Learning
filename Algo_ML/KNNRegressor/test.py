import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from KNNRegressor import KNNRegressor
import numpy as np

url="https://drive.google.com/file/d/1VD06DjyGegNAWdJxFqKW-BtNsSbZsbez/view?usp=drive_link"
url='https://drive.google.com/uc?id=' + url.split('/')[-2]
ad = pd.read_csv(url)

X = ad.drop('sales' , axis=1)
y = ad['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101) 

knn = KNNRegressor(k=20)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))