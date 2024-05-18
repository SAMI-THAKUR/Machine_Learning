import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from LinearRegression import LinearRegression_GradientDescent as Linear_Regression

url = "https://drive.google.com/file/d/1VD06DjyGegNAWdJxFqKW-BtNsSbZsbez/view?usp=drive_link"
url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
ad = pd.read_csv(url)

X = ad.drop('sales' , axis=1)
y = ad['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101) 

linear = Linear_Regression()
linear.fit(X_train, y_train)
predicted_y = linear.predict(X_test)
mse = mean_squared_error(predicted_y,y_test)
print(np.sqrt(mse))