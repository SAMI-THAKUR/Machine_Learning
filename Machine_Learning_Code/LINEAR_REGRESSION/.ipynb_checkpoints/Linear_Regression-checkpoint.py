import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

print("Linear Regression from Scratch")

url="https://drive.google.com/file/d/1VD06DjyGegNAWdJxFqKW-BtNsSbZsbez/view?usp=drive_link"
url='https://drive.google.com/uc?id=' + url.split('/')[-2]
ad = pd.read_csv(url)
print(ad)
ad['total_spent'] = ad.loc[:, 'TV':'newspaper'].sum(axis=1)
sns.regplot(data=ad , x='total_spent' , y="sales") # line of bestFit(Linear Regression line)