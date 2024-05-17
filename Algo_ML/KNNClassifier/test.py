import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay,classification_report
from KNNClassifier import KNNClassifier

url = 'https://drive.google.com/file/d/1oXrF8KHsElHXI_u-BGuijMe4F83U7dvj/view?usp=drive_link'
url = 'https://drive.google.com/uc?id='+url.split('/')[-2]
df = pd.read_csv(url)

X = df.drop('Cancer Present',axis=1)
y = df['Cancer Present']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNNClassifier(k=20)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)