import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('train.csv',header=None)#.head(10)
data.isnull().sum()
data=data.dropna()
data.isnull().sum()

x=data.iloc[:,3:11].values
y=data.iloc[:,[12]].values

'''
#Missing Value Treatment
from sklearn.preprocessing import Imputer
Imp = Imputer(missing_values="NaN", strategy="mean", axis=1) 

Imp = Imp.fit(x[:,[4]])
x[:,4:]= Imp.transform(x[:,[4]])

Imp = Imp.fit(x[:,[5]])
x[:,5:]= Imp.transform(x[:,[5]])

Imp = Imp.fit(x[:,[8]])
x[:,8:]= Imp.transform(x[:,[8]])

Imp = Imp.fit(x[:,[9]])
x[:,9:]= Imp.transform(x[:,[9]])

Imp = Imp.fit(x[:,[10]])
x[:,10:]= Imp.transform(x[:,[10]])
'''
data.describe()

from sklearn.preprocessing import LabelEncoder,OneHotEncoder 

le1=LabelEncoder()
x[:,0]=le1.fit_transform(x[:,[0]])

le2=LabelEncoder()
x[:,1]=le2.fit_transform(x[:,[1]])

le3=LabelEncoder()
x[:,2]=le3.fit_transform(x[:,[2]])

le4=LabelEncoder()
x[:,8]=le4.fit_transform(x[:,[8]])

OneHotEncoder = OneHotEncoder(categorical_features=[0])
x=OneHotEncoder.fit_transform(x).toarray()

from sklearn.preprocessing import LabelEncoder,OneHotEncoder 
le5=LabelEncoder()
y[:,0]=le5.fit_transform(y[:,[0]])
OneHotEncoder = OneHotEncoder(categorical_features=[0])
y=OneHotEncoder.fit_transform(y).toarray()

#dummy variable
x=x[:,1:]

y=y[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from  sklearn.preprocessing import StandardScaler
SC_X = StandardScaler()
X_train = SC_X.fit_transform(x_train)
X_test = SC_X.transform(x_test)

#fitting Logistic regression model
from sklearn.linear_model import LogisticRegression
Classifier = LogisticRegression(random_state=0)
Classifier.fit(x_train,y_train)

#fitting Logistic regression model
from sklearn.neighbors import KNeighborsClassifier
Classifier = KNeighborsClassifier(n_neighbors= 5,metric= 'minkowski',p=2)
Classifier.fit(X_train,y_train)

from sklearn.svm import SVC
Classifier = SVC(kernel='poly',degree=6,random_state=0)
Classifier.fit(X_train,y_train)

#fitting Naive Bayes
from sklearn.naive_bayes import GaussianNB
Classifier = GaussianNB()
Classifier.fit(X_train,y_train)

#Predicting result
y_pred = Classifier.predict(x_test)

#Checking Accuracy
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred, normalize=False))




