# EX 9  Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:

To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: vidhyasri.k
RegisterNumber: 212222230170 
```
```
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:

Encoding:

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477817/d3717f36-ac19-4bfe-979a-362a933dc9eb)

Head():

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477817/0a694537-2052-41e4-8cd5-b5f3208ddaf9)

Info():

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477817/db21e1d4-7505-4f18-a042-5e8859340cfb)

isnull().sum():

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477817/ca8f8fe6-9ebf-445a-a881-2a0fb9a7a4c7)

Prediction of y:

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477817/7c965be3-d28d-4f71-b416-2cfbb0f7b4ca)

Accuracy:

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477817/9d8650ab-54e5-42f2-9029-ef8a7e96d603)

## Result:

Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
