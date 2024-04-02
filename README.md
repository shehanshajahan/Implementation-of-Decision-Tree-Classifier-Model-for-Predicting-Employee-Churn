# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Shehan Shajahan
RegisterNumber: 212223240154
*/
import pandas as pd
data=pd.read_csv('/content/Employee.csv')
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
## 1) Head:
![Screenshot 2024-04-02 161750](https://github.com/shehanshajahan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/139317389/8a2945b4-8230-4e7e-a99d-fe4377891dd4)
## 2) Accuracy:
![Screenshot 2024-04-02 161808](https://github.com/shehanshajahan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/139317389/e06b3126-795b-4887-a82d-b2ee7f62d334)
## 3) Predict:
![Screenshot 2024-04-02 161837](https://github.com/shehanshajahan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/139317389/da35d440-129b-4f86-b3dc-9cdde51d0573)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
