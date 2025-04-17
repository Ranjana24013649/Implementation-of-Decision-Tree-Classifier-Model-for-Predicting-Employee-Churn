# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Ranjana R
RegisterNumber:  212224040270
*/
```
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, plot_tree

data=pd.read_csv("Employee_EX6.csv")

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

plt.figure(figsize=(18,6))

plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)

plt.show()


## Output

![318889806-f8005b20-85c6-444d-86ef-05988de2914d](https://github.com/user-attachments/assets/f01a9ba0-0d83-4d1d-8808-4492282333db)

![318889955-6227bcd1-239e-4610-b769-27294162049a](https://github.com/user-attachments/assets/250bad11-058c-4e94-b450-e15db25e2c4d)



![318890102-e26eeddb-0b03-42c5-955b-860c919822b0](https://github.com/user-attachments/assets/edd0c893-8ff5-4f70-b354-9c48e10c3ac3)


![318890396-e67d3205-ec86-4b99-956b-00b9fb923498](https://github.com/user-attachments/assets/f4cc5ee0-2ea3-47e3-9cc7-f404e763dacb)

![318890396-e67d3205-ec86-4b99-956b-00b9fb923498](https://github.com/user-attachments/assets/5b58cda9-4838-4f0a-81e8-b216173ab1db)

![318890578-7c93ff98-b7b4-455b-aa42-d1c38c6391f2](https://github.com/user-attachments/assets/6b222960-9a5f-450c-926c-4eddf38a1a38)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
