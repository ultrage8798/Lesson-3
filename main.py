#logistic regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("titanic2.csv")
print(data.info())

data["Age"].fillna(data["Age"].median(skipna = True), inplace = True)
data["Embarked"].fillna(data["Embarked"].value_counts().idxmax(), inplace = True)

#finding null values
print(data.isnull().sum())

#dropping things we don't need
data.drop("Cabin", axis =1, inplace = True)
data.drop("PassengerId", axis =1, inplace = True)
data.drop("Name", axis =1, inplace = True)
data.drop("Ticket", axis =1, inplace = True)
data.drop("SibSp", axis =1, inplace = True)
data.drop("Parch", axis =1, inplace = True)

#conver string values to numbers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data["Sex"] = le.fit_transform(data["Sex"])
data["Embarked"] = le.fit_transform(data["Embarked"])

print(data.head())

#setting inputs as x and out puts as y
x = data[["Pclass","Sex","Fare","Embarked"]]
y = data["Survived"]

from sklearn.model_selection import train_test_split

#training and testing
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 2)

from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()

LR.fit(x_train,y_train)

y_predict = LR.predict(x_test)

import seaborn as sns
from sklearn.metrics import confusion_matrix,accuracy_score

#making confusion matrix
matrix = confusion_matrix(y_test,y_predict)
sns.heatmap(matrix, annot = True, fmt = "d")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#getting accuracy
accuracy = accuracy_score(y_test,y_predict)
accuracy = accuracy*100
print(accuracy)