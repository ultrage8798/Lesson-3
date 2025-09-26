#logistic regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("titanic2.csv")
print(data.info())

data["Age"].fillna(data["Age"].median(skipna = True), inplace = True)
data["Embarked"].fillna(data["Embarked"].value_counts().idxmax(), inplace = True)

print(data.isnull().sum())

data.drop("Cabin", axis =1, inplace = True)
data.drop("PassengerId", axis =1, inplace = True)
data.drop("Name", axis =1, inplace = True)
data.drop("Ticket", axis =1, inplace = True)
data.drop("SibSp", axis =1, inplace = True)
data.drop("Parch", axis =1, inplace = True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data["Sex"] = le.fit_transform(data["Sex"])
data["Embarked"] = le.fit_transform(data["Embarked"])

print(data.head())

x = data[["Pclass","Sex","Fare","Embarked"]]
y = data["Survived"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 2)