# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.svm import SVC, LinearSVC

data = pd.read_csv("train.csv")

# preproccessing
data["Fare"]=data["Fare"].fillna(data["Fare"].dropna().median())
data["Age"]=data["Age"].fillna(data["Age"].dropna().mean())
data.loc[data["Sex"]=="male","Sex"]=0
data.loc[data["Sex"]=="female","Sex"]=1
data["Embarked"]=data["Embarked"].fillna("5")
data.loc[data["Embarked"]=="S","Embarked"]=0
data.loc[data["Embarked"]=="C","Embarked"]=1
data.loc[data["Embarked"]=="Q","Embarked"]=2

feature_names1 =["Pclass","Sex","Age","Fare","SibSp" ,"Parch","Embarked"]
x = data[feature_names1].values
y = data["Survived"].values

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=5)

#SVM linear
from sklearn import svm

svclassifier = svm.SVC(kernel='linear')
svclassifier.fit(X_train, Y_train)

Y_prediction = svclassifier.predict(X_test)

print("Accuracy linear kernel on train:", svclassifier.score(X_train, Y_train))
print("Accuracy linear kernel on test:",  metrics.accuracy_score(Y_test, Y_prediction))

#SVM rbf
svclassifier2 = svm.SVC(kernel='rbf' ,C=1E3)
svclassifier2.fit(X_train, Y_train)

Y_prediction = svclassifier2.predict(X_test)

print("Accuracy linear kernel on train:", svclassifier2.score(X_train, Y_train))
print("Accuracy linear kernel on test:",  metrics.accuracy_score(Y_test, Y_prediction))