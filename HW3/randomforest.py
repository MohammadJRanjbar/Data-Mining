# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn import metrics
import numpy as np

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
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

# part 1
feature_names1 =["Pclass","Sex","Age","Fare","SibSp" ,"Parch","Embarked"]
x = data[feature_names1].values
y = data["Survived"].values

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=5)

random_forest1 = RandomForestClassifier(n_estimators=100,criterion="gini",max_depth=7)
random_forest1.fit(X_train, Y_train)

Y_prediction = random_forest1.predict(X_test)

random_forest1.score(X_train, Y_train)

print("Accuracy for 7 features with gini and max_depth=7 on train:", random_forest1.score(X_train, Y_train))
print("Accuracy for 7 features with gini and max_depth=7 on test:",  metrics.accuracy_score(Y_test, Y_prediction))

#Part2
random_forest2 = RandomForestClassifier(n_estimators=100,criterion="gini",max_depth=2)
random_forest2.fit(X_train, Y_train)

Y_prediction = random_forest2.predict(X_test)

print("Accuracy for 7 features with gini and max_depth=2 on train:", random_forest2.score(X_train, Y_train))
print("Accuracy for 7 features with gini and max_depth=2 on test:",  metrics.accuracy_score(Y_test, Y_prediction))

#part 3
random_forest3 = RandomForestClassifier(n_estimators=100,criterion="entropy",max_depth=7)
random_forest3.fit(X_train, Y_train)

Y_prediction = random_forest3.predict(X_test)

print("Accuracy for 7 features with entropy and max_depth=7 on train:", random_forest3.score(X_train, Y_train))
print("Accuracy for 7 features with entropy and max_depth=7 on test:",  metrics.accuracy_score(Y_test, Y_prediction))

#part 4
random_forest3 = RandomForestClassifier(n_estimators=100,criterion="entropy",max_depth=2)
random_forest3.fit(X_train, Y_train)

Y_prediction = random_forest3.predict(X_test)

print("Accuracy for 7 features with entropy and max_depth=2 on train:", random_forest3.score(X_train, Y_train))
print("Accuracy for 7 features with entropy and max_depth=2 on test:",  metrics.accuracy_score(Y_test, Y_prediction))

import time
#Calculating Time 
start = time.time()

random_forest1 = RandomForestClassifier(n_estimators=100,criterion="gini",max_depth=7)
random_forest1.fit(X_train, Y_train)

Y_prediction = random_forest1.predict(X_test)

random_forest1.score(X_train, Y_train)
end = time.time() 

print(f"Runtime for random forest is {end - start}")

start = time.time()
dt11 = DecisionTreeClassifier(random_state=5,max_depth=7,criterion="entropy")
dt11=dt11.fit(X_train,Y_train)
y_pred = dt11.predict(X_test)

end = time.time()
print(f"Runtime for decision tree is {end - start}")