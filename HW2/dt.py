import pandas as pd
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn import metrics
import graphviz
import numpy as np





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

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

dt1 = DecisionTreeClassifier(random_state=5,max_depth=7,min_samples_split=2,criterion="gini")
dt1=dt1.fit(X_train,y_train)
y_pred = dt1.predict(X_test)

print("Accuracy for 7features with gini:",metrics.accuracy_score(y_test, y_pred))

dt11 = DecisionTreeClassifier(random_state=5,max_depth=7,min_samples_split=2,criterion="entropy")
dt11=dt11.fit(X_train,y_train)
y_pred = dt11.predict(X_test)

print("Accuracy for 7features with entropy:",metrics.accuracy_score(y_test, y_pred))


fig = plt.figure(figsize=(100,80))
_ = tree.plot_tree(dt11, 
                   feature_names=feature_names1,  
                   class_names="survived",
                   filled=True)


fig.savefig("decistion_tree.png")












# part2



feature_names2 =["Pclass","Sex","Age","Fare" ,"Parch"]

x = data[feature_names2].values
y = data["Survived"].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

dt2 = DecisionTreeClassifier(random_state=5,max_depth=5,min_samples_split=2,criterion="gini")
dt2=dt2.fit(X_train,y_train)
y_pred = dt2.predict(X_test)

print("Accuracy for 5features with gini:",metrics.accuracy_score(y_test, y_pred))


dt22 = DecisionTreeClassifier(random_state=5,max_depth=5,min_samples_split=2,criterion="entropy")
dt22=dt22.fit(X_train,y_train)
y_pred = dt22.predict(X_test)

print("Accuracy for 5features with entropy:",metrics.accuracy_score(y_test, y_pred))




# part 2

feature_names3 =["Pclass","Sex","Age"]

x = data[feature_names3].values
y = data["Survived"].values


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

dt3 = DecisionTreeClassifier(random_state=3,max_depth=3,min_samples_split=2,criterion="gini")
dt3=dt3.fit(X_train,y_train)
y_pred = dt3.predict(X_test)

print("Accuracy for 3features with gini:",metrics.accuracy_score(y_test, y_pred))


dt33 = DecisionTreeClassifier(random_state=3,max_depth=3,min_samples_split=2,criterion="entropy")
dt33=dt33.fit(X_train,y_train)
y_pred = dt33.predict(X_test)

print("Accuracy for 3features with entropy:",metrics.accuracy_score(y_test, y_pred))










