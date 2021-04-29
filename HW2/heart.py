from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn import metrics
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame




data = pd.read_csv("heart.csv")

# sns.set(style="ticks", color_codes=True)
# plot=sns.pairplot(data)
# plot.savefig("heart.png")

# pd.crosstab(data.sex,data.target).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])
# plt.title('Heart Disease Frequency for Sex')
# plt.xlabel('Sex (0 = Female, 1 = Male)')
# plt.xticks(rotation=0)
# plt.legend(["Haven't Disease", "Have Disease"])
# plt.ylabel('Frequency')
# plt.savefig("heart1.png")

# pd.crosstab(data.age,data.target).plot(kind="bar",figsize=(20,6))
# plt.title('Heart Disease Frequency for Ages')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.savefig('heartDiseaseAndAges.png')



feature_names =["age","sex","cp","trestbps","chol" ,"fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
x = data[feature_names].values
y = data["target"].values


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5,shuffle=True)

feature_scaler = StandardScaler()
X_train = feature_scaler.fit_transform(X_train)
X_test = feature_scaler.transform(X_test)

# Krange = range(1,30)
# scores = {}
# scores_list = []
# for k in Krange:
#     knn = KNeighborsClassifier(n_neighbors = k)
#     knn.fit(X_train,y_train)
#     y_pred = knn.predict(X_test)
#     scores[k] = metrics.accuracy_score(y_test,y_pred)
#     scores_list.append(metrics.accuracy_score(y_test,y_pred))
    
# plt.plot(Krange,scores_list)
# plt.xlabel("Value of K")
# plt.ylabel("Accuracy")
# plt.savefig("k.png")
# plt.show()


model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train,y_train)
y_pred= model.predict(X_test) 

print("Accuracy KNN:",metrics.accuracy_score(y_test, y_pred))


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5,shuffle=True)
#Create a Gaussian Classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print("Accuracy NB:",metrics.accuracy_score(y_test, y_pred))


