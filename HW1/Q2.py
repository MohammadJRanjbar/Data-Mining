import pandas
import os 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression as LA
from sklearn.model_selection import train_test_split

#reading the dataset
df = pandas.read_csv('student.csv', sep = ';')
#deleting columns without numeric data
df=df.drop(['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason',
            'guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic'], axis=1)
data=df.to_numpy()
#spliting features and desire outputs
X=data[:,0:-1]
Y=data[:,-1]
#spliting train and test data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
#making a Linear Regression model
model = LA()
#training our model
model.fit(X_train, Y_train)
#for determinig accuracy on test 
Y_Pred = model.predict(X_test)
error=model.score(X_train, Y_train)
accuracy=0
for i in range(Y_test.shape[0]):
    if(abs(Y_test[i]-Y_Pred[i])>error):
        accuracy=accuracy+1
r_sq = model.score(X_test, Y_test)
accuracy=1-accuracy/Y_test.shape[0]

print("accuracy=",round(accuracy*100,4))

