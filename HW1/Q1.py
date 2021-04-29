# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 20:30:02 2020

@author: Mohammad
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import subprocess

df = pd.read_csv('covid.csv')
#printing the dataset
print(df)
df.to_html('DatasetTable.html')
subprocess.call(
    'wkhtmltoimage -f png --width 0 DatasetTable.html DatasetTable.png', shell=True)
#printing the dataset length and the names' of columns
print("Number of columns=" ,len(df.columns))
print("Number of data=" , len(df))
print("names of columns=" ,df.columns.tolist())
#printing birth_years' mean max and std
birth_years = df["birth_year"]
infected_by = df["infected_by"]
print("The max year of birth=",birth_years.max())
print("The mean year of birth=",birth_years.mean())
print("The std=", birth_years.std())
#filling the null data with relevant values
df["birth_year"].fillna(round(birth_years.median()), inplace=True)
df["region"].fillna("Unknown", inplace=True)
df["infection_reason"].fillna("Unknown", inplace=True)
df["infected_by"].fillna(infected_by.min(), inplace=True)
"""
data=df.to_numpy()
plt.subplot(4, 2, 1)
plt.scatter(2020-data[:,2], data[:,-1])
plt.subplot(4, 2, 2)
plt.hist(2020-data[:,2])
plt.subplot(4, 2, 3)
plt.scatter(data[:,1], data[:,-1])
plt.subplot(4, 2, 4)
plt.hist(data[:,1])
fig, axs = plt.subplots(2)
axs[0].scatter(data[:,-1], data[:,5])
axs[1].hist(data[:,4])
"""
#making a matrix plot 
g=sns.pairplot(df,diag_kind="hist",vars=["state", "birth_year","country",'sex',"infection_reason","region"])
plt.show()

