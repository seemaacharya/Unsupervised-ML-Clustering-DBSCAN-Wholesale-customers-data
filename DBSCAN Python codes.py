# -*- coding: utf-8 -*-
"""
Created on Wed May 19 15:10:57 2021

@author: DELL
"""

#Import the libraries
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import .csv file and convert it to a DataFrame object
df = pd.read_csv("Wholesale customers data.csv");

print(df.head())
print(df.info())
df.drop(['Channel','Region'],axis=1,inplace=True)
array=df.values
array
stscaler = StandardScaler().fit(array)
X = stscaler.transform(array)
X
dbscan = DBSCAN(eps=0.8, min_samples=6)
dbscan.fit(X)
#Noisy samples are given the label -1.
dbscan.labels_
cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])
cl
pd.concat([df,cl],axis=1)
