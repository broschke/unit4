# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:24:31 2017

@author: bernardo.roschke
"""

import pandas as pd
# from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestClassifier
# import sklearn.metrics as skm
# import pylab as pl

# le = LabelEncoder()

samtest = pd.read_csv('C:\\Users\\Bernardo.Roschke\\Dropbox\\Thinkful\\data\\unit 4\\samtest.csv')
samtrain = pd.read_csv('C:\\Users\\Bernardo.Roschke\\Dropbox\\Thinkful\\data\\unit 4\\samtrain.csv')
samval = pd.read_csv('C:\\Users\\Bernardo.Roschke\\Dropbox\\Thinkful\\data\\unit 4\\samval.csv')

map_dict = {'laying':1, 'sitting':2, 'standing':3, 'walk':4, 'walkup':5, 'walkdown':6} 

def remap_col(df,colname, mapping=None):
  if not mapping:
    global map_dict
    mapping = map_dict.copy()
    
  df[colname] = df[colname].map(lambda x: mapping[x]) 
  return df

samtest = remap_col(samtest,'activity')
samtrain = remap_col(samtrain,'activity')
samval = remap_col(samval,'activity')
    
#print(samtrain[samtrain.columns[1:-2]])
#print(samtrain.head())

rfc = RandomForestClassifier(n_estimators=500, oob_score=True)
train_data = samtrain[samtrain.columns[1:-2]]
train_truth = samtrain['activity']
model = rfc.fit(train_data, train_truth)

