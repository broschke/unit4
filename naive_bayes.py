# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 09:13:21 2017

@author: bernardo.roschke
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import numpy as np

df = pd.read_csv('C:\\Users\\Bernardo.Roschke\\Dropbox\\Thinkful\\data\\unit 4\\weights.csv')

#plt.hist(df['actual'], label='actual')
#plt.hist(df['ideal'], label='ideal')
#plt.legend(loc='upper right')
#plt.show()

df['sex_cat'] = [1 if i == 'Male' else 2 for i in df['sex']]

#plt.hist(df['sex_cat'], label='sex_cat')
#plt.show()

df1 = df[['actual','ideal','diff']].copy()

#print(df1.head())

Y = np.array(df['sex_cat'])
X = np.array(df1)

clf = GaussianNB()
clf.fit(X, Y)

y_pred = clf.fit(X, Y).predict(X)

print(clf.predict([[145, 160, -15]]))
print(clf.predict([[160, 145, 15]]))
print(y_pred)

print("Number of mislabeled points out of a total %d points : %d" % (X.shape[0],(Y != y_pred).sum()))