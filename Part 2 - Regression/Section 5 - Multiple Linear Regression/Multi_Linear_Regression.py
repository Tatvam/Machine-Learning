#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 22:13:10 2018

@author: tatvam
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

    
# import the dataset

dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

# Encoding Catagorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding dummy variable trap
X=X[:, 1:]

# Splitting the data into training set and test set

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

# Fitting multi linear regression in training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Optimal model using Backward Elimination

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(float), values = X,axis=1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y,exog = X_opt).fit()
regressor_OLS.summary()

while 1:
    flag = 0
    for i in range(1,int(X_opt.size/50)):
        cmp = regressor_OLS.summary().tables[1][i][4].data
        if float(cmp) > 0.05:
            X_opt = np.delete(X_opt,i,axis = 1)
            flag = 1
            break
    if flag == 0:
        break
    regressor_OLS = sm.OLS(endog = Y,exog = X_opt).fit()

X_opt = X_opt[:, 1 : ]
from sklearn.model_selection import train_test_split
X_train_B,X_test_B,Y_train,Y_test = train_test_split(X_opt,Y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor_BE = LinearRegression()
regressor_BE.fit(X_train_B, Y_train)

y_pred_BE = regressor_BE.predict(X_test_B)

    
    

#pd.read_html(Z, header=0, index_col=0)