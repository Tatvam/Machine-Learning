#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 15:15:29 2018

@author: tatvam
importing the libraries
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset

dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values


# Splitting the data into training set and test set

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

"""# feature scaling 

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""