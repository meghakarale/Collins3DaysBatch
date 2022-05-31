# -*- coding: utf-8 -*-
"""
Created on Tue May 31 09:49:25 2022
Tunning Neural Network HyperParameter
@author: TSE
"""

# =============================================================================
# Import of Libaries
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Data Import and Feature Extraction
# =============================================================================
dataset = pd.read_csv('Churn Modelling-Bank Customers.csv')

X = dataset.iloc[:, 3:13]     # X dataframe 
y = dataset.iloc[:, 13].values  # y array

# =============================================================================
# Encoding Georgraphy and Gender Column
# =============================================================================
X = pd.get_dummies(X, columns=['Geography','Gender'],drop_first=True)

X = X.values   # convert dataframe to Numpy array

# =============================================================================
# Train-Test 80-20 Split
# =============================================================================
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# =============================================================================
# Scaling of Values
# =============================================================================
from sklearn.preprocessing import StandardScaler
scObj = StandardScaler()

X_train = scObj.fit_transform(X_train)
X_test = scObj.transform(X_test)

# =============================================================================
# Hyper Parameter Tunning
# =============================================================================
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer, h1_size, h2_size):
    classifier = Sequential()
    classifier.add(Dense(input_dim=11, units = h1_size, kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units = h2_size, kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units = 1, kernel_initializer='uniform',activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

##---> Type Cast Keras to sklearn
classifier = KerasClassifier(build_fn=build_classifier)

parameters = {"h1_size":[6,12,24,48],
              "h2_size":[6,12,24,48],
              "optimizer":['adam','rmsprop'],
              "batch_size":[10,25,50],
              "epochs":[100,150,500]}

gridSearch = GridSearchCV(estimator=classifier, param_grid=parameters,scoring='accuracy',cv=10)

gridSearch = gridSearch.fit(X_train,y_train)

best_parameters = gridSearch.best_params_
best_accuracy = gridSearch.best_score_




