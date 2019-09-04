"""
Created on Tue Sep  3 16:42:11 2019

@author: RileyBallachay
"""


import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn import metrics
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pdb
import math
import dataprep_dropacid as dp
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from keras.regularizers import l1
from keras.regularizers import l1_l2

start = time.time()
  
def validate(X,Y):
    dimension=X.shape[1]
    print(dimension)
    print('Starting Execution of CV')
    kfold = KFold(n_splits=5)
    cvscores = []
    trainingscores =[]
    best_lr = 0.005
    best_bs = 64
    dropout=0.001
    epoch=1
    for train, test in kfold.split(X,Y):
        model = Sequential()
        model.add(Dense(units=96, activation='sigmoid', input_dim=dimension))
        model.add(Dropout(dropout))
        model.add(Dense(units=96, activation='sigmoid'))
        model.add(Dense(units=48, activation='sigmoid'))        
        model.add(Dense(units=48, activation='sigmoid'))        
        sgd = SGD(lr=best_lr)
        model.add(Dense(units=1, activation='linear'))
        model.compile(optimizer=sgd,loss='mean_squared_error')
    	# Fit the model
        X_train = X[train]
        Y_train = Y[train]
        model.fit(X_train, Y_train,batch_size=best_bs,epochs=epoch,verbose=False)
        X_test = np.array(X[test])
        y_pred = model.predict(X_test)
        y_train = model.predict(X_train)
        y_train = y_train.flatten()
        y_pred = y_pred.flatten()
        training_error = metrics.mean_absolute_error(Y[train], y_train)
        error = metrics.mean_absolute_error(Y[test], y_pred)
        trainingscores.append(training_error)
        cvscores.append(error)
    print("Validation Score: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print("Training Score: %.2f%% (+/- %.2f%%)" % (np.mean(trainingscores), np.std(trainingscores)))
    return

    

data=["data_monomer","data_all"]
for data_set in data:
    for i in range(1,6):
        
        if data_set == "data_all":
            X = pd.read_csv('FinalFiles/PreparedDataAll.csv')
            Y = np.array(X['Yield'])
            X.drop('Yield',axis=1,inplace=True)
            X=np.array(X)
            X = np.delete(X,0,1)
        if data_set == "data_monomer":
            X = pd.read_csv('FinalFiles/PreparedMonomer.csv')
            Y = np.array(X['Yield'])
            X.drop('Yield',axis=1,inplace=True)
            X=np.array(X)
            X = np.delete(X,0,1)        
        
        print('Dataset', data_set)
        print('Iteration number', i)
        length = X.shape[0]
        validate(X,Y)
        end1 = time.time()
        duration = end1 - start
        print("Execution Time of Neural Net is:", duration /60, "min\n")
