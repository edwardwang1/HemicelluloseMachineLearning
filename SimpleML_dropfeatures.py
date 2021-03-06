import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn import metrics
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pdb
import math
import dataprep_1 as dp
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from keras.regularizers import l1
from keras.regularizers import l1_l2

start = time.time()

def validate(X,Y,epoch,l1l2,dropout):
        kfold = KFold(n_splits=5, shuffle=True)
        cvscores = []
        trainingscores =[]
        split=0
        for train, test in kfold.split(X,Y):
            model=Sequential()
            model.add(Dense(units=96, activation='sigmoid', input_dim=39, kernel_initializer=initializer,kernel_regularizer=l1_l2(l1=l1l2,l2=l1l2)))
            model.add(Dropout(dropout))
            model.add(Dense(units=96, activation='sigmoid', kernel_initializer=initializer,kernel_regularizer=l1_l2(l1=l1l2,l2=l1l2)))
            model.add(Dense(units=48, activation='sigmoid', kernel_initializer=initializer,kernel_regularizer=l1_l2(l1=l1l2,l2=l1l2))) 
            model.add(Dense(units=48, activation='sigmoid', kernel_initializer=initializer,kernel_regularizer=l1_l2(l1=l1l2,l2=l1l2))) 
            model.add(Dense(units=1, activation='linear'))
            model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

            model.fit(X[train], Y[train], epochs=epoch,validation_data=(X[test],Y[test]), batch_size=best_bs, verbose=0)
            y_pred = model.predict(X[test])
            y_train = model.predict(X[train])
            y_train = y_train.flatten()
            y_pred = y_pred.flatten()
            try:
                training_error = metrics.mean_absolute_error(Y[train], y_train)
                error = metrics.mean_absolute_error(Y[test], y_pred)
                trainingscores.append(training_error)
                cvscores.append(error)
            except:
                print("Input contains null values. Skipping Config.")
                continue
            split=split+1
        print("L1_L2 Coefficient:", l1l2)
        print("Dropout Rate:", dropout)
        print("Validation Score: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        print("Training Score: %.2f%% (+/- %.2f%%)" % (np.mean(trainingscores), np.std(trainingscores)))
        return 
    
    
# Prepping Data
data_start = pd.read_csv("2048data.csv")
data_start = data_start.sample(frac=.85).reset_index(drop=True)
XLabels = ['TotalT', 'Temp', 'LSR', 'CA', 'Size', 'IsoT', 'HeatT', 'Ramp', 'F_X', 'Ro', 'logRo', 'P','Acid','Acetyl','Wood','Yield']
X_raw = data_start[XLabels]
# The data preparation function
sgd = SGD(lr=0.01)
rms = RMSprop(lr=0.01)
adagrad = Adagrad(lr=0.01)
adam = Adam(lr=0.01)
optimizer = RMSprop(lr=0.01)
op = 0
l1l2s = [0.0001,0.001,0.01,0.1]
dropouts = [0.0001,0.001,0.01,0.1]

for dropout in dropouts:
    for l1l2 in l1l2s:
        X,Y,data,XLabels=dp.prep(X_raw,True)   
        best_lr = 0.01
        best_bs = 256
        initializer='lecun_uniform'
        epoch = 2500
        validate(X,Y,epoch,l1l2,dropout)
        end1 = time.time()
        duration = end1 - start
        print("Execution Time of Neural Net is:", duration /60, "min\n")
        start = end1
        op=op+1

