
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
  
def validate(X,Y,epoch):
    dimension=X.shape[1]
    print('Starting Execution of CV')
    kfold = KFold(n_splits=5, shuffle=True)
    cvscores = []
    trainingscores =[]
    split=0
    best_lr = 0.01
    best_bs = 256
    dropout=0.1
    initializer='lecun_uniform'
    for train, test in kfold.split(X,Y):
        model = Sequential()
        model.add(Dense(units=96, activation='softsign', input_dim=dimension, kernel_initializer=initializer,kernel_regularizer=l1_l2(l1=0.001,l2=0.001)))
        model.add(Dropout(dropout))
        model.add(Dense(units=96, activation='softsign', kernel_initializer=initializer,kernel_regularizer=l1_l2(l1=0.001,l2=0.001)))
        model.add(Dense(units=48, activation='softsign', kernel_initializer=initializer,kernel_regularizer=l1_l2(l1=0.001,l2=0.001)))
        model.add(Dense(units=48, activation='softsign', kernel_initializer=initializer,kernel_regularizer=l1_l2(l1=0.001,l2=0.001)))
        sgd = SGD(lr=best_lr)
        model.add(Dense(units=1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    	# Fit the model
        model.fit(X[train], Y[train], epochs=epoch, batch_size=best_bs, verbose=False)
        y_pred = model.predict(X[test])
        y_train = model.predict(X[train])
        y_train = y_train.flatten()
        y_pred = y_pred.flatten()
        try:
            training_error = metrics.mean_absolute_error(Y[train], y_train)
            error = metrics.mean_absolute_error(Y[test], y_pred)
            trainingscores.append(training_error)
        except:
            print("Input contains null values. Skipping Config.")
            continue
        cvscores.append(error)
        split=split+1
    print("Validation Score: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print("Training Score: %.2f%% (+/- %.2f%%)" % (np.mean(trainingscores), np.std(trainingscores)))
    return

    
# Prepping Data
# data_start = data_start.sample(frac=.85).reset_index(drop=True)
data_start = pd.read_csv("2048data.csv")
XLabels = ['Acid','Wood','TotalT', 'Temp', 'LSR', 'CA', 'Size', 'IsoT', 'HeatT', 'Ramp', 'F_X', 'Ro', 'logRo', 'P','Acetyl','Yield']
X_raw = data_start[XLabels]
# The data preparation function
XLabels_no = XLabels + ['NO']
for drop_this in XLabels_no:
    X_raw=data_start[XLabels]
    print("Dropped %s from the data" % drop_this)
    
    if (drop_this != 'Acid') and (drop_this != 'Wood'):
        haha_boolean = True
    else:
        haha_boolean = False
    
    X,Y,data,XLabels_notog=dp.prep(X_raw,haha_boolean,drop_this)
    
    if (drop_this != 'Acid') and (drop_this != 'Wood') and (drop_this != 'NO'):
        index=XLabels.index(drop_this)
        X=np.delete(X,index,axis=1)
   
    epoch=3000
    print('Remaining features: ', XLabels_notog)
    validate(X,Y,epoch)
    end1 = time.time()
    duration = end1 - start
    print("Execution Time of Neural Net is:", duration /60, "min\n")
