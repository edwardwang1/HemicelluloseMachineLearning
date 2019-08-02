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
import dataprep_1 as dp
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from keras.regularizers import l1
from keras.regularizers import l1_l2

start = time.time()
  
def validate(X,Y,modelname,epoch):
        kfold = KFold(n_splits=4, shuffle=True)
        cvscores = []
        trainingscores =[]
        split=0
        for train, test in kfold.split(X,Y):
        	# Fit the model
            modelname.fit(X[train], Y[train], epochs=epoch, batch_size=best_bs, verbose=0)
            y_pred = modelname.predict(X[test])
            y_train = modelname.predict(X[train] )
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
XLabels = ['TotalT', 'Temp', 'LSR', 'CA', 'Size', 'IsoT', 'HeatT', 'Ramp', 'F_X', 'Ro', 'logRo', 'P','Acid','Acetyl','Wood','Yield']
X_raw = data_start[XLabels]
# The data preparation function
for drop_this in XLabels:
    
    print("\n Dropped %s from the data" % drop_this)
    
    X,Y,data,XLabels=dp.prep(X_raw,True)
    
    index=XLabels.index(drop_this)
    
    X=np.delete(X,index,axis=1)
    
    best_lr = 0.01
    best_bs = 256
    best_dr = 0
    dropout=0.001
    initializer='lecun_uniform'
    epoch = 3000
    
     
    
    try:
        model = Sequential()
        model.add(Dense(units=96, activation='softsign', input_dim=38, kernel_initializer=initializer))
        model.add(Dropout(dropout))
        model.add(Dense(units=96, activation='softsign', kernel_initializer=initializer))
        model.add(Dense(units=48, activation='softsign', kernel_initializer=initializer))
        model.add(Dense(units=48, activation='softsign', kernel_initializer=initializer))
        model.add(Dense(units=1, activation='linear'))
    except:
        print("That isn't a valid initializer, stupid")
    sgd = SGD(lr=best_lr)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    validate(X,Y,model,epoch)
    end1 = time.time()
    duration = end1 - start
    print("Execution Time of Neural Net is:", duration /60, "min","\n")
    start = end1
