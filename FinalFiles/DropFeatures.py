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
    epoch=3000
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
    return np.mean(cvscores), np.std(cvscores), np.mean(trainingscores),np.std(trainingscores)

    
# Prepping Data
# data_start = data_start.sample(frac=.85).reset_index(drop=True)
data_start = pd.read_csv("PreparedDataAll.csv")
woods = ['acacia', 'aspen', 'basswood', 'beech', 'birch', 'carob', 'eucalyptus', 'maple', 'meranti', 'mixed', 'oak', 'olive', 'paperbark', 'paulownia', 'pine', 'poplar', 'salix', 'vine', 'willow' ]
acids = ['acetic', 'fenton', 'formic', 'malic', 'none', 'oxalic', 'phosphoric', 'sulfuric']
factors = ['Ro', 'logRo', 'P', 'logP', 'H', 'logH']



labels_to_drop = [['TotalT'],['Temp'], ['LSR'], ['CA'], ['Size'], ['IsoT'], ['Ramp'], ['F_X'], ['Ro, logRo'],  ['P, logP'], ['H, logH'], ['Acetyl'], woods, acids, factors  ]

Y = np.array(data_start['Yield'])
cols = ['dropped', 'testMean', 'testStd', 'trainMean', 'trainStd']
errorsFrame = pd.DataFrame(columns = cols)

for labels in labels_to_drop:
    X = data_start[~labels]
    testMeanE, testStdE, trainMeanE, trainStdE = validate(X, Y)
    row = [[labels, testMeanE, testStdE, trainMeanE, trainStdE]]
    tempDf = pd.DataFrame(row, columns=cols)
    error_Frame = pd.concat([error_Frame, tempDf], ignore_index=True)
    end1 = time.time()
    duration = end1 - start
    print("Execution Time of Neural Net is:", duration /60, "min\n")

errorsFrame.to_csv("DropingFeaturesSolubleXylose")

    

