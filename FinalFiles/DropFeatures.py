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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pdb
import math
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
        X_train = X.iloc[train]
        Y_train = Y.iloc[train]
        model.fit(X_train, Y_train,batch_size=best_bs,epochs=epoch,verbose=False)
        X_test = np.array(X.iloc[test])
        y_pred = model.predict(X_test)
        y_train = model.predict(X_train)
        y_train = y_train.flatten()
        y_pred = y_pred.flatten()
        training_error = metrics.mean_absolute_error(Y.iloc[train], y_train)
        error = metrics.mean_absolute_error(Y.iloc[test], y_pred)
        trainingscores.append(training_error)
        cvscores.append(error)

def validate2(X,Y):
    best_lr = 0.005
    best_bs = 64
    dropout=0.001
    epoch=3000
    dimension=X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
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
    model.fit(X_train, y_train,batch_size=best_bs,epochs=epoch,verbose=False)
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    #y_train_pred = y_train.flatten()
    #y_pred = y_pred.flatten()
    training_error = metrics.mean_absolute_error(y_train, y_train_pred)
    error = metrics.mean_absolute_error(y_test, y_pred)

    return error, 0, training_error, 0

    
# Prepping Data
# data_start = data_start.sample(frac=.85).reset_index(drop=True)
df = pd.read_csv("PreparedDataAll.csv")


woods = ['acacia', 'aspen', 'basswood', 'beech', 'birch', 'carob', 'eucalyptus', 'maple', 'meranti', 'mixed', 'oak', 'olive', 'paperbark', 'paulownia', 'pine', 'poplar', 'salix', 'vine', 'willow' ]
acids = ['acetic', 'fenton', 'formic', 'malic', 'none', 'oxalic', 'phosphoric', 'sulfuric']
factors = ['Ro', 'logRo', 'P', 'logP', 'H', 'logH']
labels_to_drop_front = [['TotalT'],['Temp'], ['LSR'], ['CA'], ['Size'], ['IsoT'], ['Ramp'], ['F_X'],]
labels_to_drop_back = [['Ro', 'logRo'],  ['P', 'logP'], ['H', 'logH'], ['Acetyl'], woods, acids, factors  ]
labels_to_drop_all = labels_to_drop_front + labels_to_drop_back


Y = (df['Yield'])
cols = ['dropped', 'testMean', 'testStd', 'trainMean', 'trainStd']
error_Frame = pd.DataFrame(columns = cols)

labels_short = [factors]

for labels in labels_to_drop_all:
    print(labels)
    X = df[df.columns[~df.columns.isin(labels)]]
    #z = df.columns.isin(labels)
    #assert len(labels) == z.tolist().count(True)
    
    testMeanE, testStdE, trainMeanE, trainStdE = validate2(X, Y)
    row = [[labels, testMeanE, testStdE, trainMeanE, trainStdE]]
    tempDf = pd.DataFrame(row, columns=cols)
    error_Frame = pd.concat([error_Frame, tempDf], ignore_index=True)
    end1 = time.time()
    duration = end1 - start
    print("Execution Time is", duration /60, "min\n")


error_Frame.to_csv("DropingFeaturesSolubleXylose.csv")

no_factors_frame = df.copy()
no_factors_frame = df[df.columns[~df.columns.isin(factors)]]

error_Frame2 = pd.DataFrame(columns = cols)


for labels in labels_to_drop_front:
    print(labels)
    X = df[df.columns[~df.columns.isin(labels)]]

    #z = df.columns.isin(labels)
    #assert len(labels) == z.tolist().count(True)
    
    testMeanE, testStdE, trainMeanE, trainStdE = validate2(X, Y)
    row = [[labels, testMeanE, testStdE, trainMeanE, trainStdE]]
    tempDf = pd.DataFrame(row, columns=cols)
    error_Frame2 = pd.concat([error_Frame2, tempDf], ignore_index=True)
    end1 = time.time()
    duration = end1 - start
    print("Execution Time is", duration /60, "min\n")

error_Frame2.to_csv("DropingFeaturesSolubleXyloseNoFactor.csv")

    

