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

def validate(X_raw, Y_raw, frac, cols):
    comb = pd.concat([X_raw, Y_raw], axis=1)
    cols = cols + ['Yield']
    comb.columns = cols
    data = comb.sample(frac=frac)
    Y = data['Yield']
    X = data.drop(columns=['Yield'])
    dimension = X.shape[1]
    print('Starting Execution of CV')
    kfold = KFold(n_splits=5)
    cvscores = []
    trainingscores =[]
    best_lr = 0.005
    best_bs = 64
    dropout=0.001
    epoch=3000
    for i in range(3):
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
    print("Validation Score: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print("Training Score: %.2f%% (+/- %.2f%%)" % (np.mean(trainingscores), np.std(trainingscores)))
    return np.mean(cvscores), np.std(cvscores), np.mean(trainingscores), np.std(trainingscores)
        
  
# Prepping Data
# data_start = data_start.sample(frac=.85).reset_index(drop=True)
df = pd.read_csv("PreparedData.csv")


woods = ['acacia', 'aspen', 'basswood', 'beech', 'birch', 'carob', 'eucalyptus', 'maple', 'meranti', 'mixed', 'oak', 'olive', 'paulownia', 'pine', 'poplar', 'salix', 'vine', 'willow' ]
acids = ['acetic', 'fenton', 'formic', 'malic', 'none', 'oxalic', 'phosphoric', 'sulfuric']
factors = ['Ro', 'logRo', 'P', 'logP', 'H', 'logH']
labels_to_drop_front = [['TotalT'],['Temp'], ['LSR'], ['CA'], ['Size'], ['IsoT'], ['F_X'],]
labels_to_drop_back = [['Ro', 'logRo'],  ['P', 'logP'], ['H', 'logH'], ['Acetyl'], woods, acids, factors  ]
labels_to_drop_all = labels_to_drop_front + labels_to_drop_back
labels_to_drop_front_flat = [item for sublist in labels_to_drop_front for item in sublist]
labels_to_scale = labels_to_drop_front_flat + factors



Y = (df['Yield'])
cols = ['dropped', 'testMean', 'testStd', 'trainMean', 'trainStd']
error_Frame = pd.DataFrame(columns = cols)
labels_short = [factors]

X_scale = df[labels_to_scale]
scaler = StandardScaler()
scaled = scaler.fit_transform(X_scale)
X_scale[labels_to_scale] = scaled

finalCols = labels_to_scale + acids + woods

X = pd.concat([X_scale, df[df.columns[df.columns.isin(acids)]],df[df.columns[df.columns.isin(woods)]]], ignore_index=True,axis=1)
X.columns = finalCols

fractions = [0.1, .25, .5, .75, 1.0]

for fraction in fractions:
    print(fraction)
    testMeanE, testStdE, trainMeanE, trainStdE = validate(X, Y, fraction, finalCols)
    row = [[fraction, testMeanE, testStdE, trainMeanE, trainStdE]]
    print(row)
    tempDf = pd.DataFrame(row, columns=cols)
    error_Frame = pd.concat([error_Frame, tempDf], ignore_index=True)
    end1 = time.time()
    duration = end1 - start
    print("Execution Time is", duration /60, "min\n")


error_Frame.to_csv("EffectOfDataSize.csv")


    

