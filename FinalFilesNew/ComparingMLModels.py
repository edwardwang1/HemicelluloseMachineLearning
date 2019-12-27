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

def run_NN(X,Y):
    dimension=X.shape[1]
    print(dimension)
    print('Starting Execution of NN')
    kfold = KFold(n_splits=5)
    cvscores = []
    trainingscores =[]
    best_lr = 0.005
    best_bs = 64
    dropout=0.001
    epoch=3000
    master_actual = []
    master_pred = []

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

        master_actual.extend(Y.iloc[test])
        master_pred.extend(y_pred)
        
    meanTest = np.mean(cvscores)
    stdTest = np.std(cvscores)
    cvscores.append(meanTest)
    cvscores.append(stdTest)
    
    meanTrain = np.mean(trainingscores)
    stdTrain = np.std(trainingscores)
    trainingscores.append(meanTrain)
    trainingscores.append(stdTrain)

    saveDF = pd.DataFrame(list(zip(master_actual, master_pred)), columns =['Actual', 'Pred'])
    saveDF.to_csv("NNRawResults.csv", index=False)
    
    print("Validation Score: %.2f%% (+/- %.2f%%)" % (meanTest, stdTest))
    print("Training Score: %.2f%% (+/- %.2f%%)" % (meanTrain, stdTrain))
    return cvscores, trainingscores    

def run_SVR(X,Y):
    dimension=X.shape[1]
    print('Starting Execution of SVR')
    kfold = KFold(n_splits=5)
    cvscores = []
    trainingscores =[]
    kern = 'rbf'
    ep = 1
    gam = 'auto'
    C_ = 20000

    master_actual = []
    master_pred = []

    for train, test in kfold.split(X,Y):
        model = SVR(kernel=kern, epsilon=ep, cache_size=2000, C=C_, gamma = gam)
        
        X_train = X.iloc[train]
        Y_train = Y.iloc[train]

        model.fit(X_train, Y_train)

        X_test = np.array(X.iloc[test])
        y_pred = model.predict(X_test)
        y_train = model.predict(X_train)
        y_train = y_train.flatten()
        y_pred = y_pred.flatten()
        training_error = metrics.mean_absolute_error(Y.iloc[train], y_train)
        error = metrics.mean_absolute_error(Y.iloc[test], y_pred)
        trainingscores.append(training_error)
        cvscores.append(error)

        master_actual.extend(Y.iloc[test])
        master_pred.extend(y_pred)

    meanTest = np.mean(cvscores)
    stdTest = np.std(cvscores)
    cvscores.append(meanTest)
    cvscores.append(stdTest)
    
    meanTrain = np.mean(trainingscores)
    stdTrain = np.std(trainingscores)
    trainingscores.append(meanTrain)
    trainingscores.append(stdTrain)
    print("Validation Score: %.2f%% (+/- %.2f%%)" % (meanTest, stdTest))
    print("Training Score: %.2f%% (+/- %.2f%%)" % (meanTrain, stdTrain))

    saveDF = pd.DataFrame(list(zip(master_actual, master_pred)), columns =['Actual', 'Pred'])
    saveDF.to_csv("SVRRawResults.csv", index=False)
    
    return cvscores, trainingscores


def run_Ridge(X,Y):
    dimension=X.shape[1]
    print(dimension)
    print('Starting Execution of Ridge')
    kfold = KFold(n_splits=5)
    cvscores = []
    trainingscores =[]
    alpha = 1

    master_actual = []
    master_pred = []
    
    for train, test in kfold.split(X,Y):
        model = Ridge(alpha=alpha)       
    	# Fit the model
        X_train = X.iloc[train]
        Y_train = Y.iloc[train]
        model.fit(X_train, Y_train)
        
        X_test = np.array(X.iloc[test])
        y_pred = model.predict(X_test)
        y_train = model.predict(X_train)
        y_train = y_train.flatten()
        y_pred = y_pred.flatten()
        training_error = metrics.mean_absolute_error(Y.iloc[train], y_train)
        error = metrics.mean_absolute_error(Y.iloc[test], y_pred)
        trainingscores.append(training_error)
        cvscores.append(error)

        master_actual.extend(Y.iloc[test])
        master_pred.extend(y_pred)
        
    meanTest = np.mean(cvscores)
    stdTest = np.std(cvscores)
    cvscores.append(meanTest)
    cvscores.append(stdTest)
    
    meanTrain = np.mean(trainingscores)
    stdTrain = np.std(trainingscores)
    trainingscores.append(meanTrain)
    trainingscores.append(stdTrain)

    saveDF = pd.DataFrame(list(zip(master_actual, master_pred)), columns =['Actual', 'Pred'])
    saveDF.to_csv("RidgeRawResults.csv", index=False)
    
    print("Validation Score: %.2f%% (+/- %.2f%%)" % (meanTest, stdTest))
    print("Training Score: %.2f%% (+/- %.2f%%)" % (meanTrain, stdTrain))
    return cvscores, trainingscores


# Prepping Data
# data_start = data_start.sample(frac=.85).reset_index(drop=True)
df = pd.read_csv("PreparedDataAll.csv")


woods = ['acacia', 'aspen', 'basswood', 'beech', 'birch', 'carob', 'eucalyptus', 'maple', 'meranti', 'mixed', 'oak', 'olive', 'paulownia', 'poplar', 'vine', 'willow' ]
acids = ['acetic', 'fenton', 'formic', 'malic', 'none', 'oxalic', 'phosphoric', 'sulfuric']
factors = ['Ro', 'logRo', 'P', 'logP', 'H', 'logH']
labels_to_drop_front = [['TotalT'],['Temp'], ['LSR'], ['CA'], ['Size'], ['IsoT'], ['F_X'], ['Acetyl'],]
labels_to_drop_back = [['Ro', 'logRo'],  ['P', 'logP'], ['H', 'logH'], woods, acids, factors  ]
labels_to_drop_all = labels_to_drop_front + labels_to_drop_back
labels_to_drop_front_flat = [item for sublist in labels_to_drop_front for item in sublist]
labels_to_scale = labels_to_drop_front_flat + factors



Y = (df['Yield'])
labels_short = [factors]

X_scale = df[labels_to_scale]
scaler = StandardScaler()
scaled = scaler.fit_transform(X_scale)
X_scale[labels_to_scale] = scaled

finalCols = labels_to_scale + acids + woods

X_all = pd.concat([X_scale, df[df.columns[df.columns.isin(acids)]],df[df.columns[df.columns.isin(woods)]]], ignore_index=True,axis=1)
print(len(finalCols))
print(X_all)
X_all.columns = finalCols

Ridge_test_err, Ridge_train_err = run_Ridge(X_all, Y)
NN_test_err, NN_train_err = run_NN(X_all, Y)
SVR_test_err, SVR_train_err = run_SVR(X_all, Y)



cols = ['NN Test', 'NN Train', 'SVR Test', 'SVR Train', 'Ridge Test', 'Ridge Train']
results = pd.DataFrame(columns = cols)
results['NN Test'] = NN_test_err
results['NN Train'] = NN_train_err
results['SVR Test'] = SVR_test_err
results['SVR Train'] = SVR_train_err
results['Ridge Test'] = Ridge_test_err
results['Ridge Train'] = Ridge_train_err



results.to_csv("MLResults.csv")

end1 = time.time()
duration = end1 - start
print("Execution Time is", duration /60, "min\n")

