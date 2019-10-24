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

##Changeed for Monomer

start = time.time()

def validate(X_test, Y_test, X_train, Y_train):
    dimension=X_test.shape[1]
    print(dimension)
    print('Starting Execution of CV')
    cvscores = []
    trainingscores =[]
    best_lr = 0.005
    best_bs = 64
    dropout=0.001
    epoch=3000
    for i in range(5):
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
        model.fit(X_train, Y_train,batch_size=best_bs,epochs=epoch,verbose=False)
        y_pred = model.predict(X_test)
        y_pred = y_pred.flatten()
        y_train_pred = model.predict(X_train)
        y_train_pred = y_train_pred.flatten()
        training_error = metrics.mean_absolute_error(Y_train, y_train_pred)
        error = metrics.mean_absolute_error(Y_test, y_pred)
        trainingscores.append(training_error)
        cvscores.append(error)
    print("Validation Score: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print("Training Score: %.2f%% (+/- %.2f%%)" % (np.mean(trainingscores), np.std(trainingscores)))
    return np.mean(cvscores), np.std(cvscores), np.mean(trainingscores), np.std(trainingscores)
        
  
# Prepping Data
# data_start = data_start.sample(frac=.85).reset_index(drop=True)
df = pd.read_csv("PreparedDataAll.csv")


woods = ['acacia', 'aspen', 'basswood', 'beech', 'birch', 'carob', 'eucalyptus', 'maple', 'meranti', 'mixed', 'oak', 'olive', 'paulownia', 'pine', 'poplar', 'salix', 'vine', 'willow' ]
acids = ['acetic', 'fenton', 'formic', 'malic', 'none', 'oxalic', 'phosphoric', 'sulfuric']
factors = ['Ro', 'logRo', 'P', 'logP', 'H', 'logH']
labels_to_drop_front = [['TotalT'],['Temp'], ['LSR'], ['CA'], ['Size'], ['IsoT'], ['F_X'],]
labels_to_drop_back = [['Ro', 'logRo'],  ['P', 'logP'], ['H', 'logH'],['Acetyl'], woods, acids, factors  ]
labels_to_drop_all = labels_to_drop_front + labels_to_drop_back
labels_to_drop_front_flat = [item for sublist in labels_to_drop_front for item in sublist]
labels_to_scale = labels_to_drop_front_flat + factors


papers = df['Source'].unique()

# Removing papers from test list if they have less  than 25 points
papersWithLessThanXPoints = []
for paper in papers:
    dataFromPaper = df[df['Source'] == paper]
    if len(dataFromPaper.index) < 25:
        papersWithLessThanXPoints.append(paper)
papers = [x for x in papers if x not in papersWithLessThanXPoints]
print(len(papers), "num papers")

test_df = []
train_df = []
for paper in papers:
    
    train_df.append(df[df['Source'] != paper])
    test_df.append(df[df['Source'] == paper])
    
finalCols = labels_to_scale + acids + woods
cols = ['dropped', 'testMean', 'testStd', 'trainMean', 'trainStd', 'numPointsTest']
error_Frame = pd.DataFrame(columns = cols)

for i in range(len(test_df)):
    test_df[i] = test_df[i].drop(columns=['Source'])
    train_df[i] = train_df[i].drop(columns=['Source'])
    
    Y_test = test_df[i]['Yield']
    Y_train = train_df[i]['Yield']


    print(len(test_df[i].index), len(Y_test.index))
    print(len(train_df[i].index), len(Y_train.index))
    
    scaler = StandardScaler()

    X_test = test_df[i][labels_to_scale]
    scaled = scaler.fit_transform(X_test)
    X_test[labels_to_scale] = scaled

    X_train = train_df[i][labels_to_scale]
    scaled = scaler.fit_transform(X_train)
    X_train[labels_to_scale] = scaled

    X_all_test = pd.concat([X_test, test_df[i][test_df[i].columns[test_df[i].columns.isin(acids)]],test_df[i][test_df[i].columns[test_df[i].columns.isin(woods)]]], ignore_index=True,axis=1)
    X_all_test.columns = finalCols

    X_all_train = pd.concat([X_train, train_df[i][train_df[i].columns[train_df[i].columns.isin(acids)]],train_df[i][train_df[i].columns[train_df[i].columns.isin(woods)]]], ignore_index=True,axis=1)
    X_all_train.columns = finalCols

    print(len(test_df[i].index), len(Y_test.index), len(X_all_test.index))
    print(len(train_df[i].index), len(Y_train.index), len(X_all_train.index))

    testMeanE, testStdE, trainMeanE, trainStdE = validate(X_all_test, Y_test, X_all_train, Y_train)

    row = [[test_df[i], testMeanE, testStdE, trainMeanE, trainStdE, len(test_df[i].index)]]
    print(row)
    tempDf = pd.DataFrame(row, columns=cols)
    error_Frame = pd.concat([error_Frame, tempDf], ignore_index=True)
    end1 = time.time()
    duration = end1 - start
    print("Execution Time is", duration /60, "min\n")


error_Frame.to_csv("DataMismatch.csv")
