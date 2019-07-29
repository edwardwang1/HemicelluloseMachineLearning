
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

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
import dataprep_2 as dp
import wooddeleter as wd
import pydot
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

start = time.time()

data_start = pd.read_csv("/Users/RileyBallachay/Documents/Third Year/data1.csv")

# data = wd.deleter(data)


# Prepping Data
data_start = data_start.sample(frac=1).reset_index(drop=True)
XLabels = ['TotalT', 'Temp', 'LSR', 'CA', 'Size', 'IsoT', 'HeatT', 'Ramp', 'F_X', 'Ro', 'logRo', 'P','Acid','Acetyl','Wood','Yield']
X = data_start[XLabels]
# The data preparation function



XLabels = ['TotalT', 'Temp', 'LSR', 'CA', 'Size', 'IsoT', 'HeatT', 'Ramp', 'F_X', 'Ro', 'logRo', 'P','Acid','Acetyl','Wood','Yield']
X = data_start[XLabels]

X,Y,data,XLabels=dp.prep(X,False)

numData = len(data.index)
numTrain = int(numData * 0.7)
numTest = int(numData * .15)
# print(numTest, numTrain)

train_Frame, valid_Frame, test_Frame, train_valid_Frame = data.iloc[:numTrain, :], data.iloc[numTrain:-numTest,
                                                                                   :], data.iloc[-numTest:,
                                                                                       :], data.iloc[:-numTest:, :]

y_train, y_valid, y_test, y_train_valid = train_Frame['Yield'], valid_Frame['Yield'], test_Frame['Yield'], \
                                          train_valid_Frame['Yield']

X_train, X_valid, X_test, X_train_valid = X[:numTrain, :], X[numTrain:-numTest, :], X[-numTest:, :], X[:-numTest, :]


learningRates = [0.002, 0.005, 0.01, 0.02]
batchSizes = [64, 128, 256, 512, 1024]
dropoutRates = [0.00, 0.001, 0.01, 0.1]
errors = []

index_of_lowest_error = 39
best_lr = learningRates[int(index_of_lowest_error / (len(batchSizes) * len(dropoutRates)))]  # Good
best_bs = batchSizes[int((index_of_lowest_error % (len(batchSizes) * len(dropoutRates))) / len(dropoutRates))]  # Good
best_dr = dropoutRates[index_of_lowest_error % len(dropoutRates)]  # Good
print("ANN Best Learning Rate is: ", best_lr)
print("ANN Best Batch Size is: ", best_bs)
print("ANN Best Dropout Rate is: ", best_dr)


kfold = KFold(n_splits=3, shuffle=True, random_state=47)
cvscores = []
trainingscores =[]
split=0
NNarchitecture=np.zeros((6,6))

model = Sequential()
model.add(Dense(units=1200, activation='sigmoid', input_dim=39))
model.add(Dropout(0.01))
model.add(Dense(units=1200, activation='sigmoid'))
model.add(Dense(units=600, activation='sigmoid'))
model.add(Dense(units=600, activation='sigmoid'))
model.add(Dense(units=1, activation='softplus'))
	# Compile model
sgd = SGD(lr=best_lr)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    
for train, test in kfold.split(X, Y):
	# Fit the model
    model.fit(X[train], Y[train], epochs=3000, batch_size=best_bs, verbose=0)
    y_pred = model.predict(X[test], batch_size=1000)
    y_train = model.predict(X[train], batch_size=1000)
    y_train = y_train.flatten()
    y_pred = y_pred.flatten()
    training_error = metrics.mean_absolute_error(Y[train], y_train)
    error = metrics.mean_absolute_error(Y[test], y_pred)
    trainingscores.append(training_error)
    cvscores.append(error)
    split=split+1
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(trainingscores), np.std(trainingscores)))

end = time.time()

time = (start-end)/60