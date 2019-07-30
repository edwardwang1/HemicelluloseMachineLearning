

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

start = time.time()

data_start = pd.read_csv("2048data.csv")

# data = wd.deleter(data)


# Prepping Data
data_start = data_start.sample(frac=1).reset_index(drop=True)
XLabels = ['TotalT', 'Temp', 'LSR', 'CA', 'Size', 'IsoT', 'HeatT', 'Ramp', 'F_X', 'Ro', 'logRo', 'P','Acid','Acetyl','Wood','Yield']
X = data_start[XLabels]
# The data preparation function

XLabels = ['TotalT', 'Temp', 'LSR', 'CA', 'Size', 'IsoT', 'HeatT', 'Ramp', 'F_X', 'Ro', 'logRo', 'P','Acid','Acetyl','Wood','Yield']
X_raw = data_start[XLabels]


X,Y,data,XLabels=dp.prep(X_raw,True)

best_lr = 0.01
best_bs = 256
best_dr = 0
print("ANN Best Learning Rate is: ", best_lr)
print("ANN Best Batch Size is: ", best_bs)
print("ANN Best Dropout Rate is: ", best_dr)

def validate(X,Y,modelname):
    split=0
    for train, test in kfold.split(X,Y):
        # Fit the model
        modelname.fit(X[train], Y[train], epochs=3000, batch_size=best_bs, verbose=0)
        y_pred = modelname.predict(X[test], batch_size=1000)
        y_train = modelname.predict(X[train], batch_size=1000)
        y_train = y_train.flatten()
        y_pred = y_pred.flatten()
        training_error = metrics.mean_absolute_error(Y[train], y_train)
        error = metrics.mean_absolute_error(Y[test], y_pred)
        trainingscores.append(training_error)
        cvscores.append(error)
        split=split+1
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(trainingscores), np.std(trainingscores)))
    return

kfold = KFold(n_splits=7, shuffle=True, random_state=1)
cvscores = []
trainingscores =[]

model1 = Sequential()
model1.add(Dense(units=48, activation='sigmoid', input_dim=39))
model1.add(Dense(units=48, activation='sigmoid'))
model1.add(Dense(units=24, activation='sigmoid'))
model1.add(Dense(units=24, activation='sigmoid'))
model1.add(Dense(units=1, activation='linear'))
sgd = SGD(lr=best_lr)
model1.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
print('48,48,24,24')
validate(X,Y,model1)

end1 = time.time()
duration = end1 - start
print("Execution Time is:", duration /60, "min")

model2 = Sequential()
model2.add(Dense(units=48, activation='sigmoid', input_dim=39))
model2.add(Dense(units=24, activation='sigmoid'))
model2.add(Dense(units=24, activation='sigmoid'))
model2.add(Dense(units=1, activation='linear'))
sgd = SGD(lr=best_lr)
model2.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
print('48,24,24')
validate(X,Y,model2)

end2 = time.time()
duration = end1 - end2
print("Execution Time is:", -duration /60, "min")

model3 = Sequential()
model3.add(Dense(units=96, activation='sigmoid', input_dim=39))
model3.add(Dense(units=48, activation='sigmoid'))
model3.add(Dense(units=24, activation='sigmoid'))
model3.add(Dense(units=1, activation='linear'))
sgd = SGD(lr=best_lr)
model3.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
print('96,48,24')
validate(X,Y,model3)

end3 = time.time()
duration = end2 - end3
print("Execution Time is:", -duration /60, "min")

model4 = Sequential()
model4.add(Dense(units=96, activation='sigmoid', input_dim=39))
model4.add(Dense(units=96, activation='sigmoid'))
model4.add(Dense(units=1, activation='linear'))
sgd = SGD(lr=best_lr)
model4.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
print('96,96')
validate(X,Y,model4)

end4 = time.time()
duration = end3 - end4
print("Execution Time is:", -duration /60, "min")

model5 = Sequential()
model5.add(Dense(units=96, activation='sigmoid', input_dim=39))
model5.add(Dense(units=48, activation='sigmoid'))
model5.add(Dense(units=1, activation='linear'))
sgd = SGD(lr=best_lr)
model5.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
print('96,48')
validate(X,Y,model5)

end5 = time.time()
duration = end4 - end5
print("Execution Time is:", -duration /60, "min")

model6 = Sequential()
model6.add(Dense(units=48, activation='sigmoid', input_dim=39))
model6.add(Dense(units=24, activation='sigmoid'))
model6.add(Dense(units=1, activation='linear'))
sgd = SGD(lr=best_lr)
model6.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
print('48,24')
validate(X,Y,model6)

end6 = time.time()
duration = end5 - end6
print("Execution Time is:", -duration /60, "min")
