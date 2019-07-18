
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
import dataprep_1 as dp

start = time.time()

data = pd.read_csv("/Users/RileyBallachay/Documents/Third Year/data1.csv")


# Prepping Data
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
data = data.sample(frac=1).reset_index(drop=True)
XLabels = ['TotalT', 'Temp', 'LSR', 'CA', 'Size', 'IsoT', 'HeatT', 'Ramp', 'F_X', 'Ro', 'logRo', 'P','Acid','Acetyl','Wood','Yield']
X = data[XLabels]

X,Y,data,XLabels=dp.prep(X)

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


# Simple Linear Regression
###
print("Staring Linear Regression ------------------")

regr = linear_model.LinearRegression()
regr.fit(X_train_valid, y_train_valid)
coeffs = dict(zip(XLabels, regr.coef_))

y_pred = regr.predict(X_test)

predAndActual = pd.DataFrame({'Pred': y_pred, 'Test': y_test})
predAndActual.to_csv("OverallSimpleLinear.csv")

linearError = pd.Series(y_test - y_pred, name='Lin Err').abs()
test_Frame = pd.concat([test_Frame, linearError], axis=1)
test_Frame.to_csv('TestDataWithErrors.csv')

print('Linear regression Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Linear regression Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Linear regression Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("Ending Linear Regression ------------------\n")

##Ridge Regression
print("Staring Ridge Regression ------------------")

# alphas = [0]
alphas = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100]
errors = []

# Creating Different models for different alphas
for a in alphas:
    ridgeModel = Ridge(alpha=a)
    ridgeModel.fit(X_train, y_train)
    y_pred = ridgeModel.predict(X_valid)
    error = metrics.mean_absolute_error(y_valid, y_pred)
    errors.append(error)

best_alpha = alphas[np.argmin(errors)]
print("Ridge Regression best alpha is: ", best_alpha)

best_model = Ridge(alpha=best_alpha)
best_model.fit(X_train_valid, y_train_valid)
y_pred = best_model.predict(X_test)

predAndActual = pd.DataFrame({'Pred': y_pred, 'Test': y_test})
predAndActual.to_csv("OverallRidge.csv")

ridgeError = pd.Series(y_test - y_pred, name='Rid Err').abs()
test_Frame = pd.concat([test_Frame, ridgeError], axis=1)
test_Frame.to_csv('TestDataWithErrors.csv')

# Evaluation
print('Ridge Regression Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Ridge Regression Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Ridge Regression Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("Ridge Regression Coefficients: ", best_model.coef_)

print("Ending Ridge Regression ------------------\n")

# SVM Models

print("Starting Support Vector Regression ------------------")

kernels = ['poly', 'rbf', 'linear']
epsilons = [0.1, 5, 10, 20]
Cs = [0.1, 1, 10, 20]
gammas = ['scale', 'auto']
errors = []

for kern in kernels:
    for ep in epsilons:
        for C_ in Cs:
            for gam in gammas:
                svrModel = SVR(kernel=kern, gamma=gam, epsilon=ep, cache_size=2000, C=C_)
                #                 svrModel.fit(X_train, y_train, sample_weight=train_weights)
                svrModel.fit(X_train, y_train)
                y_pred = svrModel.predict(X_valid)
                error = metrics.mean_absolute_error(y_valid, y_pred)
                errors.append(error)

index_of_lowest_error = np.argmin(errors)

best_kernel = kernels[int(index_of_lowest_error / (len(epsilons) * len(Cs) * len(gammas)))]  # Good
best_ep = epsilons[
    int((index_of_lowest_error % (len(epsilons) * len(Cs) * len(gammas))) / (len(Cs) * len(gammas)))]  # Good
best_C = Cs[int((index_of_lowest_error % (len(Cs) * len(gammas))) / len(gammas))]  # Good
best_gamma = gammas[index_of_lowest_error % len(gammas)]

print("SVR Best kernel is: ", best_kernel)
print("SVR Best Epsilon is: ", best_ep)
print("SVR Best C is: ", best_C)
print("SVR Best Gamma is: ", best_gamma)

# # Make it run a little faster, hardcode best

best_model = SVR(kernel=best_kernel, gamma=best_gamma, epsilon=best_ep, cache_size=2000, C=best_C)
# best_model.fit(X_train_valid, y_train_valid, sample_weight=train_valid_weights)
best_model.fit(X_train_valid, y_train_valid)
y_pred = best_model.predict(X_test)

predAndActual = pd.DataFrame({'Pred': y_pred, 'Test': y_test})
predAndActual.to_csv("OverallSVR.csv")

svrError = pd.Series(y_test - y_pred, name='SVR Err').abs()
test_Frame = pd.concat([test_Frame, svrError], axis=1)
test_Frame.to_csv('TestDataWithErrors.csv')

print('SVR Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('SVR Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('SVR Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



print("Ending Support Vector Regression ------------------\n")

## Artificial Neural Network
# Keras
print("Starting Neural Network------------------")

learningRates = [0.002, 0.005, 0.01, 0.02]
batchSizes = [64, 128, 256, 512, 1024]
dropoutRates = [0.00, 0.001, 0.01, 0.1]
errors = []

'''
for lr_ in learningRates:
    for bs in batchSizes:
        for dr in dropoutRates:
            model = Sequential()
            model.add(Dense(units=12, activation='sigmoid', input_dim=40))
            model.add(Dropout(dr))
            model.add(Dense(units=12, activation='sigmoid'))
            model.add(Dense(units=6, activation='sigmoid'))
            model.add(Dense(units=6, activation='sigmoid'))
            model.add(Dense(units=1, activation='softplus'))

            sgd = SGD(lr=lr_)
            model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

            model.fit(X_train, y_train, epochs=3000, batch_size=bs, verbose=0)



            y_pred = model.predict(X_valid, batch_size=bs)
            y_pred = y_pred.flatten()

            error = metrics.mean_absolute_error(y_valid, y_pred)
            errors.append(error)
'''

index_of_lowest_error = 38
best_lr = learningRates[int(index_of_lowest_error / (len(batchSizes) * len(dropoutRates)))]  # Good
best_bs = batchSizes[int((index_of_lowest_error % (len(batchSizes) * len(dropoutRates))) / len(dropoutRates))]  # Good
best_dr = dropoutRates[index_of_lowest_error % len(dropoutRates)]  # Good
print("ANN Best Learning Rate is: ", best_lr)
print("ANN Best Batch Size is: ", best_bs)
print("ANN Best Dropout Rate is: ", best_dr)

# Using best values
layers = [3,6,12,24,48,96]
## Changing network architecture
for layer in layers:
    model = Sequential()
    model.add(Dense(units=layer*2, activation='sigmoid', input_dim=40))
    model.add(Dropout(best_dr))
    model.add(Dense(units=layer, activation='sigmoid'))
    model.add(Dense(units=layer, activation='sigmoid'))
    model.add(Dense(units=1, activation='softplus'))
    
    sgd = SGD(lr=best_lr)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    
    
    model.fit(X_train_valid, y_train_valid, epochs=3000, batch_size=best_bs, verbose=0)
    
    # loss_and_metrics = model.evaluate(X_test, y_test,batch_size=best_bs)
    
    y_pred = model.predict(X_test, batch_size=best_bs)
    y_pred = y_pred.flatten()
    nnError = pd.Series(y_test - y_pred, name='NN Err').abs()
    
    # test_Frame = pd.concat([test_Frame, nnError], axis=1)
    
    print('Layers',layer)
    print('ANN Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('ANN Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('ANN Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)),'\n')
  
    
    
predAndActualNN = pd.DataFrame({'Pred': y_pred, 'Test': y_test, 'Error':  nnError})
indices = predAndActualNN.index.values
# X=pd.DataFrame(X_us)
# predAndActualNN = pd.concat([predAndActualNN], axis=1)
predAndActualNN.to_csv("OverallNN.csv")   
test_Frame.to_csv('TestDataWithErrors.csv')


print("Ending Neural Network Regression ------------------\n")

end = time.time()
duration = end - start
print("Execution Time is:", duration /60, "min")
