
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn import metrics
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pdb
import math
import dataprep_1 as dp
from sklearn.model_selection import KFold
from keras.regularizers import l1_l2
import random

start = time.time()
hard_start = time.time()
    
# Prepping Data and splitting into training and test sets
data_start = pd.read_csv("2048data.csv")
data_start = data_start.sample(frac=1).reset_index(drop=True)
length=int(data_start.shape[0]*.85)
XLabels = ['TotalT', 'Temp', 'LSR', 'CA', 'Size', 'IsoT', 'HeatT', 'Ramp', 'F_X', 'Ro', 'logRo', 'P','Acid','Acetyl','Wood','Yield']
X_raw = data_start[XLabels]
X_full,Y_full,data,XLabels=dp.prep(X_raw,True)  
X=X_full[0:length,:]
Y=Y_full[0:length]
X_test=X_full[length:,:]
Y_test=Y_full[length:]

lowest_error = 1000
iterations = 75

for iteration in range(1,iterations): 
    
    # Random Hyperparameter Assignment 
    learning_rate = random.uniform(0.01,0.1)
    optimizer = RMSprop(lr=learning_rate)
    batch_size = random.randrange(200,1000)
    dropout = random.uniform(0.001,0.1)
    initializer='lecun_uniform'
    epoch = random.randrange(1500,3500)
    l1l2 = random.uniform(0.0001,0.003)
    kfold = KFold(n_splits=5, shuffle=True)
    cvscores = []
    trainingscores =[]

    for train, test in kfold.split(X,Y):
        model=Sequential()
        model.add(Dense(units=96, activation='sigmoid', input_dim=39, kernel_initializer=initializer,kernel_regularizer=l1_l2(l1=l1l2,l2=l1l2)))
        model.add(Dropout(dropout))
        model.add(Dense(units=96, activation='sigmoid', kernel_initializer=initializer,kernel_regularizer=l1_l2(l1=l1l2,l2=l1l2)))
        model.add(Dense(units=48, activation='sigmoid', kernel_initializer=initializer,kernel_regularizer=l1_l2(l1=l1l2,l2=l1l2))) 
        model.add(Dense(units=48, activation='sigmoid', kernel_initializer=initializer,kernel_regularizer=l1_l2(l1=l1l2,l2=l1l2))) 
        model.add(Dense(units=1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

        model.fit(X[train], Y[train], epochs=epoch, validation_data=(X[test],Y[test]), batch_size=batch_size, verbose=0)
        y_pred = model.predict(X[test])
        y_train = model.predict(X[train])
        y_train = y_train.flatten()
        y_pred = y_pred.flatten()
        
        try:
            training_error = metrics.mean_absolute_error(Y[train], y_train)
            error = metrics.mean_absolute_error(Y[test], y_pred)
            trainingscores.append(training_error)
            cvscores.append(error)
        except:
            print("Input contains null values. Skipping Config.")
            continue
        
    print("Validation Score: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print("Training Score: %.2f%% (+/- %.2f%%)\n" % (np.mean(trainingscores), np.std(trainingscores)))
    print("Batch size:", batch_size)
    print("Learning rate:", learning_rate)
    print("Number of epochs:",epoch)
    print("l1/l2 regularization",l1l2)
    print("Dropout:", dropout,"\n")
    
    if np.mean(cvscores)<lowest_error:
        
        lowest_error = np.mean(cvscores)
        stdev_error=np.std(cvscores)
        best_lr = learning_rate
        best_bs = batch_size
        best_epoch = epoch
        best_l1l2 = l1l2
        best_dropout = dropout
        best_training_error = np.mean(trainingscores)
        stdev_training_error = np.std(trainingscores)
        
    end = time.time()
    duration = end - start
    print("Execution Time of Neural Net is:", duration /60, "min\n")
    start = end


print("\n")
print("\n")
print("\n")


optimizer = RMSprop(lr=best_lr)
model=Sequential()
model.add(Dense(units=96, activation='sigmoid', input_dim=39, kernel_initializer=initializer,kernel_regularizer=l1_l2(l1=best_l1l2,l2=best_l1l2)))
model.add(Dropout(best_dropout))
model.add(Dense(units=96, activation='sigmoid', kernel_initializer=initializer,kernel_regularizer=l1_l2(l1=best_l1l2,l2=best_l1l2)))
model.add(Dense(units=48, activation='sigmoid', kernel_initializer=initializer,kernel_regularizer=l1_l2(l1=best_l1l2,l2=best_l1l2))) 
model.add(Dense(units=48, activation='sigmoid', kernel_initializer=initializer,kernel_regularizer=l1_l2(l1=best_l1l2,l2=best_l1l2))) 
model.add(Dense(units=1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

model.fit(X, Y, epochs=best_epoch, batch_size=best_bs, verbose=0)
Y_pred = model.predict(X_test)
Y_pred = y_pred.flatten()

try:
    error = metrics.mean_absolute_error(Y_test, Y_pred)
except:
    print("Input contains null values")
    



print("The best hyperparameters for %s iterations" % iterations)
print("Best batch size:", best_bs)
print("Best learning rate:", best_lr)
print("Best number of epochs:",best_epoch)
print("Best l1/l2 regularization",best_l1l2)
print("Best dropout:", dropout)
print("Validation Error:", lowest_error)
print("Validation Stdev:", stdev_error)
print("Training Error:", best_training_error)
print("Training Stdev:" ,stdev_training_error)
print("\n***************************************************************\n")

print("Test Score: %.2f%% " % (error))
print("Total execution time:", (time.time()-hard_start)/60,"min\n")
