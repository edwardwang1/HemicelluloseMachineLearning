#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint
from sklearn import metrics
from random import shuffle
from sklearn.model_selection import train_test_split


#Kinetic Characterization of Biomass
# Dilute Sulfuric Acid Hydrolysis:
# Mixtures of Hardwoods, Softwood,
# and Switchgrass


# In[7]:


#All

from sklearn import metrics
from openpyxl import Workbook
from openpyxl import load_workbook
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint
from scipy import optimize

acid_names = ['acetic', 'fenton', 'formic', 'malic', 'none', 'oxalic', 'phosphoric', 'sulfuric']
acid_protons = [1, _, 1, 1, 1, 1, 2, 2]
acid_mw = [60.852, _,  46.025, 134.09, 0, 90.034, 97.994, 98.079]

proton_dict = (zip(acid_names, acid_protons))
mw_dict = (zip(acid_names, acid_protons))

def getK (A, E, Ca, m, T):
    #A in min-1, E in kJ/mol, Ca in %, T in K
    if T > 450:
        T = 450
    if Ca > 0:
        return A * Ca ** m * math.exp(-E/(8.3143e-3 * T))
    else:
        return A * math.exp(-E/(8.3143e-3 * T))

    
def getX(k1, k2, H0, t):
#     print(k1, k2, H0, t)
#     num = -k1 * H0 * math.exp(t* (-k1 - k2)) * (math.exp(k2*t) - math.exp(k1*t))

    num = -k1 * H0 * (math.exp(-k1*t) - math.exp(-k2*t))
    denom = k1 - k2
    return num/denom


def get_error_train(params):
        #Params are [A1, E1, m1, A2, E2, m2]
        for i in train_frame.index:
    #         print(i)
        #for i in range(1):
            #converting mol proton/L to % (assuming sulfuric Acid)
            weightPer = train_frame.at[i, 'CA'] * 98.079 / 2 / 10
            #m = 1.75 for formation, 1 for degradation unless otherwise given

            #print(k1, k2)
            A1, E1, m1, A2, E2, m2 = params

            # initial conditions
            X0 = 0
            H0 = (train_frame.at[i, 'F_X'] / 100)/(train_frame.at[i, 'LSR'] + 1) * 1000

            k1 = getK(A=A1, E=E1, Ca=weightPer, m=m1, T=train_frame.at[i, 'Temp'])
    #         print(A1, E1, m1, A2, E2, m2, data.at[i, 'Temp'], weightPer)
            k2 = getK(A=A2, E=E2, Ca=weightPer, m=m2, T=train_frame.at[i, 'Temp'])

            X_sol = getX(k1=k1, k2=k2, H0=H0, t=train_frame.at[i, 'IsoT'])

            train_frame.at[i, 'Yield2'] = 100 * X_sol * train_frame.at[i, 'LSR'] / (1000 * (train_frame.at[i, 'F_X']/100))

        error = metrics.mean_absolute_error(train_frame['Yield'], train_frame['Yield2'])
        return error
    
def get_error_test(params):
        #Params are [A1, E1, m1, A2, E2, m2]
        for i in test_frame.index:
    #         print(i)
        #for i in range(1):
            #converting mol proton/L to % (assuming sulfuric Acid)
            weightPer = test_frame.at[i, 'CA'] * 98.079 / 2 / 10
            #m = 1.75 for formation, 1 for degradation unless otherwise given

            #print(k1, k2)
            A1, E1, m1, A2, E2, m2 = params

            # initial conditions
            X0 = 0
            H0 = (test_frame.at[i, 'F_X'] / 100)/(test_frame.at[i, 'LSR'] + 1) * 1000

            k1 = getK(A=A1, E=E1, Ca=weightPer, m=m1, T=test_frame.at[i, 'Temp'])
    #         print(A1, E1, m1, A2, E2, m2, data.at[i, 'Temp'], weightPer)
            k2 = getK(A=A2, E=E2, Ca=weightPer, m=m2, T=test_frame.at[i, 'Temp'])

            X_sol = getX(k1=k1, k2=k2, H0=H0, t=test_frame.at[i, 'IsoT'])

            test_frame.at[i, 'Yield2'] = 100 * X_sol * test_frame.at[i, 'LSR'] / (1000 * (test_frame.at[i, 'F_X']/100))

        error = metrics.mean_absolute_error(test_frame['Yield'], test_frame['Yield2'])
        return error

data = pd.read_csv("PreparedDataAll.csv")


initalGuessParams = [4.67e16, 142.58, 1.75, 6.51e16, 155.36, 1]

cvscores = []
trainingscores = []

for i in range(5):
    data = data.sample(frac=1).reset_index(drop=True)
    numData = len(data.index)
    numTrain = int(numData * 0.8)

    train_frame, test_frame, = data.iloc[:numTrain, :], data.iloc[numTrain:, :]

    initalGuessParams = [4.67e16, 142.58, 1.75, 6.51e16, 155.36, 1]
    output = optimize.fmin(get_error, initalGuessParams, maxiter=100, full_output=1)
    minimum = output[0]
    
    bestA1, bestE1, bestm1, bestA2, bestE2, bestm2 = minimum
    
    train_err = get_error_train(minimum)
    test_err = get_error_test(minimum)
    
    cvscores.append(test_err)
    trainingscores.append(train_err)
    
   
meanTest = np.mean(cvscores)
stdTest = np.std(cvscores)
cvscores.append(meanTest)
cvscores.append(stdTest)
        
meanTrain = np.mean(trainingscores)
stdTrain = np.std(trainingscores)
trainingscores.append(meanTrain)
trainingscores.append(stdTrain)


results = pd.DataFrame(columns = ['Lit Test', 'Lit Train'])
results['Lit Test'] = cvscores
results['Lit Train'] = trainingscores

results.to_csv("LitResults.csv")


# In[ ]:


#All

#Monomer
from sklearn import metrics
from openpyxl import Workbook
from openpyxl import load_workbook
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint
from scipy import optimize

acid_names = ['acetic', 'fenton', 'formic', 'malic', 'none', 'oxalic', 'phosphoric', 'sulfuric']
acid_protons = [1, _, 1, 1, 1, 1, 2, 2]
acid_mw = [60.852, _,  46.025, 134.09, 0, 90.034, 97.994, 98.079]

proton_dict = (zip(acid_names, acid_protons))
mw_dict = (zip(acid_names, acid_protons))

def getK (A, E, Ca, m, T):
    #A in min-1, E in kJ/mol, Ca in %, T in K
    if T > 450:
        T = 450
    if Ca > 0:
        return A * Ca ** m * math.exp(-E/(8.3143e-3 * T))
    else:
        return A * math.exp(-E/(8.3143e-3 * T))

    
def getX(k1, k2, H0, t):
#     print(k1, k2, H0, t)
#     num = -k1 * H0 * math.exp(t* (-k1 - k2)) * (math.exp(k2*t) - math.exp(k1*t))

    num = -k1 * H0 * (math.exp(-k1*t) - math.exp(-k2*t))
    denom = k1 - k2
    return num/denom


def get_error_train(params):
        #Params are [A1, E1, m1, A2, E2, m2]
        for i in train_frame.index:
    #         print(i)
        #for i in range(1):
            #converting mol proton/L to % (assuming sulfuric Acid)
            weightPer = train_frame.at[i, 'CA'] * 98.079 / 2 / 10
            #m = 1.75 for formation, 1 for degradation unless otherwise given

            #print(k1, k2)
            A1, E1, m1, A2, E2, m2 = params

            # initial conditions
            X0 = 0
            H0 = (train_frame.at[i, 'F_X'] / 100)/(train_frame.at[i, 'LSR'] + 1) * 1000

            k1 = getK(A=A1, E=E1, Ca=weightPer, m=m1, T=train_frame.at[i, 'Temp'])
    #         print(A1, E1, m1, A2, E2, m2, data.at[i, 'Temp'], weightPer)
            k2 = getK(A=A2, E=E2, Ca=weightPer, m=m2, T=train_frame.at[i, 'Temp'])

            X_sol = getX(k1=k1, k2=k2, H0=H0, t=train_frame.at[i, 'IsoT'])

            train_frame.at[i, 'Yield2'] = 100 * X_sol * train_frame.at[i, 'LSR'] / (1000 * (train_frame.at[i, 'F_X']/100))

        error = metrics.mean_absolute_error(train_frame['Yield'], train_frame['Yield2'])
        return error
    
def get_error_test(params):
        #Params are [A1, E1, m1, A2, E2, m2]
        for i in test_frame.index:
    #         print(i)
        #for i in range(1):
            #converting mol proton/L to % (assuming sulfuric Acid)
            weightPer = test_frame.at[i, 'CA'] * 98.079 / 2 / 10
            #m = 1.75 for formation, 1 for degradation unless otherwise given

            #print(k1, k2)
            A1, E1, m1, A2, E2, m2 = params

            # initial conditions
            X0 = 0
            H0 = (test_frame.at[i, 'F_X'] / 100)/(test_frame.at[i, 'LSR'] + 1) * 1000

            k1 = getK(A=A1, E=E1, Ca=weightPer, m=m1, T=test_frame.at[i, 'Temp'])
    #         print(A1, E1, m1, A2, E2, m2, data.at[i, 'Temp'], weightPer)
            k2 = getK(A=A2, E=E2, Ca=weightPer, m=m2, T=test_frame.at[i, 'Temp'])

            X_sol = getX(k1=k1, k2=k2, H0=H0, t=test_frame.at[i, 'IsoT'])

            test_frame.at[i, 'Yield2'] = 100 * X_sol * test_frame.at[i, 'LSR'] / (1000 * (test_frame.at[i, 'F_X']/100))

        error = metrics.mean_absolute_error(test_frame['Yield'], test_frame['Yield2'])
        return error

data = pd.read_csv("PreparedDataMonomer.csv")


initalGuessParams = [4.67e16, 142.58, 1.75, 6.51e16, 155.36, 1]

cvscores = []
trainingscores = []

for i in range(5):
    data = data.sample(frac=1).reset_index(drop=True)
    numData = len(data.index)
    numTrain = int(numData * 0.8)

    train_frame, test_frame, = data.iloc[:numTrain, :], data.iloc[numTrain:, :]

    initalGuessParams = [4.67e16, 142.58, 1.75, 6.51e16, 155.36, 1]
    output = optimize.fmin(get_error, initalGuessParams, maxiter=100, full_output=1)
    minimum = output[0]
    
    bestA1, bestE1, bestm1, bestA2, bestE2, bestm2 = minimum
    
    train_err = get_error_train(minimum)
    test_err = get_error_test(minimum)
    
    cvscores.append(test_err)
    trainingscores.append(train_err)
    
   
meanTest = np.mean(cvscores)
stdTest = np.std(cvscores)
cvscores.append(meanTest)
cvscores.append(stdTest)
        
meanTrain = np.mean(trainingscores)
stdTrain = np.std(trainingscores)
trainingscores.append(meanTrain)
trainingscores.append(stdTrain)


results = pd.DataFrame(columns = ['Lit Test', 'Lit Train'])
results['Lit Test'] = cvscores
results['Lit Train'] = trainingscores

results.to_csv("LitResultsMonomer.csv")

