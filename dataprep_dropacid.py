#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:16:36 2019

@author: RileyBallachay

First Version of data analysis and scaling 
"""

import pandas as pd
import numpy as np



from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import math
import pdb

# Function for calculating the average acetyl content of each wood type
def calculate_avg(name,X,acetyl_mean):
        name_frame = X[X.Wood== name] 
        avg = name_frame['Acetyl'].mean()
        if math.isnan(avg):
            if name == 'pine':
                avg=2.0
            else:
                avg = acetyl_mean
        return avg
 
# Function to one-hot encode categorical variables and return binary array      
def onehot(X):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(X)
    OneHotArray = enc.transform(X).toarray()
    categories = enc.categories_[0].tolist()
    OneHotArray = pd.DataFrame(OneHotArray,columns=categories)
    return OneHotArray

# Function to impute missing values in input dataframe
def impute_missing(X):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(X)
    X = imp_mean.transform(X)
    return X

# Scales data to unit variance
def scale_data(X):
    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)
    X=pd.DataFrame(data=X)
    return X



# Main function to prepare data
def prep(X,boolean,acid_wood):
    # Added functionality to choose whether or not you want to rescale data from loaded CSV or use pre-scaled data
    # Will default to True 
    if boolean == True:  
        try: 
            data = pd.read_csv("Prepared_Data.csv")
            print("\nSuccessfully loaded prepared data, skipping data prep\n")
            Y_data = data['Yield']
            X_titles = list(data.columns)
            X_titles.remove('Yield')
            del X_titles[0]
            X_data = np.array(data.drop('Yield',axis=1))
            X_data = np.delete(X_data,0,1)
        except:
            boolean = False
            print("\nCannot load prescaled data from Prepared_Data.csv\n")
            print("\nProceeding with data scaling\n")
    
    # Create new numpy arrays, with string datatype to hold acid and wood types
    if boolean ==False:
        
        Acid = np.reshape(np.array(X['Acid'],dtype=str),(-1,1))
        Wood = np.reshape(np.array(X['Wood'],dtype=str),(-1,1))
        
        # Converting all names to lowercase and stripping whitespace
        # Need to add code to check that wood types match list of included wood types 
        # and alter the name to fit the known name is it is mispelled
        Acid = np.char.lower(np.char.rstrip(Acid))
        Wood = np.char.lower(np.char.rstrip(Wood))
    
        
        # Import known acetyl concentrations and compute and store average
        Acetyl=pd.DataFrame()
        Acetyl['Acetyl'] = X['Acetyl']
        acetyl_mean = Acetyl['Acetyl'].mean()
        
        # One-hot encoding of acid types
        # Overwrites initial acid concentration array 
        # Worth considering if original array should be preserved
        
        Acid = onehot(Acid)
        print('The acid types in the data include:', Acid.columns)    
       
        Wood = onehot(Wood)
        print('The wood types in the data include:', Wood.columns)
        
        # acid_dictionary = {'sulfuric':np.mean([-3.0,1.92]),'oxalic':np.mean([1.27,4.27]),'phosphoric':np.mean([2.1,7.2]),
           #                'acetic':4.75,'formic':3.75,'malic':1.9,'none':14}
        
        # X['Acid pKa'] = X['Acid'].map(acid_dictionary)
        # Creating acetyl concentration dictionary
        # Currently set to fill unknown wood-type acetyl concentration with mean
        # Need to update to contain known acetyl concentration of wood types from literature
        

        acetyl_dictionary = {}
            
        for name in list(Wood.columns):
            avg = calculate_avg(name,X,acetyl_mean)
            acetyl_dictionary[name] = avg
        
        # Adding new acetyl column with mapped values and then reordering to move yield to the end of the dataframe
        # Dropping acid and wood categorical variables from dataframe
        X.drop('Acetyl',axis=1,inplace=True)
        X.drop('Acid',axis=1,inplace=True)
        X['Acetyl'] = X['Wood'].map(acetyl_dictionary) 
        X.drop('Wood',axis=1,inplace=True)
        
        # Creating new array with yields and dropping 
        Y_data = (X['Yield'])
        X.drop('Yield',axis=1,inplace=True)
        X_columns=(list(X.columns))
        X['Ramp'] = X['Ramp'].replace(np.inf,np.nan)
        
        # Make full set of column titles to add to final, returned X-data dataframe 
        X_titles = X_columns + list(Acid.columns) + list(Wood.columns)  
        
        # Perform imputation and scaling on data 
        # Will make this a little more advanced in the near future 
        X_data = impute_missing(X)
        X_data = scale_data(X_data)
        
        # Concatenate dataframes to make final data to pass to learning algorithms
        # Make full set of column titles to add to final, returned X-data dataframe 
        if acid_wood == 'Acid':
            X_titles = X_columns + list(Wood.columns) 
            X_data = pd.concat([X_data,Wood], ignore_index=True,axis=1)
       
        elif acid_wood == 'Wood':
            X_titles = X_columns + list(Acid.columns)
            X_data = pd.concat([X_data,Acid], ignore_index=True,axis=1)
        
        else:
            X_titles = X_columns + list(Acid.columns) + list(Wood.columns) 
            X_data = pd.concat([X_data,Acid,Wood], ignore_index=True,axis=1)
        
        
        X_pandas = X_data
        X_pandas.columns = (X_titles)
        X_data = np.array(X_data)
        data = pd.concat([X_pandas,Y_data],axis=1)
        
        # Saves prepared data in the same directory as this script
        data.to_csv("Prepared_Data.csv")
        
        # Important to note that X_data is passed as numpy array while Y_data is left as pandas dataframe
    return X_data, Y_data, data,X_titles