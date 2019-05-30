from openpyxl import Workbook
from openpyxl import load_workbook
import os
import pandas as pd
import numpy as np
import math
import string

directory_in_str = "\\\\?\\C:\\Users\\Edward\\Desktop\\Projects\\HemicelluloseMachineLearning\\RawData\\"

#print(directory_in_str)
directory = os.fsencode(directory_in_str)

COLUMN_NAMES=['TotalT','Temp','LSR','CA','Size','Mass','Moisture', 'IsoT', 'HeatT', 'Ramp', 'F_A', 'F_Gal', 'F_Glu',
            'F_X', 'F_M', 'F_R', 'A', 'Gal', 'Glu','X', 'M', 'R', 'Furf', 'HMF', 'Source']

masterDF = pd.DataFrame(columns=COLUMN_NAMES)
#print(len(masterDF.columns))
#print(masterDF.columns)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    absPath = directory_in_str + filename
    if filename.endswith(".xlsx"): 
        wb = load_workbook(filename = absPath, data_only = True) 
        ws = wb["Data"]
        ws.delete_rows(1,2)
        df = pd.DataFrame(ws.values, columns=COLUMN_NAMES[:-1])
        df['Source'] = filename
        #print("1", df.at[1, 'Temp'])
        masterDF = pd.concat([masterDF, df], ignore_index=True)
        #masterDF = masterDF.append(df)
        #print("2",masterDF.at[1, 'Temp'])
        continue
    else:
        continue
        
        
#masterDF = masterDF.astype(float)
masterDF.reset_index()

#Deleting empty rows
rowsToDelete = []
for i in masterDF.index:
    if masterDF.at[i, 'X'] is None or pd.isnull(masterDF.at[i, 'X']) or masterDF.at[i, 'F_X'] is None or pd.isnull(masterDF.at[i, 'F_X']):
        rowsToDelete.append(i)

masterDF = masterDF.drop(rowsToDelete, axis=0)
masterDF.reset_index()

masterDF.to_csv("data.csv", index=False)

for i in masterDF.index:
    if masterDF.at[i, 'Ramp'] is None or pd.isnull(masterDF.at[i, 'Ramp']):
        masterDF.at[i, 'Ramp'] = (masterDF.at[i, 'Temp'] - 25) / masterDF.at[i, 'HeatT']
    if masterDF.at[i, 'HeatT'] is None or pd.isnull(masterDF.at[i, 'HeatT']):
        masterDF.at[i, 'HeatT'] = (masterDF.at[i, 'Temp'] - 25) / masterDF.at[i, 'Ramp']
    #print(type(masterDF.at[i, 'TotalT']))
    #print((masterDF.at[i, 'Temp']))
    if masterDF.at[i, 'TotalT'] is None or pd.isnull(masterDF.at[i, 'TotalT']):
        masterDF.at[i, 'TotalT'] =  masterDF.at[i, 'HeatT'] +  masterDF.at[i, 'IsoT']
    if masterDF.at[i, 'IsoT'] is None or pd.isnull(masterDF.at[i, 'IsoT']):
        masterDF.at[i, 'IsoT'] =  masterDF.at[i, 'TotalT'] -  masterDF.at[i, 'HeatT']
    if masterDF.at[i, 'TotalT'] == 0:
        masterDF.at[i, 'IsoT'] =  0
        masterDF.at[i, 'HeatT'] =  0
        
masterDF['TotalT'].fillna(masterDF.at[i, 'HeatT'] +  masterDF.at[i, 'IsoT'])

#Filling in Moisture Content
if masterDF.at[i, 'Moisture'] is None or pd.isnull(masterDF.at[i, 'Moisture']):
    if masterDF.at[i, 'Size'] > 5:
        masterDF.at[i, 'Moisture'] = 20
    else:
        masterDF.at[i, 'Moisture'] = 10
        
    

masterDF['X'].fillna(0, inplace=True)

#Converting Celsius to Kelvin
masterDF['Temp'] = masterDF['Temp'] + 273.15

#Making An Identifier Column to use instead of paper names
unique_papers = masterDF['Source'].unique()
paper_dict = dict(enumerate(unique_papers))
#swapping key and values
paper_dict = dict((v,k) for k,v in paper_dict.items())
alphabet = string.ascii_uppercase

masterDF['ID'] = masterDF['Source']
for i in masterDF.index:
    masterDF.at[i, 'ID'] = alphabet[paper_dict[masterDF.at[i, 'Source']]]

#This makes everything numeric
masterDF.to_csv("data.csv", index=False)
masterDF = pd.read_csv("data.csv")


#Creating Yield, Ro and P Factor
masterDF['Yield'] = 100 * masterDF['X'] * masterDF['LSR'] / (1000 * (masterDF['F_X']/100)) #1000 is density of water in g/L, X is in g/L

#P = exp(40.48 - 15106/T) * t, with T in Kelvin, t in hours
masterDF['P'] = np.exp(40.48 - 15106/masterDF['Temp']) * masterDF['IsoT']/60
masterDF['logP'] = masterDF['P']

#Ro = t * exp((T - 100)/14.75), t in minutes, T in Celsius
masterDF['Ro'] = masterDF['IsoT'] * np.exp(((masterDF['Temp'] -273.15) - 100)/14.75)
masterDF['logRo'] = masterDF['Ro']



for i in masterDF.index:
    if masterDF.at[i, 'logRo'] != 0:
        masterDF.at[i, 'logRo'] = np.log( masterDF.at[i, 'logRo'])
    if masterDF.at[i, 'logP'] != 0:
        masterDF.at[i, 'logP'] = np.log( masterDF.at[i, 'logP'])


masterDF.to_csv("data.csv", index=False)




