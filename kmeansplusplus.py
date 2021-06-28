# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 17:13:20 2021

@author: Roneet
"""
import numpy as np
import math
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from scipy.spatial import distance
import pandas as pd
import os


#Read csv file and return dataframe.
def read_csv_file(csv_file):
    data_frame= None
    if os.path.exists(csv_file):
            data_frame = pd.read_csv(csv_file)

    return data_frame

#Load file and remove timestamps for clustering.
integratedVectors = read_csv_file('C:/Users/Roneet/Documents/DTU/02461/BoolAndLocationData/integratedVectorsNoEmpties.csv')
integratedVectors = integratedVectors.loc[:, ~integratedVectors.columns.str.contains('^Unnamed')]
integratedVectors = integratedVectors.sort_values(by=['Timestamp']) 
del integratedVectors['Timestamp']
del integratedVectors['Bool']
integrals_matrix = integratedVectors.to_numpy()

