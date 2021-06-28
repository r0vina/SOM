# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 22:24:38 2021

@author: Roneet
"""
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import euclidean_distances

def read_csv_file(csv_file):
    data_frame= None
    if os.path.exists(csv_file):
            data_frame = pd.read_csv(csv_file)

    return data_frame


def standardize(array):
    
    array = np.array(array)
    standard_deviation = np.std(array)
    mean = np.mean(array)
    
    standardized_array = []
    for element in array:
        standardized_element = (element - mean)/standard_deviation
        standardized_array.append(standardized_element)
    
    return standardized_array

def cluster_assignment(matrix, centers):
    
    min_distance_index = np.argmin(euclidean_distances(matrix, centers), axis = 1)
    return min_distance_index

def calculate_starting_centroids(K, matrix):
    
    centers = []
   
    for array in np.array_split(matrix, K):

        center = array.mean(0)
        center = center.tolist()
        # print('center',center)
        centers.append(center)
    
    return centers
def write_csv_file(csv_filename, data_frame):
    data_frame.to_csv(csv_filename)
    print(csv_filename + " created.")
