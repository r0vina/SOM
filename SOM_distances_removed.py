# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:20:56 2021

@author: Roneet
"""

import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import euclidean_distances
from helperFunctions import read_csv_file
from helperFunctions import calculate_starting_centroids
from helperFunctions import write_csv_file
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

def neighbourhood_coordinates(winning_index, radius, weights_size):
    
    xs = np.arange(weights_size)
    ys = np.arange(weights_size)
    
    neighbourhood_coords = []
    for x in xs:
        for y in ys:
            coords = [x,y]
            dist = euclid_dist(coords, winning_index)
            if dist < radius:
                neighbourhood_coords.append(coords)
                
    return neighbourhood_coords
        
    

def valid_coords_check(coords, matrix):
    # print('check_coords',coords)
    good_coords = []
    for coord in coords:
        # print('coord in coords', coord)
        if coord[0] < 0 or coord[1] < 0:
            continue
        else:
            try:
                matrix[coord[0]][coord[1]]
            except:
                # print('Invalid Coords Found', coord[0],coord[1])
                continue
            else:
                good_coords.append(coord)
    # print(valid_coords, 'validCoords')
    # print(good_coords)
    return good_coords
            
            
            

def find(element, matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == element:
                return (i, j)
def cluster_assignment(matrix, centers):
    
    min_distance_index = np.argmin(euclidean_distances(matrix, centers), axis=0)
    return min_distance_index


def euclid_dist(array1, array2):
    distance = []
    for i in range(0, len(array1)):
        dist =(array1[i]-array2[i])**2
        # print(dist)
        distance.append(dist)
        # print(distance)
    return math.sqrt(np.sum(distance))
   
    
def spacio_temporal_decay(winning_index, coords, radius):
    
    distance = euclid_dist(winning_index, coords)
    decay = math.exp(-(distance**2)/2*(radius**2))
    
    return decay
    
def temporal_decay(iterations, iteration, radius):
    time_constant = iterations/radius
    new_radius = radius*math.exp(-(iteration/time_constant))
    return new_radius
    
    
averaged_vectors = read_csv_file('C:/Users/Roneet/Documents/DTU/02461/BoolAndLocationData/integratedVectors.csv')
location_and_time = read_csv_file('C:/Users/Roneet/Documents/DTU/02461/locationAndTimeClustered.csv')
location_and_time_clusters = location_and_time['Clusters']
location_and_time_session_length = location_and_time['sessionLengthInSeconds']

#Load file again, remove timestamps this time for processing.
av_minus_timestamp = read_csv_file('C:/Users/Roneet/Documents/DTU/02461/BoolAndLocationData/integratedVectors.csv')


averaged_vectors = averaged_vectors.loc[:, ~averaged_vectors.columns.str.contains('^Unnamed')]
av_minus_timestamp = av_minus_timestamp.loc[:, ~av_minus_timestamp.columns.str.contains('^Unnamed')]


av_minus_timestamp = av_minus_timestamp.sort_values(by=['Timestamp']) 

del av_minus_timestamp['Timestamp']
av_minus_timestamp['sessionLengthInSeconds'] = location_and_time_session_length
av_minus_timestamp = av_minus_timestamp.sort_values(by=['sessionLengthInSeconds']) 
# print(av_minus_timestamp)
del av_minus_timestamp['sessionLengthInSeconds']
av_matrix = av_minus_timestamp.to_numpy()

scaler = StandardScaler()
scaler.fit(av_matrix)
print(scaler.transform(av_matrix))
av_matrix = scaler.transform(av_matrix)
weights = np.random.rand(3,3,8)
weights_size = len(weights)
winning_weights = []
radius = 3
# print('before', weights)
learning_rate = 0.9
iterations = 10000
weights_norm = np.linalg.norm(weights, axis=2)
pcm = ax.pcolormesh(weights_norm, cmap='hot')
ax.set_title('before')
plt.show()
for i in range (0, iterations):
    print(radius, ':radius', i, ':index')
    # print(i)
    for row in av_matrix:
        distance = 0
        x_coor = 0
        y_coor = 0
        # distances = np.random.rand(len(weights),len(weights))
        for rowIndex in range(len(weights)):
            for elementIndex in range(len(weights[rowIndex])):
         
                dist = euclid_dist(row, weights[rowIndex][elementIndex])
                if distance and (dist <= distance):
                    distance = dist
                    x_coor = rowIndex
                    y_coor = elementIndex
        
                
                # input('Press enter 1')
        # print('distances', distances)
        winning_index = [x_coor,y_coor]
        
        # print(distances)
        # print('winner',weights[winning_index])
        # print('winnerIndex',winning_index, i)
        

    winner = weights[winning_index]
    neighbour_coords = neighbourhood_coordinates(winning_index, radius, weights_size)
    # print(neighbour_coords, 'neighbour')
    
    valid_coords = valid_coords_check(neighbour_coords, weights)
    # print('valid_coords', valid_coords)
    radius = temporal_decay(iterations, i, len(weights))
    
    for coords in valid_coords:

        decay = spacio_temporal_decay(winning_index, coords, radius)
        weights[coords[0]][coords[1]] = weights[coords[0]][coords[1]] + (learning_rate*decay)*(row - weights[coords[0]][coords[1]])
    learning_rate = learning_rate * math.exp(-(i/iterations))           
    #weights[winning_index] = weights[winning_index] + learning_rate*(row-weights[winning_index])
weights_norm = np.linalg.norm(weights, axis=2)
  
fig2, ax2 = plt.subplots()
ax2.set_title('after')
ax2.pcolormesh(weights_norm, cmap='hot') 
plt.show()
weights.tofile('weights')   
# print('after', weights)
# print(len(winning_weights))



# index = cluster_assignment(av_matrix, weights)
# print(np.size(index))
# print(weights[index])