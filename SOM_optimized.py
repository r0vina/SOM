# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:20:56 2021

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
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()


#Unified distance matrix
def udistmatrix(matrix):
    umatrix_width = matrix.shape[0]+(matrix.shape[0]-1)
    
    umatrix = np.zeros([umatrix_width, umatrix_width])
    umatrix1 = np.zeros([umatrix_width, matrix.shape[0]])
    umatrix2 =  np.zeros([umatrix_width,matrix.shape[0]])
    distance_rows = np.linalg.norm(np.diff(matrix, axis=0), axis = 2)
    print(umatrix1.shape)
    umatrix1[1::2] = distance_rows
    umatrix1 = umatrix1.T
    # print(distance_rows.shape)
    print(umatrix.shape)    
    # umatrix = umatrix.T
    # print(np.diff(umatrix, axis=0))
    matrix = np.transpose(matrix,(1,0,2))
    distance_columns = np.linalg.norm(np.diff(matrix, axis=0), axis = 2)
    umatrix2[1::2] = distance_columns
    umatrix2 = np.transpose(umatrix2)
    
    umatrix[::2] = umatrix2
    umatrix = umatrix.T
    umatrix[::2] = umatrix1
    # print(umatrix1,umatrix2) 
    
    
    
    for i in range(0, len(umatrix)):
        for j in range(0, len(umatrix[i])):
            mean_array = []
            if not umatrix[i][j]:
                if(j-1)>0 and (j-1) < len(umatrix):
                  mean_array.append(umatrix[i][j-1])
                if (j+1)>0 and (j+1) <len(umatrix):
                  mean_array.append(umatrix[i][j+1])
                if(i-1)>0 and (i-1) < len(umatrix):
                    mean_array.append(umatrix[i-1][j])
                if(i+1>0) and (i+1) < len(umatrix):
                    mean_array.append(umatrix[i+1][j])
                if sum(mean_array):
                    mean = np.mean(mean_array)
                    umatrix[i][j] = mean
    umatrix = umatrix.T              
    return umatrix
 
    
    
#Read csv file and return dataframe.
def read_csv_file(csv_file):
    data_frame= None
    if os.path.exists(csv_file):
            data_frame = pd.read_csv(csv_file)

    return data_frame



# Function used to exponentially decay the radius and the learning rate over iterations
# The time constant is set to 10% of the number of iterations.
def temporal_decay(iterations, iteration, value):
    time_constant = iterations*(10/100)
    new_value = value*math.exp(-(iteration/time_constant))
    return new_value

# User-defined initial learning rate, radius (which describes the winner's neighbourhood)
# and the length of the output nodes grid (a square)

learning_rate_init = 0.9
output_row_length = 16
radius_init = output_row_length*(60/100)


#Load file and remove timestamps for clustering.
integratedVectors = read_csv_file('C:/Users/Roneet/Documents/DTU/02461/BoolAndLocationData/integratedVectorsNoEmpties.csv')
integratedVectors = integratedVectors.loc[:, ~integratedVectors.columns.str.contains('^Unnamed')]
integratedVectors = integratedVectors.sort_values(by=['Timestamp']) 
del integratedVectors['Timestamp']
del integratedVectors['Bool']
integrals_matrix = integratedVectors.to_numpy()


# # Scale input to between 0 and 1
# scaler = MinMaxScaler()
# scaler.fit(integrals_matrix)
# integrals_matrix = scaler.transform(integrals_matrix)

# Normalize input
integrals_matrix = normalize(integrals_matrix, norm=  'l1')


# Generate random weights

weight_length = len(integrals_matrix[0])
# weights = np.random.randn(output_row_length**2, weight_length)
random_indices = np.random.randint(199, size=output_row_length**2)

# Uncomment here to use vectors from input as matrix
weights = integrals_matrix[random_indices[:output_row_length**2]]
weights = np.reshape(weights,(output_row_length,output_row_length,weight_length))

# scaler2 = MinMaxScaler()
# scaler2.fit(weights)
# weights = scaler2.transform(weights)
# weights = np.reshape(weights,(output_row_length,output_row_length,weight_length))

weights_size = len(weights)
winning_weights = []
radius = radius_init
learning_rate = learning_rate_init
iterations = 10000
weights_norm = np.linalg.norm(weights, axis=2)

random_indices = np.arange(0, integrals_matrix.shape[0])
np.random.shuffle(random_indices)

training_data = integrals_matrix[random_indices[:150]]
# training_data = integrals_matrix
ax.pcolor(weights_norm,cmap='binary')


# Make an array of valid coordinates for each output node. Used to measure the 
# neighbourhood of the winner and assign weight changes to the relevant nodes.
start = 0
end = weights.shape[0]
step = 1
points = np.arange(start, end, step)  # [0.5, 1.5, 2.5, 3.5, 4.5]
weight_coords = np.c_[np.repeat(points, len(points)), np.tile(points, len(points))]

# Folding out the weights 3D matrix, to make it usable with numpy/scipy vectorization.
# This is done to avoid excessive looping which slows down the script.
weights2D = np.reshape(weights,(output_row_length**2, weight_length))

# Initial weights saved to a variable, so it can be compared with the result (for testing)
weights2Dinitial = weights2D


#Main loop - measures the distance of the vectors from the input array to the weights of each output 
# node. The node that is closest is chosen as the winner, and its weight is adjusted. The neighbouring
# nodes to the winner also have their weights adjusted (value of the change decays over distance from
# the winner). After each iteration, the radius that describes a winning node's neighbourhood as well as
# the learning rate is "decayed". Clusters are formed due to similar weights being increased together.

for i in range (0, iterations):

     for row in training_data:
         
         #row turned to a 2D matrix, and compared with all weights. The index 
         #of the weight with the shortest distance from the row is saved to a 
         #variable.
         row = [row]
         winning_index_2D = np.argmin(distance.cdist(row,weights2D))

         #The weight that wins is saved.
         winner = weights2D[winning_index_2D]
         
         #weights array is folded back to 3D to find the coordinates of the winning node.
         weights = np.reshape(weights2D,(output_row_length,output_row_length,weight_length))

         winning_index_3D = np.where(weights == winner)
         winindex = winning_index_3D
         # print(winindex)
         winning_index_3D = [winning_index_3D[0][0], winning_index_3D[1][0]]
         winning_index_3D = np.reshape(winning_index_3D, (1, len(winning_index_3D)))
         
         # Array which holds the distances of the winning node to all other nodes.
         # Note: this is not the same as the weights distance from before, this is 
         # strictly in 2D and is used to find neighbouring nodes and assign weight changes.
         cdistarray = distance.cdist(winning_index_3D,weight_coords)
         
         
         radius = temporal_decay(iterations, i, radius_init)
         decaymatrix = np.exp(-(cdistarray**2)/2*(radius**2))

         decaymatrix[0][np.where(cdistarray > radius)[1]] = 0
         row_array = np.full((weights2D.shape),row)
         
         weight_change = (learning_rate*decaymatrix[0])*((row_array - weights2D).T)
         
         weights2D = weights2D.T + weight_change
         
                   
         weights2Dnew = weights2D
 
         weights2D = weights2D.T
         
         winner_change = learning_rate*(row-winner)
         weights2D[winning_index_2D] = winner + winner_change
         
     if not i%1000:
         weights = np.reshape(weights2D,(output_row_length,output_row_length,weight_length))
         umat = udistmatrix(weights)

         ax3.pcolor(umat, cmap='Greys')
         plt.show()

         #print(np.sum(weights2Dinitial-weights2D))
         print(radius, i, winner_change, learning_rate)
         
     # learning_rate = 1/(i+1)
     # learning_rate = learning_rate_init * math.exp(-(i/iterations))
     learning_rate = temporal_decay(iterations, i, learning_rate_init)
     
     # Artificial convergence check, as SOMs do not have a target function.
     if learning_rate ==0:
         break
coordict = {}
for row in training_data:
    row = [row]
    winning_index_2D = np.argmin(distance.cdist(row,weights2D))
    
    #The weight that wins is saved.
    winner = weights2D[winning_index_2D]
    
    #weights array is folded back to 3D to find the coordinates of the winning node.
    weights = np.reshape(weights2D,(output_row_length,output_row_length,weight_length))
    
    winning_index_3D = np.where(weights == winner)
    winning_index_3D = [winning_index_3D[0][0], winning_index_3D[1][0]]
    coordict[str(row)] = winning_index_3D
    
    
weights = np.reshape(weights2D,(output_row_length,output_row_length,weight_length))
weights_norm_end = np.linalg.norm(weights, axis=2)

# scaler3 = MinMaxScaler()
# scaler3.fit(weights_norm_end)
# weights_norm_end = scaler3.transform(weights_norm_end)
# print(weights_norm_end)
# r, g, b =[weights_norm_end for _ in range(3)]
# c = np.dstack([r,g,b])
ax2.pcolor(weights_norm_end, cmap='binary')
plt.show()
weights.tofile('weights')   


# ax.set_title("Kohonen-Clustering of 30 random session vectors.\n"+size+"x"+size+" output nodes, 10K iterations.", fontsize=20)



# weights500 = np.fromfile('som/500kweights')
# # print(weights500)

# scaler = MinMaxScaler()
# scaler.fit(umat)
# umat = scaler.transform(umat)

