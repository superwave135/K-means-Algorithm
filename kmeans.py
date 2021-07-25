### Name: Tan Ngiap Chuan Alvin ###
# AML Practicum 1: K-means Algorithm # 

# import the libraries for programming
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
## -----------------------------------------------------------
def loadData(name):
    '''
    Objective: To load the file and convert to numpy array
    Input: filename (string)
    Output: X dataset (array) 
    '''
    df = pd.read_csv(name, delimiter = "\t", header = None)
    return df.to_numpy()
## -----------------------------------------------------------
file_name = '2010825.txt'                                # get the file name
col_3 = np.zeros(16259).reshape(16259, 1)                # create all zeros in col_3
X = np.concatenate((loadData(file_name), col_3), axis=1) # concatenate col_3 to X horizontally 
## -----------------------------------------------------------
def errorCompute(X, M):
    '''
    Objective: To compute the error as L2 norm
    Input: X dataset of 2 features + 1 ClusterID, mean value (M) for each cluster
           (both are arrays)
    Output: The value of objective function for the clustering (float)
    '''
    l2 = np.linalg.norm(X[:, :2] - M[X[:, 2].astype(int)], axis=1)
    return np.mean(l2)
## -----------------------------------------------------------
def Group(X, M):
    '''
    Objective: Assign each object into a cluster bases on the current set of means (M)
    Input: X dataset of 2 features + 1 ClusterID, current mean (M)
           (both are arrays)
    Output: X with updated clusterID. Assign each object into its closest cluster
    '''
    for idx, i in enumerate(X[:,0:2]):                # iterate over 10 rows of X, col 1 and col 2
        l2_norm = np.sqrt(np.square(i[0] - M[:,0]) + np.square(i[1] - M[:,1])) # get the l2 norm value
        X[idx, 2] = np.argmin(l2_norm)                                         # populates col_2 with cluster IDs 
    return X
## -----------------------------------------------------------
def calcMeans(X, M):
    '''
    Objective: Update the means (M) until there is no changes in clustering result
    Input: X dataset of 2 features + 1 ClusterID, current mean (M)
           (both are arrays)
    Output: The updated mean value(M) for each cluster
    '''
    for count in range(100):
        X = Group(X, M) # call the Group function
        centroid_XY = {i:[] for i in range(0,M.shape[0])} # create dict for xy_coords, K = M.shape[0]
        for i in X[:,0:2]:                # iterate over 10 rows of X, col 1 and col 2
            l2_norm = np.sqrt(np.square(i[0] - M[:,0]) + np.square(i[1] - M[:,1])) # get the l2 norm value
            centroid_XY[np.argmin(l2_norm)].append(i)                              # append xy_coord to their k-v pair
   
        old = M                                           # assign current M centroid matrix to old var b4 update M
        centroid_list = [np.array(i) for i in centroid_XY.values()] # extract xy coords of each centroid into a list
        M = np.array([0,0])           # dummy array for vstacking purpose
        for xy in centroid_list:      # for each pair of xy_coord in list
            x_pts = xy.T[0]           # get all x_coords of a centroid
            y_pts = xy.T[1]           # get all y_coords of a centroid
            row = np.array([np.mean(x_pts), np.mean(y_pts)]) # assign xy_mean pairs to row
            M = np.vstack((M, row))   # create M matrix

        M = M[1:]                     # get rid of dummy array
        print(f'Running {count+1} epochs in progress ...')
        if np.array_equal(old, M):  # if no change in centroids' XY coords, break loop 
            print(f'Centroids unchanged after {count+1} epochs\n')
            print(f'M after {count+1} epoch:\n{M}\n')
            break
    return M 
## -----------------------------------------------------------
print('--------------- PROGRAM STARTS -----------------')
print('-------------------------------------------')
M = np.array([[0,0]]) # use (0,0) as the initial mean 
print(f'Using M=[0,0] the error is {errorCompute(X, M)}\n')
print('-------------------------------------------')
## -----------------------------------------------------------
M=np.copy(X[0:5,0:X.shape[1]-1]) # use first 5 rows of XY as initial means
X = Group(X, M)
error = errorCompute(X, M) 
print(f'Using first 5 XY-coords of X as centroids, the error is {error}\n')
print('-------------------------------------------')
## -----------------------------------------------------------
print('Computing M and error when K=5 begins ...\n')
M = calcMeans(X, M) # Run k-means with K=5
error = errorCompute(X, M) # compute the error when K=5
print(f'After running calcMeans() and Group(), the error is {error}\n')
print('Computing M and error when K=5 completed.\n')
print('-------------------------------------------')
## -----------------------------------------------------------
print('Computing M and error when K=50 begins ...\n')
M_50_initial = np.copy(X[0:50,0:X.shape[1]-1]) # use first 50 rows of XY as initial means
X = Group(X, M_50_initial)
M_50_final = calcMeans(X, M_50_initial)
error_50 = errorCompute(X, M_50_final) # compute the error when K=50
print(f'After running K=50, the error is {error_50}\n')
print('Computing M and error when K=50 completed.\n')
print('-------------------------------------------')
## -----------------------------------------------------------
print('Computing M and error when K=100 begins ...\n')
M_100_initial = np.copy(X[0:100,0:X.shape[1]-1]) # use first 100 rows of XY as initial means
X = Group(X, M_100_initial)
M_100_final = calcMeans(X, M_100_initial)
error_100 = errorCompute(X, M_100_final) # compute the error when K=100
print(f'After running K=100, the error is {error_100}\n')
print('Computing M and error when K=100 completed.\n')
print('--------------- PROGRAM ENDS -----------------')
## -----------------------------------------------------------
## Program ends ##
