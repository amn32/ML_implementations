import numpy as np
import random
import pandas as pd
import os
import gc
from sklearn.model_selection import train_test_split

def load_data(dataset, train_size = 10000, test_size = 0, standardise = False, validation_size = 0, noise = False):

    X_val = None
    Y_val = None
    
    if dataset == 'Sarcos':

        sarcos = pd.read_csv('sarcos_inv.csv',header=None).values

        X_vals = sarcos[:,:21]
        Y_vals = sarcos[:,[21]]
        
        del sarcos
        gc.collect()

    else:
        
        random.seed(50)

        n = int(train_size + test_size)

        X_vals = np.random.uniform(low = 0, high = 1, size = (n, 3))

        W  = [[10], [2], [20]]
        b  = [5]

        if dataset == 'L_toy':
            Y_vals = X_vals.dot(W) + b
        elif dataset == 'NL_toy':
            Y_vals = X_vals.dot(W)**2/10
        else:
            print('Incorrect dataset name')
            return
        
        if noise:
            
            Y_vals += np.random.normal(0,5,[len(Y_vals)]).reshape(-1,1)

        Y_vals = Y_vals.reshape(-1, 1)
        
    X_train, X_test, Y_train, Y_test = train_test_split(X_vals, Y_vals, train_size = train_size, random_state = 50)

    if standardise == True:
        mean = X_train.mean(axis = 0)
        std = X_train.std(axis = 0)

        X_train = (X_train - mean)/std
        X_test = (X_test - mean)/std
        
    if validation_size > 0:
        X_val, X_train, Y_val, Y_train = train_test_split(X_train, Y_train, train_size = validation_size)

    if test_size>0:

        X_test = X_test[:test_size]
        Y_test = Y_test[:test_size]
        
    print(f'{dataset} Data Loaded')
    
    return X_train, X_test, X_val, Y_train, Y_test, Y_val