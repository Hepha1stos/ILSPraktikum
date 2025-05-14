#!/usr/bin/env python
# regression_airfoilnoise.py
# Musterloesung zu Versuch 2, Aufgabe 3c
# to log outputs start with: python regression_airfoilnoise.py >regression_airfoilnoise.log

import numpy as np
import pandas as pd

from polynomial_basis_functions import *
from Regression import *

# ***** MAIN PROGRAM ********
# (I) Hyper-Parameters
# (I.a) Hyper-Parameters for evaluation
seed=42           # define seed for random number generator
modeltype='lsr'   # define which model to evaluate (either 'lsr' or 'knn')
S=3               # do S-fold cross-validation
N_pred=5;         # number of predictions on the training set for testing
x_test_1 = [1250,11,0.2,69.2,0.0051];   # test vector 1
x_test_2 = [1305,8,0.1,57.7,0.0048];    # test vector 2
# (I.b) Hyper-Parameters for linear regression
deg=1             # degree of basis function polynomial phi(x) 
lmbda=0           # regularization parameter (lambda>0 avoids also singularities)
flagSTD=0         # if >0 then standardize data before training (i.e., scale X to mean value 0 and standard deviation 1)
eps=0.01          # parameter to recognize badly conditioned matrixes
# (I.c) Hyper-Parameters for KNN regression
K=1               # K for K-Nearest Neighbors
flagKLinReg = 0   # if flag==1 and K>=D then do a linear regression of the KNNs to make prediction
lr_deg=1          # degree of basis function polynomials for KNN-regression 
lr_lambda=0       # regularization parameter (lambda>0 avoids also singularities)
lr_flagSTD=0      # if >0 then standardize data before training (i.e., scale X to mean value 0 and standard deviation 1)
lr_eps=0.01       # parameter to recognize badly conditioned matrixes

# (II) Load data 
fname='../../DATA/AirfoilSelfNoise/airfoil_self_noise.xls'
airfoil_data = pd.read_excel(fname,0); # load data as pandas data frame 
T = airfoil_data.values[:,5]           # target values = noise load (= column 5 of data table)
X = airfoil_data.values[:,:5]          # feature vectors (= column 0-4 of data table)
N,D=X.shape                            # size and dimensionality of data set
idx_perm = np.random.permutation(N)    # get random permutation for selection of test vectors 
print("Data set ",fname," has size N=", N, " and dimensionality D=",D)
print("X=",X)
print("T=",T)
print("x_test_1=",x_test_1)
print("x_test_2=",x_test_2)

# (III) Create regression model 
if modeltype=='lsr':
    print("\n#### Least Squares Regression with regularization lambda=", lmbda, " ####")
    phi =None                          # !!! REPLACE THIS !!!  --> get polynomial basis functions
    regm=None                          # !!! REPLACE THIS !!!  --> create linear regression model (least squares with regularization)
elif modeltype=='knn':
    print("\n#### KNN regression with K=", K, ", flagKLinReg=", flagKLinReg, " ####")
    lr_phi=None                        # !!! REPLACE THIS !!!  --> get polynomial basis functions
    regm  =None                        # !!! REPLACE THIS !!!  --> create KNN regression model
else:
    assert 0, "unknown modeltype="+str(modeltype)
    
# (IV) Fit model and do evaluation/cross validation 
regm.fit(X,T)                                            # fit whole data set (just for demonstration)
if modeltype=='lsr':
    print("regm.W_LSR=",regm.W_LSR)                      # weight vector for least squares regression
    print("number of basisfunctions M=",len(regm.W_LSR)) # dimensionality of basis function space (=feature space)
print("IV.1) Some predictions on the training data:")
for i in range(N_pred): 
    n=idx_perm[i]
    print("Prediction for X[",n,"]=",X[n]," is y=",regm.predict(X[n]),", whereas true value is T[",n,"]=",T[n])
print("IV.2) Some predictions for new test vectors:")
print("Prediction for x_test_1 is y=", regm.predict(x_test_1))
print("Prediction for x_test_2 is y=", regm.predict(x_test_2))
print("IV.3) S=",S,"fold Cross Validation:")
MAE,MAPE = regm.crossvalidate(S,X,T)
print("MAE=",MAE, "MAPE=",MAPE) 


