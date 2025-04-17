import numpy as np
import pandas as pd
from time import perf_counter
from Classifier import *

def hz_sgn_sqrt(x):
    return x                         # !!REPLACE!! 

def hz_sgn_log(x):
    return x                         # !!REPLACE!! 

def hz_sgn_square(x):
    return x                         # !!REPLACE!!

def hz_cubic(x):
    return x                         # !!REPLACE!! 

# (I) Load data 
forestdata  = pd.read_csv('../../DATA/ForestTypes/ForestTypesData.csv'); # load data as pandas data frame 
classlabels = ['s','h','d','o'];                                      # possible class labels (C=4) 
classidx    = {classlabels[i]:i for i in range(len(classlabels))}     # dict for mapping classlabel to index 
C           = len(classlabels)        # number of classes (Note: K is now the number of nearest-neighbors!!!!!!)
T_txt = forestdata.values[:,0]           # array of class labels of data vectors (class label is first data attribute)
X = forestdata.values[:,1:]           # array of feature vectors (features are remaining attributes)
T = [classidx[t.strip()] for t in T_txt]          # transform text labels 's','h','d','o' to numeric lables 0,1,2,3
X,T=np.array(X,'float'),np.array(T,'int')  # convert to numpy arrays
N,D=X.shape                           # size and dimensionality of data set
print("Data set 'ForestData' has size N=", N, " and dimensionality D=",D, " and C=", C, " different classes")
print("X[0..9]=\n",X[0:10])
print("T_txt[0..9]=\n",T_txt[0:10])
print("T[0..9]=\n",T[0:10])

# (II) Test KNN-classifier with grid search on S-fold cross validation and K
if 1:
    S_list=[3]                            # parameter S for cross validation
    K_list=[1]                            # number K of nearest neighbors 
    accuracy = np.zeros((len(S_list),len(K_list)));   # array to save accuracy of classifier for each value of S and K
    minerr,bestS,bestK=1.0,-1,-1
    for i in range(len(S_list)):
        S=S_list[i]                        # do an S-fold cross validation
        for j in range(len(K_list)):
            K=K_list[j]
            t1=perf_counter()              # start time
            knnc = 0                       # !!REPLACE!! create KNN classifier with kd-trees 
            err,Cp = 1.0,0                 # !!REPLACE!! do S-fold cross validation for data X,T
            t2=perf_counter()              # end time
            time_comp=t2-t1                # computing time in seconds
            print("\nS=",S," fold cross validation using the",K,"-NNClassifier with KD-Trees yields the following results:")
            print("Classification error probability err = ", err)
            print("Accuracy = ", 1.0-err)
            print("Confusion Probabilities matrix Cp[i,j]=p(true class i,predicted class j) = \n", Cp)
            print("Computing time = ", time_comp, " sec")
            accuracy[i,j]=1.0-err
            if err<minerr:
                minerr=err
                bestS,bestK=S,K
    print("\naccuracy=\n",accuracy)
    print("\np_classerror=\n",1.0-accuracy)
    print("\nmininmal err=",minerr," for S=",bestS,"K=",bestK)

#exit(0)

# (III) Test KernelMLP classifier with grid search on S-fold cross validation and K
if 0:
    S_list=[3]                         # parameter S for cross validation
    hz_list=[hz_sgn_sqrt]              # activation functions in hidden layer  
    accuracy = np.zeros((len(S_list),len(hz_list)));   # array to save accuracy of classifier for each value of S and K
    minerr,bestS,best_hz=1.0,None,None
    for i in range(len(S_list)):
        S=S_list[i]                        # do an S-fold cross validation
        for j in range(len(hz_list)):
            hz=hz_list[j]
            print("S=",S,"hz=",hz)
            #!!REPLACE/INSERT YOUR CODE!!
    print("\naccuracy=\n",accuracy)
    print("\np_classerror=\n",1.0-accuracy)
    print("\nmininmal err=",minerr," for S=",bestS,"hz=",best_hz)

