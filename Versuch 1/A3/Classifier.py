#!/usr/bin/env python
# Python Module for Classification Algorithms
# Musterloesung zu Versuch 1, Aufgabe 2
import numpy as np
import scipy.spatial
import KNearestNeighborSearch as KNNS

# ----------------------------------------------------------------------------------------- 
# ----------------------------------------------------------------------------------------- 
# Base class for classifiers
# ----------------------------------------------------------------------------------------- 
# ----------------------------------------------------------------------------------------- 
class Classifier:
    """
    Abstract base class for a classifier.
    Inherit from this class to implement a concrete classification algorithm
    """

    def __init__(self,C=2): 
        """
        Constructor of class Classifier
        Should be called by the constructors of derived classes
        :param C: Number of different classes
        """
        self.C = C                     # set C=number of different classes 

    def fit(self,X,T):    
        """ 
        Train classier by training data X, T, should be overwritten by any derived class
        :param X: Data matrix, contains in each row a data vector
        :param T: Vector of class labels, must have same length/row number as X, each label should be integer in 0,1,...,C-1 
        :returns: - 
        """
        shapeX,shapeT=X.shape,T.shape  # X must be a N x D matrix; T must be a N x 1 matrix; N is number of data vectors; D is dimensionality
        assert len(shapeX)==2, "Classifier.fit(self,X,T): X must be two-dimensional array!"
        assert len(shapeT)==1, "Classifier.fit(self,X,T): T must be one-dimensional array!"
        assert shapeX[0]==shapeT[0], "Classifier.fit(self,X,T): Data matrix X and class labels T must have same length!"
        minT,maxT=np.min(T),np.max(T)
        assert minT>=0 and maxT<self.C, "Labels T[n] should be between 0 and C-1, but C="+str(self.C)+", minT="+str(minT)+", maxT="+str(maxT)

    def predict(self,x):
        """ 
        Implementation of classification algorithm, should be overwritten in any derived class
        :param x: test data vector
        :returns: label of most likely class that test vector x belongs to (and possibly additional information)
        """
        return -1,None,None

    def crossvalidate(self,S,X,T):
        """
        Do a simple S-fold cross validation (no stratification/class balancing is done!)
        :param S: Number of parts the data set is divided into
        :param X: Data matrix (one data vector per row)
        :param T: Vector of class labels; T[n] is label of X[n]
        :returns err: probability of a classification error (=1-Accuracy)
        :returns matCp: confusion probability matrix, matCp[i,j]=p(true class=i,predicted class=j] is joint probability of true class i and predicted class j 
        """
        X,T=np.array(X),np.array(T,'int')                   # make sure we have numpy arrays
        N=len(X)                                            # N=number of data vectors
        perm = np.random.permutation(N)                     # get permutation of index list [0,1,...,N-1] to get random selections of X and T
        idxS = [range(i*N//S,(i+1)*N//S) for i in range(S)] # indexes for dividing data set into S parts of equal size
        matCp = np.zeros((self.C,self.C))                   # initialize confusion probability matrix Cp[i,j]=pr[true class i and predicted class j]
        err = 0                                             # initialize probability of a classification error
        for idxVal in idxS:                                 # loop over all S validation data sets
            # (i) generate training and testing data sets and train classifier        
            if S>1: idxTrain = [i for i in range(N) if i not in idxVal]               # remaining indices (not in idxVal) are training data
            else  : idxTrain = idxVal                                                 # if S==1 use entire data set for training and validation
            self.fit(X[perm[idxTrain]],T[perm[idxTrain]])                             # train classifier using (permutated) indexes of training data            
            # (ii) evaluate classifier using validation data
            for i in range(len(idxVal)):  # loop over all validation indexes
                y_hat = self.predict(X[perm[i]])[0]     # predicted class of i-th input vector from validation set 
                t_true = T[perm[i]]                     # corresponding true class label
                matCp[t_true,y_hat]+=1                  # increase component of confusion matrix 
                if(y_hat!=t_true): err+=1               # increase counter of errors
        matCp=(1.0/N)*matCp    # divide by data number to get confusion probability matrix
        err=err/float(N)       # divide by data number to get error probability 
        return err,matCp       # return error and class confusion probability matrix


# ----------------------------------------------------------------------------------------- 
# ----------------------------------------------------------------------------------------- 
# (Naive) K-Nearest-Neighbor classifier based on simple look-up-table and exhaustive search
# ----------------------------------------------------------------------------------------- 
# ----------------------------------------------------------------------------------------- 
class KNNClassifier(Classifier):
    """
    (Naive) k-nearest-neighbor classifier based on simple look-up-table and exhaustive search
    Derived from base class Classifier
    """

    def __init__(self,C=2,K=1):
        """
        Constructor of the KNN-Classifier
        :param C: Number of different classes
        :param k: Number of nearest neighbors that classification is based on
        """
        Classifier.__init__(self,C) # call constructor of base class  
        self.K = K                  # K is number of nearest-neighbors used for majority decision
        self.X, self.T = [],[]      # initially no data is stored

    def fit(self,X,T):
        """
        Train classifier; for naive KNN Classifier this just means to store data matrix X and label vector T
        :param X: Data matrix, contains in each row a data vector
        :param T: Vector of class labels, must have same length as X, each label should be integer in 0,1,...,C-1
        :returns: - 
        """
        Classifier.fit(self,X,T);                        # call to base class to check for matrix dimensions etc.
        self.X, self.T = np.array(X),np.array(T,'int')   # just store the N x D data matrix and the N x 1 label matrix (N is number and D dimensionality of data vectors)
    
    def predict(self,x,K=None,idxKNN=None):
        """ 
        Implementation of naive KNN classification algorithm
        :param x: input data vector
        :param K: search K nearest neighbors (default self.K)
        :param idxKNN: indexes of K nearest neighbors (if None then compute indexes with naive KNN algorithm)
        :returns y_hat: label of most likely class that test vector x belongs to
        :returns pc:     A-Posteriori probabilities: pc[i]=pr[class c|input x] is the probability that input x belongs to class i
        :returns idxKNN: indexes of the K nearest neighbors (ordered w.r.t. ascending distance) 
        """
        if K      is None: K      = self.K                                   # use default parameter K?
        if idxKNN is None: idxKNN = None                                     # !!REPLACE!! get indexes of k nearest neighbors of x (in case idxKNN is not already defined)
        pc     = np.ones(self.C)/self.C                                      # !!REPLACE!! get a-posteriori class probabilities
        y_hat  = 0                                                           # !!REPLACE!! make class decision
        return y_hat, pc, idxKNN  # return predicted class, a-posteriori-distribution, and indexes of nearest neighbors


# ----------------------------------------------------------------------------------------- 
# ----------------------------------------------------------------------------------------- 
# Fast K-Nearest-Neighbor classifier based on scipy KD trees
# ----------------------------------------------------------------------------------------- 
# ----------------------------------------------------------------------------------------- 
class FastKNNClassifier(Classifier):
    """
    Fast k-nearest-neighbor classifier based on kd-trees 
    Derived from KNNClassifier
    """

    def __init__(self,C=2,K=1):
        """
        Constructor of the KNN-Classifier
        :param C: Number of different classes
        :param K: Number of nearest neighbors that classification is based on
        """
        KNNClassifier.__init__(self,C,K)     # call to parent class constructor  

    def fit(self,X,T):
        """
        Train classifier by creating a kd-tree 
        :param X: Data matrix, contains in each row a data vector
        :param T: Vector of class labels, must have same length as X, each label should be integer in 0,1,...,C-1
        :returns: - 
        """
        KNNClassifier.fit(self,X,T)                # call to parent class method (just store X and T)
        self.kdtree = None                         # !!REPLACE!! do an indexing of the feature vectors by constructing a kd-tree
        
    def predict(self,x,K=None):
        """ 
        Implementation of KD-Tree based KNN classification algorithm
        :param x: input data vector
        :param K: search K nearest neighbors (default self.K)
        :returns y_hat: label of most likely class that test vector x belongs to
        :returns pc:     A-Posteriori probabilities: pc[i]=pr[class c|input x] is the probability that input x belongs to class i
        :returns idxKNN: indexes of the K nearest neighbors (ordered w.r.t. ascending distance) 
        """
        if K==None: K=self.K                          # use default parameter K?
        idxKNN = range(K)                             # !!REPLACE!! get indexes of K nearest neighbors of x from kd-tree
        if K==1: idxKNN = [idxKNN[0]]                 # !!REPLACE!! in case K=1 cast (single) nearest neighbor index nn as a list idxNN
        return KNNClassifier.predict(self,x,K,idxKNN) # return predicted class, a-posteriori-distribution, and indexes of nearest neighbors (as in naive KNN)

# ----------------------------------------------------------------------------------------- 
# ----------------------------------------------------------------------------------------- 
# Kernel Multilayer-Perzeptron classifier 
# ----------------------------------------------------------------------------------------- 
# ----------------------------------------------------------------------------------------- 
class KernelMLPClassifier(Classifier):
    """
    Fast k-nearest-neighbor classifier based on kd-trees 
    Derived from KNNClassifier
    """

    def __init__(self,C=2,hz=np.tanh):
        """
        Constructor of the KNN-Classifier
        :param C: Number of different classes
        :param hz: activation function in hidden layer z 
        """
        Classifier.__init__(self,C)     # call to parent class constructor
        self.hz=hz                      # store activation function

    def fit(self,X,T):
        """
        Train Kernel MLP classifier (see Skript IMLS, Chapt. 1.3.4, Example MLP3/Kernel MLP pages 47-50)
        :param X: Data matrix, contains in each row a data vector
        :param T: Vector of class labels, must have same length as X, each label should be integer in 0,1,...,C-1
        :returns: -
        """
        if len(T.shape)==1:
            T_onehot=np.zeros((len(X),self.C),'int')   # allocate space for one-hot-vectors
            for n in range(len(X)): T_onehot[n,T[n]]=1 # set one hot components
            T=T_onehot                                 # replace T by one-hot label matrix
        self.Wz=None      # !!REPLACE!! weight matrix from input to hidden layer corresponds to input data matrix
        self.K=None       # !!REPLACE!! Gram matrix
        self.Wy=None      # !!REPLACE!! weight matrix from hidden to output layer 
        
    def predict(self,x):
        """ 
        Implementation of KD-Tree based KNN classification algorithm
        :param x: input data vector
        :param K: search K nearest neighbors (default self.K)
        :returns y_hat: label of most likely class that test vector x belongs to
        :returns y: Dendritic potential (=firing rates) of linear output layer
        :returns None: dummy 
        """
        z=None                        # !!REPLACE!! firing rates of hidden layer z
        y=None                        # !!REPLACE!! dendritic potentials = firing rates in output layer y
        y_hat=np.argmax(y)            # select class with maximum potential as winner class
        return y_hat,y,None 

# *******************************************************
# __main___
# Module test
# *******************************************************

if __name__ == '__main__':
    np.random.seed(20)                # initialize random number generator

    # (i) Generate some dummy data 
    X = np.array([[1,2,3],[-2,3,4],[3,-4,5],[4,5,-6],[-5,6,7],[6,-7,8]])   # data matrix X: list of data vectors (=database) of dimension D
    T = np.array( [0     ,1       ,2       ,0       ,1       ,2      ] )   # class labels (C=3 classes)
    C = np.max(T)+1                                                        # C=3 here
    x = np.array([3.5,-4.4,5.3]);                                          # a new input vector to be classified
    print("Data matrix X=\n",X)
    print("Class labels T=",T)
    print("Test vector x=",x)
    print("Euklidean distances to x: ", [np.linalg.norm(X[i]-x) for i in range(len(X))])

    # (ii) Train simple KNN-Classifier and classify vector x
    knnc = KNNClassifier(C)           # construct kNN Classifier
    knnc.fit(X,T)                     # train with given data
    K=3                               # number of nearest neighbors
    yhat,pc,idx_knn=knnc.predict(x,K) # classify 
    print("\nClassification with the naive KNN-classifier:")
    print("Test vector is most likely from class y_hat=",yhat)
    print("A-Posteriori Class Distribution: prob(x is from class i)=",pc)
    print("Indexes of the K=",K," nearest neighbors: idx_knn=",idx_knn)

    # (iii) Do the same with the FastKNNClassifier (based on KD-Trees)
    knnc_fast = FastKNNClassifier(C)        # construct fast KNN Classifier
    knnc_fast.fit(X,T)                      # train with given data
    yhat,pc,idx_knn=knnc_fast.predict(x,K)  # classify
    print("\nClassification with the fast KNN-classifier based on kd-trees:")
    print("Test vector is most likely from class y_hat=",yhat)
    print("A-Posteriori Class Distribution: prob(x is from class i)=",pc)
    print("Indexes of the K=",K," nearest neighbors: idx_knn=",idx_knn)

    # (iv) Do the same with the KernelMLPClassifier 
    kernelMLPc = KernelMLPClassifier(C)     # construct Kernel-MLP Classifier
    kernelMLPc.fit(X,T)                     # train with given data 
    yhat,y,dummy=kernelMLPc.predict(x,K)    # classify
    print("\nClassification with the Kernel-MLP:")
    print("Test vector is most likely from class y_hat=",yhat)
    print("Model outputs y=",y) 

    # (v) Do a 2-fold cross validation using the KNN-Classifier
    S=2
    err,matCp=knnc.crossvalidate(S,X,T)
    print("\nCrossValidation with S=",S," for KNN-Classifier:")
    print("err=",err)
    print("matCp=",matCp)
    
