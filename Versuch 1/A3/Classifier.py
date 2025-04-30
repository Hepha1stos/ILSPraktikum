#!/usr/bin/env python
# Python Module for Classification Algorithms
import numpy as np
import scipy.spatial
import KNearestNeighborSearch as KNNS

# -----------------------------------------------------------------------------------------
# Base class for classifiers
# -----------------------------------------------------------------------------------------
class Classifier:
    def __init__(self, C=2):
        self.C = C

    def fit(self, X, T):
        shapeX, shapeT = X.shape, T.shape
        assert len(shapeX) == 2, "Classifier.fit: X must be 2D!"
        assert len(shapeT) == 1, "Classifier.fit: T must be 1D!"
        assert shapeX[0] == shapeT[0], "Classifier.fit: X and T must have same length!"
        minT, maxT = np.min(T), np.max(T)
        assert minT >= 0 and maxT < self.C, f"Labels should be between 0 and {self.C - 1}"

    def predict(self, x):
        return -1, None, None

    def crossvalidate(self, S, X, T):
        X, T = np.array(X), np.array(T, 'int')
        N = len(X)
        perm = np.random.permutation(N)
        idxS = [range(i*N//S, (i+1)*N//S) for i in range(S)]
        matCp = np.zeros((self.C, self.C))
        err = 0
        for idxVal in idxS:
            if S > 1:
                idxTrain = [i for i in range(N) if i not in idxVal]
            else:
                idxTrain = idxVal
            self.fit(X[perm[idxTrain]], T[perm[idxTrain]])
            for i in idxVal:
                y_hat = self.predict(X[perm[i]])[0]
                t_true = T[perm[i]]
                matCp[t_true, y_hat] += 1
                if y_hat != t_true:
                    err += 1
        matCp = matCp / float(N)
        err = err / float(N)
        return err, matCp

# -----------------------------------------------------------------------------------------
# Naive KNN Classifier
# -----------------------------------------------------------------------------------------
class KNNClassifier(Classifier):
    def __init__(self, C=2, K=1):
        Classifier.__init__(self, C)
        self.K = K
        self.X, self.T = [], []

    def fit(self, X, T):
        Classifier.fit(self, X, T)
        self.X, self.T = np.array(X), np.array(T, 'int')

    def predict(self, x, K=None, idxKNN=None):
        if K is None:
            K = self.K
        if idxKNN is None:
            idxKNN = KNNS.getKNearestNeighbors(x, self.X, K)
        labels = self.T[idxKNN]
        pc = KNNS.getClassProbabilities(labels, self.C)
        y_hat = KNNS.classify(pc)
        return y_hat, pc, idxKNN

# -----------------------------------------------------------------------------------------
# Fast KNN Classifier (KD-Tree)
# -----------------------------------------------------------------------------------------
class FastKNNClassifier(KNNClassifier):
    def __init__(self, C=2, K=1):
        KNNClassifier.__init__(self, C, K)

    def fit(self, X, T):
        KNNClassifier.fit(self, X, T)
        self.kdtree = scipy.spatial.KDTree(self.X)

    def predict(self, x, K=None):
        if K is None:
            K = self.K
        dists, idxKNN = self.kdtree.query(x, K)
        if K == 1:
            idxKNN = [idxKNN]
        return KNNClassifier.predict(self, x, K, idxKNN)

# -----------------------------------------------------------------------------------------
# Kernel MLP Classifier
# -----------------------------------------------------------------------------------------
class KernelMLPClassifier(Classifier):
    def __init__(self, C=2, hz=np.tanh):
        Classifier.__init__(self, C)
        self.hz = hz

    def fit(self, X, T):
        X = np.array(X)
        if len(T.shape) == 1:
            T_onehot = np.zeros((len(X), self.C), 'int')
            for n in range(len(X)):
                T_onehot[n, T[n]] = 1
            T = T_onehot

        self.Wz = X.T                      # "Input" weights are the training samples
        self.K = self.hz(X @ X.T)          # Kernel/Gram matrix using dot-product and activation function
        self.Wy = np.linalg.pinv(self.K) @ T  # Linear regression for output weights

    def predict(self, x):
        z = self.hz(self.Wz.T @ x)         # Hidden layer: apply activation to dot-product
        y = z @ self.Wy                    # Output layer: linear combination
        y_hat = np.argmax(y)
        return y_hat, y, None


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
    yhat, y, dummy = kernelMLPc.predict(x)
    print("\nClassification with the Kernel-MLP:")
    print("Test vector is most likely from class y_hat=",yhat)
    print("Model outputs y=",y) 

    # (v) Do a 2-fold cross validation using the KNN-Classifier
    S=2
    err,matCp=knnc.crossvalidate(S,X,T)
    print("\nCrossValidation with S=",S," for KNN-Classifier:")
    print("err=",err)
    print("matCp=",matCp)
    
