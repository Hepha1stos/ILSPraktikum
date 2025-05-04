import numpy as np

def getKNearestNeighbors(x,X,K=1):  
    """
    compute the K nearest neighbors for a query vector x given a data matrix X
    :param x: the query vector x
    :param X: the N x D data matrix (in each row there is data vector) as a numpy array
    :param K: number of nearest-neighbors to be returned
    :return: return list of K row indexes referring to the K nearest neighbors of x in X
    """
    d=[]                          # !!REPLACE!! compute list of Euklidean distances between x and X[i]
    return np.array(K*[0],'int')  # !!REPLACE!! return indexes of k smallest distances     

def getClassProbabilities(t,C):     
    """
    compute class probabilities for given target/label list
    :param t: list of target values/labels (e.g., of the K nearest neighbors computed with getKNearestNeighbors(.) t[i] must be between 0 and C-1
    :param C: number of classes
    :return P: P is list of class probabilities (length C) 
    """
    assert min(t)>=0 and max(t)<C, "t must be list of integer labels between 0 and C-1"
    P=np.zeros(C)   # allocate array for class probabilities (length = number of classes)
    P[:]=1.0/C      # !!REPLACE!! P[c] should be the probability for class c=0,1,2,...,C-1 for the label list t
    return P        # return class distribution

def classify(P): 
    """
    classify for class distribution P, i.e., select most probable class
    if several classes have the same probability then choose at random
    :param P: array of class probabilities (length = number of classes), e.g., computed by getClassProbabilities(.)
    :return c: class decision (index of the most probable class)
    """
    idx_maxP=[0]              # !!REPLACE!! get list of most likely classes (having maximum probability)
    if len(idx_maxP)>1: c=0   # !!REPLACE!! if more than one maximum class then choose at random
    else: c=0                 # !!REPLACE!! else choose unique class having maximal probability
    return c                  # return class decision (between 0 and C-1, i.e., index in P)

# *****************************************************************************
# ***** MAIN PROGRAM: Test nearest neighbor search and classification  ********
# *****************************************************************************
if __name__ == '__main__':
    
    # (i) Generate some dummy data 
    X = np.array([[1,2,3],[-2,3,4],[3,-4,5],[4,5,-6],[-5,6,7],[6,-7,8]])   # data matrix X: list of data vectors (=database) of dimension D=3
    t = np.array( [0     ,1       ,2       ,0       ,1       ,2      ] )   # class labels (C=3 classes)
    C = np.max(t)+1                                                        # C=3 here
    x = np.array([3.5,-4.4,5.3]);                                          # a new input vector to be classified
    print("Data matrix X=\n",X)
    print("Class labels t=",t)
    print("Test vector x=",x)
    
    # (ii) Print all Euklidean distances to test vector x
    print("Euklidean distances to x: ", [np.linalg.norm(X[i]-x) for i in range(len(X))])
    
    # (iii) Search for K nearest neighbor
    K=3                                                    # define K
    idx_KNN = getKNearestNeighbors(x,X,K)                  # get indexes of k nearest neighbors
    print("idx_KNN=",idx_KNN)
    print("The K Nearest Neighbors of x are the following vectors:")
    for i in range(K):
        idx=idx_KNN[i]
        print("The", i+1, "th nearest neighbor is: X[",idx,"]=",X[idx]," with distance ", np.linalg.norm(X[idx]-x)," and class label ",t[idx])

    # (iv) do classification
    P=getClassProbabilities(t[idx_KNN],C=3)                # get class probabilities for input x
    c=classify(P)                                          # get most likely class
        
    print("Class distribution P=",P)
    print("Most likely class: c=",c," with P(c)=",P[c])
