import numpy as np

def getGaussData2D(N,mu1,mu2,Sig11,Sig22,Sig12,t=0,C=2,flagOneHot=0):
    """
    generate random data matrix X of 2D Gaussian data and corresponding target matrix T
    :param N: number of Gaussian data vectors
    :param mu1: mean of component 1
    :param mu2: mean of component 2
    :param Sig11: component Sigma(1,1) of covariance matrix Sigma (=variance of first component of random vectors)
    :param Sig22: component Sigma(2,2) of covariance matrix Sigma (=variance of second component of random vectors)
    :param Sig12: component Sigma(1,2)=Sigma(2,1) of covariance matrix Sigma (=covariance between two components of random vectors)
    :param t: target value (=class index) to be stored in T (integer between 0 and C-1)
    :param C: number of different classes
    :param flagOneHot: If >1 then each line of T is a one-hot-vector; ELSE T is 1D array containing t
    :return X: data matrix of size NxD for dimension D=2
    :return T: target matrix of size NxC containing N one-hot-vectors (if flagOneHot>0) or simply 1D array of size N (if flagOneHot=0) containing t
    """
    mu=np.array([mu1,mu2])                            # define expectation vector of Gaussian
    Sig=np.array([[Sig11,Sig12],[Sig12,Sig22]])       # define covariance matrix of Gaussian
    X  =np.zeros((N,2))                               # get data matrix        !!! REPLACE THIS !!!  
    if flagOneHot:
        T=np.zeros((N,nClasses),'int')                # allocate target matrix
        T[:,t]=1;                                     # set one-hot-entry of current class to 1
    else:
        T=np.zeros(N,'int')                           # allocate target vector
        T[:]=t                                        # set to target value
    return X,T                                        # return data matrix and target matrix


# **************************************************
# ***** MAIN PROGRAM: Test data generation  ********
# **************************************************
if __name__ == '__main__':
    # (i) specify data parameters
    N1     = 1                     # number of data samples from class 1             !!! REPLACE THIS !!!  
    mu1    = np.array([0,0])       # mean vector for class 1                         !!! REPLACE THIS !!!  
    Sigma1 = np.array([[0.0,0.0],        
                       [0.0,0.0]]) # covariance matrix for class 1                   !!! REPLACE THIS !!!  

    N2     = 1                     # number of data samples from class 2             !!! REPLACE THIS !!!
    mu2    = np.array([0,0])       # mean vector for class 2                         !!! REPLACE THIS !!!
    Sigma2 = np.array([[0.,0.],        
                       [0.,0.]])   # covariance matrix for class 2                   !!! REPLACE THIS !!!

    # (ii) generate data
    np.random.seed(0)              # set seed of random number generator             !!! REPLACE THIS !!!
    X1,T1 = getGaussData2D(N1,mu1[0],mu1[1],Sigma1[0,0],Sigma1[1,1],Sigma1[0,1],1)   # generate data for class 1
    X2,T2 = X1,T1                                                                    # generate data for class 2   !!! REPLACE THIS !!!
    print("X1=",X1)
    print("T1=",T1)
    print("X2=",X2)
    print("T2=",T2)

    # (iii) concatenate to data and target matrices
    X = np.concatenate((X1,X2))
    T = np.concatenate((T1,T2))
    print("X=",X)
    print("T=",T)

    # (iv) estimate expectation and covariances of concatenated data
    N,D = X.shape[0],X.shape[1]    # total data number and dimensionality
    mu = [0,0]                     # estimate of mean vector                                  !!! REPLACE THIS !!! 
    Sigma = np.zeros((D,D))        # estimate covariance matrix: initialize with zeros
    for n in range(N):
        Sigma += np.zeros((D,D))   # add up terms for covariance estimation  using np.outer   !!! REPLACE THIS !!!   
    
    print("N=",N,"D=",D)    # data size
    print("mu=",mu)         # estimate of total mean
    print("Sigma=",Sigma)   # estimate of total covariance matrix
