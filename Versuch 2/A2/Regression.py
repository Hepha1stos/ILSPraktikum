import numpy as np
import scipy.spatial
from polynomial_basis_functions import *

# ----------------------------------------------------------------------------------------- 
# base class for regressifiers
# ----------------------------------------------------------------------------------------- 
class Regressifier:
    """
    Abstract base class for regressifiers
    Inherit from this class to implement a concrete regression algorithm
    """

    def fit(self,X,T):        # train/compute regression with lists of feature vectors X and class labels T
        """
        Train regressifier by training data X, T, should be overwritten by any derived class
        :param X: Data matrix of size NxD, contains in each row a data vector of size D
        :param T: Target vector matrix of size NxK, contains in each row a target vector of size K
        :returns: -
        """
        pass

    def predict(self,x):      # predict a target vector given the data vector x 
        """
        Implementation of the regression algorithm; should be overwritten by any derived class 
        :param x: test data vector of size D
        :returns: predicted target vector y
        """
        return None           

    def crossvalidate(self,S,X,T,eps=1e-6):  # do a S-fold cross validation 
        """
        Do a S-fold cross validation
        :param S: Number of parts the data set is divided into
        :param X: Data matrix (one data vector per row)
        :param T: Matrix of target vectors; T[n] is target vector of X[n]
        :param dist: a fuction dist(t) returning the length of vector t (default=Euklidean)
        :returns MAE, MAPE: return mean absolute error and mean absolute percentage error  
        """
        X,T=np.array(X),np.array(T,'int')                   # make sure we have numpy arrays
        N=len(X)                                            # N=number of data vectors
        perm = np.random.permutation(N)                     # get permutation of index list [0,1,...,N-1] to get random selections of X and T
        idxS = [range(i*N//S,(i+1)*N//S) for i in range(S)] # indexes for dividing data set into S parts of equal size
        MAE,MAPE,nMAE=0.0,0.0,0                             # initialize mean absolute error (MAE), mean absolute percentage (MAPE), and counter 
        for idxVal in idxS:                                 # loop over all S validation data sets
            # (i) generate training and testing data sets and train classifier        
            if S>1: idxTrain = [i for i in range(N) if i not in idxVal]               # remaining indices (not in idxVal) are training data
            else  : idxTrain = idxVal                                                 # if S==1 use entire data set for training and validation
            self.fit(X[perm[idxTrain]],T[perm[idxTrain]])                             # train classifier using (permutated) indexes of training data         
            # (ii) evaluate classifier using validation data
            for i in range(len(idxVal)):  # loop over all validation indexes
                y_hat = self.predict(X[perm[i]])        # predicted class of i-th input vector from validation set 
                t_true = T[perm[i]]                     # corresponding true class label
                if not isinstance(t_true,np.ndarray): y_hat,t_true = np.array([y_hat]),np.array([t_true])   # write as arrays for unified handling
                MAE+=np.sum(np.abs(y_hat-t_true))       # add up absolute errors
                MAPE+=np.sum(np.divide(np.abs(y_hat-t_true),t_true+eps)) # add up absolute percentage error
                nMAE+=len(t_true)                       # update number of added items
        MAE/=nMAE        # compute mean
        MAPE/=nMAE       # compute mean
        return MAE,MAPE  # return mean absolute error and mean absolute percentage error

    
# -------------------------------------------------------------------------------------------- 
# DataScaler: scale data to standardize data distribution (for mean=0, standard deviation =1)  
# -------------------------------------------------------------------------------------------- 
class DataScaler: 
    """
    Class for standardizing data vectors 
    Some regression methods require standardizing of data before training to avoid numerical instabilities!
    """

    def __init__(self,X,eps=1e-6):  
        """
        Constructor: Set parameters (mean, std,...) to standardize data matrix X
        :param X: Data matrix of size NxD the standardization parameters (mean, std, ...) should be computed for 
        :param eps: small constant for numerical stability (avoid division by sd=0)
        :returns: object of class DataScaler
        """
        if len(X.shape)==1: X=np.reshape(X,(len(X),1)) # recast as data matrix for unified handling
        self.mu = np.mean(X,0)                         # mean values for each feature column
        self.sd = np.std(X,0)                          # standard deviation for each feature column
        self.sd[self.sd<eps]=1.0                       # do not scale data with zero or very small SD (that is, constant features)
        self.sd_inv = np.divide(1.0,self.sd)           # inverse standard deviation


    def scale(self,X):                  # scales data X to mean=0 and std=1
        """
        scale data vector (or data matrix) X to mean=0 and s.d.=1 
        :param X: data vector or data matrix  
        :returns: scaled (standardized) data vector or data matrix 
        """
        return np.multiply(X-self.mu,self.sd_inv)

    def unscale(self,X):                # unscale data vector X to original distribution
        """
        unscale data vector (or data matrix) X to original data ranges  
        :param X: standardized data vector or data matrix  
        :returns: unscaled data vector or data matrix 
        """
        return np.multiply(X,self.sd)+self.mu


# -----------------------------------------------------------------------------------------
# Least Squares (ML) linear regression with sum of squares Regularization,
# -----------------------------------------------------------------------------------------
class LSRRegressifier(Regressifier):
    """
    Class for Least Squares (or Maximum Likelihood) Linear Regressifier with sum of squares regularization 
    """

    def __init__(self, lmbda=0, phi=lambda x:x, flagSTD=0, eps=1e-2):
        """
        Constructor of class LSRegressifier
        :param lmbda: Regularization coefficient lambda
        :param phi: Basis-functions used by the linear model (default: identical mapping)
        :param flagSTD: If >0 then standardize data X and target values T (to mean 0 and s.d. 1)
        :param eps: maximal residual value to tolerate (instead of zero) for numerically good conditioned problems
        :returns: -
        """
        self.lmbda=lmbda       # set regression parameter (default 0)
        self.phi=phi           # set basis functions used for linear regression (default: degree 1 polynomials)
        self.flagSTD=flagSTD;  # if flag >0 then data will be standardized, i.e., scaled for mean 0 and s.d. 1
        self.eps=eps;          # maximal residual value to tolerate (instead of zero) for numerically good conditioned problems

    def fit(self,X,T): # train/compute LS regression with data matrix X and target value matrix T
        """
        Train regressifier 
        :param X: Data matrix of size NxD, contains in each row a data vector of size D
        :param T: Target vector matrix of size NxK, contains in each row a target vector of size K
        :returns: flagOK: if >0 then all is ok, otherwise matrix inversion was bad conditioned (and results should not be trusted!!!) 
        """
        # (i) scale data for mean=0 and s.d.=0 ?
        if self.flagSTD>0:                     # if yes, then...
            self.datascalerX=DataScaler(X)     # create datascaler for data matrix X
            self.datascalerT=DataScaler(T)     # create datascaler for target matrix T
            X=self.datascalerX.scale(X)        # scale all features (=columns) of data matrix X to mean=0 and s.d.=1
            T=self.datascalerT.scale(T)        # ditto for target matrix T
        # (ii) compute weight matrix and check numerical condition
        flagOK,maxZ=1,0;                       # if <1 then matrix inversion is numerically infeasible
        self.N,self.D = X.shape                # data matrix X has size N x D (N is number of data vectors, D is dimension of a vector)
        self.M = self.phi(self.D*[0]).size     # get number of basis functions
        if len(T.shape)==1: self.K = 1         # determine number of output components
        else: self.K=T.shape[1]
        try:
            # (ii.a) compute optimal least squares weights
            PHI = None                                                    # !!! REPLACE THIS !!!  --> compute design matrix
            PHIT_PHI_lmbdaI = None                                        # !!! REPLACE THIS !!!  --> compute PHI_T*PHI+lambda*I
            PHIT_PHI_lmbdaI_inv = None                                    # !!! REPLACE THIS !!!  --> compute inverse matrix (may be bad conditioned and fail) 
            self.W_LSR = None                                             # !!! REPLACE THIS !!!  --> regularized least squares weights
            # (ii.b) check numerical condition
            Z=None                                                        # !!! REPLACE THIS !!! --> compute PHIT_PHI_lmbdaI*PHIT_PHI_lmbdaI_inv-I --> should become the zero matrix if good conditioned!
            maxZ = 1000                                                   # !!! REPLACE THIS !!! --> compute maximum component of Z (<eps for good conditioned problem)
            assert maxZ<=self.eps,"MATRIX INVERSION IS BAD CONDITIONED!"  # check if matrix inversion has good condition
        except:
            # (ii.c) if exception occurs then set weights to defaults (zeros) and print warning message
            flagOK=0;
            print("EXCEPTION DUE TO BAD CONDITION:flagOK=", flagOK, "maxZ=",maxZ,"N=",self.N,"D=",self.D,"M=",self.M,"K=",self.K)
            self.W_LSR=np.zeros((self.K,self.M))
        finally:
            # (ii.d) enforce correct format for weight matrix/vector
            if len(T.shape)==1: self.W_LSR=np.reshape(self.W_LSR,(self.M))  # if targets are single values then use weight vector instead of weight matrix
            return flagOK                                                   # if ok return 1 (otherwise 0)

    def predict(self,x):      # predict a target value given data vector x 
        """
        predicts the target value y(x) for a test vector x
        :param x: test data vector of size D
        :returns: predicted target vector y of size K
        """
        if self.flagSTD>0: x=self.datascalerX.scale(x)    # scale x before computing the prediction?
        y=0                                               # !!! REPLACE THIS !!! --> compute model prediction; you can use evaluate_linear_model(.) from module polynomial_basis_functions
        if self.flagSTD>0: y=self.datascalerT.unscale(y)  # unscale prediction?
        return y                                          # return prediction y for data vector x


# -----------------------------------------------------------------------------------------
# KNN regression 
# -----------------------------------------------------------------------------------------
class KNNRegressifier(Regressifier): 
    """
    Class for fast K-Nearest-Neighbor-Regression using KD-trees 
    """

    def __init__(self,K,flagKLinReg=0, lr_lmbda=0, lr_phi=lambda x:x, lr_flagSTD=0, lr_eps=1e-2):
        """
        Constructor of class KNNRegressifier
        :param K: number of nearest neighbors that are used to compute prediction 
        :param flagKLinReg: if >0 then the do a linear (least squares) regression on the the K nearest neighbors and their target values
                      otherwise just take the mean of the K nearest neighbors target vectors
        :param lr_lmbda,lr_polydeg,lr_flagSTD,lr_eps: parameters for applying LSRRegressifier on K nearest neighbors to make a prediction 
        :returns: -
        """
        self.K = K                                 # K is number of nearest-neighbors used for majority decision
        self.X, self.T = [],[]                     # initially no data is stored
        self.flagKLinReg=flagKLinReg               # if flag is set then do a linear regression of the KNN (otherwise just return mean T of the KNN)
        if flagKLinReg>0: self.regLSR=LSRRegressifier(lr_lmbda,lr_phi,lr_flagSTD,lr_eps)  # if flag is set create least squares regressifier
        else            : self.regLSR=None         # if flag is 0 no regressifier is needed
        
    def fit(self,X,T): # train/compute regression with lists of data vectors X and target values T
        """
        Train regressifier by stroing X and T and by creating a KD-Tree based on X   
        :param X: Data matrix of size NxD, contains in each row a data vector of size D
        :param T: Target vector matrix of size NxK, contains in each row a target vector of size K
        :returns: -
        """
        self.X, self.T = np.array(X),np.array(T)   # just store feature vectors X and corresponding class labels T
        self.N, self.D = self.X.shape              # store data number N and dimension D
        self.kdtree = scipy.spatial.KDTree(self.X) # do an indexing of the feature vectors

    def predict(self,x):                           # predict a target value given data vector x 
        """
        predicts the target value y(x) for a test vector x
        :param x: input data vector of size D
        :returns: predicted target vector 
        """
        K=min(self.K,self.N)                    
        nn = self.kdtree.query(x,K)                # get indexes of K nearest neighbors of x
        if K==1: idxNN=[nn[1]]                     # cast nearest neighbor indexes nn as a list idxNN
        else: idxNN=nn[1]
        y=0
        if(self.flagKLinReg==0):
            # just take mean value of KNNs
            y=np.mean([self.T[i] for i in idxNN])          # take the mean of the KNN target values
        else:
            # do a linear regression of the KNNs
            self.regLSR.fit(self.X[idxNN],self.T[idxNN])   # train a linear regressifier using the KNN data
            y=self.regLSR.predict(x)                       # take prediction from regressifier
        return y


# *******************************************************
# __main___
# Module test
# *******************************************************

if __name__ == '__main__':
    print("\n-----------------------------------------")
    print("Example: 1D-linear regression problem")
    print("-----------------------------------------")
    # (i) generate data
    N=100
    w0,w1=4,2                 # parameters of line
    X=np.zeros((N,1))         # x data: allocate Nx1 matrix as numpy ndarray
    X[:,0]=np.arange(0,50.0,50.0/N)  # equidistant sampling of the interval [0,50)
    T=np.zeros(N)             # target values: allocate length-N vector as numpy ndarray
    sd_noise = 1.0            # noise power (=standard deviation)
    T=w1*np.reshape(X,(N))+w0 + np.random.normal(0,sd_noise,T.shape)  # generate noisy target values on line y=w0+w1*x
    par_lambda = 0            # regularization parameter
    print("X=",X)
    print("T=",T)

    # (ii) define basis functions (phi should return list of basis functions; x should be a list)
    deg=2;                               # degree of polynomial
    phi=get_phi_poly(1,deg)              # define phi by 1D polynomial basis-functions up to degree deg 
    print("phi(4)=", phi([4]))           # print basis function vector [1, x, x*x ...] for x=4

    # (iii) compute LSR regression
    print("\n-----------------------------------------")
    print("Do a Least-Squares-Regression")
    print("-----------------------------------------")
    lmbda=0;
    lsr = LSRRegressifier(lmbda,phi)
    lsr.fit(X,T)
    print("lsr.W_LSR=",lsr.W_LSR)        # weight vector (should be approximately [w0,w1]=[4,2])
    x=np.array([3.1415])
    print("prediction of x=",x,"is y=",lsr.predict(x))

    # do S-fold crossvalidation
    S=3
    MAE = lsr.crossvalidate(S,X,T)
    print("LSRRegression cross-validation: MAE=",MAE[0],"MAPE=",MAE[1])

    # (iv) compute KNN-regression
    print("\n-----------------------------------------")
    print("Do a KNN-Regression")
    print("-----------------------------------------")
    K=5;
    knnr = KNNRegressifier(K)
    knnr.fit(X,T)
    print("prediction of x=",x,"is y=",knnr.predict(x))

    # do S-fold crossvalidation
    MAE = knnr.crossvalidate(S,X,T)
    print("KNNRegression cross-validation: MAE=",MAE[0],"MAPE=",MAE[1]) 
