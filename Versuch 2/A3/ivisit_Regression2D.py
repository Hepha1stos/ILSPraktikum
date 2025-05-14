# coding: utf-8
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@IVISIT:SIMULATION  & ivisit_RegLinearRegr 
#@IVISIT:SLIDER      & seed          & [150,1] & [0,100,3,1]    & seed        & -1 & int   & 0    # parameters for data generation
#@IVISIT:SLIDER      & sqrt(N)       & [150,1] & [1,100,3,1]    & N           & -1 & int   & 10 
#@IVISIT:SLIDER      & sd_noise      & [150,1] & [0,2,3,0.01]   & sd_noise    & -1 & float & 0.0 
#@IVISIT:SLIDER      & elevation     & [150,1] & [-90,90,3,1]   & elevation   & -1 & float & 0.0 
#@IVISIT:SLIDER      & azimuth       & [150,1] & [-180,180,3,1] & azimuth     & -1 & float & 0.0 
#@IVISIT:SLIDER      & sd_noise      & [150,1] & [0,2,3,0.01]   & sd_noise    & -1 & float & 0.0 
#@IVISIT:RADIOBUTTON & Regression Model & [Least-Squares,K-Nearest-Neighbors] & str_regmodel     & Least-Squares
#@IVISIT:RADIOBUTTON & Groundtruth Function & [sin,si,plane] & str_fun_true   & sin
#@IVISIT:CHECKBOX    & Display Data     & [train-data,test-data,groundtruth,model]               & str_display_data & 1111

#@IVISIT:DICTSLIDER & LSR-Parameters & [200,20,-1,2,10] & parLSR  & 0                            # parameters for least squares regression
#@IVISIT:DICTSLIDERITEM  & deg         & [0,50,5,1]     & deg         & int   & 3 
#@IVISIT:DICTSLIDERITEM  & lmbda_log10 & [-100,100,5,1] & lmbda_log10 & int   & 0 
#@IVISIT:DICTSLIDERITEM  & lmbda_scale & [0,10,5,0.1]   & lmbda_scale & float & 0.0 
#@IVISIT:DICTSLIDERITEM  & flagSTD     & [0,1,2,1]      & flagSTD     & int   & 0 
#@IVISIT:DICTSLIDERITEM  & eps_log10   & [-10,10,5,0.1] & eps_log10   & float & -2.0 

#@IVISIT:DICTSLIDER & KNN-Parameters & [200,20,-1,2,10] & parKNN  & 0                            # parameters for KNN regression
#@IVISIT:DICTSLIDERITEM  & K              & [1,50,5,1]     & K              & int   & 1 
#@IVISIT:DICTSLIDERITEM  & flagKLinReg    & [0,1,2,1]      & flagKLinReg    & int   & 0 
#@IVISIT:DICTSLIDERITEM  & lr_deg         & [0,50,5,1]     & lr_deg         & int   & 0 
#@IVISIT:DICTSLIDERITEM  & lr_lmbda_log10 & [-100,100,5,1] & lr_lmbda_log10 & int   & 0 
#@IVISIT:DICTSLIDERITEM  & lr_lmbda_scale & [0,10,5,0.1]   & lr_lmbda_scale & float & 0.0 
#@IVISIT:DICTSLIDERITEM  & lr_flagSTD     & [0,1,2,1]      & lr_flagSTD     & int   & 0 
#@IVISIT:DICTSLIDERITEM  & lr_eps_log10   & [-10,10,5,0.1] & lr_eps_log10   & float & -2.0 

#@IVISIT:IMAGE      & Data/Model    & 1.0     & [0,255]        & im_results & int                # output widgets
#@IVISIT:TEXT_OUT   & Results       & [20,5]  & just_left      & str_results

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ivisit as iv
from ivisit.matplotlib import *
from polynomial_basis_functions import *
from Regression import *

font = {'family' : 'normal',
        'weight' : 'normal', #'bold',
        'size'   : 16}
plt.rc('font', **font)

def fun_true_sin(X,f1=0.5,f2=0.25,phi0=0.0,c=1.0,mu=0.0,sig=0): # compute 2-dim. sin-function g(x)=c*sin(2*pi*f1*x1+f2*x2+phi0)+N(mu,sig2) for Gaussian noise N(mu,sig2) mit mean mu and variance sig2 
    X1,X2=X[:,0],X[:,1]                                                   # get x1 / x2 components of data
    return c*np.sin(2*np.pi*(f1*X1+f2*X2+phi0))+np.random.normal(mu,sig,X1.shape)   # return function values (same size as X.flat)  
              
def fun_true_si(X,f=1.0,phi0=0.0,c=1.0,mu=0.0,sig=0): # compute 2-dim. si-function g(x)=c*si(2*pi*f*r+phi0)+N(mu,sig2) for r=sqrt(x1^2+x2^2) and Gaussian noise N(mu,sig2) mit mean mu and variance sig2 
    X1,X2=X[:,0],X[:,1]                                                   # get x1 / x2 components of data
    R=np.sqrt(np.multiply(X1,X1)+np.multiply(X2,X2))                      # radius (or distance from origin) for each point(x1,x2)
    return c*np.sinc(2*f*R+phi0/np.pi)+np.random.normal(mu,sig,R.shape)   # return function values (same size as X.flat) using sinc(x):=si(pi*x) 
              
def fun_true_plane(X,c=1.0,d=1.0,mu=0.0,sig=0): # compute 2-dim. si-function g(x)=c*si(2*pi*f*r+phi0)+N(mu,sig2) for r=sqrt(x1^2+x2^2) and Gaussian noise N(mu,sig2) mit mean mu and variance sig2 
    X1,X2=X[:,0],X[:,1]                                                   # get x1 / x2 components of data
    return c*X1+d*X2+np.random.normal(mu,sig,X1.shape)                    # return function values (same size as X.flat)
              
def getMAE(Y,T, eps=1e-6):                           # compute mean absolute error (and MAPE) for evaluation
    AE=np.abs(np.array(Y.flat)-np.array(T.flat))     # absolute error components
    APE=np.divide(AE,np.maximum(eps,np.abs(Y.flat))) # absolute percentage error
    return np.sum(AE)/len(AE), np.sum(APE)/len(APE)  # return MAE and MAPE  

# (I) define parameters to be controlled by IVisit
class SimParameters(iv.IVisit_Parameters):
    seed       = 13                # seed of random generator (to be able to reproduce results)
    f,phi0,c,mu = 1.0,0.0,1.0,0.0   # parameters of true function y(x)=c*sin(2*pi*f*x+phi0)+d
    xmin,xmax = -2.0, 2.0   # range of x-axis where to draw data from
    ymin,ymax = -1,1.5      # range of y-axis
    N = 5                  # number of training/testing data
    N_dense   = 20         # number of densely sampled data (for plotting model curve)
    sd_noise  = 0.2        # noise power
    elevation,azimuth = 0.0,0.0  # view angle for 3d image
    str_regmodel = 'Least-Squares' # regression model to be applied to the data (either least-squares or KNN)
    str_fun_true = 'sin'           # groundtruth function for generating data
    str_display_data = '1111'      # flags for displaying train/test-data?
    parLSR = {             # hyperparameters for the linear regression model
        'deg':3,             # degree of polyinomial
        'lmbda_log10':0,     # log10 of regularization parameter lambda=lmbda_scale*10^(lambda_log10)
        'lmbda_scale':0.0,   # scaling factor for regularization parameter lambda=lmbda_scale*10^(lambda_log10)
        'flagSTD':0,         # if >0 then standardize data and targets
        'eps_log10':-2.0     # log10 of eps:=10^eps_log10 defining the maximum allowed error for badly condition matrix
    }
    parKNN = {             # hyperparameters for the KNN regression model
        'K':1,               # number of nearest neighbors to search for
        'flagKLinReg':0,     # if >0 then make predictions using linear regression (lr) on the K nearest neighbors of input x  
        'lr_deg':3,          # degree of polynomial for linear regression model
        'lr_lmbda_log10':0,  # log10 of regularization parameter lambda=lmbda_scale*10^(lambda_log10) for linear regression model
        'lr_lmbda_scale':0.0,# scaling factor for regularization parameter lambda=lmbda_scale*10^(lambda_log10) for linear regression model
        'lr_flagSTD':0,      # if >0 then standardize data and targets for linear regression model
        'lr_eps_log10':-2.0  # log10 of eps:=10^eps_log10 defining the maximum allowed error for badly condition matrix for linear regression model
    }
    
# (II) define simulation data to be displayed by IVisit
class SimData(iv.IVisit_Data):
    im_results  = np.array([[0,0,0]])  # image of regression results
    str_results = '-'                  # string for displaying results

# (III) define IVISIT main simulation class 
class Sim(iv.IVisit_Simulation):
    def __init__(self,name_arg="demo1_gaussfilter.py"):
        iv.IVisit_Simulation.__init__(self,name_arg,SimParameters,SimData)

    def step(self):
        p = SimParameters           # short hand to simulation parameters
        pLSR,pKNN=p.parLSR,p.parKNN # short hands to hyperparmeters of least square regression and KNN regression
        d = SimData                 # short hand to simulation data arrays

        # (i) generate training and test data
        np.random.seed(p.seed)                              # set seed of random generator (to be able to regenerate data)
        if   p.str_fun_true=='sin'   : fun_true=fun_true_sin
        elif p.str_fun_true=='si'    : fun_true=fun_true_si
        elif p.str_fun_true=='plane' : fun_true=fun_true_plane
        else: assert 0, "Unknown groundtruthfunction str_fun_true="+str(p.str_fun_true)
        # (i.a) get train data
        x_train=np.linspace(p.xmin,p.xmax,p.N).reshape((p.N,1));  # linear spaced x values (on each axes)
        X1_train,X2_train=np.meshgrid(x_train,x_train)            # get meshgrid for later drawing 3D plots
        X_train=np.concatenate((np.reshape(X1_train,(p.N*p.N,1)),np.reshape(X2_train,(p.N*p.N,1))),axis=1)  # convert meshgrid format to data matrix format
        T_train=fun_true(X_train,sig=p.sd_noise)                  # compute corresponding training targets (including noise)
        # (i.b) get test data
        x_test=x_train+0.5*(p.xmax-p.xmin)/max(1,p.N-1)           # place x values for testing between those for training (on each axes)
        X1_test,X2_test=np.meshgrid(x_test,x_test)                # get meshgrid for later drawing 3D plots
        X_test=np.concatenate((np.reshape(X1_test,(p.N*p.N,1)),np.reshape(X2_test,(p.N*p.N,1))),axis=1)  # convert meshgrid format to data matrix format
        T_test=fun_true(X_test,sig=p.sd_noise)                    # compute corresponding training targets (including noise)
        # (i.c) get dense data matrix for computing ground truth data
        x_dense = np.linspace(p.xmin,p.xmax,p.N_dense).reshape((p.N_dense,1)) # densely sampled x values (just for plotting the model curve)
        X1_dense,X2_dense=np.meshgrid(x_dense,x_dense)            # get meshgrid for later drawing 3D plots
        X_dense=np.concatenate((np.reshape(X1_dense,(p.N_dense*p.N_dense,1)),np.reshape(X2_dense,(p.N_dense*p.N_dense,1))),axis=1)  # convert meshgrid to data matrix 
        
        # (ii) create and train regression model
        if p.str_regmodel == 'Least-Squares':               # which regression model should be used?
            lmbda=pLSR['lmbda_scale']*(10**pLSR['lmbda_log10']) # compute regularization coeffizient
            phi=get_phi_poly(2,pLSR['deg'])                     # define basis functions as 1D polynomial of degree p.deg
            regm = LSRRegressifier(lmbda,phi,pLSR['flagSTD'],10**pLSR['eps_log10']) # create linear regression model (least squares with regularization)
            str_model = "Least Squares with lambda="+str(lmbda)
        elif p.str_regmodel == 'K-Nearest-Neighbors':
            lr_lmbda=pKNN['lr_lmbda_scale']*(10**pKNN['lr_lmbda_log10']) # compute regularization coeffizient
            lr_phi=get_phi_poly(2,pKNN['lr_deg'])                        # define basis functions as 1D polynomial of degree p.deg
            regm = KNNRegressifier(pKNN['K'],pKNN['flagKLinReg'],lr_lmbda,lr_phi,pKNN['lr_flagSTD'],10**pKNN['lr_eps_log10'])   # create KNN regression model
            str_model = 'KNN with K='+str(pKNN['K'])
            if pKNN['flagKLinReg']>0: str_model+=' and lin. regr. with lr_lambda='+str(lr_lmbda)
        else:
            assert 0,"Unknown regression model: str_regmodel="+str(p.str_regmodel)
        regm.fit(X_train,T_train)                               # train model using training data
            
        # (iii) make predictions
        Y_train = np.array([regm.predict(x) for x in X_train])  # predictions on training data
        Y_test  = np.array([regm.predict(x) for x in X_test])   # predictions on test data
        Y_dense = np.array([regm.predict(x) for x in X_dense])  # predictions on test data (just for plotting the model curve)
        MAE_train,MAPE_train = getMAE(Y_train,T_train)                     # get mean absolute (percentage) error on train data
        MAE_test ,MAPE_test  = getMAE(Y_test ,T_test)                      # get mean absolute (percentage) error on test data
        d.str_results=str_model+'\n'                            # write results string
        if p.str_regmodel == 'Least-Squares': d.str_results+='weights w='+str(regm.W_LSR)+'\n'
        d.str_results+='MAE_train='+str(MAE_train)+', MAPE_train='+str(MAPE_train)+'\n'
        d.str_results+='MAE_test ='+str(MAE_test) +', MAPE_test ='+str(MAPE_test)+'\n'
        
        # (iv) compute actual function
        Y_true = fun_true(X_dense)                              # ground truth (without noise)

        # (v) plot results
        fig = getMatplotlibFigure(figsize=(8,6))
        fig.suptitle(str_model) 
        ax1 = fig.add_subplot(111,projection='3d')
        if p.str_display_data[0]=='1': ax1.scatter(X1_train,X2_train,T_train,c='b',marker='x',s=100)  # plot training data points (blue x)
        if p.str_display_data[1]=='1': ax1.scatter(X1_test ,X2_test ,T_test ,c='y',marker='o',s=50)   # plot test data points (small yellow o)
        if p.str_display_data[2]=='1': ax1.plot_wireframe(X1_dense,X2_dense,Y_true.reshape(X1_dense.shape), color='b', rstride=1,cstride=1)  # plot true function curve (blue dotted)
        if p.str_display_data[3]=='1': ax1.plot_wireframe(X1_dense,X2_dense,Y_dense.reshape(X1_dense.shape), color='r', rstride=1,cstride=1) # plot model curve for LSR regression (red)
        ax1.view_init(p.elevation,p.azimuth)                    # set viewing angles
        ax1.set_xlabel('x1')                                    # label on x-axis
        ax1.set_ylabel('x2')                                    # label on y-axis
        ax1.set_zlabel('y')                                     # label on y-axis
        ax1.set_xlim((p.xmin-0.01,p.xmax+0.01))                 # set x-limits
        ax1.set_ylim((p.xmin-0.01,p.xmax+0.01))                 # set x-limits
        ax1.set_zlim((p.ymin,p.ymax))                           # set y-limits
        d.im_results = getMatplotlibImage(fig)                  # transfer matplotlib image to ivisit image


# main program
sim=Sim()
sim.main_init()
sim.init()
sim.step()

iv.IVisit_main(sim=sim)
