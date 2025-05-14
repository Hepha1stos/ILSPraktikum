# coding: utf-8
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@IVISIT:SIMULATION  & ivisit_RegLinearRegr 
#@IVISIT:SLIDER      & seed          & [200,1] & [0,100,5,1]    & seed        & -1 & int   & 0    # parameters for data generation
#@IVISIT:SLIDER      & N             & [200,1] & [1,1000,5,1]   & N           & -1 & int   & 10 
#@IVISIT:SLIDER      & sd_noise      & [200,1] & [0,2,5,0.01]   & sd_noise    & -1 & float & 0.0 
#@IVISIT:RADIOBUTTON & Regression Model & [Least-Squares,K-Nearest-Neighbors] & str_regmodel     & Least-Squares
#@IVISIT:CHECKBOX    & Display Data     & [train-data,test-data]              & str_display_data & 11

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
import ivisit as iv
from ivisit.matplotlib import *
from polynomial_basis_functions import *
from Regression import *

font = {'family' : 'normal',
        'weight' : 'normal', #'bold',
        'size'   : 16}
plt.rc('font', **font)

def fun_true(X,f=1.0,phi0=0.0,c=1.0,mu=0.0,sig=0):   # compute 1-dim. sin-function f(x)=c*sin(2*pi*f*x+phi0)+N(mu,sig^2)
    X=np.array(X.flat)                                                   # use flat vectors to generate target values 
    return c*np.sin(2*np.pi*f*X+phi0)+np.random.normal(mu,sig,X.shape)   # return function values (same size as X.flat)
              
def getMAE(Y,T, eps=1e-6):                           # compute mean absolute error (and MAPE) for evaluation
    AE=np.abs(np.array(Y.flat)-np.array(T.flat))     # absolute error components
    APE=np.divide(AE,np.maximum(eps,np.abs(Y.flat))) # absolute percentage error
    return np.sum(AE)/len(AE), np.sum(APE)/len(APE)  # return MAE and MAPE  

# (I) define parameters to be controlled by IVisit
class SimParameters(iv.IVisit_Parameters):
    seed       = 13                # seed of random generator (to be able to reproduce results)
    f,phi0,c,d = 1.0,0.0,1.0,0.0   # parameters of true function y(x)=c*sin(2*pi*f*x+phi0)+d
    xmin,xmax = 0.0, 1.0   # range of x-axis where to draw data from
    ymin,ymax = -2, 2      # range of y-axis
    N         = 10         # number of training/testing data
    N_dense   = 500        # number of densely sampled data (for plotting model curve)
    sd_noise  = 0.2        # noise power
    str_regmodel = 'Least-Squares' # regression model to be applied to the data (either least-squares or KNN)
    str_display_data = '11'        # flags for displaying train/test-data?
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
        X_train=np.linspace(p.xmin,p.xmax,p.N).reshape((p.N,1));  # linear spaced x values
        T_train=fun_true(X_train,sig=p.sd_noise);                 # target values (including noise) 
        X_test=X_train+0.5*(p.xmax-p.xmin)/max(1,p.N-1)           # place x values for testing between those for training
        T_test=fun_true(X_test,sig=p.sd_noise)                    # test targets (with same amount of noise as for training data) 
        
        # (ii) create and train regression model
        if p.str_regmodel == 'Least-Squares':               # which regression model should be used?
            lmbda=pLSR['lmbda_scale']*(10**pLSR['lmbda_log10']) # compute regularization coeffizient
            phi=get_phi_poly(1,pLSR['deg'])                     # define basis functions as 1D polynomial of degree p.deg
            regm = LSRRegressifier(lmbda,phi,pLSR['flagSTD'],10**pLSR['eps_log10']) # create linear regression model (least squares with regularization)
            str_model = "Least Squares with lambda="+str(lmbda)
        elif p.str_regmodel == 'K-Nearest-Neighbors':
            lr_lmbda=pKNN['lr_lmbda_scale']*(10**pKNN['lr_lmbda_log10']) # compute regularization coeffizient
            lr_phi=get_phi_poly(1,pKNN['lr_deg'])                        # define basis functions as 1D polynomial of degree p.deg
            regm = KNNRegressifier(pKNN['K'],pKNN['flagKLinReg'],lr_lmbda,lr_phi,pKNN['lr_flagSTD'],10**pKNN['lr_eps_log10'])   # create KNN regression model
            str_model = 'KNN with K='+str(pKNN['K'])
            if pKNN['flagKLinReg']>0: str_model+=' and lin. regr. with lr_lambda='+str(lr_lmbda)
        else:
            assert 0,"Unknown regression model: str_regmodel="+str(p.str_regmodel)
        regm.fit(X_train,T_train)                               # train model using training data
            
        # (iii) make predictions
        Y_train = np.array([regm.predict(x) for x in X_train])  # predictions on training data
        Y_test  = np.array([regm.predict(x) for x in X_test])   # predictions on test data
        X_dense = np.linspace(p.xmin,p.xmax,p.N_dense).reshape((p.N_dense,1)) # densely sampled x values (just for plotting the model curve)
        Y_dense = np.array([regm.predict(x) for x in X_dense])  # predictions on test data (just for plotting the model curve)
        MAE_train,MAPE_train = getMAE(Y_train,T_train)          # get mean absolute (percentage) error on train data
        MAE_test ,MAPE_test  = getMAE(Y_test ,T_test)           # get mean absolute (percentage) error on test data
        d.str_results=str_model+'\n'                            # write results string
        if p.str_regmodel == 'Least-Squares': d.str_results+='weights w='+str(regm.W_LSR)+'\n'
        d.str_results+='MAE_train='+str(MAE_train)+', MAPE_train='+str(MAPE_train)+'\n'
        d.str_results+='MAE_test ='+str(MAE_test) +', MAPE_test ='+str(MAPE_test)+'\n'
        
        # (iv) compute actual function
        Y_true = fun_true(X_dense).flat                         # ground truth (without noise)

        # (v) plot results
        fig = getMatplotlibFigure(figsize=(8,6))
        fig.suptitle(str_model) 
        ax1 = fig.add_subplot(111)
        if p.str_display_data[0]=='1': ax1.scatter(X_train.flat,T_train.flat,c='b',marker='x',s=100)  # plot training data points (blue x)
        if p.str_display_data[1]=='1': ax1.scatter(X_test.flat ,T_test.flat ,c='y',marker='o',s=50)   # plot test data points (small yellow o)
        ax1.plot(X_dense,Y_dense, c='r', linewidth=2)           # plot model curve for LSR regression (red)
        ax1.plot(X_dense,Y_true , c='b', linestyle='--')        # plot true function curve (blue dotted)
        ax1.set_xlabel('x')                                     # label on x-axis
        ax1.set_ylabel('y')                                     # label on y-axis
        ax1.grid(which='both')                                  # draw a grid
        ax1.set_xlim((p.xmin-0.01,p.xmax+0.01))                 # set x-limits
        ax1.set_ylim((p.ymin,p.ymax))                           # set y-limits
        ax1.set_xticks(np.linspace(0,1,10+1))                   # set x-ticks
        ax1.set_yticks(np.linspace(-2,2,10+1))                  # set y-ticks
        d.im_results = getMatplotlibImage(fig)                  # transfer matplotlib image to ivisit image


# main program
sim=Sim()
sim.main_init()
sim.init()
sim.step()

iv.IVisit_main(sim=sim)
