# coding: utf-8
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np, scipy.stats
import matplotlib.pyplot as plt
from time import perf_counter

import ivisit as iv
from ivisit.matplotlib import *

from GaussDataGeneration import *
from KNearestNeighborSearch import *
from Classifier import *

# *************************************************************************************************************
# (I) IVISIT GUI elements
# *************************************************************************************************************

#@IVISIT:SIMULATION & ivisit_LinearClassifiers 

#@IVISIT:SLIDER     & seed            & [200,1] & [0,100,5,1]        & seed            & -1 & int   & 0      # seed for reproducible results
#@IVISIT:SLIDER     & K               & [200,1] & [1,100,5,1]        & K               & -1 & int   & 1      # number of nearest neighbors (for K-NN classification)
#@IVISIT:SLIDER     & S               & [200,1] & [1,100,5,1]        & S               & -1 & int   & 2      # number data partitions for cross validation 
#@IVISIT:SLIDER     & t_avg           & [200,1] & [1,100,5,1]        & t_avg           & -1 & int   & 10     # number of time steps to average 
#@IVISIT:CHECKBOX   & Classifiers     & [KNN,FastKNN,KernelMLP]      & str_classifiers & 100                 # selected classifiers

#@IVISIT:DICTSLIDER & Class1-GaussData-Parameters & [200,20,-1,2,10] & par_GaussData1 & 0                    # class1 data parameters
#@IVISIT:DICTSLIDERITEM  & N          & [0,200, 3,1]    & N        & int   & 75 
#@IVISIT:DICTSLIDERITEM  & mu_1       & [-5, 5, 3,0.1]  & mu_1     & float & -1.0 
#@IVISIT:DICTSLIDERITEM  & mu_2       & [-5, 5, 3,0.1]  & mu_2     & float & 2.0 
#@IVISIT:DICTSLIDERITEM  & Sigma_11   & [0, 5, 3,0.1]   & Sigma_11 & float & 2.0
#@IVISIT:DICTSLIDERITEM  & Sigma_22   & [0, 5, 3,0.1]   & Sigma_22 & float & 3.0
#@IVISIT:DICTSLIDERITEM  & Sigma_12   & [-5, 5, 3,0.1]  & Sigma_12 & float & 0.5

#@IVISIT:DICTSLIDER & Class2-GaussData-Parameters & [200,20,-1,2,10] & par_GaussData2 & 0                    # class2 data parameters
#@IVISIT:DICTSLIDERITEM  & N          & [0,200, 3,1]    & N        & int   & 100 
#@IVISIT:DICTSLIDERITEM  & mu_1       & [-5, 5, 3,0.1]  & mu_1     & float & 2.0 
#@IVISIT:DICTSLIDERITEM  & mu_2       & [-5, 5, 3,0.1]  & mu_2     & float & -1.0 
#@IVISIT:DICTSLIDERITEM  & Sigma_11   & [0, 5, 3,0.1]   & Sigma_11 & float & 2.0
#@IVISIT:DICTSLIDERITEM  & Sigma_22   & [0, 5, 3,0.1]   & Sigma_22 & float & 1.0
#@IVISIT:DICTSLIDERITEM  & Sigma_12   & [-5, 5, 3,0.1]  & Sigma_12 & float & 0.5

#@IVISIT:IMAGE       & Data Vectors   & 1.0     & [0,255]          & im_results      & int                   # image for displaying data
#@IVISIT:TEXT_OUT    & Results        & [20,5]  & just_left        & str_results

# *************************************************************************************************************
# (II) Auxiliary Data and Functions 
# *************************************************************************************************************

font = {'family' : 'normal',
        'weight' : 'normal', #'bold',
        'size'   : 16}
plt.rc('font', **font)                         # define font for plots in matplotlib


# *************************************************************************************************************
# (III) IVISIT Classes for Parameters, Data, and Simulation  
# *************************************************************************************************************

# (III.1) define parameters to be controlled by IVisit
class SimParameters(iv.IVisit_Parameters):
    seed = 14           # seed of random generator (to be able to reproduce results)
    K    = 1            # number of nearest neighbors (for KNN-classification)
    S    = 2            # number data partitions for cross validation 
    str_classifiers='100'             # selected classifiers (KNN/FastKNN/KernelMLP)
    par_GaussData1={'N':75,           # number of data points
                    'mu_1':-1.0,      # expectation (component 1)
                    'mu_2':2.0,       # expectation (component 2)
                    'Sigma_11':2.0,   # variance (component 1)
                    'Sigma_22':3.0,   # variance (component 2)
                    'Sigma_12':0.5    # covariance (between components 1 and 2)
    }
    par_GaussData2={'N':100,          # number of data points
                    'mu_1':2.0,       # expectation (component 1)
                    'mu_2':-1.0,      # expectation (component 2)
                    'Sigma_11':2.0,   # variance (component 1)
                    'Sigma_22':1.0,   # variance (component 2)
                    'Sigma_12':0.5    # covariance (between components 1 and 2)
    }
    xmin,xmax,ymin,ymax = -8,8,-8,8   # axis limits
    t_avg,t_avg_max = 20,100          # time window for averaging (and slider maximum)
    
# (III.2) define simulation data to be displayed by IVisit
class SimData(iv.IVisit_Data):
    im_results  = np.array([[0,0,0]])  # image of data vectors

# (III.3) define Simulation Class
class Sim(iv.IVisit_Simulation):
    def __init__(self,name_arg="ivisit_GaussDataGeneration.py"):
        iv.IVisit_Simulation.__init__(self,name_arg,SimParameters,SimData)
        self.datvec_clicked_x, self.datvec_clicked_y=-1,-1 # clicked coordinates on image canvas
        self.datvec_canvas=None                            # image canvas
        self.ax1=None                                      # reference to matplotlib axis
        self.x=[0.0,0.0]                                   # new input vector to be classified (will be defined by clicking on image im_results)
        p = SimParameters                                  # short hand to simulation parameters
        self.list_t_datagen =np.zeros(p.t_avg_max)         # time list for averaging over t_datagen  (time for data generation)
        self.list_t_train   =np.zeros(p.t_avg_max)         # time list for averaging over t_train    (time for creating and training classifiers)
        self.list_t_classify=np.zeros(p.t_avg_max)         # time list for averaging over t_classify (time for a single classification of x)
        self.list_t_crossval=np.zeros(p.t_avg_max)         # time list for averaging over t_crossval (time for a cross validation of all data)
        self.list_t_pos=-1                                 # current position in times lists
        
    def bind(self,parent=None,display=None):    # parent is typically the ivisit object (having a display object where to bind to)
        if parent!=None: self.parent=parent
        if display==None: display=self.display
        if display==None and self.parent!=None: display=parent.display
        if display!=None:
            w=display.getWidget("Data Vectors")
            self.datvec_canvas=display.bind2Widget("Data Vectors","<Button-1>",self.onPressedB1_datvec,"imgcanvas")   # bind mouse clicks on image to function 
            self.display=display     # store reference to display for later manipulations etc.

    def onPressedB1_datvec(self,event):
        d = SimData          # short hand to simulation data arrays
        x,y=event.x,event.y  # clicked position (in pixel coordinates)
        self.x[0],self.x[1] = getDataPos_from_PixelPos(x,y, d.im_results.shape[0], self.ax1)  # clicked position (axis coordinates) defines input vector
            
    def step(self):
        p = SimParameters    # short hand to simulation parameters
        d = SimData          # short hand to simulation data arrays
        self.list_t_pos=(self.list_t_pos+1)%p.t_avg # increment position pointer in time lists
        
        # (i) generate data
        t_start=perf_counter()
        C=2                                                  # number of classes
        np.random.seed(p.seed)                               # set seed of random generator (to be able to regenerate data)
        dp=p.par_GaussData1                                  # reference to data parameters for class 1
        X1,T1=getGaussData2D(dp['N'],
                             dp['mu_1'],dp['mu_2'],
                             dp['Sigma_11'], dp['Sigma_22'], dp['Sigma_12'],
                             t=0,C=C,flagOneHot=0)           # create data for class 1
        dp=p.par_GaussData2                                  # reference to data parameters for class 2
        X2,T2=getGaussData2D(dp['N'],
                             dp['mu_1'],dp['mu_2'],
                             dp['Sigma_11'], dp['Sigma_22'], dp['Sigma_12'],
                             t=1,C=C,flagOneHot=0)           # create data for class 2
        X = np.concatenate((X1,X2))
        T = np.concatenate((T1,T2))
        self.list_t_datagen[self.list_t_pos]=perf_counter()-t_start # measure time and store in list
        
        # (ii) create and train classifiers to evaluate
        t_start=perf_counter()
        list_clf=[]
        if p.str_classifiers[0]=='1': list_clf+=[['KNN'      ,KNNClassifier(C,p.K)]]           # create naive KNN classifier?
        if p.str_classifiers[1]=='1': list_clf+=[['FastKNN'  ,FastKNNClassifier(C,p.K)]]       # create fast KNN classifier?
        if p.str_classifiers[2]=='1': list_clf+=[['KernelMLP',KernelMLPClassifier(C,np.tanh)]] # create Kernel-MLP classifier?
        for clf in list_clf: clf[1].fit(X,T)
        self.list_t_train[self.list_t_pos]=perf_counter()-t_start   # measure time and store in list

        # (iii) classify input x
        t_start=perf_counter()
        list_classify_x = [clf[1].predict(self.x) for clf in list_clf]        # classify for each selected classifier
        self.list_t_classify[self.list_t_pos]=perf_counter()-t_start # measure time and store in list

        # (iv) do cross validation on whole data set
        t_start=perf_counter()
        list_crossval = [clf[1].crossvalidate(p.S,X,T) for clf in list_clf]  # cross validation results for each selected classifier
        self.list_t_crossval[self.list_t_pos]=perf_counter()-t_start # measure time and store in list

        # (iv) write results
        d.str_results="-------------------------------------\n"
        d.str_results+="Input x="+str(self.x)+"\n\n"
        for i in range(len(list_clf)):
            d.str_results+=str(list_clf[i][0])+": y_hat(x)="+str(list_classify_x[i][0])+"; p(c|x)="+str(list_classify_x[i][1])+"\n"
            d.str_results+="   crossval: err="+str(list_crossval[i][0])+" Cp=["+str(list_crossval[i][1][0])+";"+str(list_crossval[i][1][1])+"]\n\n"
        d.str_results+="t_datagen ="+str(np.mean(self.list_t_datagen [0:p.t_avg]))+"\n"
        d.str_results+="t_train   ="+str(np.mean(self.list_t_train   [0:p.t_avg]))+"\n"
        d.str_results+="t_classify="+str(np.mean(self.list_t_classify[0:p.t_avg]))+"\n"
        d.str_results+="t_crossval="+str(np.mean(self.list_t_crossval[0:p.t_avg]))+"\n"
        d.str_results+="(averaged over "+str(p.t_avg)+" steps)\n"
        
        # (v) plot data
        fig = getMatplotlibFigure(figsize=(6,5))                 # create figure
        ax1 = fig.add_subplot(111)                               # create axis for contour plot of posterior distribution
        ax1.scatter(X1[:,0],X1[:,1],c='r',marker='x',s=20)                                     # plot data from class 1
        ax1.scatter(X2[:,0],X2[:,1],marker='o',s=20, facecolors='none', edgecolors='b')        # plot data from class 2
        ax1.scatter([self.x[0]], [self.x[1]], marker='*',s=40, facecolors='k', edgecolors='k') # plot new input vector
        #ax1.scatter(X_KNN[:,0],X_KNN[:,1],marker='o',s=60, facecolors='none', edgecolors='c')  # plot data of the K neareste neighbors 
        ax1.set_xlabel('x1'); ax1.set_ylabel('x2')               # labels on x/y-axis
        ax1.grid(which='both')                                   # draw a grid
        ax1.set_xlim((p.xmin,p.xmax))                            # limits for axis
        ax1.set_ylim((p.ymin,p.ymax))                            # limits for axis
        #circleKNN=plt.Circle((self.x[0],self.x[1]), r_KNN ,fill=False, edgecolor='c', linestyle='--') # Circle around input x with radius r_KNN
        #ax1.add_artist(circleKNN)                                # add circle to plot
        d.im_results = getMatplotlibImage(fig)                   # send matplotlib image to ivisit image
        self.ax1=ax1

        
# *************************************************************************************************************
# Main Program: Just start IVISIT simulation
# *************************************************************************************************************

sim=Sim()
sim.main_init()
sim.init()
sim.step()

iv.IVisit_main(sim=sim)
