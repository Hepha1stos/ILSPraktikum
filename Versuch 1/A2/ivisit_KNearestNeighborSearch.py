# coding: utf-8
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np, scipy.stats
import matplotlib.pyplot as plt
import ivisit as iv
from ivisit.matplotlib import *

from GaussDataGeneration import *
from KNearestNeighborSearch import *

# *************************************************************************************************************
# (I) IVISIT GUI elements
# *************************************************************************************************************

#@IVISIT:SIMULATION & ivisit_LinearClassifiers 

#@IVISIT:SLIDER     & seed            & [200,1] & [0,100,5,1]        & seed            & -1 & int   & 0      # seed for reproducible results
#@IVISIT:SLIDER     & K               & [200,1] & [1,100,5,1]        & K               & -1 & int   & 1      # number of nearest neighbors (for K-NN classification)

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
plt.rc('font', **font)                         # define font for plost in matplotlib


# *************************************************************************************************************
# (III) IVISIT Classes for Parameters, Data, and Simulation  
# *************************************************************************************************************

# (III.1) define parameters to be controlled by IVisit
class SimParameters(iv.IVisit_Parameters):
    seed = 14           # seed of random generator (to be able to reproduce results)
    K    = 1            # number of nearest neighbors (for KNN-classification)
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

        # (i) generate data
        np.random.seed(p.seed)                               # set seed of random generator (to be able to regenerate data)
        dp=p.par_GaussData1                                  # reference to data parameters for class 1
        X1,T1=getGaussData2D(dp['N'],
                             dp['mu_1'],dp['mu_2'],
                             dp['Sigma_11'], dp['Sigma_22'], dp['Sigma_12'],
                             t=1,C=2,flagOneHot=0)           # create data for class 1
        dp=p.par_GaussData2                                  # reference to data parameters for class 2
        X2,T2=getGaussData2D(dp['N'],
                             dp['mu_1'],dp['mu_2'],
                             dp['Sigma_11'], dp['Sigma_22'], dp['Sigma_12'],
                             t=2,C=2,flagOneHot=0)           # create data for class 2
        X = np.concatenate((X1,X2))
        T = np.concatenate((T1,T2))

        # (ii) find nearest neighbor for input x
        idx_KNN = [0]                                        # !!REPLACE!! get indexes of K nearest neighbors
        X_KNN = X[idx_KNN]                                   # nearest neighbor vectors
        T_KNN = T[idx_KNN]                                   # nearest neighbor class labels
        r_KNN = np.linalg.norm(X_KNN[-1]-self.x)             # largest Euklidean distance defines radius of sphere containing all K nearest neighbors

        # (iii) do classification
        P=[0.5, 0.5]                                         # !!REPLACE!! get class probabilities; subtract 1 from T_KNN to get class labels 0,1 (instead of 1,2) as required
        c=classify(P)+1                                      # select class decision (add 1 to have again class labels 1,2 instead of 0,1)

        # (iv) write results
        d.str_results="-------------------------------------\n"
        d.str_results+="Input x="+str(self.x)+"\n"
        d.str_results+="Class distribution: P="+str(P)+"\n"
        d.str_results+="Class decision: c="+str(c)+" with P(c)="+str(P[c-1])+"\n"
        
        # (iii) plot data
        fig = getMatplotlibFigure(figsize=(6,5))                 # create figure
        ax1 = fig.add_subplot(111)                               # create axis for contour plot of posterior distribution
        ax1.scatter(X1[:,0],X1[:,1],c='r',marker='x',s=20)                                     # plot data from class 1
        ax1.scatter(X2[:,0],X2[:,1],marker='o',s=20, facecolors='none', edgecolors='b')        # plot data from class 2
        ax1.scatter([self.x[0]], [self.x[1]], marker='*',s=40, facecolors='k', edgecolors='k') # plot new input vector
        ax1.scatter(X_KNN[:,0],X_KNN[:,1],marker='o',s=60, facecolors='none', edgecolors='c')  # plot data of the K neareste neighbors 
        ax1.set_xlabel('x1'); ax1.set_ylabel('x2')               # labels on x/y-axis
        ax1.grid(which='both')                                   # draw a grid
        ax1.set_xlim((p.xmin,p.xmax))                            # limits for axis
        ax1.set_ylim((p.ymin,p.ymax))                            # limits for axis
        circleKNN=plt.Circle((self.x[0],self.x[1]), r_KNN ,fill=False, edgecolor='c', linestyle='--') # Circle around input x with radius r_KNN
        ax1.add_artist(circleKNN)                                # add circle to plot
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
