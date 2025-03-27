# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 17:41:16 2023

@author: gopalsamy
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:55:57 2023

@author: gopalsamy
"""



##dynamcis 
## find velocity of crack 



import os
import sys

import numpy as np
#from mesh import simplexMesh
import pathlib
import matplotlib.pyplot as plt

mesh_file_name = r'D:\VBox shared folder\dynamics_liger\msh\rec_plate_crack_r3.msh'  ## path to mesh file

respath = r'D:\VBox shared folder\dynamics_liger\results_liger\Kalthoff_r2_v1_l1' 

 




with open( respath/pathlib.Path('time_cracktip.npz'),'rb') as f :
        data = np.load(f)
        crack_tip_pos = data['tip']
        time = data['time']

time = time[:315*5]
crack_tip_pos = crack_tip_pos[:315*5]
        
## calculate distance of crack tip
dist = [0]
for i, coord in enumerate(crack_tip_pos[1:]):
    dis = dist[-1]+np.sqrt((coord[0]-crack_tip_pos[i][0])**2 + (coord[1]-crack_tip_pos[i][1])**2)
    dist.append(dis)
dist = np.array(dist)

## remove repated elements
mask = []
tmp = None
for i,j in enumerate(dist):
    if tmp != j:
        mask.append(i)
        tmp =j

##Now:  np.unique(crack_tip_pos) = crack_tip_pos[mask] ;; masked time = time[mask]
## remove repeated values
time = time[mask]
dist = dist[mask]



## least square fiting of data points in the set of 3

ls_fitted_pos = [0]
for i in range(int(len(time)/3)):
    if i%2 ==0:
        a11 = time[2*i]**2+time[2*i+1]**2+time[2*i+2]**2
        a12 = time[2*i]+time[2*i+1]+time[2*i+2]
        a22 = 3
        b1  = time[2*i]*dist[2*i]+time[2*i+1]*dist[2*i+1]+time[2*i+2]*dist[2*i+2]
        b2 =  dist[2*i]+dist[2*i+1]+dist[2*i+2]
        det = a22*a11- a12**2
        
        adj_a = (1/det)*np.array([[a22, -a12], [-a12, a11]])
        b = np.array([b1,b2])
        sol = adj_a.dot(b)
        m = sol[0]
        c = sol[1]
        
        #ls_fitted_pos.append(m*time[2*i]+c)
        ls_fitted_pos.append(m*time[2*i+1]+c)
        ls_fitted_pos.append(m*time[2*i+2]+c)


vel = [0]
for i,j in enumerate(dist[:-1]):
    vel.append((dist[i+1]-dist[i])/(time[i+1]-time[i]))
    
    
## smoothened dist and velcity profiles
pfit = np.polyfit(time,dist,deg=15)

fit_dist = np.polyval(p=pfit, x=time)

fit_vel =[0]
for i,j in enumerate(fit_dist[:-1]):
    fit_vel.append((fit_dist[i+1]-fit_dist[i])/(time[i+1]-time[i]))



