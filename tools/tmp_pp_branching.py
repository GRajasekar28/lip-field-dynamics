# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 13:34:38 2023

@author: gopalsamy
"""


"""psot proc for paper dynamic crack branching symmetry (multiply by 2 for energies)"""

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

"""
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
"""


font = {'size'   : 16}

matplotlib.rc('font', **font)

file = 'time_energy.npz'
#file = 'time_cracktip.npz'

file1 = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\symmetry\rect_plate\imposed_sigma_r1'+'\\'+file
file2 = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\symmetry\rect_plate\imposed_sigma'+'\\'+file
file3 = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\symmetry\rect_plate\imposed_sigma_r3'+'\\'+file

a1 = np.load(file1)
a2 = np.load(file2)
a3 = np.load(file3)


t1 = a1['time'] 
t2 = a2['time']
t3 = a3['time']



def crack_tip_velocity(time, crack_tip_pos, smooth = True):
    
    time_copy = time.copy()
    ## calculate distance of crack tip
    dist = [0]
    for i, coord in enumerate(crack_tip_pos[1:]):
        dis = dist[-1]+np.sqrt((coord[0]-crack_tip_pos[i][0])**2 + (coord[1]-crack_tip_pos[i][1])**2)
        dist.append(dis)
    dist = np.array(dist)
    
    
    if not smooth:
        dt = time[1]-time[0]
        vel = [0]
        for i in range(len(dist[1:])):
            vel.append((dist[i+1]-dist[i])/dt)
        return time, vel
    

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
    
    
    vel = [0]
    for i,j in enumerate(dist[:-1]):
        vel.append((dist[i+1]-dist[i])/(time[i+1]-time[i]))
        
    #return time, vel
        
    ## smoothened dist and velcity profiles
    pfit = np.polyfit(time,dist,deg=39)
    
    fit_dist = np.polyval(p=pfit, x=time)
    
    fit_vel =[0]
    for i,j in enumerate(fit_dist[:-1]):
        fit_vel.append((fit_dist[i+1]-fit_dist[i])/(time[i+1]-time[i]))
    
    #return list(time), fit_vel
    return list(time)+list(time_copy[mask[-1]+1 :]), fit_vel+[0]*len(time_copy[mask[-1]+1 :])



## part 1 (time vs crack-tip velocity)
"""
t1,v1 = crack_tip_velocity(t1, a1['tip'])
t2,v2 = crack_tip_velocity(t2, a2['tip'])
t3,v3 = crack_tip_velocity(t3, a3['tip'])

t1 = np.array(t1)
t2 = np.array(t2)
t3 = np.array(t3)

v1 = np.array(v1)
v2 = np.array(v2)
v3 = np.array(v3)

E = 3.2e10
rho = 2450. #kg/m3
wavespeed = np.sqrt(E/rho)
r_wspd = 2125

plt.plot(t1*1e6,v1/r_wspd,'--')
plt.plot(t2*1e6,v2/r_wspd,'--')
plt.plot(t3*1e6,v3/r_wspd,'--')

plt.legend(['Mesh 1','Mesh 2','Mesh 3'])

plt.xlabel('Time '+ r"$\mu$" + 's' )
plt.ylabel(r"$v/c_R$")

"""


## part 2 (time vs energies)  (multiply by 2 due to symmetry)
ep1 = [2*i for i in a1['ep']]
ep2 = [2*i for i in a2['ep']]
ep3 = [2*i for i in a3['ep']]




ed1 = [2*i for i in a1['ed']]
ed2 = [2*i for i in a2['ed']]
ed3 = [2*i for i in a3['ed']]

ec1 = [2*i for i in a1['ec']]
ec2 = [2*i for i in a2['ec']]
ec3 = [2*i for i in a3['ec']]


## total energy
et1 = [2*i for i in a1['ec']+a1['ep']+a1['ed']]
et2 = [2*i for i in a2['ec']+a2['ep']+a2['ed']]
et3 = [2*i for i in a3['ec']+a3['ep']+a3['ed']]


plt.plot(t1*1e6,ep1)
plt.plot(t2*1e6,ep2,'--')
plt.plot(t3*1e6,ep3,'-.')

plt.xlim([-1,80])
plt.ylim([-1e-3,.18])

plt.legend(['Mesh 1','Mesh 2','Mesh 3'])

plt.xlabel('Time '+ r"$[\mu$" + 's]' )
plt.ylabel("Potential energy [J/m]")
