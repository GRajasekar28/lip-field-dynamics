# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:05:28 2023

@author: gopalsamy
"""

import sys
sys.path.append('../lip2d')
#sys.path = ['/home/nchevaug/.local/lib/python3.8/site-packages/cvxopt-1.2.7+0.gd5a21cf.dirty-py3.8-linux-x86_64.egg'] + sys.path
#sys.path = ['/home/nchevaug/.local/lib/python3.8/site-packages/cvxopt-0+unknown-py3.8-linux-x86_64.egg']+sys.path
import cvxopt
import os
import matplotlib.pyplot as plt
import numpy as np
import material_laws_planestrain as mlaws
from   mesh import simplexMesh, dualLipMeshTriangle
import mechanics2D as mech2d
import lipdamage as lipprojector
import liplog    as liplog
import dynamics as dynamics
import shutil
import pathlib

import scipy
import scipy.sparse



meshfilename = '../msh/Kalthoff_r3.msh'
results_path = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\new_law\Kalthoff_v0\fields - Copie\changed_files' 


mesh = simplexMesh.readGMSH(meshfilename)

L = 0.1 # lenght of the Domain
lc = 20.*L/1000. 
E = 190e9 #Pa
rho = 8000.
Gc = 22.2e3 
Yc = Gc/4./lc

hmin = np.min(mesh.innerCirleRadius())
wavespeed = np.sqrt(E/rho)



"""IMP: requires cahnge as per the simualtion.. during simulation keep costant saveperiod"""
dt = .9*hmin/wavespeed


law = mlaws.SofteningElasticityAsymmetric(Yc = Yc)



#mech = mech2d.Mechanics2D(mesh, law,lipprojector=None, lc=lc, log = None)

areas = mesh.areas()


## for primary crack at top
## element id's where  crack_length is not to be calculated
"""
videl_d0 = []
a_low = 0.025*.75    ;   a_left = 0
a_uppe = np.inf     ;   a_right = np.inf
for ele_num,ele in enumerate(mesh.triangles):
    for ver in ele:
        x_ver = mesh.xy[ver,0]
        y_ver = mesh.xy[ver,1]
        if y_ver<a_low or y_ver> a_uppe or x_ver< a_left or x_ver > a_right :
            videl_d0.append(ele_num)
            break

## element id's where  crack_length is to be calculated
videl = list(set(range(mesh.ntriangles)).difference(set(videl_d0)))
"""


## for secondary crack at the bottom
## element id's where  crack_length is to be calculated
videl_d0 = []
a_low = 0.025*.75    ;   a_left = 0
a_uppe = np.inf     ;   a_right = np.inf
for ele_num,ele in enumerate(mesh.triangles):
    for ver in ele:
        x_ver = mesh.xy[ver,0]
        y_ver = mesh.xy[ver,1]
        if y_ver<a_low or y_ver> a_uppe or x_ver< a_left or x_ver > a_right :
            videl_d0.append(ele_num)
            break
videl = videl_d0    
  
"""
videl_d1 = []
a_low = 0.025*.2    ;   a_left = 0
a_uppe = np.inf     ;   a_right = np.inf 
for ele_num,ele in enumerate(mesh.triangles):
    for ver in ele:
        x_ver = mesh.xy[ver,0]
        y_ver = mesh.xy[ver,1]
        if y_ver<a_low or y_ver> a_uppe or x_ver< a_left or x_ver > a_right :
            videl_d1.append(ele_num)
            break
## element id's where  crack_length is to be calculated
#videl = list(set(range(mesh.ntriangles)).difference(set(videl_d0)))
videl = list(set(videl_d0).difference(set(videl_d1)))
"""





os.chdir(results_path)


a = [i for i in os.listdir(results_path) if os.path.isfile(i)]



## get steps
steps = []   ## list of steps
for fle_name in a:
    step = int(fle_name.split('_')[1].split('.')[0])
    steps.append(step)

## do some sorting 
## (as the files names might not be in the order of steps)
srtd_indx = sorted(range(len(steps)), key = lambda k: steps[k])

srtd_files = [a[i] for i in srtd_indx]


crack_length = []
time = []

for it,fle_name in enumerate(srtd_files):
    #if it%5 ==0:     ##skip some files
        print('\rPreparing file '+ str(it+1) +'/'+ str(len(srtd_files)), end = '\r')
        ld = np.load(fle_name, allow_pickle=1)
        
        d = ld['d']   ## face damage
        
        cl = np.dot(d[videl] , areas[videl] )/lc
        crack_length.append(cl)
        time.append(dt*steps[it])
        

crack_length = np.array(crack_length)
time = np.array(time)

crack_vel = np.diff(crack_length)/np.diff(time)
crack_vel = [0] + list(crack_vel)


crack_vel = np.array(crack_vel)

import matplotlib


font = {'size'   : 16}

matplotlib.rc('font', **font)


r_wspd = 2799
plt.plot(time*1e6,crack_vel/r_wspd)


plt.xlabel('Time ['+ r"$\mu$" + 's]' )
plt.ylabel(r"$v/c_R$")
        
        

        
