# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 12:43:41 2023

@author: gopalsamy
"""

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



meshfilename = '../msh/rec_plate_crack_symmetry_11.msh'
results_path = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\symmetry\rect_plate\imposed_sigma_r1\fields'
branching_index = 139     ##mesh r1
#branching_index = 280    ## mesh r2
#branching_index = 408    ## mesh r3

mesh = simplexMesh.readGMSH(meshfilename)

L = 0.1 # lenght of the Domain
H = .04 #height of domain
#nnodes = 102.  ## number of nodes on H for r1 mesh
lc = 5*2.5e-4  # lenght scale of the lip constrain (m)
E = 3.2e10 #Pa
nu = 0.2
lamb, mu = (E*nu/(1.+nu)/(1.-2*nu), E/2./(1+nu))
eta = .1
rho = 2450. #kg/m3 
Gc = 3  #J/m2

Yc = Gc/4./lc

hmin = np.min(mesh.innerCirleRadius())
wavespeed = np.sqrt(E/rho)



"""IMP: requires cahnge as per the simualtion.. during simulation keep costant saveperiod"""
dt = .8*hmin/wavespeed


law = mlaws.SofteningElasticityAsymmetric(Yc = Yc)



#mech = mech2d.Mechanics2D(mesh, law,lipprojector=None, lc=lc, log = None)

areas = mesh.areas()






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
        
        if it< branching_index:
            cl = np.dot(d , areas )/(2*lc)
        else:
            cl = np.dot(d , areas )/(lc)
        crack_length.append(cl)
        time.append(dt*steps[it])
        

crack_length = np.array(crack_length)
time = np.array(time)

crack_vel = np.diff(crack_length)/np.diff(time)
crack_vel = [0] + list(crack_vel)

r_wspd = 2125

#plt.plot(time,np.array(crack_vel)/r_wspd)

"""
import matplotlib


font = {'size'   : 16}

matplotlib.rc('font', **font)
plt.plot(np.array(time1)*1e6,np.array(crack_vel1)/r_wspd)
plt.plot(np.array(time2)*1e6,np.array(crack_vel2)/r_wspd,'--')
plt.plot(np.array(time3)*1e6,np.array(crack_vel3)/r_wspd,'-.')
plt.legend(['Mesh 1','Mesh 2', 'Mesh 3'])
plt.xlim([-1,80])
plt.ylim([-1e-3,1.3])
plt.xlabel('Time ['+ r"$\mu$" + 's]' )
plt.ylabel(r"$v/c_R$")
        
"""   

        
