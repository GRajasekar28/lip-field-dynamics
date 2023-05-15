# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 12:13:54 2023

@author: gopalsamy
"""





### create group vtk files

import os
import sys
sys.path.append('../lip2d')

import post_process_vtk_format as pp

import numpy as np
from mesh import simplexMesh


 ## path where all the results file in .npz format are stored (each file corresponds to each time step)
folder_path = r'D:\VBox shared folder\dynamics_liger\test_PF\data\OG2111300\rect_plate\fields' 
folder_path = r'D:\VBox shared folder\dynamics_liger\results_liger\rect_plate_pf\fields' 
folder_path = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\test_B\fields' 
#folder_path = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\new_law\Kalthoff_v2\fields' 
folder_path = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\new_law\Kalthoff_v0\fields - Copie' 
#folder_path = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\crack_arrest_060323_1\fields' 
#folder_path = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\new_law\Kalthoff_v2\Kalthoff_v1\fields'
#folder_path = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\symmetry\rect_plate\imposed_sigma\fields'
#folder_path = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\PF\crack_arrest\fields' 
#folder_path = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\crack_arrest\test_91\fields'
#folder_path = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\symmetry\rect_plate\imposed_sigma_r3\fields'
#folder_path = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\symmetry\rect_plate\imposed_v7\fields'
#folder_path = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\old_law\new\fields'

 
#post_pr_file_name = 'output_post_proc'  ## final file name to open in paraview























os.chdir(folder_path)


a = [i for i in os.listdir(folder_path) if os.path.isfile(i)]


##get applied dispalcements

steps = []   ## list of steps
for fle_name in a:
    step = int(fle_name.split('_')[1].split('.')[0])
    steps.append(step)

## do some sorting 
## (as the files names might not be in the order of steps)
srtd_indx = sorted(range(len(steps)), key = lambda k: steps[k])

srtd_files = [a[i] for i in srtd_indx]
srtd_files_1 = list(reversed(srtd_files))


## post_processes results stored in a new directory within given path
post_pr_dir = './changed_files'
os.makedirs(post_pr_dir,exist_ok = 1)
os.chdir(post_pr_dir)




steps = srtd_indx
name = []
nodal_data= None



for it,fle_name in enumerate(srtd_files_1):
    #if it%5 ==0:     ##skip some files
        print('\rPreparing file '+ str(it+1) +'/'+ str(len(srtd_files)), end = '\r')
        ld = np.load('../'+fle_name, allow_pickle=1)
        
        
        
        u = ld['u']   ## nodal dispalcements
        
        v = ld['v']   ## nodal velocities
        
        a = ld['a']   ## nodal accelaration
        
        d = ld['d']   ## face damage
        
        stress = ld['stress']
        
        if it==0:
            with open('./'+fle_name,'wb') as f :
                np.savez(f,u=u,v=v,a=a,stress=stress,d=d)
            d1 = d.copy()
        else:
            d = np.minimum(d,d1)
            with open('./'+fle_name,'wb') as f :
                np.savez(f,u=u,v=v,a=a,stress=stress,d=d)
            d1 = d.copy()
            