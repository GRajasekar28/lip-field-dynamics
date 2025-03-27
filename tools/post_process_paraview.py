# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:55:57 2023

@author: gopalsamy
"""



#### given the folder path to the results files in .npz format provides the post processing file for paraview
## dynamics

### create group vtk files

import os
import sys
sys.path.append('../lip2d')

import post_process_vtk_format as pp

import numpy as np
from mesh import simplexMesh

mesh_file_name = r'D:\VBox shared folder\dynamics_liger\test_LF\msh\rec_plate_crack3_r2.msh'  ## path to mesh file
mesh_file_name = r'D:\VBox shared folder\dynamics_liger\test_LF\msh\rec_plate_crack3_r3.msh'  ## path to mesh file
mesh_file_name = r'D:\VBox shared folder\dynamics_liger\test_LF\msh\Kalthoff_r3.msh'  ## path to mesh file
#mesh_file_name = r'D:\VBox shared folder\dynamics_liger\test_LF\msh\crack_arrest.msh'  ## path to mesh file
#mesh_file_name = r'D:\VBox shared folder\dynamics_liger\test_LF\msh\rec_plate_crack_symmetry_12.msh'  ## path to mesh file

 ## path where all the results file in .npz format are stored (each file corresponds to each time step)
folder_path = r'D:\VBox shared folder\dynamics_liger\test_PF\data\OG2111300\rect_plate\fields' 
folder_path = r'D:\VBox shared folder\dynamics_liger\results_liger\rect_plate_pf\fields' 
folder_path = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\test_B\fields' 
#folder_path = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\new_law\Kalthoff_v2\fields' 
folder_path = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\new_law\Kalthoff_v0\fields - Copie\changed_files' 
#folder_path = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\crack_arrest_060323_1\fields' 
#folder_path = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\new_law\Kalthoff_v2\Kalthoff_v1\fields'
#folder_path = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\symmetry\rect_plate\imposed_sigma\fields'
#folder_path = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\PF\crack_arrest\fields' 
#folder_path = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\crack_arrest\test_91\fields'
#folder_path = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\symmetry\rect_plate\imposed_sigma_r3\fields'
#folder_path = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\symmetry\rect_plate\imposed_v7\fields'
#folder_path = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\old_law\new\fields'

 
post_pr_file_name = 'output_post_proc'  ## final file name to open in paraview




















##load mesh
mesh_file = simplexMesh.readGMSH(mesh_file_name)





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



## post_processes results stored in a new directory within given path
post_pr_dir = './post_proc'
os.makedirs(post_pr_dir,exist_ok = 1)
os.chdir(post_pr_dir)



vtk_instance = pp.store_data_as_vtk(mesh_file)
steps = srtd_indx
name = []
nodal_data= None

for it,fle_name in enumerate(srtd_files[:]):
    #if it%5 ==0:     ##skip some files
        print('\rPreparing file '+ str(it+1) +'/'+ str(len(srtd_files)), end = '\r')
        ld = np.load('../'+fle_name, allow_pickle=1)
        
        
        
        u = ld['u']   ## nodal dispalcements
        ux = u[:,0] ; uy = u[:,1];
        v = ld['v']   ## nodal velocities
        vx = v[:,0] ; vy = v[:,1];
        a = ld['a']   ## nodal accelaration
        ax = a[:,0] ; ay = a[:,1];
        d = ld['d']   ## face damage
        
        stress = ld['stress']
        sx = stress[:,0];  sy = stress[:,1]; sxy = stress[:,2];
        
        nodal_data = {'ux':ux, 'uy':uy, 'vx':vx, 'vy':vy, 'ax':ax, 'ay':ay}   
        face_data = {'damage':d, 'sx':sx,'sy':sy,'sxy':sxy} 
        #face_data = {'damage':d}
        #global_data = {'uF':np.array([uimp[it], Fy[it]])}
        
        name.append('sim'+str(it) )
        vtk_instance.save_vtk(path = name[-1], point_data=nodal_data, cell_data= face_data)
    
    

vtk_instance.group_vtk(source =name, dest = post_pr_file_name, indices= steps)
print('\nFind the file in '+ os.getcwd() + "\\" + post_pr_file_name + '.pvd')
    