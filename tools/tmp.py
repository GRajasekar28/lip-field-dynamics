# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 11:21:31 2023

@author: gopalsamy
"""

"""Just for processng Kalthoff old law results"""


#### given the folder path to the results files in .npz format provides the post processing file for paraview
## dynamics

### create group vtk files

import os
import sys
import shutil
sys.path.append('../lip2d')

import post_process_vtk_format as pp

import numpy as np

folder_path = r'D:\VBox shared folder\dynamics_liger\results_liger\test_030323\old_law\fields'






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
post_pr_dir = './arranged_files'
os.makedirs(post_pr_dir,exist_ok = 1)
os.chdir(post_pr_dir)



steps = srtd_indx

for it,fle_name in enumerate(srtd_files[:]):
    if it%10 ==0:     ##skip some files
        print('\rPreparing file '+ str(it+1) +'/'+ str(len(srtd_files)), end = '\r')
        shutil.copy('../'+fle_name, './'+fle_name)
        