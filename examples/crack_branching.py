# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 09:30:11 2023

@author: gopalsamy
"""

# -*- coding: utf-8 -*-

### PF testing  ::: IMP:  Yc and GC are used intercahngeably for PF (use of same variable names)

"""
Created on Thu Feb  9 15:47:52 2023

@author: gopalsamy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
// Copyright (C) 2021 Chevaugeon Nicolas
Created on Mon APRIL 17 9:54:10 2022

@author: chevaugeon

This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


#note cohesive model : lm = 2.*Yc*lc/Gc

#Init timing :
#INFO - name       inc time(s)  inc cpu time(s) tot time(s)  tot cpu time(s)
#INFO - Time Loop     7056.13           3118.00     7056.13          3118.00
#INFO - Compute D     6931.83           2993.85     6931.83          2993.85
#INFO - Compute Acc      71.64             71.53       71.64            71.53

#PAtch fix :
#INFO - name       inc time(s)  inc cpu time(s) tot time(s)  tot cpu time(s)
#INFO - Time Loop     5056.73           3478.22     5056.73          3478.22
#INFO - Compute D     4931.19           3352.80     4931.19          3352.80
#INFO - Compute Acc      72.25             72.17       72.25            72.17
#Dissipated energie 3.572430e+04 J

# Serial, my cvxopt.
#INFO - name       inc time(s)  inc cpu time(s) tot time(s)  tot cpu time(s)
#INFO - Time Loop     7763.83          12127.74     7763.83         12127.74
#INFO - Compute D     7633.48          11997.21     7633.48         11997.21
#INFO - Compute Acc      72.52             72.83       72.52            72.83


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


def initProblem(logger, timer = None):    
    logger.info('Setting up rectangular plate with crack')
    #meshfilename = '../msh/rec_plate_crack_r3.msh'
    meshfilename = '../msh/rec_plate_crack1_r2.msh'
    
    mesh = simplexMesh.readGMSH(meshfilename)
    L = 0.1 # lenght of the Domain
    H = .04 #height of domain
    #nnodes = 102.  ## number of nodes on H for r1 mesh
    lc = 2.*2.5e-4 # lenght scale of the lip constrain (m)
    E = 3.2e10 #Pa
    nu = 0.2
    lamb, mu = (E*nu/(1.+nu)/(1.-2*nu), E/2./(1+nu))
    
    rho = 2450. #kg/m3 
    Gc = 3  #J/m2
    
    Yc = 3*Gc/(4*lc)
    
    #v0 = 16.54 # Impactor initial speed (m/s)
    #v0 = 50
    #v0 = 100. # Impactor initial speed (m/s)
    pressure = 1e6 ## Pa   ##for Neumann BCD
    load = pressure*L  
    
    massfac = 0. #added mass on boundary = total mass sample*massfac
    
    law = mlaws.SofteningElasticityAsymmetric(lamb, mu, Yc = Yc,
                                            h = mlaws.HQuadratic2(), 
                                            g = mlaws.GQuadratic())
    
    method = "LF"
    
    if method == "LF":
        solverd = mech2d.Mechanics2D.solveDLipBoundPatch 
        solverdoptions ={'kernel': lipprojector.minimizeCPcvxopt,'mindeltad':1.e-3, 'lip_reltole':1.e-6,
                        'lipmeasure':'triangle', 'FMSolver':'edge', 
                        'parallelpatch':False, 'snapthreshold':0.999,
                        'kktsolveroptions': {'mode':'schur', 'linsolve':'cholmod'},
                        'fixpatchbound': False,
                        }
    elif method == "PF":
        solverd_pf = mech2d.Mechanics2D.phase_field_AT2 
        solverdoptions ={'linsolve':'cholmod'}
    
    nv = mesh.nvertices
    nf = mesh.ntriangles
    logger.info('Mesh Loaded from file ' + meshfilename)
    logger.info('Mesh size : nv: '+ str(nv) + ' nf: '+ str(nf))
    lipmesh = dualLipMeshTriangle(mesh)
    logger.info('LipMesh constructed from Mesh')
    logger.info('LipMesh size : nv: '+ str(lipmesh.nvertices)+' nf: '+ str(lipmesh.ntriangles))
    
    lipproj = lipprojector.damageProjector2D(lipmesh)
    mech = mech2d.Mechanics2D(mesh, law, lipproj, lc, log = logger)       
         
    idb = 10#id bottom (load in negtive y )
    #idr = 11#id right 
    idt = 12 #id top (load in positive y)
    
    bc = mech2d.boundaryConditions()
    bc.neumanns = mech2d.NeumannConditions(mesh)
    vids1 = mesh.getVerticesOnClassifiedEdges(idb) 
    n1 = len(vids1)
    bc.neumanns.add([idb],{'y':-load/n1})
    vids2 = mesh.getVerticesOnClassifiedEdges(idt) 
    n2 = len(vids2)
    bc.neumanns.add([idt],{'y':load/n2})
    
    # Added Mass on impacted edge
    totalmass = sum(mesh.areas())*rho
    addedmass  = scipy.sparse.dok_matrix((nv,nv))
    
    for i in  vids1 : addedmass[i,i] = totalmass*massfac/n1
    for i in  vids2 : addedmass[i,i] = totalmass*massfac/n2
    #set dynamical system    
    dynsys = dynamics.lipfieldDynamicSystem(mech, bc, rho, solverd, solverdoptions,method, lump = True, addedmass= addedmass, timer =timer)
    # set initial velocity
    u, v, a, d = dynsys.zeros()
    
    wavespeed = np.sqrt(E/rho) # m/s
    tend = 4.*L/wavespeed #s
    hmin = np.min(mesh.innerCirleRadius())
    dt = 1.*hmin/wavespeed
    t = 0.
    step = 0
    #print(tend, dt, tend/dt)
    #raise
    return dynsys, step, t, tend,dt, u,v,a, law, mech

def Kplot(mech, u, d, name, dpi = 100, pmin = None, pmax = None):
        plt.close('all')
        mesh = mech.mesh
        strain = mech.strain(u)
        stress = mech.law.trialStress(strain, d)
        p = stress[:,0] + stress[:,1]
        fig = plt.figure()
        axes =    fig.subplots(1,2)
        axd = axes[0]
        axs = axes[1]
        mesh.plotScalarElemField(d, Tmin= 0., Tmax = 1., showmesh = False, fig=fig, ax = axd)
        cexx, ignore, ignore  =mesh.plotScalarElemField(p, u*2, showmesh = False, fig=fig, ax = axs, eraseifFalse =  np.where(d < 0.99)[0], Tmin = pmin, Tmax = pmax)

        for axi in axes.flatten() :
            axi.axis('equal')
            axi.set_axis_off()
        fig.savefig(name, format = 'png', dpi =dpi, bbox_inches='tight')
        
def Kplot_s(mech, u, d, name, dpi = 100, pmin = None, pmax = None):
        plt.close('all')
        mesh = mech.mesh
        strain = mech.strain(u)
        stress = mech.law.trialStress(strain, d)
        p = stress[:,0] + stress[:,1]
        fig = plt.figure()
        axs =    fig.subplots(1,1)
       
       
        #mesh.plotScalarElemField(d, Tmin= 0., Tmax = 1., showmesh = False, fig=fig, ax = axd)
        cexx, ignore, ignore  =mesh.plotScalarElemField(p, u*2, showmesh = False, fig=fig, ax = axs, eraseifFalse =  np.where(d < 0.99)[0], Tmin = pmin, Tmax = pmax)

        #for axi in axes.flatten() :
        axs.axis('equal')
        axs.set_axis_off()
        fig.savefig(name, format = 'png', dpi =dpi, bbox_inches='tight')
        
def post(respath, steps):
    modelfilename = respath/ pathlib.Path('model')
    dynsys = dynamics.lipfieldDynamicSystem.load(modelfilename)
    mech = dynsys
    loadfilename = respath / pathlib.Path("fields")
    fieldload = liplog.stepArraysFiles(loadfilename)
    pmin, pmax  = np.inf, -np.inf
    for step in steps :
        rstep =  fieldload.load(step)
        u  = rstep['u']
        d  = rstep['d']
        strain = mech.strain(u)
        stress = mech.law.trialStress(strain, d)
        p = stress[:,0] + stress[:,1]
        pmin = min(pmin, np.amin(p))
        pmax = max(pmax, np.amax(p))  
    for step in steps :
        print('export step %08d'%step)
        rstep =  fieldload.load(step)
        u  = rstep['u']
        d  = rstep['d']
        Kplot_s(mech, u,d, respath/pathlib.Path("dyn_S_step_%08d.png"%step), dpi = 400, pmin = pmin, pmax = pmax )
    # make movie :    
    #ffmpeg -framerate 5 -pattern_type  glob -i "dyn*.png" output.mp4
    

def crack_tip_position(d, mech, crack_ini , crack_snap = .75):
    ## finds crack tip coordinates (using iso-surface of damage field given by crack_snap)
    ## works only for crack growing in the positve x-direction and symmetric crack growth
    ## crack tip position is given by the maximum x-coordinate amoung the iso-surface
    ## not reliable when multiple branches/ multiple cracks are present 
    
    ## crack_ini is the position of pre-existing crack
    
    
    iso_surf = np.where(d>=crack_snap)
    iso_coord = mech.lipprojector.mesh.getVertices()[iso_surf]
    
    crack_tip_pos = None
    if iso_coord.size !=0:
        crack_tip_index = np.argmax(iso_coord[:,0])
        crack_tip_pos = iso_coord[crack_tip_index]
    else:
        crack_tip_pos = crack_ini
    return  crack_tip_pos


def Redo_load_data(respath, startstep):
    
    respath1 = respath/pathlib.Path('fields')
    
    
    loadfilename = respath1/pathlib.Path('fields')
    fieldload = liplog.stepArraysFiles(loadfilename)
    rstep =  fieldload.load(startstep)
    u  = rstep['u']
    v =  rstep['v']
    a =  rstep['a']
    d  = rstep['d']
    
    
    with open(respath/pathlib.Path('time_energy.npz'), 'rb') as f:
        data = np.load(f)
        time, ec, ep, ed  = data['time'], data['ec'], data['ep'], data['ed']
        
    with open( respath/pathlib.Path('time_cracktip.npz'),'rb') as f :
        data = np.load(f)
        crack_tip_pos = list(data['tip'])
    
    time = list(time)
    ec = list(ec)
    ep = list(ep)
    ed = list(ed)
    
    dt = time[startstep] -time[startstep-1]
    
    return u,v,a,d,dt,time,ec,ep,ed,crack_tip_pos
     
     
    
    
      
    
    
    
        
if __name__ == '__main__': 
#    print(cvxopt)
#    raise
    plt.close('all')
    
    resume_simulation = False
    if resume_simulation:
        ## path for loading data
        load_respath = None
        loadstep = None
    
    
    ## path for storing data
    respath = pathlib.Path('/data/OG2111300/rect_plate_LF_r2')
    respath1 = pathlib.Path('/data/OG2111300/rect_plate_LF_r2/fields')
    os.makedirs(respath, exist_ok = 1)
    os.makedirs(respath1, exist_ok = 1)
    shutil.copy(__file__, respath / "Run_script.py")
    logger = liplog.setLogger(respath / "rect_plate.log")  
    logger.info('Starting Program rectangular plate with crack')
    logger.info('All results saved in ' + str(respath))
    timer = liplog.timer()  
    dynsys, step, t, tend ,dt, u,v,a, law, mech = initProblem(logger, timer)   
    dynamics.lipfieldDynamicSystem.save(dynsys, respath/pathlib.Path('dynmodel') )
    
    saveperiod = 10 # save field every saveperiod steps
    
    fieldsave  = liplog.stepArraysFiles(respath1 / pathlib.Path("fields"))
    dynsolver  = dynamics.NewmarkExplicit(dynsys) 
    
    if not resume_simulation:
        ec = [dynsys.ec(v)]
        ep = [dynsys.ep(u)]
        ed = [dynsys.ed(dynsys.d)]
        time = [t]
        crack_tip_pos = [[0.05,.02]]   ## initial position of crack 
        stress = law.trialStress(mech.strain(u), dynsys.d)
        fieldsave.save(step,  u = u, v=v, a =a, d = dynsys.d, stress = stress)
        
    else:
        u,v,a,d,dt,time,ec,ep,ed,crack_tip_pos = Redo_load_data(load_respath, loadstep)
        logger.info('Loaded data for resuming simulation')
        step = loadstep+1
        t = time[-1]+dt
        
    
    logger.info('Starting time iteration. t =%e, tend =%e, dt = %e, nstep = %d'%(t,tend,dt, int((tend-t)/dt)+1 ))
    timer.start("Time Loop")
    while t <= tend :
        logger.info("step %08d, time = %010.8f"%(step,t))
        u,v,a,t = dynsolver.step(u,v,a,t,dt)
        stress = law.trialStress(mech.strain(u), dynsys.d)
        ec.append(dynsys.ec(v))
        ep.append(dynsys.ep(u))
        ed.append(dynsys.ed(dynsys.d))
        time.append(t)
        crack_tip_pos.append(crack_tip_position(dynsys.d, mech, crack_tip_pos[0]))
        
        if not  (step%saveperiod):  
            fieldsave.save(step,  u = u, v=v, a=a, d = dynsys.d, stress = stress)
        
            with open( respath/pathlib.Path('time_cracktip.npz'),'wb') as f :
                np.savez(f,time = np.array(time), tip= np.array(crack_tip_pos))
    
            with open( respath/pathlib.Path('time_energy.npz'),'wb') as f :
                np.savez(f,time = np.array(time), ec = np.array(ec), ep = np.array(ep), ed = np.array(ed))
            
        step  +=1
        
            
        
    timer.end("Time Loop")
        
    ec = np.array(ec)
    ep = np.array(ep)
    ed = np.array(ed)
    time = np.array(time)
    crack_tip_pos  = np.array(crack_tip_pos)
    timer.log(logger)
    
    with open(respath/pathlib.Path('time_energy.npz'), 'rb') as f:
        data = np.load(f)
        time, ec, ep,ed  = data['time'], data['ec'], data['ep'],data['ed']
    
    
    
    e = ep+ec
    print('Dissipated energie %e J'%(e[0] - e[-1]) )
    print('Dissipated energy %e J'%(ed[-1]))
        
    plt.ion()
    fig = plt.figure()
    plt.plot(time, ec, label ='ec')
    plt.plot(time, ep, label ='ep')
    plt.plot(time, ep+ec, label = 'e')
    plt.legend()
    fig.savefig(respath/pathlib.Path("dyn_D_glob.pdf"), format = 'pdf', bbox_inches='tight')
    
    
    

def Redo(respath, startstep, tend, pr = None):
    
    respath1 = respath/pathlib.Path('fields')
    redo_respath = respath/pathlib.Path('redo')
    redo_respath1 = redo_respath/pathlib.Path('fields')
    os.makedirs(redo_respath1, exist_ok = True)
    logger = liplog.setLogger(redo_respath / "redo_rect_plate_crack.log")  
    logger.info('Starting Program Redo_rect_plate_crack')
    timer = liplog.timer() 
    
    modelfilename = respath/ pathlib.Path('dynmodel')
    dynsys = dynamics.lipfieldDynamicSystem.load(modelfilename)
    print(dynsys.solverdoptions)
    
    solverdlip = mech2d.Mechanics2D.solveDLipBoundPatch 
    solverdlipoptions ={'mindeltad':1.e-3, 'lip_reltole':1.e-6,
                        'lipmeasure':'triangle', 'FMSolver':'edge', 
                        'parallelpatch':False, 'snapthreshold':0.999,
                        #'kktsolveroptions': {'mode':'direct', 'linsolve':'umfpack'}
                        'kktsolveroptions': {'mode':'schur', 'linsolve':'cholmod'}
                        
                        }
    
    solverd_pf = mech2d.Mechanics2D.phase_field_AT2 
    solverdoptions ={'linsolve':'cholmod'}
    dynsys.solverd = solverd_pf
    dynsys.solverdoptions = solverdoptions
    
    fieldsave  = liplog.stepArraysFiles(respath1 / pathlib.Path("fields"))
    dynsolver  = dynamics.NewmarkExplicit(dynsys)
    
    print(dynsys.solverdoptions)
    saveperiod =10
    dynsys.mech.logger = logger
    loadfilename = respath1/pathlib.Path('fields')
    fieldload = liplog.stepArraysFiles(loadfilename)
    rstep =  fieldload.load(startstep)
    u  = rstep['u']
    v =  rstep['v']
    a =  rstep['a']
    d  = rstep['d']
    dynsys.dmin = d.copy()
    dynsys.dk = d.copy()
    dynsys.d  = d.copy()
    
    with open(respath/pathlib.Path('time_energy.npz'), 'rb') as f:
        data = np.load(f)
        time, ec, ep  = data['time'], data['ec'], data['ep']
        
    with open( respath/pathlib.Path('time_cracktip.npz'),'rb') as f :
        data = np.load(f)
        crack_tip_pos = list(data['tip'])
    
    time = list(time)
    ec = list(ec)
    ep = list(ep)
    
    dt = time[startstep] -time[startstep-1]
    t = time[startstep] + dt     ## current time step
    #dynsys.solverdoptions['parallelpatch'] = 
    
    step = startstep+1
    
    logger.info('Starting time iteration. t =%e, tend =%e, dt = %e, nstep = %d'%(t,tend,dt, int((tend-t)/dt)+1 ))
    timer.start("Time Loop")
    while t <= tend :
        logger.info("step %08d, time = %010.8f"%(step,t))
        u,v,a,t = dynsolver.step(u,v,a,t,dt)
        stress = law.trialStress(mech.strain(u), dynsys.d)
        if not  (step%saveperiod):  fieldsave.save(step,  u = u, v=v, a=a, d = dynsys.d, stress = stress)
        ec.append(dynsys.ec(v))
        ep.append(dynsys.ep(u))
        time.append(t)
        crack_tip_pos.append(crack_tip_position(dynsys.d, mech, crack_tip_pos[0]))
        step  +=1
    timer.end("Time Loop")
        
    ec = np.array(ec)
    ep = np.array(ep)
    time = np.array(time)
    crack_tip_pos  = np.array(crack_tip_pos)
    timer.log(logger)
    
    
    with open( redo_respath/pathlib.Path('time_cracktip.npz'),'wb') as f :
        np.savez(f,time = time, tip= crack_tip_pos)
    
    with open( redo_respath/pathlib.Path('time_energy.npz'),'wb') as f :
        np.savez(f,time = time, ec = ec, ep = ep)
        
    with open(redo_respath/pathlib.Path('time_energy.npz'), 'rb') as f:
        data = np.load(f)
        time, ec, ep  = data['time'], data['ec'], data['ep']
    
    e = ep+ec
    print('Dissipated energie %e J'%(e[0] - e[-1]) )
        
    plt.ion()
    fig = plt.figure()
    plt.plot(time, ec, label ='ec')
    plt.plot(time, ep, label ='ep')
    plt.plot(time, ep+ec, label = 'e')
    plt.legend()
    fig.savefig(respath/pathlib.Path("dyn_D_glob.pdf"), format = 'pdf', bbox_inches='tight')
    
    
    
      
    
    