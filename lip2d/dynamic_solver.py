#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 22:39:24 2022

@author: nchevaug
"""

import numpy as np
import cvxopt
import lip2d.linsolverinterface as lin


class massSpringDynamicSystem():
    ''' example of a very simple dynamical system So that we can test the interface with dynamical sovlver'''
    def __init__(self, m, k):
        self.m = m
        self.k = k
    def kinetic(self, x, v, t):
        return 0.5*self.m*v*v
    def potential(self, x, v, t):
        return 0.5*self.k*x*x
    def a(self, x, v= None, t=None):
        return -self.k*x/self.m
    def updateInternalVariables(x):
        return
    def dadx(self, x = None, v = None, t = None):
        return -self.k/self.m
    def dadv(self, x = None, v = None, t = None):
        return 0
    def dadt(self, x = None, v = None, t = None):
        return 0

class NewmarkExplicit():
    def __init__(self, dyn):
        self.dyn = dyn
        
    def step(self, xn, vn, an, tn, dt):
        xp  = xn + dt*vn +dt*dt/2.*an
        vp = vn + dt/2.*an
        t = tn+dt
        x = xp
        self.dyn.updateInternalVariables(x)
        a  = self.dyn.a(xp, vp, t)
        v = vp + dt*0.5*a
        return x, v, a, t
    
class lipfieldDynamicSystem():
    def __init__(self, mech, bc,  rho, solverd, solverdoptions, lump=False, addedmass = None, timer = None):
        self.mech = mech
        self.nv = self.mech.mesh.nvertices
        self.bc = bc
        ignore, self.d  = mech.zeros()
        self.dmin =self.d.copy()
        self.dk = self.d.copy()
        self.solverd = solverd
        self.solverdoptions = solverdoptions
        self.rho = rho
        self.timer = timer
        M = rho*mech.M()
        if addedmass is not None : M = M+addedmass 
        M = lin.convert2cvxoptSparse(M)
        if lump :
            diag = M*cvxopt.matrix(np.ones(M.size[0]) )
            M = cvxopt.spdiag(diag)
        self.M = M
        Mfac = cvxopt.cholmod.symbolic(self.M)
        cvxopt.cholmod.numeric(self.M,Mfac)
        self.Mfac = Mfac
    
    def zeros(self):
        u, d =self.mech.zeros()
        v = u.copy()
        a = u.copy()
        return u,v,a,d
        
#        self.v = self.u.copy()
#        
#        if self.bc.dirichlets is not None : 
#            keys,values = self.bc.dirichlets.impose()
#            u = self.u.reshape(nv*2)
#            u[keys] = values
#            self.u = u.reshape((nv,2))
        
    def compute_d(self, strain, dmin, dguess):
        resd = self.solverd (self.mech, dmin, dguess, strain, self.solverdoptions)
        if not resd['Converged'] :
            print('d solver did not converge')
            raise
        return resd['d']
    
    def updateInternalVariables(self,x):
        if self.timer is not None : self.timer.start("Compute D")
        strain = self.mech.strain(x)
        self.d = self.compute_d(strain, self.dmin, self.dk)
        self.dmin = self.d.copy()
        self.dk = self.d.copy()
        if self.timer is not None : self.timer.end("Compute D")
   
    def ec(self, v):
        '''return Kinetic Energy'''
        vx = cvxopt.matrix(v[:,0])
        vy = cvxopt.matrix(v[:,1])
        return 0.5 * (vx.T*(self.M*vx))[0] + 0.5 * (vy.T*(self.M*vy))[0]
    
    def ep(self, u):
        '''return strain Energy'''
        strain = self.mech.strain(u)
        Fint  =  np.array(self.mech.F(strain, self.dk)).squeeze()
        return 0.5*Fint.dot(u.flatten())
    
    def a(self, x, v= None, t=None):
        '''return acceleration for given displacement (d is fixed)'''  
        if self.timer is not None : self.timer.start("Compute Acc")
        strain = self.mech.strain(x)
        Fint  = -np.array(self.mech.F(strain, self.dk)).squeeze()
        Fint = Fint.reshape((self.nv,2))
        Fintx = cvxopt.matrix(Fint[:,0])
        Finty = cvxopt.matrix(Fint[:,1])
        cvxopt.cholmod.solve(self.Mfac, Fintx, sys = 0)
        cvxopt.cholmod.solve(self.Mfac, Finty, sys = 0)
        ak = np.zeros((self.nv,2))
        ak[:,0] = np.array(Fintx[:,0]).flatten()
        ak[:,1] = np.array(Finty[:,0]).flatten()
        
        # impose dirichlet bc.
        if self.bc.dirichlets is not None :
            ak = ak.flatten()
            keys, value = self.bc.dirichlets.impose()
            ak[keys] = 0.
            ak = ak.reshape((self.nv, 2))
        if self.timer is not None : self.timer.end("Compute Acc")
        return ak