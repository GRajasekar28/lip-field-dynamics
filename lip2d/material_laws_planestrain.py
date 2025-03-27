#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
// Copyright (C) 2021 Chevaugeon Nicolas
Created on Tue Feb  2 08:56:14 2021
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
import numpy as np
import scipy as sp
import scipy.optimize
import liplog as liplog
logger = liplog.logger
from abc import ABC, abstractmethod

warning_need_overload = set()

def warn_once_need_overload(name, logger):
    """ Helper function to use in default base implementation of a member function to waen the user that he should overload 
        a member function for increase speed/ precision
    """
    if name not in warning_need_overload:
        logger.warning(name+' should be overloaded in derived class for better perf/precision')
        warning_need_overload.add(name)
        raise
        

def approx_f_onevariable_prime(x, f, epsilon):
    """ compute a numercoal derivative of a function f at x, using centered difference scheme, eps is the step""" 
    return 0.5*(f(x+epsilon)-f(x-epsilon))/epsilon

def eigenSim2D_voigt(eps,  vector = False):
    """ Compute eigen values and optionally eigen vector for each 2d plane strain state given in voigt format on entry 
        if vector = false, return l0, l1 the first and second eigen value of eps.
        if vector = true return l0, l1 and N0, N1 where N0 N1 are the eigen vectors.
    """
    
    eps00 = eps[...,0]
    eps11 = eps[...,1]
    eps01 = 0.5*(eps[...,2])
    t = eps00 + eps11
    d = eps00*eps11-eps01*eps01
    delt =  np.sqrt(t*t-4*d)
    l0 = ((t - delt)/2.)
    l1 = ((t + delt)/2.)
    if not vector : return l0, l1
    t2 = np.arctan2(eps01, eps00-t/2.)
    c = np.cos(-t2/2.)
    s = np.sin(-t2/2.)
    N0 = np.column_stack([s,  c]).squeeze()
    N1 = np.column_stack([c, -s]).squeeze()
    return l0, l1, N0, N1
    
                
def toVoight2D( T ) :  
    """ assumming T is of shape x,2,2,2,2,  ... as an array of 2nd order two dimensional order 4 tensor with full symmetries, return
        an array of shape x,3,3, representing the tensor in voigt notation.
    """ 
    TV =  np.zeros(T.shape[:-4] + (3,3))
    TV[...,0,0]  = T[...,0,0,0,0]
    TV[...,0,1]  = T[...,0,0,1,1]
    TV[...,0,2] = 0.5*(T[...,0,0,0,1] + T[...,0,0,1,0])    
    TV[...,1,0]  = T[...,1,1,0,0]
    TV[...,1,1]  = T[...,1,1,1,1]
    TV[...,1,2]  = 0.5*(T[...,1,1,0,1] + T[...,1,1,1,0])    
    TV[...,2,0]  = 0.5*(T[...,0,1,0,0] + T[...,1,0,0,0])
    TV[...,2,1]  = 0.5*(T[...,0,1,1,1] + T[...,1,0,1,1])
    TV[...,2,2]  = 0.25*(T[...,0,1,0,1] + T[...,0,1,1,0] + T[...,1,0,0,1] +  T[...,1,0,1,0]  )    
    return TV

class HBase(ABC):
    """ represent function of 1 variable (typically d) that can be derived twice """
    def __init__(self, name='hbase', epsilon = 1.e-6):
        self.name, self.epsilon = (name, epsilon)
    @abstractmethod
    def __call__(self,d):
        pass
    def jac(self, d) :
        warn_once_need_overload('HBase.jac')
        return approx_f_onevariable_prime(d, lambda d : self(d), self.epsilon)
    def hess(self, d):
        warn_once_need_overload('HBase.hess')
        return approx_f_onevariable_prime(d, lambda d : self.jac(d), self.epsilon)

class HPoly(HBase):
    def __init__(self, coef=[0,2.], name='hpoly'):
        super().__init__(name)
        self._poly = np.polynomial.Polynomial(coef)
        self._deri1, self._deri2  = (self._poly.deriv(),self._poly.deriv(2))
    def __call__(self,d, compute = True, computejac =False, computehess =False):
        v = None; vjac = None; vhess = None
        if compute :
            v = self._poly(d)
        if computejac :
            vjac = self._deri1(d)
        if computehess :
            vhess = self._deri2(d)
        return v, vjac, vhess
    def jac(self, d):     return self._deri1(d)
    def hess(self,d):     return self._deri2(d)

class HRPoly(HBase):
    def __init__(self, coefn, coefd, name='hrpoly'):
        super().__init__(name)
        self._num, self._den =  (np.polynomial.Polynomial(coefn), np.polynomial.Polynomial(coefd))
        self._deriv1_num, self._deriv2_num = (self._num.deriv(),self._num.deriv(2))
        self._deriv1_den, self._deriv2_den = (self._den.deriv(),self._den.deriv(2))
    def __call__(self,d, compute = True, computejac =False, computehess =False) : 
        v = None; vjac = None; vhess = None
        f  = self._num(d)
        g=  self._den(d)
        if (computejac or computehess):
            df = self._deriv1_num(d)
            dg= self._deriv1_den(d)
        if compute : v = f/g
        if computejac : 
            vjac = (df*g-f*dg)/g**2
        if computehess : 
            d2f = self._deriv2_num(d)
            d2g = self._deriv2_den(d)
            vhess = ((d2f*g-f*d2g)*g -2.*(df*g-f*dg)*dg)/g**3    
        return v, vjac, vhess
    def jac(self, d) : 
        f, df = (self._num(d), self._deriv1_num(d))
        g, dg = (self._den(d), self._deriv1_den(d))
        return (df*g-f*dg)/g**2
    def hess(self, d) : 
        f, df, d2f = (self._num(d), self._deriv1_num(d), self._deriv2_num(d) )
        g, dg, d2g = (self._den(d), self._deriv1_den(d), self._deriv2_den(d))
        return ((d2f*g-f*d2g)*g -2.*(df*g-f*dg)*dg)/g**3    

def HLinear():      return HPoly(coef=[0.,2.],name='h(d) = 2d') # Gc = 2YcLc
#def HQuadratic():   return HPoly(coef=[0,2,3],name='h(d) = 2d+3d²') #Gc = 4yc lc ??

class HQuadratic(HBase):
    def __init__(self):
        super().__init__('h(d) = 2d²')   
    def __call__(self, d, compute = True, computejac =False, computehess =False) :
        h = None; jach = None; hessh = None
        if compute : h = d *(2 +3*d)
        if computejac : jach = 2. + 6*d
        if computehess : hessh = 6*np.ones(np.atleast_1d(d).shape)
        return h, jach, hessh
    def jac(self, d) : 
       return 2. + 6*d;
    def hess(self, d) : 
        if type(d) == np.ndarray :
            return 6*np.ones(d.shape)
        return 6.
        
def HQuadratic2():  return HPoly(coef=[0,0,2],name='h(d) = 2d²')   #Gc = 2/3 Yclc damage imediat
# to get a convex function we need, 0<lm<0.5
# for a given Gc, lc, lm = 2.Yc*lc/Gc 
# for example, Yc = 1, Gc =1, 0.1<lc < 0.25 
# -> lc =0.2 to be safe for the coarser mesh. then lm = 0.4
#    now, refine h<- h/2, we can take lc = 0.1 and  lm = 0.2
def HCohesive(lm) : return HRPoly([0.,2.,-1.],[1.,-2,(1.+2*lm),-2.*lm, lm**2], name = 'cohesive l='+str(lm)) 

def GLinear():      return HPoly(coef=[1.,-1.],name= 'g(d) = 1-d')
def GQuadratic():   return HPoly(coef=[1.,-2.,1.],name= 'g(d) = (1-d)^2') #1.-2*d+d^2
#typical value for eta = 0.5h/l
def GO3Eta(eta):   return HPoly(coef=[1.,-2., 1.+eta, -eta],name= '(1.-d)^2 +eta*(1.-d)*d**2') #1.-2d+d^2 +eta*d^2 - eta*d^3
#def GO4Eta(eta):   return HPoly(coef=[1.,-2., 1.,      eta, -eta],name= 'O4_LE') # g[0] = 1. g'(0) = 1 g(1) = 0 g'(1) = eta

class GO4Eta(HBase):   #g[0] = 1. g'(0) = 1 g(1) = 0 g'(1) = eta
    def __init__(self, eta):
        super().__init__('G04eta')
        self.eta = eta
    def __call__(self, d, compute = True, computejac =False, computehess =False) :
        eta = self.eta
        g = None; jacg = None; hessg = None;
        if compute :         g  = 1. + d*(-2 + d*( 1. + d*(eta -eta*d  )))
        if computejac :   jacg  = -2. + d*(2. + d *(3*eta  +d*(-4*eta)))   
        if computehess : hessg  = 2. +d*(6*eta +d*(-12*eta))
        return  g, jacg, hessg
    
    def jac(self,d):
        eta = self.eta
        return -2. + d*(2. + d *(3*eta  +d*(-4*eta)))
    def hess(self,d):
        eta = self.eta
        return 2. +d*(6*eta +d*(-12*eta))
    
    
class SofteningElasticity(ABC):
    """ Base class for Softeneing elastic material. Everything is supposed to be computabel starting from the potential which is a function
        of strain and the softeningvaraible d. Us this class as a base class and overload at least potential.
    """
    def __init__(self, lamb = 1., mu =1., Yc = 1., h = HLinear(), g = GQuadratic(), log = logger):
        self.lamb = lamb
        self.mu = mu
        self.Yc = Yc
        self.h = h
        self.g = g
        self.logger = log
   
    @abstractmethod 
    def potential(self, strain, d) :
        """return strain elastic energy potential at each gp for which strain and d are given"""
        pass
    
    def trialStress(self, strain, softeningvariables=None):
        warn_once_need_overload("trialStress", self.logger)
        strain = np.atleast_2d(strain)
        stress = np.zeros(strain.shape)
        eps =  1.e-6
        for i in range(3) :
            strainpeps = strain.copy()
            strainpeps[:,i] += eps
            strainmeps = strain.copy()
            strainmeps[:,i] -= eps
            stress[:,i] = (self.potential(strainpeps, softeningvariables) - self.potential(strainmeps, softeningvariables))/2./eps
        return stress
    
    def dTrialStressDStrain(self, strain, softeningvariables, localvariables =None):
        """ return dtrialstress/dstrain as d2phi/(dstrain)2 at fixed soft and local var """
        warn_once_need_overload("dTrialStressDStrain", self.logger)
        if len(strain.shape) ==1 :  strain  = strain.reshape(1,3)
        n = strain.shape[0]
        if softeningvariables is None:  softeningvariables = np.zeros((n))
        D = np.zeros((strain.shape[0],3,3))
        eps =  1.e-6
        d = softeningvariables
        for i in range(3) :
            strainpeps = strain.copy()
            strainpeps[:,i] += eps
            strainmeps = strain.copy()
            strainmeps[:,i] -= eps
            D[:,:,i] = (self.trialStress(strainpeps, d) - self.trialStress(strainmeps, d))/2./eps
            #D[:,:,i] = (self.trialStress(strainpeps, d) + self.trialStress(strainmeps, d) - )/2./eps
        return D
    
    def Y(self, strain, softeningvariables=None):
        """return the Yield energy (Y = - dphidd)  at each gp for which strain and d are given"""
        warn_once_need_overload("Y",self.logger)
        eps =  1.e-6
        d = softeningvariables
        return -(self.potential(strain, d+eps) - self.potential(strain, d-eps))/2./eps
    
    def dY(self, strain, softeningvariables=None) :
        """return the  derivative of the Yield energy (Y = - dphidd)  at each gp for which strain and d are given"""
        warn_once_need_overload("dY", self.logger)
        eps =  1.e-6
        d = softeningvariables
        return -(self.potential(strain, d+eps) + self.potential(strain, d-eps) - 2*self.potential(strain, d) )/eps/eps
       
    def potentialFixedStrain(self, strain):
        """ 
        return a function phid that permits to compute the potential as a function of d, for strain fixed at strain
        phid takes d as a mandatory parameter and to optional boolean Y and dYdd defaulted to false.
        if calling phid(d, Y=False, dY = False) return the potential
        if calling phid(d, Y=True, dY = False) return a potential, Y pair
        if calling phid(d, Y=True, dY = True) return a potential, Y, dY pair
        """
        warn_once_need_overload("potentialFixedStrain", self.logger)
        def phid(d, cphi = True, cY = False, cdY = False):
            phi = None; Y = None; dY = None
            if cphi :  phi =  self.potential(strain, d)
            if cY : Y = self.Y(strain, d)
            if cdY : dY = self.dY(strain,d)
            return phi, Y, dY
        return phid
    
    def solveSoftening(self, strain, softeningvariablesn, softeningvariablesguess = None, deriv =False):
        """ at fixed strain and value of d at previous stap (dn), compute d that minimize the potential, sich as dn <= d <= 1."""
        dn = np.atleast_1d(softeningvariablesn)
        d = dn.copy()
        Ydn = self.Y(strain, dn)
        index = ((Ydn > 0.)*(d<1.)).nonzero()[0]
        for k in index :
            s = strain[k]
            fun = lambda x: self.Y(s, x)
            if fun(1.)>0.: d[k]=1.
            else:    d[k] = sp.optimize.brentq(fun, dn[k], 1.)
        return d
           
    
class SofteningElasticitySymetric0(SofteningElasticity):
    """ Softening elasticity material law ... Slow version for test purpose, 
        using default numerical implementation of the derivatives from the base class  """
    def __init__(self, lamb = 1., mu =1., Yc = 1., h = HLinear(), g = GQuadratic(), log = logger):
        super().__init__( lamb, mu, Yc, h, g, log)
    
    def potential(self, strain, d) :
        """return strain elastic energy potential at each gp for which strain and d are given"""
        phie = self._elastic_potential( strain)
        g, ignore, ignore  = self.g(d)
        h, ignore, ignore = self.h(d)
        return g*phie + self.Yc*h
    
    def _elastic_potential(self, strain):
        if len(strain.shape) ==1 :
            strain  = strain.reshape(1,3)
        if strain.shape[1] != 3 : raise
        eps11 = strain[:,0]
        eps22 = strain[:,1]
        eps12 = strain [:,2]/2.
        return 0.5*self.lamb*((eps11+eps22)**2)+self.mu*(eps11**2+eps22**2+2*eps12**2)        
            
class SofteningElasticitySymmetric(SofteningElasticity):
    """ Symetric traction/compression """
    def __init__(self, lamb = 1., mu =1., Yc = 1., h = HLinear(), g = GQuadratic(), log = logger):
        super().__init__( lamb, mu, Yc, h, g, log)
        
    def _elastic_potential(self, strain):
        if len(strain.shape) ==1 :
            strain  = strain.reshape(1,3)
        if strain.shape[1] != 3 : raise
        eps11 = strain[:,0]
        eps22 = strain[:,1]
        eps12 = strain [:,2]/2.
        return 0.5*self.lamb*((eps11+eps22)**2)+self.mu*(eps11**2+eps22**2+2*eps12**2)        
        
    def potential(self, strain, d) :
        """return strain elastic energy potential at each gp for which strain and d are given"""
        phi, ignore, ignore  = self.potentialFixedStrain(strain)(d)
        return phi
    
    def fe(self,strain, d) :
        """ return the stored elastic energy """
        return self.g(d)*self._elastic_potential(strain)
    
    def fs(self,d) : 
        """ return the dissipated energy (dissipation potential ??) """
        return self.Yc*self.h(d)
    
    def Y(self, strain, d) :
        """return the Yield energy (Y = - dphidd)  at each gp for which strain and d are given"""
        ignore, Y, ignore  =self.potentialFixedStrain(strain)(d, cphi = False, cY = True)
        return Y
    
    def dY(self, strain, d) :
        """return the  derivative of the Yield energy (Y = - dphidd)  at each gp for which strain and d are given"""
        ignore, ignore, dY =self.potentialFixedStrain(strain)(d, cphi = False, cY = False, cdY = True)
        return dY
   
    def potentialFixedStrain(self, strain):
        """ 
        return a function phid that permits to compute the potential as a function of d, for strain fixed at strain
        phid takes d as a mandatory parameter and to optional boolean Y and dYdd defaulted to false.
        if calling phid(d, Y=False, dY = False) return the potential
        if calling phid(d, Y=True, dY = False) return a potential, Y pair
        if calling phid(d, Y=True, dY = True) return a potential, Y, dY pair
        """
        phie = self._elastic_potential(strain)
        def phid(d, cphi = True, cY = False, cdY = False):
            phi = None; Y = None; dY = None
            g, jacg, hessg = self.g(d, compute = cphi, computejac =cY, computehess = cdY)
            h, jach, hessh = self.h(d, compute = cphi, computejac =cY, computehess = cdY)
            if cphi :  phi =  g*phie      + self.Yc*h
            if cY :    Y   =  -jacg*phie  - self.Yc*jach
            if cdY :   dY  =  -hessg*phie - self.Yc*hessh
            return phi, Y, dY
        return phid
    
    def solveSoftening(self, strain, softeningvariablesn, softeningvariablesguess = None, deriv =False):
        """ at fixed strain and value of d at previous stap (dn), compute d that minimize the potential, sich as dn <= d <= 1."""
        dn = np.atleast_1d(softeningvariablesn)
        d = dn.copy()
        phid = self.potentialFixedStrain(strain)
        ignore, Ydn, ignore =  phid(dn, cphi = False, cY = True, cdY = False)
        index = ((Ydn > 0.)*(d<1.)).nonzero()[0]
        for k in index :
            s = strain[k]
            phid = self.potentialFixedStrain(s)
            def fun(x) : 
                ignore, Y, ignore =  phid(x, cphi = False, cY = True, cdY = False)
                return Y
            if fun(1.)>0.: d[k]=1.
            else:    d[k] = sp.optimize.brentq(fun, dn[k], 1.)
        return d
           

    def trialStress(self, strain, softeningvariables=None):
        """ return trial stress as dphi/dstrain at fixed soft and local var """
        if len(strain.shape) ==1 :
             strain  = strain.reshape(1,3)
        mu = self.mu
        lamb = self.lamb
        eps11 = strain[:,0]
        eps22 = strain[:,1]
        eps12 = strain[:,2]/2.
        sig11 = lamb*(eps11+eps22)+2*mu*eps11
        sig22 = lamb*(eps11+eps22)+2*mu*eps22
        sig12 = 2.*mu*eps12
        if softeningvariables is None :
            g, ignore, ignore = self.g(0.)
        else :
            d = softeningvariables
            g, ignore, ignore = self.g(d)
        return np.array([g*sig11, g*sig22, g*sig12 ]).T
 
    def dTrialStressDStrain(self, strain, softeningvariables, localvariables =None):
        """ return dtrialstress/dstrain as d2phi/(dstrain)2 at fixed soft and local var """
        mu = self.mu
        lamb = self.lamb
        if len(strain.shape) ==1 :  strain  = strain.reshape(1,3)
        n = strain.shape[0]
        if softeningvariables is None:  d= np.zeros((n))
        d = softeningvariables
        g, ignore, ignore = self.g(d)
        return   np.tensordot( g,  np.array( [[lamb+2*mu, lamb, 0.],
                   [lamb, lamb+2*mu, 0.],
                   [0.,0., mu]]
                   ), axes = 0)
        
    # This member is not needed to solve a problem using mechanics2D ...  This is for testing new stuff
    def dtrialStressdD(self, strain, softeningvariables=None):
        """ return trial stress as dphi/dstrain at fixed soft and local var """
        if len(strain.shape) ==1 :
             strain  = strain.reshape(1,3)
        mu = self.mu
        lamb = self.lamb
        eps11 = strain[:,0]
        eps22 = strain[:,1]
        eps12 = strain[:,2]/2.
        sig11 = lamb*(eps11+eps22)+2*mu*eps11
        sig22 = lamb*(eps11+eps22)+2*mu*eps22
        sig12 = 2.*mu*eps12
        if softeningvariables is None :
            dg = self.g.jac(1.)
        else :
            d = softeningvariables
            dg = self.g.jac(d)
        return np.array([dg*sig11, dg*sig22, dg*sig12 ]).T
    
    # This member is not needed to solve a problem using mechanics2D ...  This is for testing new stuff Might be used latter for cases where internal variables are needed.
    def solveStressFixedSoftening(self, strain, softeningvariablesn = None, localvariablesn=None, localvariablesguess=None, deriv = False, usebasesolver = False):
        if usebasesolver : return super().solve_stress_fixed_softening(strain, softeningvariablesn, localvariablesn, localvariablesguess, deriv, usebasesolver)
        d = softeningvariablesn
        stress = self.trialStress(strain, d)
        if not deriv : return stress, None
        return stress, None, self.dTrialStressdStrain(strain, d) 
            
        
class SofteningElasticityAsymmetric(SofteningElasticity):
    """ Damage only when in traction, recovert strenght in compression"""
    def __init__(self, lamb = 1., mu =1., Yc = 1., h = HLinear(), g = GQuadratic() ):
        super().__init__(self, lamb, mu, Yc, h, g)
        self.lamb = lamb
        self.mu = mu
        self.Yc = Yc
        self.g  = g
        self.h  = h
   
    
    def potential(self, strain, d) :
        """return strain elastic energy potential at each gp for which strain and d are given"""
        phi0, phid  = self._split_elastic_potential(strain)
        g, ignore, ignore = self.g(d)
        h, ignore, ignore = self.h(d)
        return phi0 + g*phid + self.Yc*h
    
    def potentialFixedStrain(self, strain):
        """ 
        return a function phid that permits to compute the potential as a function of d, for strain fixed at strain
        phid takes d as a mandatory parameter and to optional boolean Y and dYdd defaulted to false.
        if calling phid(d, Y=False, dY = False) return the potential
        if calling phid(d, Y=True, dY = False) return a potential, Y pair
        if calling phid(d, Y=True, dY = True) return a potential, Y, dY pair
        """
        phic, phit = self._split_elastic_potential(strain)
        def phid(d, cphi = True, cY = False, cdY = False):
            phi = None; Y = None; dY = None
            g, jacg, hessg = self.g(d, compute = cphi, computejac =cY, computehess = cdY)
            h, jach, hessh = self.h(d, compute = cphi, computejac =cY, computehess = cdY)
            if cphi : phi =  phic + g*phit + self.Yc*h
            if cY   : Y    = -jacg*phit    - self.Yc * jach
            if cdY  : dY   = -hessg*phit      - self.Yc *hessh
            return phi, Y, dY
        return phid
    
    """
    def _split_elastic_potential(self, strain, derivsigma = 0) :
         
         ##derivsigma 0 : return phi0, phid
         ##derivsigma 1 : return phi0, phid, sigma0, sigmad
         ##derivsigma 2 : return phi0, phid, sigma0, sigmad, D0, Dd
         
            
         #phi(eps, d) = mu (Sum_i f(d) * (e_i^+)^2  + (e_i^-)^2 + lambda/2 f(d) tr(eps)^2 +   Yc*h(d)
         # sit toute les vp sont > 0 , mu(Sum_i f(d) * (e_i^+)^2  + (e_i^-)^2) = mu f(d) * eps:eps 
         # sit toute les vp sont < 0 , mu(Sum_i f(d) * (e_i^+)^2  + (e_i^-)^2) = mu  * eps:eps 
         mu = self.mu
         lamb = self.lamb
         eps = np.atleast_2d(strain)
         eps11 = np.atleast_1d(eps[..., 0])
         eps22 = np.atleast_1d(eps[..., 1])
         eps12 = np.atleast_1d(eps[..., 2]/2.)
         trace = eps11 + eps22
         det   = eps11*eps22 -  eps12**2
         ss = det >= 0.
         iss = np.where(ss)  # indexes where the vp are of same sign
         ios = np.where(np.logical_not(ss))  # indexes where the vp are of opposite sign
         
         if derivsigma :         
             l0, l1, N0, N1 = eigenSim2D_voigt(eps[ios], True)
         else:
             l0, l1 = eigenSim2D_voigt(eps[ios])
         
         phi0 = np.zeros(( eps.shape[:-1]))
         phid = lamb/2.*trace**2 
         I2iss = (eps11[iss]**2 + eps22[iss]**2 + 2.*eps12[iss]**2)
         #print(l0**2+l1**2 - eps11**2 + eps22**2+2.*eps12**2)
         phi0[iss] += mu *np.where(trace[iss] >=0., 0., 1.) * I2iss 
         phid[iss] += mu *np.where(trace[iss] >= 0., 1., 0.) * I2iss
         phi0[ios] += mu*l0**2
         phid[ios] += mu*l1**2
         
         if derivsigma == 0 : return phi0, phid
         stress0 = np.zeros(( eps.shape[:-1] + (3,)))
         stressd = (lamb*trace)[..., np.newaxis] * np.array([1.,1.,0.]) 
         stress0[iss]  += np.where(trace[iss] >=0., 0., 2.*mu)[..., np.newaxis]*np.array([eps11[iss],eps22[iss], eps12[iss]]).T
         stressd[iss]  += np.where(trace[iss] >=0., 2.*mu, 0.)[..., np.newaxis]*np.array([eps11[iss],eps22[iss], eps12[iss]]).T
         N0xN0 = np.array([N0[...,0]**2, N0[...,1]**2, N0[...,0]*N0[...,1]]).T
         N1xN1 = np.array([N1[...,0]**2, N1[...,1]**2, N1[...,0]*N1[...,1]]).T
         stress0[ios] +=  2.*mu*l0[:, np.newaxis]*N0xN0 
         stressd[ios] +=  2.*mu*l1[:, np.newaxis]*N1xN1
         if derivsigma == 1 : return phi0, phid, stress0, stressd
         
         D0 = np.zeros(( eps.shape[:-1] + (3,3)))
         Dd = lamb*np.ones(eps.shape[:-1] + (1,1))*  np.array([[1.,1.,0.],[1.,1.,0.], [0.,0.,0.]] ) 
         D0[iss] +=  np.where(trace[iss] >=0., 0., 2.*mu)[..., np.newaxis, np.newaxis]*np.array([[1.,0.,0.],[0.,1.,0.], [0.,0.,0.5] ]) 
         Dd[iss] +=  np.where(trace[iss] >=0., 2.*mu, 0.)[..., np.newaxis, np.newaxis]*np.array([[1.,0.,0.],[0.,1.,0.], [0.,0.,0.5] ]) 
    
         N0000 = toVoight2D(np.einsum('...i,...j,...k,...l -> ...ijkl', N0, N0, N0, N0 ))
         N1111 = toVoight2D(np.einsum('...i,...j,...k,...l -> ...ijkl', N1, N1, N1, N1 ))
                        
         N0101 = np.einsum('...i,...j,...k,...l -> ...ijkl', N0, N1, N0, N1 )
         N1010 = np.einsum('...i,...j,...k,...l -> ...ijkl', N1, N0, N1, N0 )
         SN0101 = toVoight2D(N0101+N1010)
    
         
         smindex = np.where( (l1 - l0) < 1.e-6)[0]
         lmindex = np.where( (l1 - l0) >= 1.e-6)[0]
         if (len(smindex) > 0)  :
             D0[ios] += 2*mu*np.ones(ios[0].shape+(1,1))*N0000
             Dd[ios] += 2*mu*np.ones(ios[0].shape+(1,1))*N1111
        
             print('difficulty ') 
             D0[ios][smindex] +=   2.*mu*np.ones( ( len(smindex),1,1) )* SN0101[smindex,:,:]
             D0[ios][lmindex] +=   2.*mu*(l0[lmindex]/(l0[lmindex]-l1[lmindex]))[:, np.newaxis, np.newaxis]* SN0101[lmindex]
         
             D0[ios][smindex] +=   2.*mu*np.ones( ( len(smindex),1,1))* SN0101[smindex]
             Dd[ios][lmindex] +=   -2.*mu*(l1[lmindex]/(l0[lmindex]-l1[lmindex]))[:, np.newaxis, np.newaxis]* SN0101[lmindex]
             
             
         D0[ios] += 2*mu*(np.ones(ios[0].shape+(1,1))*N0000 +  (l0/(l0-l1))[:, np.newaxis, np.newaxis]* SN0101)
         Dd[ios] += 2*mu*(np.ones(ios[0].shape+(1,1))*N1111 -  (l1/(l0-l1))[:, np.newaxis, np.newaxis]* SN0101)
        
         
            
         
         # ! last term above would be dangerous in more general context, when l0-l1 -> 0, here it's probably always ok, since in the above context, we have 
         # l0*l1 < 0. if We hit trouble we need to separate again for the case where 2.(l0-l1)/(l0+l1) large ... Using l'hospital rule we could write :
         # D0[ios] += 2*mu*(np.ones(ios[0].shape+(1,1))*N0000 + np.ones( eps.shape[:-1] + (1,1))* SN0101)
         # Dd[ios] += 2*mu*(np.ones(ios[0].shape+(1,1))*N1111 + np.ones( eps.shape[:-1] + (1,1))* SN0101)
          
    
        
         return phi0, phid, stress0, stressd, D0, Dd
     
        """
     
        
    def _split_elastic_potential(self, strain, derivsigma = 0) :
         """ 
         derivsigma 0 : return phi0, phid
         derivsigma 1 : return phi0, phid, sigma0, sigmad
         derivsigma 2 : return phi0, phid, sigma0, sigmad, D0, Dd
         """
            
         #phi(eps, d) = mu (Sum_i f(d) * (e_i^+)^2  + (e_i^-)^2 + lambda/2 f(d) tr(eps)^2 +   Yc*h(d)
         # sit toute les vp sont > 0 , mu(Sum_i f(d) * (e_i^+)^2  + (e_i^-)^2) = mu f(d) * eps:eps 
         # sit toute les vp sont < 0 , mu(Sum_i f(d) * (e_i^+)^2  + (e_i^-)^2) = mu  * eps:eps 
         mu = self.mu
         lamb = self.lamb
         eps = np.atleast_2d(strain)
         eps11 = np.atleast_1d(eps[..., 0])
         eps22 = np.atleast_1d(eps[..., 1])
         eps12 = np.atleast_1d(eps[..., 2]/2.)
         trace = eps11 + eps22
         det   = eps11*eps22 -  eps12**2
         ss = det >= 0.
         ps1 = trace>=0.
         iss = np.where(ss)  # indexes where the vp are of same sign
         ios = np.where(np.logical_not(ss))  # indexes where the vp are of opposite sign
         ips1 = np.where(ps1)  # indexes where the traces are of +ve sign
         ins1 = np.where(np.logical_not(ps1))  # indexes where the traces are of -ve sign
         
         if derivsigma :         
             l0, l1, N0, N1 = eigenSim2D_voigt(eps[ios], True)
         else:
             l0, l1 = eigenSim2D_voigt(eps[ios])
         
         phi0 = np.zeros(( eps.shape[:-1]))
         phid = np.zeros(( eps.shape[:-1]))
         phid[ips1] = lamb/2.*trace[ips1]**2
         phi0[ins1] = lamb/2.*trace[ins1]**2
         I2iss = (eps11[iss]**2 + eps22[iss]**2 + 2.*eps12[iss]**2)
         #print(l0**2+l1**2 - eps11**2 + eps22**2+2.*eps12**2)
         phi0[iss] += mu *np.where(trace[iss] >=0., 0., 1.) * I2iss 
         phid[iss] += mu *np.where(trace[iss] >= 0., 1., 0.) * I2iss
         phi0[ios] += mu*l0**2
         phid[ios] += mu*l1**2
         
         if derivsigma == 0 : return phi0, phid
         stress0 = np.zeros(( eps.shape[:-1] + (3,)))
         stressd = np.zeros(( eps.shape[:-1] + (3,)))
         stress0[ins1] = (lamb*trace[ins1])[..., np.newaxis] * np.array([1.,1.,0.]) 
         stressd[ips1] = (lamb*trace[ips1])[..., np.newaxis] * np.array([1.,1.,0.]) 
         stress0[iss]  += np.where(trace[iss] >=0., 0., 2.*mu)[..., np.newaxis]*np.array([eps11[iss],eps22[iss], eps12[iss]]).T
         stressd[iss]  += np.where(trace[iss] >=0., 2.*mu, 0.)[..., np.newaxis]*np.array([eps11[iss],eps22[iss], eps12[iss]]).T
         N0xN0 = np.array([N0[...,0]**2, N0[...,1]**2, N0[...,0]*N0[...,1]]).T
         N1xN1 = np.array([N1[...,0]**2, N1[...,1]**2, N1[...,0]*N1[...,1]]).T
         stress0[ios] +=  2.*mu*l0[:, np.newaxis]*N0xN0 
         stressd[ios] +=  2.*mu*l1[:, np.newaxis]*N1xN1
         if derivsigma == 1 : return phi0, phid, stress0, stressd
         
         if derivsigma ==2: raise('Algebraic differentiation not implmented for Stiffness matrix')
           
    def Y(self, strain, d) :
        """return the Yield energy (Y = - dphidd)  at each gp for which strain and d are given"""
        phi0, phid  = self._split_elastic_potential(strain)
        return self.Yphisplited(phid, d)
    
    def Yphisplited(self, phid, d) :
        return  -self.g.jac(d)*phid - self.Yc*self.h.jac(d)
    
    def dY(self, strain, d) :
        """return the  derivative of the Yield energy (Y = - dphidd)  at each gp for which strain and d are given"""
        phi0, phid  = self._split_elastic_potential(strain)
        return  -self.g.hess(d)*phid - self.Yc*self.h.hess(d)
    
    def trialStress(self, strain, d) :
        """return strain elastic energy potential at each gp for which strain and d are given"""
        phi0, phid, stress0, stressd  = self._split_elastic_potential(strain, 1)    
        g, ignore, ignore = self.g(d)
        return stress0 + g[:, np.newaxis]*stressd
      
            
    def solveSoftening(self, strain, softeningvariablesn, softeningvariablesguess = None, localvariables = None, deriv =False):
        dn = softeningvariablesn
        d = dn.copy()
        phi0, phid = self._split_elastic_potential(strain)
        Ydn = self.Yphisplited(phid, dn)
        index = ((Ydn > 0.)*(d<1.)).nonzero()[0]
        for k in index :
            phidk = phid[k]
            fun = lambda x: self.Yphisplited(phidk, x).squeeze()
            if fun(1.)>0.: d[k]=1.
            else:    d[k] = sp.optimize.brentq(fun, dn[k], 1.)
        return d
    
    # This member is not needed to solve a problem using mechanics2D ...  This is for testing new stuff
    def dTrialStressDStrain(self, strain, d, localvariables =None):
        """ return dtrialstress/dstrain as d2phi/(dstrain)2 at fixed soft and local var """
        phi0, phid, stress0, stressd, D0, Dd  = self._split_elastic_potential(strain, 2)
        #dcorr = np.where(d<0.1,0.1,d)
        g,ignore, ignore = self.g(d)
        return D0 +g[:, np.newaxis, np.newaxis]*Dd
    # This member is not needed to solve a problem using mechanics2D ...  This is for testing new stuff Might be used latter for cases where internal variables are needed.
    def solveStressFixedSoftening(self, strain, softeningvariablesn = None, localvariablesn=None, localvariablesguess=None, deriv = False, usebasesolver = False):
        if usebasesolver : return super().solve_stress_fixed_softening(strain, softeningvariablesn, localvariablesn, localvariablesguess, deriv, usebasesolver)
        d = softeningvariablesn
        stress = self.trialStress(strain, d)
        if not deriv : return stress, None
        return stress, None, self.dTrialStressdStrain(strain, d) 
    
