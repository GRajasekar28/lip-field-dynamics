#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
// Copyright (C) 2021 Chevaugeon Nicolas
Created on Wed Dec 23 12:35:12 2020
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
import scipy.optimize
import cvxopt
from sortedcontainers import SortedList
import scipy
import scipy.optimize
from liplog import logger
from mesh import numberingTriPatch, numberingEdgePatch, partitionGraph
import multiprocessing
import linsolverinterface as lin

cvxopt.solvers.options['show_progress'] = False
cvxopt.solvers.options['abstol'] = 1.e-7
cvxopt.solvers.options['reltol'] = 1.e-6
cvxopt.solvers.options['feastol'] = 1.e-7
cvxopt.solvers.options['maxiters'] = 10000

cvxopt.solvers.options['show_progress'] = False
cvxopt.solvers.options['abstol'] = 1.e-6
cvxopt.solvers.options['reltol'] = 1.e-5
cvxopt.solvers.options['feastol'] = 1.e-6
cvxopt.solvers.options['maxiters'] = 10000


def getIntegratedPotentialFunctionFixedStrain(areas, localpotentialfixedstrain, d0):
    ''' return a cvxopt objective function F that compute the energy at fixed strain for a given
        vector x that store the values of d.
        input :
          - areas : an np.array that contain the area associated to each integration point (for the integral)
          - localpotentialfixedstrain : a callable which from a vector of damage values can compute the energy density.
          - d0 :  starting value of d for the optimization
    '''
    n = lin.size(d0)[0]
    def F(x=None, z= None):
        if x is None: return 0, cvxopt.matrix(d0) #return the number of nl constrain (0) and an initial x
        d = np.array(x).squeeze()
        phi, Y, dY = localpotentialfixedstrain(d, cphi = True, cY = True, cdY = z is not None)
        f =   cvxopt.matrix(areas.dot(phi), size=(1,1))
        Df =  cvxopt.matrix(-areas*Y, size =(1,n))
        if z is None: return  f, Df
        H = cvxopt.spdiag(cvxopt.matrix(-z[0]*areas*dY, size=(n,1)))
        return f,Df,H
    return F
    

def projectionOnEllipse(M, c, x):
    ''' projection de x0 sur l'ellipse d'equation x.T M x  = c(M symetric defini positif)'''
    
    x0 = x[0]
    y0 = x[1]
    M00 = M[0,0]
    M11 = M[1,1]
    M01 = M[0,1]
    eps = 1.e-6
    def residu(z, cjac = False):
        ''' residu z= [x0, x1, l ]
            r(z)     =  [x-x0 + l Mx , xtMx -c
        '''

        x = z[0]
        y = z[1]
        Mx_0 = M00*x + M01*y
        Mx_1 = M01*x + M11*y
        xMx  = x*Mx_0 + y* Mx_1 
        l = z[2]
        res = np.array([ x-x0+2*l*Mx_0, y-y0 + 2*l*Mx_1, xMx -c ])
        if not cjac  :return res
        jac = np.array([[ 1.+ 2.*l*M00,  2.*l*M01,   2*Mx_0  ],
                        [ 2.*l*M01,      1.+ 2.*l*M11, 2*Mx_1],
                        [2.*M00*x + 2*M01*y,      2.*M11*y + 2.*M01*x,  0.]])
        return res, jac
    
    z = np.array([x0, y0, 0.])
    r, j  = residu(z, True)
    nr0 = np.linalg.norm(r)
    nr = nr0
    while ((nr >= nr0*eps ) and (nr >=eps)):
        z = z - np.linalg.solve(j, r)
        r, j = residu(z, True)
        nr = np.linalg.norm(r)
    return z[:2]

def combineConeIneq(iterableConeIneq):
    "From an iterable collection of coneIneq (following cvxopt description of coneineq) build one groupe of coneineq combining them all "
    l = sum( [ it['dims']['l']  for it in iterableConeIneq ] ,0)
    q = sum( [ it['dims']['q']  for it in iterableConeIneq ], [])
    s = sum( [ it['dims']['s']  for it in iterableConeIneq ], [])
    dims = {'l': l, 'q': q, 's': s}
    if len(s) != 0 : raise #s type coneineq are not taken in charge in the library !
    Gl  = cvxopt.sparse([ it['G'][:it['dims']['l'],:]  for it in iterableConeIneq ] )
    Gq  = cvxopt.sparse([ it['G'][it['dims']['l']:,:]  for it in iterableConeIneq ] )
    hl  = cvxopt.matrix([ it['h'][:it['dims']['l'],:]  for it in iterableConeIneq ] )
    hq  = cvxopt.matrix([ it['h'][it['dims']['l']:,:]  for it in iterableConeIneq ] )
    G = cvxopt.sparse([Gl, Gq])
    h = cvxopt.matrix([hl, hq])
    return {'G':G, 'h':h, 'dims':dims}
    
def kktToolsBuildW(W):
    ''' from W in cvxopt formt, build the corresponding sparse matrix W (block diagonal) only linear and quadratic cones are considered (semidefinite cone are not considered)'''
    Wq = [  beta*(2.*v*v.T - J) for v, beta in zip(W['v'], W['beta'] ) for J in [cvxopt.spdiag([1]+[-1]*(v.size[0]-1)) ] ]
    if W['d'].size[0] != 0:
        Wl  = cvxopt.spdiag(W['d'])
        return cvxopt.spdiag([Wl]+Wq)
    else :
        return cvxopt.spdiag(Wq)

def kktToolsBuildinvW(W):
    ''' from W in cvxopt formt, build the corresponding sparse matrix W^-1 (block diagonal) only linear and quadratic cones are considered (semidefinite cone are not considered)'''
    invWl  = cvxopt.spdiag(W['di'])
    def bv(v, beta):
        n = v.size[0]
        v[0] = -v[0]
        res = 2.*v*v.T
        res[0] -= 1.
        res[n+1::n+1] +=1.
        res /=beta 
        return res
    invWq = [bv(-v, beta) for v, beta in zip(W['v'], W['beta']) ]
    return cvxopt.spdiag([invWl]+invWq)

def kktsol(H,G,Wcvx, mode = 'direct', linsolver= 'umfpack'):
    ''' return a kkt solver, abble to solve [ H, G.T] |x| = | rx| 
                                            [ G    W] |z|   | ry|
        H, square matrix (Hessian of the objective function)
        G, rectangular matrix (inequality constraint ..)
        Wcvx terms of the "weight" square matrix in the cvxopt special format
        This is used to replace the default kkt solver in cvxopt which is to slow ...
        two version can be used by setting the mode to 'direct' or 'schur'
        each version need a linear solver. It can be selected by setting the linsolver parameter 
        to either 'umfpack' or 'cholmod'
    '''
    if mode == 'direct' : return kktsolDirect(H,G,Wcvx, linsolver)
    elif mode == 'schur' : return kktsolSchur(H,G,Wcvx, linsolver)
    else : raise

def kktsolDirect(H,G,Wcvx, linsolver= 'umfpack'):
    W = kktToolsBuildW(Wcvx)
    K = cvxopt.sparse([[H, G],[G.T, -W.T*W]])
    if  linsolver == 'umfpack' :    
        sLU = cvxopt.umfpack.symbolic(K)
        LU = cvxopt.umfpack.numeric(K,sLU)
        def solve(x, y, z): 
            X = cvxopt.matrix([x,z])
            cvxopt.umfpack.solve(K,LU, X)    
            x[:] = X[:x.size[0]]
            z[:] = W*X[x.size[0]:]
        return solve
    elif linsolver == 'cholmod' :
        LTL = cvxopt.cholmod.symbolic(K)
        cvxopt.cholmod.numeric(K,LTL)
        def solve(x, y, z):
            X = cvxopt.matrix([x,z])
            cvxopt.cholmod.solve(LTL, X, sys = 0)
            x[:] = X[:x.size[0]]
            z[:] = W*X[x.size[0]:]
        return solve
    else  : raise

def kktsolSchur(H,G,Wcvx, linsolver= 'umfpack'):
    invW = kktToolsBuildinvW(Wcvx)
    invW2 = invW*invW
    K = H + G.T*invW2*G
    if  linsolver == 'umfpack' :    
        sLU = cvxopt.umfpack.symbolic(K)
        LU = cvxopt.umfpack.numeric(K,sLU)
        def solve(x, y, z):
            X = x+ G.T*invW2*z
            cvxopt.umfpack.solve(K,LU, X)
            x[:] = X[:x.size[0]]
            z[:] = -invW*(z-G*x)
        return solve    
    elif linsolver == 'cholmod' :
        LTL = cvxopt.cholmod.symbolic(K)
        cvxopt.cholmod.numeric(K,LTL)
        def solve(x, y, z):
            X = x+ G.T*invW2*z
            cvxopt.cholmod.solve(LTL, X, sys = 0)
            x[:] = X[:x.size[0]]
            z[:] = -invW*(z-G*x)
        return solve
    else : raise
    
def slack(coneineq,  d, check = False ):
        """ Compute the slack vector (the difference between Gd and h, in the sense of the eventual second order cones defined by dim)"""
        # Warning ; the code suppose to be fast that all second order cones (q) are of the same size.
        G = coneineq['G']
        h = coneineq['h']
        dims = coneineq['dims']
        d = cvxopt.matrix(d)
        ml = dims['l']
        mq = len(dims['q'])
        if check : 
            m = G.size[0]
            n = G.size[1]
            if h.size[0] != m : raise
            if d.size[0] != n : raise
            if (ml+sum(dims['q']) != m ): raise
            if(mq > 0) :
                if max(dims['q']) != min(dims['q']) :raise
            if len(dims['s']) != 0 : raise
        s= h - G*d
        if (mq == 0) : return s     
        sdim = dims['q'][0]
        snltmp = s[ml:]
        snltmp.size = (sdim, mq)
        s0 = np.abs(snltmp[0,:]).squeeze()
        s1 = np.linalg.norm(snltmp[1:,:], axis=0).squeeze()
        s[ml:ml+mq ] = s0 -s1
        return s[:ml+mq]
    
def jacslack(coneineq,  d, check = False ):
        """Compute the derivative of the slack variable. Usefull for second order cone constrain represented has generic nonlinear constrain 
            Or to linearise the second order cone constrain
            #Warning : We make the assumption that all the quadratic cones have the same size for spped.
            This can be check by setting chek to True
        """
        G = coneineq['G']
        h = coneineq['h']
        dims = coneineq['dims']
        ml = dims['l']
        mq = len(dims['q'])
        #print('ml', ml, 'mq', mq)
        n = G.size[1]
        if check : 
            m = G.size[0]
            if h.size[0] != m : raise
            if d.size[0] != n : raise
            if (ml+sum(dims['q']) != m ): raise
            if(mq > 0) :
                if max(dims['q']) != min(dims['q']) :raise
            if len(dims['s']) != 0 : raise
        d = cvxopt.matrix(d, size = (n,1))
        s= h - G*d
        jacl = -G[:ml,:]
        if not mq :  return jacl
        sdim = dims['q'][0]
        snl = s[ml:]
        snl.size = (sdim, mq)
        s1 = np.array(snl[1:3,:])
        norm_s1 = np.linalg.norm(s1, axis=0)
        fac = np.zeros((sdim,mq))
        fac[ 0, :] = -np.sign(snl[0,:])
        with np.errstate(divide='ignore', invalid='ignore'):
            fac[1:,: ] = np.where(norm_s1> 1.e-5, s1/norm_s1, np.sign(s1))         
#        F = cvxopt.spmatrix(fac.T.flatten(), sum( [[i]*sdim for i in range(mq) ], []), np.arange(mq*sdim) , (mq, sdim*mq) )
#        jacq = F*G[ml:,:]
#        jac = cvxopt.sparse([  jacl, jacq]   )
        
        F = cvxopt.spmatrix(list(-np.ones(ml)) + list(fac.T.flatten()),
                            [i for i in range(ml)] +sum( [[i]*sdim for i in range(ml,ml+mq) ], []),
                            [i for i in range(ml)]  +[i for i in range(ml, ml+mq*sdim)] , (ml+mq, ml+sdim*mq) )
        jac = F*G
         
#        Fq = cvxopt.spmatrix( fac, sum( [[i]*sdim for i in range(mq) ], []), np.arange(mq*sdim) , (mq, sdim*mq) )
#        F = cvxopt.spdiag([Fl, Fq])
#        jac = F*G
        return jac
    
def linearizeConeIneq(G, h, dims, x0):
    '''return a linearisation of the possibly second order cone problem defined by G, h, dims around point x0 
       Gx-h less than 0 were less is in the sense of the cones defined by dims (linear and quadratic cones)
       is transform to :
       Gl(xO) dx -hl(x0) <= 0 where less than refer to the regular inequalities
    '''
    
    s0 =   slack({'G':G, 'h':h, 'dims':dims},  x0, check = False )
    ds0 = -jacslack({'G':G, 'h':h, 'dims':dims},  x0, check = False )
    return ds0, s0, {'l':ds0.size[0], 'q':[], 's':[]}
    

def coneQPcvxopt(P, q, coneineq, x0=None,  
                 kktsolveroptions = {'mode':'direct', 'linsolve':'umfpack'},
                 logger = logger, **kwargs):
    ''' solve a cone qp problem using cvxopt note x0 is not used
    optional argument in kwargs : 
    if 'check_faisible' is set to True, a cone lp is solve first to check faisability.
       in this case if the lp do not converge, res['status'] is set to the res['status'] of the conelp.
       and the function return.'''
    G = coneineq['G']
    h = coneineq['h']
    dims = coneineq['dims']
    check_faisable = kwargs.get('check_faisible', False)
    if check_faisable :
        res = cvxopt.solvers.conelp(c = q, G = G, h =h, dims =dims) 
        if (res['status'] != 'optimal'):  
            print('conelp status', res['status'])
            return {'status':res['status']}
    
    # define the kkt solver
    mode =  kktsolveroptions['mode'] 
    if mode  == 'cvxdefault' : kktsolver = None
    else : 
        def kktsolver(Wcvx):
           solve = kktsol(P,G,Wcvx, mode,  linsolver= kktsolveroptions['linsolve'])
           return solve
    #call cvxopt cone quadratic problem solver.
    res = cvxopt.solvers.coneqp(P = P, q = q, G = G, h =h, dims =dims, kktsolver = kktsolver)
    if (res['status'] != 'optimal') : 
           logger.error('ERROR : cvxopt.solvers.coneqp did not converge ! status : '+ res['status'])
           conv =False
    else : conv = True
    return {'d':np.array(res['x']).squeeze(), 'Converged':conv}
    
    
    
def coneQPscipy(P, q, coneineq, x0=None, logger=logger, **kwargs):
    ''' minimize a cone quadratic poblem using the generic scipy minimize interface
        The coneQP is translated into the scipy interface to non-linear minimization
        'SLSQP' solver is used. Very slow since SLSQP's implementation in scipy is dense ...
    '''
    n = x0.size[0] 
    if (n == 0) : return
    x0 = np.array( x0).reshape(n) 
    def fun (x):
        x = cvxopt.matrix(x, size = (n,1))
        return (.5*x.T*P*x+q.T*x)[0]     
    def jac(x):
        x = cvxopt.matrix(x, size = (n,1))
        return np.array(x.T*P+q.T).reshape(n) 
    def hess(x):
        return np.array(P) 
    def ineqfunnl(x):
        s = slack(coneineq,  x, check = False)
        return np.array(s).reshape(s.size[0]) 
    def ineqjacnl(x):     
        jac_cvxoptsparse = jacslack(coneineq,  x, check = False)
        return np.array(cvxopt.matrix(jac_cvxoptsparse)).reshape(jac.size[0], jac.size[1])     
    nlc = {'type':'ineq', 'fun':ineqfunnl, 'jac':ineqjacnl}
    res = scipy.optimize.minimize(fun, x0, method='SLSQP', jac = jac, constraints = [ nlc])   
    if (res.success) : return {'d': res.x.squeeze(), 'status':'optimal'}
    logger.error('ERROR : scipy.optimize.minimize did not converge ! status : '+ res.message)
    raise

def minimizeCPscipy(F, coneineq, x0=None, logger= logger, **kwargs):
    ''' minimize a convex problem with (upto ) quadratic cone inequality constraintusing the generic scipy minimize interface
        The cp is translated into the scipy interface to non-linear minimization
        'SLSQP' solver is used. Very slow since SLSQP's implementation in scipy is dense ...
        Note : since SLSQP does not care about convexity, it can work even if F is not convex
    '''
    n = x0.size[0] 
    if (n == 0) : return
    x0 = np.array( x0).reshape(n) 
    def fun (x):
        x = cvxopt.matrix(x, size = (n,1))
        f,Df = F(x)
        return f[0]   
    def jac(x):
        x = cvxopt.matrix(x, size = (n,1))
        f,Df = F(x)
        return np.array(Df).reshape(n) 
    def hess(x):
        x = cvxopt.matrix(x, size = (n,1))
        z = cvxopt.matrix([1.], size = (n,1))
        f,Df,H = F(x,z)
        return np.diagflat(H[::(H.size()+1)])
    def ineqfunnl(x):
        s = slack(coneineq,  x, check = False)
        return np.array(s).reshape(s.size[0]) 
    def ineqjacnl(x):     
        jac = damageProjector2D.jacslack(coneineq,  x, check = False)
        return np.array(jac).reshape(jac.size[0], jac.size[1])     
    nlc = {'type':'ineq', 'fun':ineqfunnl, 'jac':ineqjacnl}
    res = scipy.optimize.minimize(fun, x0, jac = jac, constraints = [ nlc])   #method='SLSQP'
    if (res.success) : return {'d': res.x.squeeze(), 'Converged':True, 'status':'optimal', 'res': res}
    logger.error('ERROR : scipy.optimize.minimize did not converge ! status : '+ res.message)
    return {'d': res.x.squeeze(), 'Converged':False, 'status':'notconverged', 'res':res}
    raise
    
def minimizeCPcvxopt(F, lipconstrain, x0 = None, 
                     kktsolveroptions = {'mode':'direct', 'linsolve':'umfpack'}, 
                     logger=logger, **kwargs):
       ''' solve a convex problem with inequality convex constrains 'using cvxopt.solvers.cp
           F is the convex objective function as defined in cvxopt.
           constrain contain the inequality constrain description. we only assume linear ans second order cone constrain.
           init is meant to represent an initial guess but is not used by cvxopt solver.
           kktsolveroption control how the kkt linear problem at the kernel of cvxopt.solvers.cp is solved.
           possible values for kktsolveroptions : 
               'cvxopdefault' : use default cvxopt kktsolver. usefull for dense or small sparse problem
               'direct_umfpack' : directly solve the kkt problem using umfpack
               'direct_cholmod' : directly solve the kkt problem using cholmod. 
                   Might fail since there is no guaranties that the kkt matrix is positive definite
               'schur_umfpack' and 'schur_cholmod' first condense the kkt problem on the primary unknowns then solve with either
               umfpack or cholmod.   The condensed problem is positive definite and cholmod should work.        
       '''
       G = lipconstrain['G']
       h = lipconstrain['h']
       dims = lipconstrain['dims']       
       mode =  kktsolveroptions['mode']       
       if mode  == 'cvxdefault' :
           kktsolvernl = None
       else :
          def kktsolvernl(x,z,Wcvx) :
              f,Df,H = F(x,z)
              solve = kktsol(H,G,Wcvx, mode,  linsolver= kktsolveroptions['linsolve'])
              return  solve

       res = cvxopt.solvers.cp(F, G = G, h =h, dims =dims, kktsolver = kktsolvernl)
       if (res['status'] != 'optimal') : 
           logger.error('ERROR : cvxopt.solvers.cp did not converge ! status : '+ res['status'])
           conv =False
       else : conv = True
       return {'d':np.array(res['x']).squeeze(), 'Converged':conv, 'res':res}
   
def improvebyQuadApproxCPcvxopt(F, lipconstrain, x0 = None, 
                     kktsolveroptions = {'mode':'direct', 'linsolve':'umfpack'}, 
                     logger=logger, **kwargs):
       ''' approximate the solution of convex problem with inequality convex constrains 
           by developing a linear or quadratic approximation of F around x0 and then use coneqp or conelp to find an approx
           solution, then use a line search in to improve the solution
           
           F is the convex objective function as defined in cvxopt.
           
           constrain contain the inequality constrain description. we only assume linear ans second order cone constrain.
           init is meant to represent an initial guess but is not used by cvxopt solver.
           kktsolveroption control how the kkt linear problem at the kernel of cvxopt.solvers.cp is solved.
           possible values for kktsolveroptions : 
               'cvxopdefault' : use default cvxopt kktsolver. usefull for dense or small sparse problem
               'direct_umfpack' : directly solve the kkt problem using umfpack
               'direct_cholmod' : directly solve the kkt problem using cholmod. 
                   Might fail since there is no guaranties that the kkt matrix is positive definite
               'schur_umfpack' and 'schur_cholmod' first condense the kkt problem on the primary unknowns then solve with either
               umfpack or cholmod.   The condensed problem is positive definite and cholmod should work.        
       '''
       if  x0 is None:
           tmp, x0 =  F()  
       
       quad = False
       G = lipconstrain['G']
       h = lipconstrain['h']
       dims = lipconstrain['dims']
       n = x0.size[0]
       if not quad :
           f0, J0 =  F(x0)
           H0 = cvxopt.spmatrix([],[],[], (n,n))
       else :
           f0, J0, H0 =  F(x0, cvxopt.matrix([1.]))
       
       mode =  kktsolveroptions['mode']
       def Fq(Dx=None,  z = None):
               if Dx is None: return 0, cvxopt.matrix(0, (n,1)) #return the number of nl constrain (0) and an initial x
              # f = f0+J0*Dx+0.5*Dx.T*H0*Dx
               H0Dx = H0*Dx
               f = f0+Dx.T*(J0.T+0.5*H0Dx)  
               df = J0.T + H0Dx
               if z is None : return f, df.T
              # print('ss',f.size, df.size, H0.size)
               return f, df.T, z[0]*H0  
        
       
       if mode  == 'cvxdefault' :
           kktsolverl = None
       else :
          def kktsolverl(Wcvx) :
              solve = kktsol(H0,G,Wcvx, mode,  linsolver= kktsolveroptions['linsolve'])
              return  solve
          def kktsolvernl(x,z,Wcvx) :
              f,Df,H = Fq(x,z)
              solve = kktsol(H,G,Wcvx, mode,  linsolver= kktsolveroptions['linsolve'])
              return  solve
       if not quad :
           res = cvxopt.solvers.conelp(J0.T, G, h - G*x0, dims, kktsolver = kktsolverl)
       else:
           res = cvxopt.solvers.coneqp(H0, J0.T, G, h - G*x0, dims, kktsolver = kktsolverl)
           
       if (res['status'] != 'optimal') : 
           logger.error('ERROR : cvxopt.solvers.cp did not converge ! status : '+ res['status'])
           conv =False
       else : conv = True
       Dx  = res['x']
       xapp = x0+Dx
       #return {'d':np.array(xapp).squeeze(), 'Converged':conv}
   
       def Falpha(x=None,  z = None):
           if x is None: return 0, cvxopt.matrix([1.]) #return the number of nl constrain (0) and an initial x
           d = x0 + x[0]*Dx
           if z is None:
               f, df = F(d)
               dfdalpha = df*Dx
               return f, dfdalpha
           else :
               f, df, H = F(d, z)
               dfdalpha = df*Dx
               Halpha = Dx.T*H*Dx
               return f, dfdalpha , Halpha
           
       f0,df0 = Falpha(cvxopt.matrix([0.]))
       f1,df1 = Falpha(cvxopt.matrix([1.]))
       
       fXapp, dfXapp = Fq(Dx)
       fX, dfX= F(xapp)
       
       print('f1', f1[0], 'fX',  fX[0], "fXapp", fXapp[0])
       if np.abs( (f1[0]-fXapp[0]) / f1[0] ) < 1.e-8:
           return {'d':np.array(xapp).squeeze(), 'Converged':conv}
       Galpha = cvxopt.spmatrix([-1.,1.],[0,1],[0,0],(2,1))
       halpha = cvxopt.matrix([0.,1.])
       dimsalpha = {'l':2,'q':[], 's':[]}   
       resalpha = cvxopt.solvers.cp(Falpha, G = Galpha, h =halpha, dims =dimsalpha)
       alpha = resalpha['x'][0]
       print('linesearch alpha',alpha)
       x = x0 + alpha*Dx
       return {'d':np.array(x).squeeze(), 'Converged':conv}

   

def minimizeQuadApproxCPcvxopt(F, lipconstrain, x0 = None, 
                     kktsolveroptions = {'mode':'direct', 'linsolve':'umfpack'}, 
                     logger=logger, prevres = None, **kwargs):
       ''' approximate the solution of convex problem with inequality convex constrains 
           by deveolpping a linear or quadratic approximation of F and then use coneqp or conelp to find an approx
           solution, then use a line search in to improve the solution
           
           F is the convex objective function as defined in cvxopt.
           
           constrain contain the inequality constrain description. we only assume linear ans second order cone constrain.
           init is meant to represent an initial guess but is not used by cvxopt solver.
           kktsolveroption control how the kkt linear problem at the kernel of cvxopt.solvers.cp is solved.
           possible values for kktsolveroptions : 
               'cvxopdefault' : use default cvxopt kktsolver. usefull for dense or small sparse problem
               'direct_umfpack' : directly solve the kkt problem using umfpack
               'direct_cholmod' : directly solve the kkt problem using cholmod. 
                   Might fail since there is no guaranties that the kkt matrix is positive definite
               'schur_umfpack' and 'schur_cholmod' first condense the kkt problem on the primary unknowns then solve with either
               umfpack or cholmod.   The condensed problem is positive definite and cholmod should work.        
       '''
       if  x0 is None:
           tmp, x0 =  F()

       G = lipconstrain['G']
       h = lipconstrain['h']
       dims = lipconstrain['dims']
       f0, J0, H0 =  F(0.*x0, cvxopt.matrix([1.]))
       mode =  kktsolveroptions['mode']
     
       if mode  == 'cvxdefault' :
           kktsolverl = None
       else :
          def kktsolverl(Wcvx) :
              solve = kktsol(H0,G,Wcvx, mode,  linsolver= kktsolveroptions['linsolve'])
              return  solve
       initvals = None
       res = cvxopt.solvers.coneqp(H0, J0.T, G, h, dims, initvals = initvals, kktsolver = kktsolverl)
       if (res['status'] != 'optimal') : 
           logger.error('ERROR : cvxopt.solvers.cp did not converge ! status : '+ res['status'])
           conv =False
       else : conv = True
       x  = res['x']
       return {'d':np.array(x).squeeze(), 'Converged':conv, 'cvxoptres':res}
   

def lineariseCPcvxopt(F, constrain, x0):
    '''return linearized problem around x0'''
    #x0 = x0
    f, Df =  F(x0)
    G = constrain['G']
    h = constrain['h']
    dims = constrain['dims']
    if len(dims['s']) != 0 : raise
    nl = dims['l']
    nq = len(dims['q'])
    Gx0 = G*x0    
    Gl = G[:nl,:]
    hl = h[:nl]-Gx0[:nl]    
    hql = cvxopt.matrix(0., (nq,1) )
    index = nl
    i = 0
    GqlI =    [cvxopt.matrix([])]
    GqlJ =    [cvxopt.matrix([])]
    GqlVal =  [cvxopt.matrix([])]
    
    for size in  dims['q']:
        Gqx0i = Gx0[index: index+size]
        hql[i] = h[index] - Gqx0i.T*Gqx0i
        glqi =  cvxopt.sparse(Gqx0i.T)*G[index: index+size, :]
        GqlVal.append( glqi.V )
        GqlI.append( glqi.I+i )
        GqlJ.append( glqi.J )
        i = i+1
        index = index +size
    GqlVal = cvxopt.matrix(GqlVal)
    GqlI = cvxopt.matrix(GqlI)
    GqlJ = cvxopt.matrix(GqlJ)
    Gql = cvxopt.spmatrix(GqlVal, GqlI, GqlJ, (nq, x0.size[0] ) )
    Gql = cvxopt.sparse([Gl, Gql])
    hql = cvxopt.matrix([hl, hql])   
    dims = {'l': Gql.size[0], 'q': [], 's': []}
    return f,  Df, {'G':Gql, 'h':hql, 'dims':dims}
       
class lipProblem:
    def __init__(self,P,q,G,h,dims):
        n = P.size[0]
        if(n != P.size[1]) : raise
        if(n != q.size[0]) : raise
        if(n != G.size[1]) : raise
        m = G.size[0]
        if(m!= h.size[0]) : raise
        if ((dims['l'] + sum(dims['q']) + sum(dims['s'] )) != m) : raise
        self.P = P
        self.q = q
        self.G = G
        self.h = h
        self.dims = dims
        
        self.n = n
        self.m = m
        
    def constrainGap(self, d):
        print('d',d)
        print(self.n)
        if (d.shape[0] != self.n) : raise
        Gd = self.G*cvxopt.matrix(d, size=(self.n,1))
        print(self.G)
        print('grads',Gd)
        for ie in range(self.dims['l'], self.dims['l'] + sum(self.dims['q']), 3):
            print('ie', ie)
            print('grad',Gd[ie+1:ie+3])
            gdx = Gd[ie+1]
            gdy = Gd[ie+2]
            normg = np.sqrt(gdx**2 + gdy**2)
            print('normgrad',normg)
    

def lipConstant(mesh, d):
    """ Compute the lip-constant (max_{ij} |di-dj|/|xi-xj]) This only give accurate results for convex domain (with no holes)"""
    lipc = 0.
    x =  mesh.xy[:,0]
    y =  mesh.xy[:,1]
    for i in range(d.shape[0]-1):
        di = d[i]
        ddi = np.abs(d[i+1:] - di)
        dxi = x[i+1:] - x[i]
        dyi = y[i+1:] - y[i]
        lipc = max(lipc, np.max(ddi/np.sqrt(dxi**2 + dyi**2)))
    return lipc


class damageProjector2D :
    def __init__(self, mesh) : 
        self.mesh = mesh
        self._globalLipTriIneq = None
        self._globalLipEdgeIneq = None
        self.n = mesh.nvertices
        self.cvxopt_prev = None
        self._v2vdata = None
        self._edgedeltaop = None
        self.prevres = None 
        
    def islip(self,d, lc):
        ineq = self.getGlobalLipTriIneq(lc)
        G = ineq['G']
        nf = G.size[0]
        nv = G.size[1]
        Gd = G*cvxopt.matrix(d, size=(nv,1))
        ng = np.zeros(nf//3)
        
        for ie in range(ineq['dims']['l'], ineq['dims']['l'] + sum(ineq['dims']['q']), 3):
            gdx = Gd[ie+1]
            gdy = Gd[ie+2]
            normg = np.sqrt(gdx**2 + gdy**2)
            ng[ie//3] = normg
        
        ng = ng*lc
        return (ng.max()<=1., ng)
        
    def getLipV2VData(self):
        if self._v2vdata is None :
            v2vdata = [None]*self.n
            v2vs = self.mesh.getVertex2Vertices()
            for idv in range(self.n):
                v2v = v2vs[idv]
                le  = np.linalg.norm( self.mesh.xy[v2v] - self.mesh.xy[idv], axis =1)
                v2vdata[idv] = (v2v, le)
            self._v2vdata = v2vdata
        return self._v2vdata            
            
    def getGlobalLipTriIneq(self,lc):
        if self._globalLipTriIneq is None:
            t = self.mesh.triangles
            nt = t.shape[0]
            GX = self.mesh.elementaryGradOperators().flatten()
            GI = np.empty((3, nt*2), dtype ='int')
            GI[:,::2] = np.arange(1,3*nt, 3)
            GI[:,1::2] = np.arange(2,3*nt, 3)
            GI = GI.flatten('F')
            GJ = np.empty((nt*2, 3), dtype ='int')
            GJ[::2] =   t
            GJ[1::2] =  t
            GJ = GJ.flatten()
            G  = cvxopt.spmatrix(GX, GI, GJ)
            h = cvxopt.matrix([1.,0.,0.]*nt, size=(3*nt,1))
            dims ={'l':0, 'q':[3]*nt, 's':[]}
            self._globalLipTriIneq = {'G': G, 'h':h, 'dims':dims}
        gLipTriIneq = self._globalLipTriIneq.copy()
        gLipTriIneq['G'] = gLipTriIneq['G']*lc
        return gLipTriIneq
        
    def getPatchLipTriIneq(self, triangles, vg2l, lc):
        gt = self.mesh.triangles
        m = len(triangles)
        n = len(vg2l)
        t = np.array(list(map(vg2l.get, gt[triangles].flatten()))).reshape((m,3))
        GX = self.mesh.elementaryGradOperators()[triangles].flatten()
        GI = np.empty((3, m*2), dtype ='int')
        GI[:,::2] = np.arange(1,3*m, 3)
        GI[:,1::2] = np.arange(2,3*m, 3)
        GI = GI.flatten('F')
        GJ = np.empty((m*2, 3), dtype ='int')
        GJ[::2] =   t
        GJ[1::2] =  t
        GJ = GJ.flatten()
        G = lc*cvxopt.spmatrix(GX, GI, GJ, size=(3*m,n))
        dims = {'l': 0, 'q': [3]*m, 's':  []}  
        h = cvxopt.matrix([1.,0.,0.]*m, size=(3*m,1))
        return {'G':G, 'h':h, 'dims':dims}
    
    def getGlobalLipEdgeIneq(self, lc) :
        if self._globalLipEdgeIneq is None :
            edges = self.mesh.getEdge2Vertices()
            v_glob2loc = np.arange(self.mesh.nvertices)
            self._globalLipEdgeIneq = self.getPatchLipEdgeIneq(list(range(edges.shape[0])), v_glob2loc, lc = 1.)           
        globalLipEdgeIneq= self._globalLipEdgeIneq.copy()
        globalLipEdgeIneq['h'] = globalLipEdgeIneq['h']/lc
        return globalLipEdgeIneq
        
    def getPatchLipEdgeIneq(self, edges, v_glob2loc, lc = 1.) :
        nedge = len(edges)
        n = len(v_glob2loc)
        m = nedge
        x = np.empty(2*m)
        I = np.empty(2*m, dtype='int')
        J = np.empty(2*m, dtype='int')
        for ie, e2v in enumerate(self.mesh.getEdge2Vertices()[edges]):
            I[2*ie:2*ie+2]     = 2*ie+1
            v0id  = e2v[0]
            v1id  = e2v[1]
            J[2*ie] = v_glob2loc[v0id]
            J[2*ie+1] = v_glob2loc[v1id]
            xy0 = self.mesh.xy[v0id]
            xy1 = self.mesh.xy[v1id]
            d01 = np.linalg.norm(xy1-xy0)
            x[2*ie:2*ie+2] = [1./d01, -1./d01]
        G = cvxopt.spmatrix(x,I, J, (2*m,n))
        h = cvxopt.matrix([1./lc,0.]*m, size=(2*m,1))
        dims = {'l':0, 'q': [2]*m, 's' : []}
        return {'G': G, 'h':h, 'dims': dims}  
            
    def lipProjClosestToTarget(self,dmin, dtarget, lc, lipmeasure = 'triangle', init = None, kktsolveroptions = {'mode':'direct', 'linsolve':'umfpack'}, logger=logger):
        """ Find the closest d to dtarget (in the sense of the L2Norm) That fullfill the lipconstrain Warning dmin not used ..."""
        n = self.n
        if ( (dmin.size != n) | (dtarget.size != n) | (lc < 0.)) :  raise
        xt = cvxopt.matrix(dtarget, (n, 1))
        P = cvxopt.spdiag([1]*n)
        if lipmeasure == 'edge' :
            lipineq = self.getGlobalLipEdgeIneq(lc)    
        elif  lipmeasure == 'triangle' :
            lipineq = self.getGlobalLipTriIneq(lc)   
        
        if init is not None:  init0 = init
        else :                init0 = None
        
        res = coneQPcvxopt(P, -xt, lipineq, x0=init0, kktsolveroptions =  kktsolveroptions, logger= logger)
        
        return res['d']
        
   
    def lipProjFM(self, dtarget, lc, seeds =None, lipmeasure = 'edge', side ='up', logger = logger, verbose=False) :
        """ compute the upper or lower bound ('side':['up','lo']) of the lip projection of dtarget using a Fast Marching approach,
            either imposing that 'edge' lip contrain are verified (lipmeasure='edge')
            or imposing that edge and triangle constran lip contrain are verified (lipmeasure='traingle')
            lc is the characteristic lenght.
            seeds is a list of node to start from. if not given, all the nodes of the mesh are put in the front.
            return a dictionnary containing 
            - 'd' : the projected field, as an array containing value at each node
            - 'visitedvertices' : an boolean array contaning True for all nodes visited by the FM
        """
        nfailed = [0]   
        def triald2(x01, x02, d0, d1, lc): 
                
                invdet = 1./ (x02[0]*x01[1] - x01[0]*x02[1])  
                iG00 =  x01[1]*invdet
                iG01 = -x02[1]*invdet
                iG10 = -x01[0]*invdet
                iG11 =  x02[0]*invdet
                A0 = -d0*iG00 + (d1-d0)* iG01
                A1 = -d0*iG10 + (d1-d0)* iG11
                B0 = iG00
                B1 = iG10
                a = B0**2+B1**2
                b = 2.*(A0*B0+A1*B1)
                c = A0**2 + A1**2 -1./lc/lc
                delta = b**2 - 4.*a*c
                if delta < 0 : return None
                sdelta = np.sqrt(delta)
                return [(-b-sdelta)/2./a, (-b+sdelta)/2./a]
        #print(nfailed)     
        d = dtarget.copy()
        visited = np.zeros(d.shape, dtype='bool')
        if (side == 'lo'): 
            front = SortedList(key=lambda x: (-x[0],x[1]))
            sidecoef = 1
        elif(side == 'up'):
            front = SortedList()
            sidecoef = -1
        else:
            logger.error('side '+side+' unknown in lipProjFMopt')
            raise
        if seeds is None :
            for (iv, div) in enumerate(d) :   front.add((div,iv))
        else :
            for  iv in seeds : front.add((d[iv], iv))     
            
        def enforceedgelip(idfront, xfront, dfront):
            idns, les = self.getLipV2VData()[idfront] #idns : vertex ids of neibhors, les lenghts of corresponding edge            
            for idv, l in zip(idns, les) : 
                dv = d[idv]
                dv_limit = dfront +  sidecoef * l/lc
                if (sidecoef * dv) >  (sidecoef *dv_limit) :
                   d[idv] = dv_limit
                   front.discard( (dv, idv ) )
                   front.add( (dv_limit, idv ) )
        ### the below 'enforce trainglelip function are tentatives to generalize the lip measure that we want to full-fill to triangle lip ...
        ## up to now the result is not satisfactory .... I need to impove on that. Fo now only use the edge measure at FM/Dijistra stage,
        ## and check after the minimization is the taingle constrain is veryfyed. if not, repeat ...
        def enforcetrianglelip(idfront, xfront, dfront, nfailed):    
            triangles = self.mesh.getVertex2Triangles(idfront)
            for t in triangles :                        
                idf = -1
                idb = -1
                for idv in self.mesh.triangles[t] :
                    if idv != idfront :
                        dv = d[idv]
                        if sidecoef*dv <=  sidecoef * dfront : 
                            idb = idv
                            db = dv
                        else :
                            idf = idv
                            df = dv                            
                if (idf != -1) and (idb !=-1) :
                    xfront2b = (xfront - self.mesh.xy[idb]).squeeze()
                    xfront2f = (xfront - self.mesh.xy[idf]).squeeze()
                    r = triald2(xfront2b, xfront2f, dfront, db, lc)
                    if r is None :
                        #print('ARG')
                        nfailed[0] +=1
                    if r is not None :
                            dbound = r[max(sidecoef,0)]
                            if ((sidecoef * dbound >= sidecoef*dfront) and (sidecoef*df > sidecoef*dbound  )) :
                                front.discard((df, idf))
                                d[idf] = dbound
                                front.add((dbound, idf)) 
                                
                                
        def enforcetrianglelip_2(idfront, xfront, dfront):    
            triangles_id = self.mesh.getVertex2Triangles(idfront)
            for tid in triangles_id :
                t= self.mesh.triangles[tid]
                id0 = idfront
                lid0 =  np.where(t==id0)[0][0]
                id1 = t[(lid0+1)%3]
                id2 = t[(lid0+2)%3]
                did1 = d[id1]
                did2 = d[id2]
                # this is the intent, but tooslow !
                #Ge = np.array([ self.mesh.xy[id1]- xfront ,self.mesh.xy[id2] - xfront] )
                #M = np.linalg.inv(Ge.T.dot(Ge))
                #lmax = np.max(scipy.linalg.eigvals(M))
                
                x1 = (self.mesh.xy[id1]- xfront).squeeze()
                x2 = (self.mesh.xy[id2] - xfront).squeeze()
                a = x1[0]*x1[0]+x1[1]*x1[1]
                c = x1[0]*x2[0]+x1[1]*x2[1]
                b = x2[0]*x2[0]+x2[1]*x2[1]
                lmax = 2./(a+b -np.sqrt((a+b)*(a+b)-4*(a*b-c*c)))
                deltadmax = 1./np.sqrt(2*lmax)/lc
#                Ge = np.array([ self.mesh.xy[id1]- xfront ,self.mesh.xy[id2] - xfront] )
#                M = np.linalg.inv(Ge.T.dot(Ge))
#                deltadmax1 = 1./2./np.sqrt(M[0,0]+M[1,1]+ 2.*M[0,1])/lc
#                print(deltadmax, deltadmax1)
                
                dlim = dfront+sidecoef*deltadmax
                if sidecoef* dlim < sidecoef*did1 :
                    front.discard((did1, id1))
                    d[id1] = dlim
                    front.add((dlim, id1))
                if sidecoef* dlim < sidecoef*did2 :
                    front.discard((did2, id2))
                    d[id2] = dlim
                    front.add((dlim, id2))
                    
        def enforcetrianglelip_3(idfront, xfront, dfront):    
            triangles_id = self.mesh.getVertex2Triangles(idfront)
            for tid in triangles_id :
                t= self.mesh.triangles[tid]
                id0 = idfront
                lid0 =  np.where(t==id0)[0][0]
                id1 = t[(lid0+1)%3]
                id2 = t[(lid0+2)%3]
                d1 = d[id1]
                d2 = d[id2]
               
                # this is the intent, but tooslow !
                #Ge = np.array([ self.mesh.xy[id1]- xfront ,self.mesh.xy[id2] - xfront] )
                #M = np.linalg.inv(Ge.T.dot(Ge))
                #lmax = np.max(scipy.linalg.eigvals(M))
                
                x1 = (self.mesh.xy[id1]- xfront).squeeze()
                x2 = (self.mesh.xy[id2] - xfront).squeeze()
                Ge = np.array([ x1, x2] )
                M = np.linalg.inv(Ge.T.dot(Ge))
                
#                if sidecoef*dfront >= sidecoef*d1 :
#                    r = triald2(-x1, -x2, dfront, d1, lc)
#                    if r is None :
#                        print('ARG')
#                        #nfailed[0] +=1
#                    if r is not None :
#                            dbound = r[max(sidecoef,0)]
#                            if ((sidecoef * dbound >= sidecoef*dfront) and (sidecoef*d2 > sidecoef*dbound  )) :
#                                front.discard((d2, id2))
#                                d[id2] = dbound
#                                front.add((dbound, id2)) 
#                    
#                elif sidecoef*dfront >= sidecoef*d2 :
#                    r = triald2(-x2, -x1, dfront, d2, lc)
#                    if r is None :
#                        print('ARG')
#                        #nfailed[0] +=1
#                    if r is not None :
#                            dbound = r[max(sidecoef,0)]
#                            if ((sidecoef * dbound >= sidecoef*dfront) and (sidecoef*d1 > sidecoef*dbound  )) :
#                                front.discard((d1, id1))
#                                d[id1] = dbound
#                                front.add((dbound, id1))     
#                else :
                if (True):
                    deltad1max = 1./np.sqrt(M[0,0])/lc
                    deltad2max = 1./np.sqrt(M[1,1])/lc
                    deltad1_0 = sidecoef*(d1 - dfront)
                    deltad2_0 = sidecoef*(d2 - dfront)
                    deltad1 = min(deltad1_0, deltad1max)
                    deltad2 = min(deltad2_0, deltad2max)
                    if (deltad1*M[0,0]*deltad1 + 2.*deltad1*M[0,1]*deltad2 + deltad2*M[1,1]*deltad2)- 1./lc/lc > 1.e-8:
                        deltad1, deltad2 = projectionOnEllipse(M, 1./lc/lc, np.array([deltad1, deltad2]))  
                    
                    if deltad1_0 != deltad1 :
                         front.discard((d1, id1))
                         d1 = dfront +sidecoef*deltad1
                         d[id1] = d1
                         front.add((d1, id1))
                    
                    if deltad2_0 != deltad2 :
                         front.discard((d2, id2))
                         d2 = dfront +sidecoef*deltad2
                         d[id2] = d2
                         front.add((d2, id2))
                
                
                   
        if lipmeasure == 'edge' :
            enforcelip = enforceedgelip
        elif lipmeasure =='triangle':
            def enforcelip (idfront, xfront, dfront) :
                enforceedgelip(idfront, xfront, dfront)
                enforcetrianglelip(idfront, xfront, dfront, nfailed)
        
        elif lipmeasure =='triangle2':
                enforcelip = enforcetrianglelip_2
        
        elif lipmeasure =='triangle3':
                enforcelip = enforcetrianglelip_3
               
                
        else : raise
        
        while len(front) > 0:
            dfront, idfront = front.pop()
            visited[idfront] = True
            xfront = self.mesh.xy[idfront]
            enforcelip(idfront, xfront, dfront)
            
        #print('rate ', nfailed)
        return {'d':d, 'visitedvertices': visited}
    
    def lipBoundFM(self, dmin, dprec, dbar, lc, options={'lipmeasure':'edge'}, logger = logger):
        """construct bounds dup, dlow for d, such as 1>= dup >= d >= dlow => dmin and  and dup, dlo verify lip  """
        lipmeasure = options['lipmeasure']
        if  lipmeasure =='edge' :
            lipineq = self.getGlobalLipEdgeIneq(lc)
            cell2v  = self.mesh.getEdge2Vertices()
        elif (lipmeasure == 'triangle' or  lipmeasure == 'triangle2' or lipmeasure == 'triangle3') :
            lipineq = self.getGlobalLipTriIneq(lc)  
            cell2v  = self.mesh.triangles     
        else :
            logger.error('lipmeasure '+lipmeasure +' unknown in lipBoundFM' )
            raise
        s = np.array(slack( lipineq, dbar, check = False))  
        if(np.all(s>=0.)) : return {'d':dbar, 'dbar':dbar, 'dtop':dbar, 'dbot':dbar, 'local':True, 'nvisitedvertices': 0}
        seeds = set()
        for (ie, sie) in enumerate(s) :
            t= cell2v[ie]
            if ((sie <0.))  : seeds.update(t)
            #if ((sie < 0.) and (  len(np.where(dbar[t]>(dmin[t]+1.e-12))[0]  ))) : seeds.update(t)
            
            
        resFMup = self.lipProjFM(dbar, lc, seeds=seeds,  lipmeasure= lipmeasure,  side ='up')
        resFMlo = self.lipProjFM(dbar, lc, seeds=seeds,  lipmeasure= lipmeasure,  side ='lo')
        dtop = resFMup['d']
        dbot = resFMlo['d']
        res = {'dbar':dbar,'dtop':dtop, 'dbot':dbot, 'local':False,
               'visitedvertices' : resFMup['visitedvertices'] +resFMlo['visitedvertices'],
                'visited_top':resFMup['visitedvertices'], 'visited_bottom':resFMlo['visitedvertices'], 'seeds':list(seeds)}
        return res
    
    def solveDLipBoundPatch(self, dmin, dprec, strain, lc, localDamageSolver, potentialFixedStrain, getIntegratedPotentialFunctionFixedStrain, areas, log,
                            solverdoptions = {
                             'kernel': minimizeCPcvxopt,
                             'lip_reltole':1.e-6, 
                             #'mindeltad':1.e-3, 
                             #'snapthreshold':0.999,
                             'fixpatchbound':False, 
                             'lipmeasure':'triangle', 
                             'FMSolver':'edge', 
                             'parallelpatch':True, 
                             'kktsolveroptions': {'mode':'direct', 'linsolve':'umfpack'}
                             }):
        """minimize F(u,d) under lip constrain and d>=dmin, is min phi at fixed eps"""
       
        lipmeasure = solverdoptions['lipmeasure']
        if  lipmeasure == 'edge' : 
            glipineq = self.getGlobalLipEdgeIneq(lc) 
            cell2v = self.mesh.getEdge2Vertices()
        elif  lipmeasure == 'triangle' : 
            glipineq = self.getGlobalLipTriIneq(lc)
            cell2v = self.mesh.triangles
        else  : 
            log.error('lipmeasure :%s is unknown in solveDLipBoundPatch'%lipmeasure)
            raise
            
        dbar = localDamageSolver(strain, dmin)
        gs = np.array(slack( glipineq, dbar, check = False))
        gsmin = np.min(gs)
        it_infos = [{'global_lipslack':gs, 'patches':[[]] }]
        d = dbar.copy()
        
        maxlipslack = solverdoptions['lip_reltole'] #/self.lc
        log.info('     D solver, iter 0, Global slack: %.2e  '%gsmin)
        if(gsmin>= -maxlipslack) : return {'d':d, 'local':True, 'Converged':True, 'it_infos': it_infos, 'dup':d, 'dlo':d}
        
        iterpatch =0
        deltadmin = maxlipslack/100.
       
        # use fastmarching to determine the initial patch
        resFM = self.lipBoundFM( dmin, dprec, dbar, lc, options ={'lipmeasure':solverdoptions['FMSolver']} )
        dup = np.array(resFM['dtop'].squeeze())
        dlo = np.array(resFM['dbot'].squeeze())
        d = (dup +dlo)*0.5
        gsfm = np.array(slack( glipineq, d, check = False))
        gsminfm = np.min(gsfm)
        log.info('     D solver, iter 0, Global slack post FM : %.2e  '%gsminfm)
        
        set_verts = set( np.where((dup-dlo)> 0. )[0])
        
        lipmesh = self.mesh
        parallel = solverdoptions['parallelpatch']
        conv = True
        
        def patchsolver(verts, iproc, return_dict):
            if lipmeasure == 'edge' : 
                edges = set()
                for iv in verts :  edges.update( lipmesh.getVertex2Edges()[iv])
                edges = list(edges)
                edges, vg2l, vl2g = numberingEdgePatch(edges, lipmesh.getEdge2Vertices())
                ineqLip   =  self.getPatchLipEdgeIneq(edges, vg2l, lc)
                nvpatch = len(vl2g)
            elif  lipmeasure == 'triangle' :
                triangles = set()
                for iv in verts :  triangles.update( list(lipmesh.getVertex2Triangles(iv)))
                triangles = list(triangles)
                triangles, vg2l, vl2g = numberingTriPatch(triangles, lipmesh.triangles)
                ineqLip   =  self.getPatchLipTriIneq(triangles, vg2l, lc)           
                nvpatch = len(vl2g)
            else :
                log.error('lipmeasure '+ lipmeasure+' Not Defined in solveDLipBoundPatch')
                raise
            nlipconstrain = len(ineqLip['dims']['q'])
            fixbound = solverdoptions['fixpatchbound']
            if fixbound & (lipmeasure == 'triangle') :
                log.warning('fixpatchbound:True and Patchsolver:triangle is instable as of now ...')
             
            if fixbound :
                vgbound = list(set(vl2g)-set(verts))
                vgfree= verts
                vlfree  =  [ vg2l[ig] for ig in vgfree ] 
                vlbound =  [ vg2l[ig] for ig in vgbound ] 
                nfree      = len(vlfree)
                lipG = ineqLip['G']
                lipGfree = lipG[:, vlfree]
                liph  = ineqLip['h']
                dtopbound = cvxopt.matrix(dup[vgbound], (len(vgbound), 1) )
                #dbotbound = cvxopt.matrix(dlo[vgbound], (len(vgbound), 1) )
                liphbound = lipG[:,vlbound]*dtopbound
                liphfree = liph - liphbound
                ineqLip = {'G':lipGfree, 'h':liphfree, 'dims':ineqLip['dims']}            
                patchAreas = areas[vgfree]
                phiOfD = self.law.potentialFixedStrain(strain[vgfree])
            else :
                vgfree = vl2g
                nfree = nvpatch
                patchAreas = areas[vl2g]
                phiOfD = potentialFixedStrain(strain[vl2g])
            
            dmincvx  = cvxopt.matrix(dmin[vgfree], size=(nfree,1))
            dpreccvx = cvxopt.matrix(dprec[vgfree], size=(nfree,1))
            dbotcvx  = cvxopt.matrix(dlo[vgfree], size=(nfree,1))
            dtopcvx  = cvxopt.matrix(dup[vgfree], size=(nfree,1))           
            donecvx  = cvxopt.matrix(1., size=(nfree,1))
            dcvx  = cvxopt.matrix(d[vgfree], size=(nfree,1))  
                  
            Id = cvxopt.spdiag([1.]*nfree)
            donecvx    = cvxopt.matrix(1., size=(nfree,1))
            if not fixbound :
                ineqGt = {'G':-Id,  'h': -dmincvx, 'dims': {'l': nfree, 'q': [], 's':  []} }
                ineqLt  = {'G':Id,  'h': donecvx,  'dims': {'l': nfree, 'q': [], 's':  []} }
            else:   
                ineqGt = {'G':-Id,  'h': -dbotcvx, 'dims': {'l': nfree, 'q': [], 's':  []} }
                ineqLt  = {'G':Id,  'h': dtopcvx,  'dims': {'l': nfree, 'q': [], 's':  []} }
            lipconstrain = combineConeIneq([ineqLip, ineqGt, ineqLt])
                
            smin0 = np.array(slack(ineqLip,  dcvx, check = False)).min()           
            #F = Mechanics2D.getIntegratedPotentialFunctionFixedStrain(areas, phiOfD, 0.5*(dbotcvx + dtopcvx))
            F = getIntegratedPotentialFunctionFixedStrain(patchAreas, phiOfD, dcvx)
            #dpatch_res =  minimizeCPcvxopt(F, lipconstrain, kktsolveroptions = solverdoptions['kktsolveroptions'])   
            #dpatch_res =  linImprovCPcvxopt(F, lipconstrain, x0 = dpreccvx, kktsolveroptions = solverdoptions['kktsolveroptions'])   
            dpatch_res = solverdoptions['kernel'](F, lipconstrain, x0 = dpreccvx, kktsolveroptions = solverdoptions['kktsolveroptions'])   
            
            dpatchcvx = cvxopt.matrix(np.atleast_1d(dpatch_res['d']))
            smin1 = np.array(slack(ineqLip,  dpatchcvx, check = False)).min()
            log.info('     Patch : unknown :%d, lipconstrain : %d, slack:  %.2e -> %.2e '%(nfree,nlipconstrain, smin0, smin1))
            dpatch = np.array(dpatchcvx).squeeze()
            conv = dpatch_res['Converged']
            #snap d to dmin or 1. where needed
            dpatch = np.where(dpatch < dmin[vgfree]+deltadmin, dmin[vgfree], dpatch)
            dpatch = np.where(dpatch >= 1.-deltadmin, 1., dpatch)
            return_dict[iproc] = {'dpatch':dpatch, 'vgfree':vgfree, 'Converged':conv}
            return
        
        while gsmin< -maxlipslack :
            iterpatch += 1
            for (ie, sie) in enumerate(gs) : 
                if ((sie < 0.)) : set_verts.update(cell2v[ie])
            verts = np.array(list(set_verts))
            verts_patches = partitionGraph(verts, lipmesh.getVertex2Vertices())
            
            if not parallel :
                return_dict = dict()
                for ipatch,  verts in enumerate(verts_patches) :  patchsolver(verts, ipatch, return_dict)
            else :
                manager = multiprocessing.Manager()
                return_dict = manager.dict()       
                patch_procs = [multiprocessing.Process(target = patchsolver, args = (verts, iproc, return_dict)) for iproc, verts in enumerate(verts_patches) ]
                for pp in patch_procs : pp.start()
                for pp in patch_procs : pp.join()
            
            conv = True
            for ires, res in return_dict.items() :
                convi = res['Converged']
                if not convi : log.warning('     Patchsolve for patch %d did not converge !'%ires)
                conv = conv&convi
                d[res['vgfree']] = res['dpatch']
            
                #check lip  constraint
            gs = np.array(slack( glipineq, d, check = False)) 
            it_infos.append({'global_lipslack':gs, 'patches':verts_patches})
            gsmin1 = np.min(gs)
            log.info('     D solver, iter %d, Global slack:  %.2e -> %.2e '%(iterpatch, gsmin, gsmin1))
            
            gsmin = gsmin1
            
        
        return {'d':d, 'local':False, 'Converged':conv, 'it_infos': it_infos, 'dup':dup, 'dlo':dlo }
    
  

#    def lipProjFMTriangle2(self, dtarget, lc, seeds =None,  side ='up', logger = logger, verbose=False) :
#        """ compute the upper or lower bound ('side':['up','lo']) of the lip projection of dtarget using a Fast Marching approach,
#            either imposing that 'edge' lip contrain are verified (lipmeasure='edge')
#            or imposing that edge and triangle constran lip contrain are verified (lipmeasure='traingle')
#            lc is the characteristic lenght.
#            seeds is a list of node to start from. if not given, all the nodes of the mesh are put in the front.
#            return a dictionnary containing 
#            - 'd' : the projected field, as an array containing value at each node
#            - 'visitedvertices' : an boolean array contaning True for all nodes visited by the FM
#        """
#      
#        
#        def enforceTrianglelipUp(idfront, d, front, visited):    
#            triangles = self.mesh.getVertex2Triangles(idfront)
#            triangles, vg2l, vl2g = numberingTriPatch(triangles, self.mesh.triangles)
#            nv = len(vg2l)
#            idfrontlocal = vg2l[idfront]
#            triineq = self.getPatchLipTriIneq(triangles, vg2l, lc)
#            lvisited = [idfrontlocal]+ [ vg2l[ig] for ig in vl2g if visited[ig] ]
#            gvisited = [idfront]+ [ ig for ig in vl2g if visited[ig] ]
#            nvfixed = len(lvisited)
#            print(nv, vl2g, lvisited, gvisited, idfront, idfrontlocal)
#            A = cvxopt.spmatrix([1.]*nvfixed, [ i for i in range(nvfixed)], lvisited,(nvfixed,nv))
#            b = cvxopt.matrix( d[gvisited], (nvfixed,1) )
#            
#            P = cvxopt.spdiag([1.]*nv)
#            q = cvxopt.matrix(-dtarget[vl2g], (nv,1))
#            
#            growingineq = {}
#            growingineq['dims']={'l':nv-1, 'q':[], 's':[]}
#            
#            growingineq['G']  = cvxopt.spmatrix( [1.]*(nv-1), [i for i in range(nv-1) ],  [i  for i in range(nv) if i != idfrontlocal ], (nv-1,nv)  )
#            growingineq['h']  = cvxopt.matrix( [d[i]-1.e-4 for i in vl2g if i !=idfront ],  (nv-1, 1))
#            
#            ineq = combineConeIneq([triineq, growingineq])
#            ineq = triineq
#            res = cvxopt.solvers.coneqp(P = P, q = q,A=A, b=b, G = ineq['G'], h = ineq['h'], dims = ineq['dims'])
#            conv = res['status'] == 'optimal'
#            if (not conv) : 
#                 logger.error('ERROR : cvxopt.solvers.coneqp did not converge ! status : '+ res['status'])
#                 
#            dloc = np.array(res['x']).squeeze()
#            for il, ig in enumerate(vl2g):
#                if dloc[il] > d[ig]+1.e-4 :
#                    front.discard((d[ig], ig))
#                    d[ig] = dloc[il]
#                    front.add((dloc[il], ig))
#            
#        d = dtarget.copy()
#        visited = np.zeros(d.shape, dtype='bool')
#        front = SortedList()
#        if seeds is None :
#            for (iv, div) in enumerate(d) :   front.add((div,iv))
#        else :
#            for  iv in seeds : front.add((d[iv], iv))          
#    
#        while len(front) > 0:
#            dfront, idfront = front.pop()
#            visited[idfront] = True
#            enforceTrianglelipUp(idfront, d,  front, visited)
#
#        return {'d':d, 'visitedvertices': visited}      
#
