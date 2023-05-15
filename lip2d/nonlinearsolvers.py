#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

// Copyright (C) 2021 Chevaugeon Nicolas
Created on Fri Nov 12 11:02:35 2021

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

from liplog import logger
import linsolverinterface as linsol
import numpy as np

class newton:
    def __init__(self, solveroptions = {'linsolve':'cholmod', 'itmax':20, 'maxnormabs':1.e-8, 'maxnormrel':1.e-6}, logger =logger):
            self.solveroptions = solveroptions
            self.logger = logger
    
    def solve(self, x0, R, dR):
        epsabs = self.solveroptions['maxnormabs']
        epsrel = self.solveroptions['maxnormrel']
        itmax  = self.solveroptions['itmax']
        it = 0
        x = +x0
        r = R(x)
        nr = np.linalg.norm(r)
        nr0 = nr
        logger.info('     NR Solver iter %d Residual Norm abs : %.3e'%(it, nr))
        while (nr > epsabs ) and (nr > epsrel*nr0) and (it < itmax):
            K = dR(x)
            res = linsol.solve(K=K, F=-r, x0 = x, 
                                           solver=self.solveroptions['linsolve'],  
                                           solveroptions=dict(), 
                                           logger = self.logger)
            if not res['converged'] : 
                logger.error('Linear solver '+ self.solveroptions['linsolve']+'did not converge inside newton solver while computing iteration it %d'%it)
                raise  
            x = x + res['x']
            r = R(x)
            nr = np.linalg.norm(r)
            it +=1
            logger.info('     NR Solver iter %d Residual Norm abs : %.3e rel : %.3e '%(it, nr, nr/nr0))
        
        converged = ((nr <= epsabs ) or (nr <= epsrel*nr0))
        if not converged :
            logger.warning('     NR Solver failed to converge after %d iterations. Residual Norm abs :%.3e rel : %.3e '%(it, nr, nr/nr0))    
        else : 
            logger.info('     NR Solver converged after %d iterations.'%it)
        res = {'x':x, 'R': r, 'converged': converged, 'it': it} 
        return res
    
#class kktnewton:
#    def __init__(self, f, g):
#        self.f = f
#        self.g = g
#    def step(u, l):
#        l  = max(0,l)
#        gu, dgu = g(u)
#        active = gu < 0. and l>0.
#        r = f(u) - dgu.T[:,active]*l[active]
#        if (np.linalg.norm(r) < eps and np.all(gu >=0.) and gu*l <= eps) return u, l
#        
#        KKT = cvxopt.sparse()
        
