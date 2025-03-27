#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface to some linear solver

// Copyright (C) 2021 Chevaugeon Nicolas
Created on Fri Sep 24 11:11:46 2021
@author: nchevaug
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

import scipy
import numpy as np
import cvxopt
from cvxopt import umfpack
from cvxopt import cholmod
from liplog import logger


def convert2cvxoptSparse(A):
    """ convert a scipy sparse matrix to cvxopt.spmatrix """
    if scipy.sparse.issparse(A) :
        Acoo = A.tocoo()
        return cvxopt.spmatrix(Acoo.data, Acoo.row, Acoo.col, size=Acoo.shape)
    elif type(A) == cvxopt.base.spmatrix:
        return A
    else: raise

def convert2scipy(A):
    """ convert a cvxopt.spmatrix  to  a scipy sparse matrix (scipy.sparse.coo_matrix) """
    if type(A) == cvxopt.base.spmatrix :
        X = list(A.V)
        I = list(A.I)
        J = list(A.J)
        Ascipy = scipy.sparse.coo_matrix( ( X, (I, J)))
        return Ascipy
    else : raise

def size(A):
    """ return the size of a matrix, weither it is a numpy, scipy or cvxopt matrix """
    if type(A) == cvxopt.base.spmatrix or type(A) == cvxopt.base.matrix:
        return A.size
    else :return A.shape

def solve(K,F, x0 = None, solver='cholmod',  solveroptions=dict(), logger = logger):
    """ 
    Solve a linear system K x = F
    K the matrix, F the right handside, x0 an initial guess (usefull for iterative methods)
    return a dictionary with entry 'x' that contain the solution and entry 'converged' with qassociated value True
    if solver succeed or False if it failed.
    """
    sizeA = size(K)
    sizeF = size(F)
    
    n = sizeA[0]
    if sizeA[1] != n or sizeF[0] !=n :
        logger.error('systems size invalid in linsolverinterface.solve()')
        raise
    if solver == 'direct':
       x  = scipy.sparse.linalg.spsolve(K,F)
       res = {'x':x, 'converged':True}
       return res
    elif solver == 'cg':
        if x0 is  None :   x0 = np.zeros((n))
        x, info = scipy.sparse.linalg.cg(K, F, x0 =x0, M = None)
        if info != 0 :
            logger.error('cg failed, info :'+str(info))
            raise
        res ={'x':x, 'converged':True}
        return res  
    elif solver == 'cholmod':
        cholmodoptions = cvxopt.cholmod.options.copy()
        if 'supernodal' in solveroptions :
            cvxopt.cholmod.options['supernodal'] = 0 # This set cholmod to L*D*L^T mode
        LTL = cvxopt.cholmod.symbolic(K)
        cvxopt.cholmod.numeric(K,LTL)
        F = cvxopt.matrix(F)
        cvxopt.cholmod.solve(LTL, F, sys = 0)
        cvxopt.cholmod.options = cholmodoptions
        x = +F
        res ={'x':np.array(x).squeeze(), 'converged':True}
        return res
    elif solver == 'umfpack':
        FSK= cvxopt.umfpack.symbolic(K)
        FK = cvxopt.umfpack.numeric(K,FSK) 
        F = cvxopt.matrix(F)
        cvxopt.umfpack.solve(K,FK, F)
        x = +F
        res ={'x':np.array(x).squeeze(), 'converged':True}
        return res
    else :
        logger.error('Linear Solver '+solver+' not known !')
        raise
        
    
