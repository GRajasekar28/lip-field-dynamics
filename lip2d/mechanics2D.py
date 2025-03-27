#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

// Copyright (C) 2021 Chevaugeon Nicolas
Created on Tue May  4 10:46:36 2021
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

import scipy
import numpy as np
import cvxopt
import matplotlib.pylab as plt

import multiprocessing
import linsolverinterface as lin
import liplog as liplog
import lipdamage as lip
import mesh
import nonlinearsolvers  as nls


import phasedamage as phased

logger = liplog.logger

class dirichletConditions:
    def __init__(self, mesh):
        self.dcs = []
        self.mesh = mesh
        
    def add(self, listoflineid, dic):
        vids = sum( [self.mesh.getVerticesOnClassifiedEdges(idl)  for idl in listoflineid ],[]) 
        imposed_displacement= dict()
        imposedx = dic.get('x')
        if imposedx is not None:  imposed_displacement.update({2*int(i):imposedx for i in vids})
        imposedy = dic.get('y')
        if imposedy is not None:  imposed_displacement.update({2*int(i)+1:imposedy for i in vids})
        dcid = len(self.dcs)
        self.dcs.append({'vids':vids, 'dispdic':imposed_displacement})
        return dcid
    
    def addPoints(self,  listofpointid, dic):
        vids = sum( [list(self.mesh.getClassifiedPoint(idv))  for idv in listofpointid ],[]) 
        imposed_displacement= dict()
        imposedx = dic.get('x')
        if imposedx is not None:  imposed_displacement.update({2*int(i):imposedx for i in vids})
        imposedy = dic.get('y')
        if imposedy is not None:  imposed_displacement.update({2*int(i)+1:imposedy for i in vids})
        dcid = len(self.dcs)
        self.dcs.append({'vids':vids, 'dispdic':imposed_displacement})
        return dcid
    
    def update(self, dcid, dic ):
        dc = self.dcs[dcid]
        vids = dc['vids']
        imposed_displacement = dc['dispdic']
        imposedx = dic.get('x')
        if imposedx is not None:  imposed_displacement.update({2*int(i):imposedx for i in vids})
        imposedy = dic.get('y')
        if imposedy is not None:  imposed_displacement.update({2*int(i)+1:imposedy for i in vids})
        self.dcs[dcid] = {'vids':vids, 'dispdic':imposed_displacement}
    
    def impose(self):
        imposed_dispdof=  sum([list(dc['dispdic'].keys()) for dc in self.dcs],[])
        imposed_dispval=  sum([list(dc['dispdic'].values()) for dc in self.dcs],[])
        return imposed_dispdof, imposed_dispval
    
    def getReactions(self,dcid, R):
        imposed_dispdof= list(self.dcs[dcid]['dispdic'].keys())
        return R[imposed_dispdof]
        
# Dirichlet Boundary conditions. degree of freedom are indexed by vertice Id  : ux(id) -> u(id*2), uy(id)   = u(id*2+1)



class NeumannConditions:
    def __init__(self, mesh):
        self.ncs = []
        self.mesh = mesh
        
    def add(self, listoflineid, nc):
        vids = sum( [self.mesh.getVerticesOnClassifiedEdges(idl)  for idl in listoflineid ],[]) 
        imposed_force= dict()
        imposed_fx = nc.get('x')
        if imposed_fx is not None:  imposed_force.update({2*int(i):imposed_fx for i in vids})
        imposed_fy = nc.get('y')
        if imposed_fy is not None:  imposed_force.update({2*int(i)+1:imposed_fy for i in vids})
        ncid = len(self.ncs)
        self.ncs.append({'vids':vids, 'force':imposed_force})
        return ncid
    
    def addPoints(self,  listofpointid, nc):
        vids = sum( [list(self.mesh.getClassifiedPoint(idv))  for idv in listofpointid ],[]) 
        imposed_force= dict()
        imposed_fx = nc.get('x')
        if imposed_fx is not None:  imposed_force.update({2*int(i):imposed_fx for i in vids})
        imposed_fy = nc.get('y')
        if imposed_fy is not None:  imposed_force.update({2*int(i)+1:imposed_fy for i in vids})
        ncid = len(self.ncs)
        self.ncs.append({'vids':vids, 'force':imposed_force})
        return ncid
    
    def update(self, ncid, nc ):
        ncs = self.ncs[ncid]
        vids = ncs['vids']
        imposed_force = ncs['force']
        imposed_fx = nc.get('x')
        if imposed_fx is not None:  imposed_force.update({2*int(i):imposed_fx for i in vids})
        imposed_fy = nc.get('y')
        if imposed_fy is not None:  imposed_force.update({2*int(i)+1:imposed_fy for i in vids})
        self.ncs[ncid] = {'vids':vids, 'force':imposed_force}
    
    def impose(self):
        imposed_force_dof=  sum([list(nc['force'].keys()) for nc in self.ncs],[])
        imposed_force_val=  sum([list(nc['force'].values()) for nc in self.ncs],[])
        return imposed_force_dof, imposed_force_val
    
    def getDisplacement(self,ncid, u):
        imposed_force_dof= list(self.ncs[ncid]['force'].keys())
        return u[imposed_force_dof]

    
class rigidBodyConditions:
    def __init__(self, mesh):
        self.rbs= []
        self.mesh = mesh
    
    def addRigidBody(self, listoflineid):
        vids = sum( [self.mesh.getVerticesOnClassifiedEdges(idl)  for idl in listoflineid ],[])
        C    = np.sum(self.mesh.xy[vids], axis =0)/len(vids)
        rb = {'vids': vids, 'ref':C}
        rbid = len(self.rbs)
        self.rbs.append(rb)
        return rbid
    
    def updateRigidBody(self, rbid, dic):
        self.rbs[rbid].update(dic)
    
    def getRigidBody(self, rbid):
        return self.rbs[rbid]
    
    def addLinearizedConstrains(self,K,F):
        for rbc in self.rbs : 
            start_index =  K.size[0]
            # index of the rb dofs
            iurbc = start_index 
            ivrbc = start_index +1
            isrbc = start_index +2
            
            rbvert = rbc['vids']
            nrbvert = len(rbvert)
            xref      = rbc['ref'][0]
            yref      = rbc['ref'][1]
            aI = np.array(sum( [ [2*iv]*3 + [2*iv+1]*3 for iv in range(nrbvert) ], []), dtype = 'int')
            aJ = np.array(sum( [ [2*ivid, iurbc, isrbc, 2*ivid+1, ivrbc, isrbc] for ivid in rbvert ], []), dtype = 'int')
            aX = np.array(sum( [ [1., -1., self.mesh.xy[ivid,1] - yref, 1., -1., -self.mesh.xy[ivid,0] +xref] for ivid in rbvert ], []))
            A = cvxopt.spmatrix(aX,aI,aJ, (2*nrbvert, start_index +3 ) ) 
            Kru = cvxopt.spmatrix([],[],[], (3, start_index ) )
            Krr = cvxopt.spmatrix([],[],[], (3, 3 ) )
            Kll = cvxopt.spmatrix([],[],[], (2*nrbvert, 2*nrbvert ) )
            KK = cvxopt.sparse( [[K, Kru],[Kru.T,  Krr]])
            K = cvxopt.sparse( [[KK, A],[A.T,  Kll]])
            Fr = cvxopt.matrix([0]*(3+2*nrbvert))
            F = cvxopt.matrix( [F, Fr ] )
            
            imposedx = rbc.get('x')
            if imposedx is not None :
                Ax = cvxopt.spmatrix([1.],[0],[iurbc],(1, K.size[0] ))
                Fx = cvxopt.matrix([imposedx])
                Axx = cvxopt.spmatrix([],[],[],(1,1))
                K = cvxopt.sparse( [[K, Ax],[Ax.T, Axx ]])
                F  = cvxopt.matrix([F,Fx])
            imposedy = rbc.get('y')
            if imposedy is not None :
                Ax = cvxopt.spmatrix([1.],[0],[ivrbc],(1, K.size[0] ))
                Fx = cvxopt.matrix([imposedy])
                Axx = cvxopt.spmatrix([],[],[],(1,1))
                K = cvxopt.sparse( [[K, Ax],[Ax.T, Axx ]])
                F  = cvxopt.matrix([F,Fx]) 
            imposedsin = rbc.get('teta') 
            if imposedsin is not None :
                Ax = cvxopt.spmatrix([1.],[0],[isrbc],(1, K.size[0] ))
                Fx = cvxopt.matrix([imposedsin])
                Axx = cvxopt.spmatrix([],[],[],(1,1))
                K = cvxopt.sparse( [[K, Ax],[Ax.T, Axx ]])
                F  = cvxopt.matrix([F,Fx]) 
        return K, F

    def extractDisplacementsAndReactions(self, x):
        ''' extract from the sol x of the linear sys the reactions and the displacement associated to the rgid body displacement '''
        for i, rbc in reversed(list(enumerate(self.rbs))): 
            M = 0.
            Fx = 0.
            Fy = 0.
            if rbc.get('teta') is not None :
                M = x[-1]
                x = x[:-1]
            if rbc.get('y') is not None :
                Fy = x[-1]
                x = x[:-1]                          
            if rbc.get('x') is not None :
                Fx = x[-1]
                x = x[:-1]   

            x = x[:-2*len(rbc['vids'])]
            rbu = x[-3:]
            x = x[:-3]
            self.rbs[i]['Displacements'] = rbu
            self.rbs[i]['Reactions'] =    -np.array([Fx, Fy, M])
        return x
                     
class boundaryConditions:
    def __init__(self):
        self.dirichlets  = None
        self.rigidbodies = None
        self.neumanns    = None
        self.linearconstraints = None


class Mechanics2D:
    def __init__(self, mesh, law, lipprojector=None, lc=0., log = logger):
        self.mesh = mesh
        self.law = law
        self.lipprojector = lipprojector
        self.lc = lc
        self.log = log
        self._strainop = None
        self._D_index = None
        self._M = None
        self._lipconstrains = None
        
    def zeros(self):
        '''
        Returns u, d
        -------
        return a zeros np.array of shape (nvertices,2) for u and p.array of shape (ntriangles) for d.
        '''
        return np.zeros((self.mesh.nvertices,2)),  np.zeros(self.mesh.ntriangles)
        
    def strainOp(self) :
        """
          return the strain operator D: a cvxopt sparse matrix, such as D*u give the strain in each element :
          eps = D*u
          with eps = eps[3*ie:3*ie+3] : eps contain the strain : eps[0] = dudx,  eps[1] = dvdy, eps[2] = 0.5*(dudy + dvdx)
        """
        if self._strainop is None :    
           self._strainop = self._strainOp()
        return self._strainop
        
    def _strainOp(self) :
        G = self.mesh.elementaryGradOperators()
        t = self.mesh.triangles
        nt = t.shape[0]
        nv =self.mesh.xy.shape[0]
        tmp0 =  np.arange(0,3*nt,3).reshape((nt,1))
        tmp1 = np.arange(1,3*nt,3).reshape((nt,1))
        tmp2 = np.arange(2,3*nt,3).reshape((nt,1))
        BI = np.hstack( (tmp0,tmp0, tmp0, tmp1,tmp1, tmp1, tmp2, tmp2, tmp2, tmp2, tmp2, tmp2)).flatten()
        BJ = np.hstack((2*t, 2*t+1, 2*t, 2*t+1)).flatten()
        BX = np.hstack((G[:,0,:],G[:,1,:], G[:,1,:], G[:,0,:])).flatten() 
        return cvxopt.spmatrix(BX,BI,BJ, (3*nt,2*nv ) )  

    def areas(self): 
        """
        return the area of all triangles in the mesh

        Returns
        -------
        TYPE np.array, indexed by triangle id in the mesh
            areas[i] : area of triangle i.

        """
        return self.mesh.areas()
    
    def displacementContraintsOnNodePairs(self, verticespairing):
        """ 
            Given a pairing of nodes (a list of pair (nodeA, NodeB)  return a set of linear constraints, such as u(NodeA) = u(nodeB)"
            expressed as a set of linear equations : Au = b to be verifyed by u. 
            A and b are returned in a dictionary indexed by "A" and "b"
        Parameters
        ----------
        self : the mechanical problem. not realy used, but vertices in the verticespairing refer to vertices id of the mesh of the mechanical problem
               and dof are supposed to be numbered using vertice number vid(NodeA) -> dofs= 2vid (x) and 2vid+1 (y)
        verticespairing : (a list of pair (nodeA, NodeB) of the vertices
        Returns
        -------
            a set of linear constraints, such as u(NodeA) = u(nodeB)"
            expressed as a set of linear equations : Au = b to be verifyed by u. 
            A and b are returned in a dictionary indexed by "A" and "b"
        """
        npair = len(verticespairing)
        b = cvxopt.spmatrix([],[],[],(2*npair, 1)) 
        aI = np.array( sum([[2*k, 2*k, 2*k+1, 2*k+1] for k in range(npair)],[]), dtype = 'int') 
        aJ = np.array( sum([[2*v0, 2*v1, 2*v0+1, 2*v1+1] for (v0, v1) in verticespairing], []), dtype = 'int')
        aX = np.array( [1., -1.] * 2*npair) 
        A = cvxopt.spmatrix(aX,aI,aJ, (2*npair,2*self.mesh.nvertices ) ) 
        lincon ={'A':A, 'b':b}
        return lincon
            
    def D(self, strain = None, d = None, capd = None):
        ''' compute the local stifness De at each element and store them in a sparse matrix where 3*3 diagonal bloc is De  
            by De we mean the tangent operator in voight notation :
                    |sxx|     |Dxxxx Dxxyy Dxxxy|      |epsxx|
                    |syy|   =  |Dyyxx Dyyyy Dyyxy| .   |epsyy|
                    |sxy|     |Dxyxx Dxyyy Dxyxy|      |2epsxy|
            
            if strain  is None assume they are eps = 0, 0,0 at each element
            if d is None assume d = 0. at each element
            if capd is not None, capd must be a float between 0. and 1. it can be used to 'cap' the value of d so that max (d) <= capd
        ''' 
        nf = self.mesh.ntriangles
        def getDindex():
            if self._D_index is None:
                def gaI():
                    for ie in range(nf):
                        for i in [3*ie, 3*ie, 3*ie, 3*ie+1, 3*ie+1, 3*ie+1, 3*ie+2, 3*ie+2, 3*ie+2] : yield i
                def gaJ():
                    for ie in range(nf):
                        for j in [3*ie, 3*ie+1, 3*ie+2, 3*ie, 3*ie+1, 3*ie+2, 3*ie, 3*ie+1, 3*ie+2] : yield j
                aI = np.fromiter(gaI(), dtype ='int', count = nf*9)
                aJ = np.fromiter(gaJ(), dtype ='int', count = nf*9)
                self._D_index = [aI, aJ]
            return self._D_index[0], self._D_index[1]
        
        if strain is None : strain = np.zeros((nf, 3))
        if d is None : d = np.zeros((nf))
        if capd is not None :
            d[d> capd] = capd
        H = self.law.dTrialStressDStrain(strain, d)
        A = self.areas()
        aX = (H*A[:,None, None]).flatten()
        
        aI, aJ = getDindex()
        return cvxopt.spmatrix(aX, aI, aJ, (3*nf, 3*nf ) )
        #return cvxopt.spdiag([ cvxopt.matrix(H[ie]*A[ie]) for ie in range(nf)])
    
    def F(self, strain, d): 
        ''' return the vector of internal nodal forces'''
        Astress = self.areas()[:, np.newaxis]*self.law.trialStress(strain, d)
        R      = (self.strainOp().T)*cvxopt.matrix((Astress.flatten()))
        return R
    
    def K(self, strain =None, d=None, mindiag = None):
        """ return the tangent stifness matix as a cvxopt sparse matrix"""
        nf = self.mesh.ntriangles
        if strain is None : strain = np.zeros((nf, 3))
        if d is None :      d = np.zeros((nf))
        d = d.squeeze()
        B = self.strainOp()
        D = self.D(strain, d) 
        Kuu = B.T*(D*B)
        if mindiag is not None :
            n = Kuu.size[0]
            diag = np.array(Kuu[::n+1].V).flatten()
            toosmall =diag < mindiag
            if  np.any(toosmall) :
                smallest = np.argmin(diag)
                self.log.warning('%d small pivot corrected in K : smallest %d :  %f. Corrected to -> %f '%(np.sum(toosmall), smallest, diag[smallest], 1.))
                diag[toosmall] = 1.
                Kuu[::n+1] = diag 
        return Kuu
    
    def KDD(self, strain, d):
        d2phidd = -self.law.dY( strain, d)
        KDD = cvxopt.spdiag(list(d2phidd))
        return KDD
    
    def KDU(self, strain, d):
        nf = strain.shape[0]
        dstressdD = self.law.dtrialStressdD(strain, d)
        B = lin.convert2cvxoptSparse(self.strainOp())
        I = [ i  for ii in range(nf) for i in [ii]*3]
        J = [ j for j in range(3*nf)]
        dstressdD = cvxopt.spmatrix( list(dstressdD.flatten()), I , J)
        KDU = dstressdD*B
        return KDU

    def eigenAnalysis(self, strain, d, dofs, nvp = 15) :    
        KUU = self.K(strain, d)[dofs, dofs]
        KDD = self.KDD(strain, d)
        KDU = self.KDU(strain, d)[:, dofs] 
        KFF = cvxopt.sparse([[KUU, KDU], [KDU.T, KDD]])
        
        cholmodoptions = cvxopt.cholmod.options.copy()
        cvxopt.cholmod.options['supernodal'] = 0 # This set cholmod to L*D*L^T mode
        FKFF = cvxopt.cholmod.symbolic(KFF)
        cvxopt.cholmod.numeric(KFF,FKFF)
        n =len(d) + len(dofs)
        def KFFsolve(v):
            res = cvxopt.matrix( v, (n,1))
            cvxopt.cholmod.solve(FKFF, res, sys = 0)
            return res
        
        
        KFFsolveOp = scipy.sparse.linalg.LinearOperator((n,n), KFFsolve)
        smalleigs = scipy.sparse.linalg.eigsh(KFFsolveOp, 2*nvp, which='BE', return_eigenvectors= False)
        
        cvxopt.cholmod.options = cholmodoptions
        return 1./smalleigs
    
    def M(self):
        if self._M is None :
            n = self.mesh.nvertices
            M = scipy.sparse.dok_matrix((n,n))
            Mref = np.array([[2.,1.,1.],[1.,2.,1.],[1.,1.,2.]])/24.
            A = self.areas()
            for it, t in enumerate(self.mesh.triangles) :
                M[t[0],t] += 2.*A[it]*Mref[0,:]
                M[t[1],t] += 2.*A[it]*Mref[1,:]
                M[t[2],t] += 2.*A[it]*Mref[2,:]
            self._M = M.tocsr()
        return self._M
    
    def projectionL2Tris2Vertices(self, tri_values):
        nv = self.mesh.nvertices
        nt = self.mesh.ntriangles
        if (tri_values.shape[0] != nt):
            raise
        p = tri_values.flatten().shape[0]//nt
        F=np.zeros((nv, p))
        M = self.M()
        A = self.areas()
        for it, t in enumerate(self.mesh.triangles) :
            F[t,:] +=  A[it]*np.ones((3,1)).dot(tri_values[it])/3.
        return scipy.sparse.linalg.spsolve(M, F).squeeze()
    
    def energy(self, strain = None, d= None) :
         """ return the potential energy, integrated over the mesh """
         nf = self.mesh.ntriangles
         if strain is None : strain = np.zeros((nf, 3))
         if d is None : d = np.zeros((nf))
         phi = self.law.potential(strain, d.squeeze())
         return self.integrate(phi)
         
    def integrate(self, elementfield):
        """ Given a constant field per element, integrate over the mesh """
        A = self.areas()
        if (elementfield.shape[0] != len(A)) :
            print('Error in integrate')
            raise
        return A.dot(elementfield)
            
    def solveDisplacementFixedDNonLinear(self, u0 = None, d =None,
                                         bc = None,
                                         solveroptions = {'linsolve':'cholmod', 'mindiag': None, 'itmax':20, 'maxnormabs':1.e-8, 'maxnormrel':1.e-6},
                                         ):
                                         
        nv = self.mesh.nvertices
        nf = self.mesh.ntriangles
        if u0 is None : u0 = np.zeros((nv, 2))
        if d is None : d = np.zeros((nf, 1))
        if bc is None : self.log.warning("No boundary conditions defined")
        imposed_dispdof= []
        imposed_dispval= []
        if bc.dirichlets is not None:
            imposed_dispdof,imposed_dispval   = bc.dirichlets.impose()
        free_dispdof = list(set(range(0,2*nv)).difference(set(imposed_dispdof)))
        u = u0.copy().reshape((nv*2))
        u[imposed_dispdof] = imposed_dispval     
        if bc.neumanns is  None:  Fext = np.zeros(nv*2)
        else : 
            self.log.error('Neumanns bc are not implemented in solveDisplacementFixedDNonLinear')
            #Fext = bc.neumanns.impose()
        if bc.rigidbodies is not None :
            self.log.error('linearized rigid body constraints not implemented in solveDisplacementFixedDNonLinear')
        if bc.linearconstraints is not None :
            self.log.error('linearized constraints not implemented in solveDisplacementFixedDNonLinear')
            
        def R(x):
            u[free_dispdof] = x
            strain = self.strain(u)
            Rt = np.array(self.F(strain, d)).squeeze() - Fext
            return Rt[free_dispdof]
        mindiag = solveroptions.get('mindiag')
        def dR(x):
            u[free_dispdof] = x
            Kuu = self.K(self.strain(u), d, mindiag)
            
            Kff = Kuu[free_dispdof, free_dispdof]
            return Kff
        
        x0 = u[free_dispdof]
        nlsolver = nls.newton(solveroptions, self.log) 
        res = nlsolver.solve(x0, R, dR)
        converged = res['converged']
      
        u[free_dispdof] = res['x']
        u = u.reshape((nv,2))
        strain = self.strain(u)
        R  = np.array(self.F(strain,d)).squeeze() - Fext
        return {'u':u, 'R': R, 'Converged': converged, 'it': res['it']}  

    def solveDisplacementFixedDLinear(self, u0= None, d=None, bc =None,
                                      solveroptions = {'linsolve':'cholmod', 'mindiag':None}):
        nf = self.mesh.ntriangles
        nv = self.mesh.nvertices
        if u0 is None : u0 = np.zeros((nv, 2))
        if d is None :  d = np.zeros((nf, 1))
        if bc is None : self.log.warning("No boundary conditions defined")
        strain = self.strain(u0)
        Kuu = self.K(strain, d, mindiag = solveroptions.get('mindiag'))
        n,n =Kuu.size
        fixeddofs = []
        freedofs = list(range(n))
        imposedval = []
        if bc.dirichlets is not None :
            fixeddofs, imposedval = bc.dirichlets.impose()
        freedofs = list(set(freedofs).difference(set(fixeddofs)))
        nfree = len(freedofs)
        nfixed = len(fixeddofs)
        imposedval = cvxopt.matrix(list(imposedval), (nfixed,1))
        Kff = Kuu[freedofs, freedofs]
        Kui = Kuu[freedofs, fixeddofs]
        K = Kff
        F = -Kui*imposedval
        if bc.linearconstraints is None : c = 0
        else : c = bc.linearconstraints['A'].size[0]
        if c != 0 :
            Klf = bc.linearconstraints['A'][:, freedofs]
            Kfl = Klf.T
            Kll = cvxopt.spmatrix( [],[],[],  (c,c))
            K = cvxopt.sparse( [[Kff, Klf],[Kfl,  Kll ]])
            F = cvxopt.matrix( [F, bc.linearconstraints['b']] )
        if bc.rigidbodies is not None :
            K, F = bc.rigidbodies.addLinearizedConstrains(K,F)
        
        #### calling the linsolver
        res = lin.solve(K,F, solver = solveroptions['linsolve'], solveroptions= solveroptions, logger=self.log)
            
        if res['converged'] :
            x = res['x']
            u=cvxopt.matrix(0., (n,1))
            if bc.rigidbodies is not None :
                bc.rigidbodies.extractDisplacementsAndReactions(x)
           
            u[freedofs] =    x[:nfree]
            lagmul = +x[nfree:]
            u[fixeddofs] = imposedval[:,0]
            R = np.array(Kuu*(u)).squeeze()
            u  = np.array(u).reshape(n//2, 2)
            res ={'u':u, 'R': R, 'lagmul':lagmul, 'Converged':True}
            return res
        self.log.error('Linear Solver failed')
        raise
        
    def lipConstrains(self, dmincvx):
        if self._lipconstrains is None:
            self._lipconstrains = self.lipprojector.setUpLipConstrain(dmincvx, self.lc)
        self._lipconstrains['h'][:self.lipprojector.n]  = dmincvx
        return self._lipconstrains
    
    def solveDv0(self, dmin, dprec, strain,  **kwargs):
        """minimize norm(d-dtarget) under lip constrain, were dtarget is min phi at fixed eps"""
        dbar = self.law.solveSoftening(strain, dmin)
        d = self.lipprojector.lipProjClosestToTarget(dmin, dbar, self.lc, init = None)
        return {'d':d, 'Converged':True}
    
    def solveDLocal(self, dmin, dprec, strain, check = False, **kwargs):
        """minimize F(u,d) under  d<=1. and d>=dmin, is min phi at fixed eps"""
        dbar = self.law.solveSoftening(strain, dmin)   
        return {'d':dbar, 'local':True, 'Converged':True}   
    
    def solveDGlobal(self, dmin, dprec, strain, solverdoptions={'kernel':lip.minimizeCPcvxopt, 'lipmeasure':'edge', 'snapthreshold':0.999, 'kktsolveroptions':{'mode':'direct', 'linsolve':'umfpack'}}):
        """minimize F(u,d) under lip constrain and d>=dmin, is min phi at fixed eps"""
        n = dmin.size 
        dbar = self.law.solveSoftening(strain, dmin)
        lipmeasure = solverdoptions['lipmeasure']
        if lipmeasure  == 'triangle': lipineq = self.lipprojector.getGlobalLipTriIneq(self.lc)    
        elif lipmeasure  == 'edge'  : lipineq = self.lipprojector.getGlobalLipEdgeIneq(self.lc) 
        else :
            self.log.error('lipmeasure '+lipmeasure+' Not Defined in solveDGlobal')
            raise
        
        smin = np.array(lip.slack( lipineq,  dbar, check = False)).min()
        if(smin >=0.) : return {'d':dbar, 'local':True, 'Converged':True}
        phiOfD = self.law.potentialFixedStrain(strain)
        areas = self.areas()
        dmincvx = cvxopt.matrix(dmin, size=(n,1))
        dprecvx  = cvxopt.matrix(dprec, size=(n,1))
        F = lip.getIntegratedPotentialFunctionFixedStrain(areas, phiOfD, dmincvx)
        
        Idn = cvxopt.spdiag([1.]*n)
        ineqGtDmin = {'G':-Idn, 'h': -dmincvx, 'dims': {'l': n, 'q': [], 's':  []} }
        ineqltOne  = {'G':Idn,  'h':cvxopt.matrix(1., size=(n,1)), 'dims' : {'l': n, 'q': [], 's':  []} }
        lipconstrain = lip.combineConeIneq([lipineq, ineqGtDmin, ineqltOne])
        resopt =solverdoptions['kernel'](F, lipconstrain, dprecvx, kktsolveroptions = solverdoptions['kktsolveroptions'], prevres = self.lipprojector.prevres)
        
        if not resopt['Converged'] : self.log.warning('minimizer did not converge in solveDGlobal')
        d= np.array(resopt['d']).squeeze()
        smin2 = np.array(lip.slack(lipineq,  d, check = False)).min()
        self.log.info('   Constrains satisfied  : '+'%.2e'%smin+' -> ' +'%.2e'%smin2)
        #reclip d, numerical error in cvxopt could push values of d on the wrong side, making the subsequent newton fail !
        d = np.where(d < dmin, dmin, d)
        snapthreshold = solverdoptions['snapthreshold']
        d = np.where(d >= snapthreshold, 1., d)
        #self.lipprojector.prevres = resopt['cvxoptres']
        return {'d':d, 'local':False,'Converged':resopt['Converged']}
        
    def solveDLipBoundPatch(self, dmin, dprec, strain, 
                            solverdoptions = {
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
        
        
        return self.lipprojector.solveDLipBoundPatch(dmin, dprec, strain, self.lc, self.law.solveSoftening,
                                                    self.law.potentialFixedStrain,
                                                    lip.getIntegratedPotentialFunctionFixedStrain, 
                                                    self.areas(), self.log,
                                                    solverdoptions)
    
    
    ################### PHASE FIELD DAMAGE SOLVER AT2 ######################################
    def phase_field_AT2(self,dmin, d, strain, 
                               solverdoptions ,imposed_d_0 =None  ):
        
        ## imposed_d_0 -- faces where damage is imposed to 0. (damage calculation avoided !)
        
        
        logger = self.log
        G_c = self.law.Yc
        l_c = self.lc
        driving_force = self.law._split_elastic_potential(strain)[1]
        
        ## phase field solver solves damage on lipmesh vertices
        
        phase_mesh = self.mesh
        
        ## an instance for the mechanics file of phase field
        try:
            self.at2
        except:
            self.at2 = phased.phase_damage_AT2(phase_mesh, G_c, l_c)
            
        at2 = self.at2  
        
        nv = phase_mesh.nvertices
        nf = phase_mesh.ntriangles
        
        
        
        if imposed_d_0 is None:
            imposed_d_0 = []
        
        
        
        ## faces where damage calc. are performed;;; others imposed d = 0
        free_faces = list(set(range(0,nf)).difference(set(imposed_d_0)))
        
        ## nodes where damage calculations are to performed (coressponding to faces indexed as 'free')
        free =[]
        for t in phase_mesh.triangles[free_faces]:
            free += [int(i) for i in t]
        free = list(set(free))   ##sorting and remove repetitve nodes
        nfree = len(free)
        #print(free)
        #print(type(free[0]))
       
        
        M = at2.massMatrix(driving_force)[free,free]
        K = at2.stiffness()[free,free]
        F = at2.forceVector(driving_force)[free]
        
        ## solve (K+M) * d = F
        
        #solverdoptions = {'linsolve':'umfpack'}
        lhs = M+K
        res = lin.solve(lhs,cvxopt.matrix(F), solver = solverdoptions['linsolve'], solveroptions= solverdoptions, logger=logger)
        
        if  1 :
                
                x = cvxopt.matrix(0., (nv,1))
                x[free] = res['x'][:nfree]
                x = np.array(x).squeeze()
                d=np.zeros(nf)
                       
                ## averaging to map nodal d values to face d values
                for it,t in enumerate(phase_mesh.triangles[free_faces]):
                    #print(t)
                    #np.mean(x[t])
                    d[free_faces[it]] =    np.mean(x[t])
                #lagmul = +x[nfree:]
                
        
                res ={'d':d, 'Converged':True,'d_nodes':x}
                return res
        
       

    def plotPatches(self, res, savefilename, ui):
        mesh = self.mesh
        lipmesh = self.lipprojector.mesh
        for k, itr in  enumerate(res['itresults']):
            dres = itr['resd']
            if not dres['local']:
                for kk in range( len(dres['it_infos'])-1 ):
                    dup = dres['dup'].squeeze()
                    dlo = dres['dlo'].squeeze()
                    infos_it0 = dres['it_infos'][kk]
                    infos_it1 = dres['it_infos'][kk+1]
                    s0 = infos_it0['global_lipslack'].squeeze()
                    patches = np.zeros(dup.shape[0], dtype = 'int')
                    for i, patch in enumerate(infos_it1['patches']):
                        patches[patch] = i+1
                    s1 =  infos_it1['global_lipslack'].squeeze()
                    fig, axes = plt.subplots(1, 4, figsize=(18,7), squeeze = False)
                    s0s = np.where(s0<0.,1., 0.)
                    c, fig, ax = lipmesh.plotScalarElemField(s0s, Tmin = 0., Tmax = 1.,showmesh =False, fig =fig, ax =axes[0,0])
                    ax.set_axis_off()
                    ax.axis('equal')
                    c, fig, ax = mesh.plotScalarElemField(((dup-dlo)>0.)*1,showmesh =False, fig =fig, ax =axes[0,1] )
                    ax.set_axis_off()
                    ax.axis('equal')
                    c, fig, ax = mesh.plotScalarElemField(patches, showmesh =False, fig =fig, ax =axes[0,2] )
                    ax.set_axis_off()
                    ax.axis('equal')
                    s1s = np.where(s1<0.,1., 0.)
                    c, fig, ax = lipmesh.plotScalarElemField(s1s,Tmin = 0., Tmax = 1., showmesh =False, fig =fig, ax =axes[0,3])
                    ax.set_axis_off()
                    ax.axis('equal')
                
                    fig.savefig(savefilename+'slack_u_%f_%d_%d.png'%(ui,k,kk), format='png')
                   
    def alternedDispDSolver(self, dmin, dguess, un=None, 
                            bc = None,
                            incstr = '', #string to put in the log at the beggining of each increment
                            alternedsolveroptions ={'abstole':1.e-9, 'reltole':1.e-6,  'deltadtol':1.e-5,
                                                    'outputalliter':False, 'verbose':False, 'stoponfailure':False, 'maxiter':10000},
                            solverdisp = solveDisplacementFixedDLinear,
                            solverdispoptions = {'linsolv':'cholmod'},
                            solverd= solveDGlobal, 
                            solverdoptions = {'kernel':lip.minimizeCPcvxopt, 'abstole':1.e-9, 'reltole':1.e-6,'fixpatchbound':True, 
                                              'Patchsolver':'edge', 'FMSolver':'edge', 'parallelpatch':True},
                            timer  = liplog.timer()
                            ):
        outputalliter = alternedsolveroptions.get('outputalliter')
        if outputalliter is None :  outputalliter =False
        verbose = alternedsolveroptions.get('verbose')
        if verbose is None :  outputalliter = False
        stoponfailure = alternedsolveroptions.get('stoponfailure')
        if stoponfailure is None :  stoponfailure = False
        
        if verbose : self.log.info("  Starting Alterned Displacement Damage Solver.")
        #nv = self.mesh.nvertices
        converged = False
        info=''
        d= dguess.copy()
        abstole   = alternedsolveroptions['abstole']
        reltole   = alternedsolveroptions['reltole']
        deltadtol = alternedsolveroptions['deltadtol']
        maxiter =   alternedsolveroptions['maxiter']
        it = 0
        u = un.copy() 
        itresults = []
        resd=dict()

        while(not converged and it <= maxiter):
            it +=1
            timer.start('disp solve')
            resu  = solverdisp(self, u0 = u, d = d,  
                               bc = bc,
                               solveroptions = solverdispoptions)
            timer.end('disp solve')
            
            if not resu['Converged'] :
                    self.log.error('Displacement solver did not converge at iteration :' + str(it))
                    if stoponfailure :
                         converged = False
                         info = "displacement solver failed"
                         return {'u':u,'d':d, 'iter':it, 'Converged':converged, 'info': info}   
                     
            delta_u = np.linalg.norm(u - resu['u'])    
            u = resu['u']
            R = resu['R']
            strain = self.strain(u)
            eu = self.energy(strain,d)  
            self.log.info("   "+incstr+ " Iter u "+'%' '4d'%it+", delta u "+ '%.2e'%delta_u +", eu "+'%.2e'%eu )
         
         
        
        
            timer.start('d solve')
            resd = solverd(self, dmin, d, strain, solverdoptions = solverdoptions)
            timer.end('d solve')
            
            if not resd['Converged'] :
                print('d solver did not converge')
                if stoponfailure :
                     converged =False
                     info = "d solver failed"
                     return {'u':u,'d':d, 'iter':it, 'Converged': converged, 'info' : info}   
            delta_d  = np.linalg.norm(resd['d']-d, np.inf)
            d = resd['d']
            ed = self.energy(strain,d)
            
            
            self.log.info("   "+incstr+" Iter d "+'%' '4d'%it+", delta d "+ '%.2e'%delta_d +", ed "+'%.2e'%eu + ", eu-ed: " + '%.2e'%(eu -ed) + ", (eu-ed)/ed: " + '%.2e'%((eu -ed)/max(ed,1.e-12)))
      
            if outputalliter : itresults.append({'it':it, 'resu':resu, 'resd':resd, 'eu':eu, 'ed':ed}) 
             
            if ( (ed >= eu) or (((abs(ed -eu) <= abstole) or (abs(ed -eu) <= abs(reltole*ed))) and (delta_d < deltadtol) )):
                converged = True
                
#            if (  (((abs(ed -eu) <= abstole) or (abs(ed -eu) <= abs(reltole*ed))) and (delta_d < deltadtol) )):
#                converged = True
#            
        if not outputalliter : itresults.append({'it':it, 'resu':resu, 'resd':resd, 'eu':eu, 'ed':ed})
        
        return {'u':u,'R':R,'d':d, 'iter':it, 'delta_d':delta_d, 'Converged':converged, 'itresults':itresults, 'info':info}           

    def strain(self, u):
        """ return the strain in each element, as an array such as :
            strain[i,0] is eps_xx in element i
            strain[i,1] is eps_yy in element i
            strain[i,2] is 2*eps_xy in element i
            eps_xx = dux/dx
            eps_yy = duy/dy
            2*eps_xy = dux/dy +duy/dx
            where 
             u is an array shape (2*nv) where nv is the number of vertices in the mesh,
              such as u[2*i] is the displacement component in direction x of vertice,
              u[2*i+1] is the displacement component in direction y of vertice i
        """
        nv = self.mesh.nvertices
        nf = self.mesh.ntriangles
        B = self.strainOp()
        strain = np.array(B * cvxopt.matrix( u.reshape(2*nv))).reshape(nf,3)
        #strain = B.dot(u.reshape(2*nv)).reshape(nf,3)
        return strain
    
    def stress(self, u, d):
        """ return the stress in each element, as an array such as :
            stress[i,0] is sxx in element i
            stress[i,1] is syy in element i
            stress[i,2] is sxy in element i
            where 
            - u is an array shape (2*nv) where nv is the number of vertices in the mesh,
              such as u[2*i] is the displacement component in direction x of vertice,
              u[2*i+1] is the displacement component in direction y of vertice i
              d is an array of shape (ne) where ne is the number of element.
              such as d[i] is the damage variable in element i
        """
        strain = self.strain(u)
        return self.law.trialStress( strain, d)
    
    
    def plots(self, u, d, u1, path, fields =np.array([['d_1','stressxx' ],['Y', 'd']]), showmesh = False, format = 'pdf'):
        """From u and d, respectively the displacement nodal field and damage element field plots e, 
        sigxx and d on 3 axis of the  same figure and save them on a [name].pdf file
        u1 stand for a loading factor of an increment for example, used for the title and to tune the name of the figure
        """
        strain = self.strain(u)
        stress = self.stress(u,d)
        e = self.law.potential(strain,d)
        Y = self.law.Y(strain,d)
        mesh = self.mesh
        shape = fields.shape
        if len(shape) ==1 : 
            shape = (1,shape)
            fields = fields.reshape((1,shape))
        if len(shape) > 2 : raise
            
        fig, axes = plt.subplots(shape[0], shape[1], figsize=(18,7), squeeze = False)
        axes.reshape(shape)
        
        with  np.nditer(fields, flags=['multi_index']) as it :
            for field in it :
                ax = axes[it.multi_index]
                if field == 'stressxx':
                    c, fig, ax = mesh.plotScalarField(stress[:,0], fig =fig, ax = ax, showmesh =showmesh)
                    ax.set_title(r"$\sigma_{xx}$") 
                elif field == 'stressyy':
                    c, fig, ax = mesh.plotScalarField(stress[:,1], fig =fig, ax = ax, showmesh =showmesh)
                    ax.set_title(r"$\sigma_{yy}$") 
                elif field == 'stressxy':
                    c, fig, ax = mesh.plotScalarField(stress[:,2], fig =fig, ax = ax, showmesh =showmesh)
                    ax.set_title(r"$\sigma_{xy}$")
                elif field =='d':
                    c, fig, ax = mesh.plotScalarField(d, fig =fig, ax = ax, Tmin=0., Tmax = 1., showmesh =showmesh)
                    ax.set_title(r"$d$")
                elif field =='d_1': # higlight element for which d==1.
                    c, fig, ax = mesh.plotScalarField(np.where(d>=1.-1.e-9,1.,0.), fig =fig, ax = ax, Tmin=0., Tmax = 1., showmesh =showmesh)
                    ax.set_title(r"$d = 1$")
                elif field =='Y': 
                    c, fig, ax = mesh.plotScalarField(Y, fig =fig, ax = ax, showmesh =showmesh)
                    ax.set_title(r"$Y$")  
                elif field =='Y_filtered':
                    saturated = (d< 0.9)[0]
                    Ysat = Y[saturated].max()
                    Yfiltered = np.where(d> 1.-1.e-9, Ysat, Y) 
                    c, fig, ax = mesh.plotScalarField(Yfiltered, fig =fig, ax = ax, showmesh =showmesh)
                    ax.set_title(r"$Y_f$") 
                elif field =='e': #strain energy
                    c, fig, ax = mesh.plotScalarField(e, fig =fig, ax = ax,  showmesh =showmesh)
                    ax.set_title(r"$e$")
                elif field =='u':
                    unorm = np.linalg.norm(u, axis = 1) 
                    c, fig, ax = mesh.plotScalarField(unorm, disp=u, fig =fig, ax = ax, showmesh=showmesh)
                    ax.set_title(r"$displacement$") 
                    
                    
                fig.colorbar(c, ax=ax, orientation = 'vertical')
                ax.set_axis_off()
                ax.axis('equal')
                    
        fig.suptitle('Loadingfactor : '+str(u1))       
        #fig.tight_layout()
        fig.savefig(path.with_name(path.name +'u_%f'%u1+'_.'+format), format=format)   
        plt.close(fig) 
            
