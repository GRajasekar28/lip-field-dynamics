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

    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import matplotlib.tri
import matplotlib.pylab as plt
import numpy as np
import gmshParser as gmshParser
import triangle
import cvxopt
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection

def innerCircleRadius(x):
    a = np.linalg.norm(x[1]-x[0])
    b = np.linalg.norm(x[2]-x[1])
    c = np.linalg.norm(x[0]-x[2])
    s = (a+b+c)/2.
    A = np.sqrt(s*(s-a)*(s-b)*(s-c))
    r = A/s
    return r


    
def numberingTriPatch(triangles, tris2vertices):
        """ triangles contains a list of triangle id, tris2 vertices is the connectivity table
            return a new numbering of the vertices touched by the triangles
            in the form of the tuple (triangles, vg2l, vl2g)
            were triangles  is the input triangle, vg2l is a map associating global numbering of nodes to a 'local ' one that number all the nodes
            participating in the triangles given as input from 0 to nl-1
        """
        v_loc2glob = np.array(np.unique(tris2vertices[triangles]), dtype='int')
        v_glob2loc = dict(zip(v_loc2glob, np.arange(v_loc2glob.shape[0]) ))
        return (triangles, v_glob2loc, v_loc2glob)

def numberingEdgePatch(edges, edge2vertices):
        """ edges contains a list of edge id, edge2vertices is the connectivity table of edge to vertices
            return a new numbering of the vertices touched by the edges
            in the form of the tuple (edges, vg2l, vl2g)
            were triangles  is the input edges, vg2l is a map associating global numbering of nodes to a 'local ' one that number all the nodes
            participating in the edges given as input from 0 to n-1, where n is the number of touched vertices
        """
        return numberingTriPatch(edges, edge2vertices)

def getEdge2Vertices(triangles):
    """ from an array of shape (n, 3) where line i contain the 3 nodes of triangle i, return an array of the edges of this triangles. 
      Each edge appears once int the returned array """
    e2v = set()
    for t in triangles :
        e0 = (min(t[0], t[1] ), max(t[0],t[1]) )
        e1 = (min(t[1], t[2] ), max(t[1],t[2]) )
        e2 = (min(t[2], t[0] ), max(t[2],t[0]) )
        e2v.update(  [e0, e1, e2])
    return  np.array(list(e2v))

class simplexMesh:
    def readGMSH(gmsh_file_name):
        """ Read a mesh from a gmsh (.msh file , format =2.2) """
        gmsh_mesh = gmshParser.Mesh()
        gmsh_mesh.read_msh(gmsh_file_name)
        tris = gmsh_mesh.Elmts[gmshParser.triLinElementTypeId][1].copy()
        edges = gmsh_mesh.Elmts[1]
        classvertices = dict()
        # check if there are geometric nodes, and add them to the list
        if gmsh_mesh.Elmts.get(15):
            for label, vid in zip(gmsh_mesh.Elmts[15][0], gmsh_mesh.Elmts[15][1]):
                classvertices[label] = vid
        return simplexMesh(gmsh_mesh.Verts[:,[0,1]], tris, classedges= edges, classvertices=classvertices)
    
    def __init__(self, xy, tris, topedges = None, classedges=None, classvertices=dict()):
        if topedges is None:
            topedges = np.zeros((0,2), dtype ='int')
        if classedges is None :
            classedges = (np.zeros(0, dtype='int'), np.zeros( (0,0), dtype='int'))
        self.classvertices = classvertices
        self.xy = xy
        self.triangles  = tris
        self.topedges = topedges
        self.classedges = classedges
        self.nvertices  = xy.shape[0]
        self.ntopedges = topedges.shape[0]
        self.ntriangles = tris.shape[0]
        # storage of cached results. (like adjacencies)
        self._cached = {}
        
        
    def areas(self): 
        """
        return the area of all triangles in the mesh

        Returns
        -------
        TYPE np.array, indexed by triangle id in the mesh
            areas[i] : area of triangle i.

        """
        a = self._cached.get("areas")
        if a is None :
            xy0 = self.xy[self.triangles[:,0]]
            xy1 = self.xy[self.triangles[:,1]]
            xy2 = self.xy[self.triangles[:,2]]
            a  = np.abs(np.cross(xy1-xy0, xy2-xy0)/2.) 
            self._cached["areas"] = a
        return a
    
    def innerCirleRadius(self):
        nt = self.ntriangles
        r = np.zeros(nt)
        for i in range(nt):
            r[i] = innerCircleRadius(self.xy[self.triangles[i]])
        return r

    
    def elementaryGradOperators(self):
        """Return an array of n matrices, where n is the number of triangles. For each triangle in the mesh, G  is 2*3 matrix, such as given 3 nodal values
           of the scalar field in a vector T , G*T is the gradient of the lineary interpolated nodal values. 
        """
        G = self._cached.get("G")
        if G is None:
             t = self.triangles
             xy = self.xy
             nt = t.shape[0]
             exy = np.empty((nt,2,2))
             exy[:,0,:] = xy[t[:,1],:]-xy[t[:,0],:]
             exy[:,1,:] = xy[t[:,2],:]-xy[t[:,0],:]
             G =  np.linalg.inv(exy).dot( np.array([[-1,1.,0.],[-1.,0.,1.]]))
             self._cached["G"] = G
        return G
            
    def integrateElementField(self, elementfield):
        """ Given a constant field per element, integrate over the mesh """
        A = self.areas()
        if (elementfield.shape[0] != len(A)) :
            print('Error in integrate')
            raise
        return A.dot(elementfield)
    
    def integrateVertexField(self, vertexfield):
        """ Given a field per vertices, integrate over the mesh, assuming linear per elements """
        
        if (vertexfield.shape[0] !=  self.nvertices) :
            print('Error in integrate')
            raise
        average =  self.averageOp()*(cvxopt.matrix(vertexfield, (self.nvertices, 1)))
        A = self.areas()
        return A.dot(average)
    
    def averageOp(self): 
        """ return the averaging operator :  A,  sparse matrix (cvxopt.spmatrix) of size (n, m), where n is the number 
  of triangles and m is the number of nodes, such as T_e = A * T_v  where T_v is a vector of nodal value, 
  and T_e is a vector of element values. T_e[i] is the average of T in element [i], T, assumed linear per element.
  use full to project a nodal field to an element field. """
        aOp = self._cached.get("averageOp")
        if aOp is None :
            aX = np.array(self.ntriangles*3*[1./3.])
            aI = np.array(sum([ [i,i,i] for i in range(self.ntriangles)], []), dtype = 'int')
            aJ = np.array(sum( [list(self.triangles[i]) for i in range (self.ntriangles)],[] ), dtype = 'int')
            aOp = cvxopt.spmatrix(aX,aI,aJ, (self.ntriangles, self.nvertices ) ) 
            self._cached["averageOp"] = aOp
        return aOp
        
        
    def getMPLTriangulation(self):
        """ return a matplotlib trianguluation from the current mesh """
        MPLTriangulation = self._cached.get("MPLTriangulation")
        if MPLTriangulation is None : 
            MPLTriangulation  = matplotlib.tri.triangulation.Triangulation(self.xy[:,0], self.xy[:,1], self.triangles)
            self._cached["MPLTriangulation"] = MPLTriangulation
        return MPLTriangulation
    
    def getTop(self):
        if (self.ntriangles and (self.ntopedges == 0)) :
            return self.triangles
        if (self.ntopedges and (self.ntriangles ==0 )) :
            return self.topedges
        
    def getVertices(self):
        return self.xy
    
    def getClassifiedEdges(self, phys = None):
        """Return  the list of edges classifyed on physical line phys or all the classifyed edge"""
        if phys is None  : return self.classedges[1]
        onphys = self.classedges[0]==phys
        return self.classedges[1][onphys]
    
    def getVerticesOnClassifiedEdges(self, phys = None):
        """Return  the list of vertices classifyed on physical line phys """
        return list(set([ vid for  e in  self.getClassifiedEdges(phys) for vid in e]))
    
    
    def getClassifiedPoint(self, phys):
        """Return  the list of vertices classifyed on phys point """
        if phys is None  : return None
        return self.classvertices[phys]
        
    def getVertices2Triangles(self):
        """ return  a list that contain for each entry i the list of triangles connected to i"""
        vtotri = self._cached.get("vert2tri")
        if vtotri is None :
            vtotri = [None]*self.nvertices
            for (tid, t) in enumerate(self.triangles) :
                for i in [0,1,2] :
                        a = vtotri[t[i]]
                        if  a is None :
                            vtotri[t[i]] = set([tid])
                        else :
                            a.add(tid)
                            vtotri[t[i]] = a.copy()
            self._cached["vert2tri"] = vtotri
        return vtotri
    
    def getVertex2Triangles(self, i):
        """ return the list of triangles connected to vertex i"""
        return self.getVertices2Triangles()[i]
    
    def getTris2NeibTri(self):
        ''' return an array, whos entry i contain the list of neighbors of triangle i'''
        tri2neib = self._cached.get("tri2neib") 
        if tri2neib is None :
            tri2neib = self.getMPLTriangulation().neighbors
            self._cached["tri2neib"] = tri2neib
        return tri2neib
        
    def getTopCOG(self):
        COG = self._cached.get("COG")
        if COG is  None :
            COG = []
            for e in self.getTop():
                x = [ self.xy[kv] for kv  in e] 
                x = sum(x)/len(x)
                COG.append(x)
            COG = np.array(COG)
            self._cached["COG"] = COG
        return COG
    
    def getVertex2Vertices(self):
        v2vs  = self._cached.get("v2vs")
        if v2vs is None :
            nv = self.nvertices
            v2vs = [None]*nv
            for vi in range(nv):
                vi2v = set()
                tris= self.getVertex2Triangles(vi)
                for t in tris:
                    vi2v.update(self.triangles[t])
                vi2v.remove(vi)
                v2vs[vi] = list(vi2v)
            self._cached["v2vs"] = v2vs
        return v2vs
    
    def getEdge2Vertices(self):
        e2vs = self._cached.get("e2vs") 
        if e2vs is None:
            e2vs = getEdge2Vertices(self.triangles)
            self._cached["e2vs"] = e2vs
        return e2vs
    
    def getVertex2Edges(self):
        v2es = self._cached.get("v2es")
        if v2es is None :
            edge2vertices = self.getEdge2Vertices()
            nv = self.nvertices
            v2es = [ list() for i in range(nv)]
            for ei, ev in enumerate(edge2vertices):
                v2es[ev[0]].append(ei)
                v2es[ev[1]].append(ei)
            self._cached["v2es"] = v2es
        return v2es
    
    def getTriangle(self, v0, v1, v2):
        """Return the triangle id that as node id v0, v1, v2 if any. return None otherwise"""
        tv0 = set(self.getVertex2Triangles(v0))
        tv1 = set(self.getVertex2Triangles(v1))
        tv2 = set(self.getVertex2Triangles(v2))
        sett = set.intersection(tv0,tv1,tv2)
        if len(sett) == 0 : return None
        if len(sett) == 1 : return list(sett)[0]
        raise
        
    def link2LinesVertices(self, IdL0, IdL1, eps = 1.e-9, sortaxis = None):
        """ IdL0 and IdL1 are the physical Id of 2 lines (L0, L1), supposed to be geometrically identical, with nodes at the same positions. 
            This function return a list containing vertices pairs of corresponding nodes id from L0 and L1
            Both line should have the same number of nodes.
            if x0 is the coordinate of a node on Line0, with id vid0, and x1 is the coordinate of a node on Line1 with  id vid1, 
            the list will contain the pair(id0, id1)
            if sortaxis is None, the order in the list correspond to the input order.
            if sortaxis is a np.array of size2, the node will be sorted according to their position along the line of axis sortaxis
            The function raise an exception if it can't pair all the node of L1 with all the node of L2. Complexoty is n^2 where n is the number of node on L0'
        """
        vidsl0 = self.getVerticesOnClassifiedEdges(IdL0)
        vidsl1 = self.getVerticesOnClassifiedEdges(IdL1)
       
        if sortaxis is not None :
            vidsl0.sort( key = lambda id : self.xy[id].dot(sortaxis))
            vidsl1.sort( key = lambda id : self.xy[id].dot(sortaxis))
            
        
        
        if len(vidsl0)!=len(vidsl1) :
            print("error in link2LinesVertice : the lines don't have the same number of vertices")
            raise
            
            
        pairing = list()
        for id0 in vidsl0:
            x0 = self.xy[id0]
            for id1 in vidsl1:
                x1 = self.xy[id1]
                if np.linalg.norm(x1-x0) < eps :    
                    pairing.append((id0, id1))
                    break
        return pairing
    
    def zeroVertexField(self, nvar= 1 ):
        """Return an array to store a nodal field. nvar is the number of variables per node (and .shape[1] of the np array)"""
        return np.zeros((self.nvertices, nvar)).squeeze()
    
    def zeroTriangleField(self, nvar = 1):
        """Return an array to store a triangle field. nvar is the number of variables per face (and .shape[1] of the np array)"""
        return np.zeros((len(self.getTopCOG()), nvar)).squeeze()
    
    def plotScalarVertexField(self, T,disp=None, fig =None, ax = None, showmesh =True, meshplotstyle = 'b-', Tmin =None, Tmax = None, ncontour = 20, clabel = False):
        if (T.shape[0] != self.nvertices) : 
            raise
        if ax is None :
            fig, ax = plt.subplots()
       
        if Tmin is None :
            Tmin = T.min()
        if Tmax is None :
            Tmax = T.max()
        
        if disp is not None:
            mpl_tri = matplotlib.tri.triangulation.Triangulation(self.xy[:,0], self.xy[:,1], self.triangles)            
            mpl_tri.x =  mpl_tri.x + disp[:,0]
            mpl_tri.y =  mpl_tri.y + disp[:,1]
            
        else : mpl_tri=self.getMPLTriangulation()
        if Tmin == Tmax : 
            c = ax.tripcolor(mpl_tri, T, vmin = Tmin, vmax =Tmax, cmap=plt.cm.rainbow)
        else :
            #c = ax.tricontour(mpl_tri, T,np.linspace(Tmin, Tmax, ncontour), cmap=plt.cm.rainbow)       
            if clabel:
                ax.clabel(c, inline=True,  fontsize=10)
            c = ax.tricontourf(mpl_tri, T,np.linspace(Tmin, Tmax, ncontour), cmap=plt.cm.rainbow)   
        #fig.colorbar(c, ax=ax)
        
        if showmesh :
            self.plot(fig, ax, style =  meshplotstyle)
        ax.axis('equal') 
        return c, fig, ax
    
    def plot(self, fig =None, ax = None, style = 'b-', color =None):
        """ plot the mesh """
        if ax is None:
            fig, ax = plt.subplots()
        if self.ntriangles :
            ax.triplot(self.getMPLTriangulation(), style, lw=1, color = color)
        for e in self.topedges :
            v0 = self.xy[e[0]]
            v1 = self.xy[e[1]]
            x = [v0[0], v1[0]]
            y = [v0[1], v1[1]]
            ax.plot(x,y, 'r')
        ax.axis('equal') 
        return fig, ax
        
    def plotScalarElemField(self, T, disp = None, fig =None, ax = None, showmesh =True,  meshplotstyle = 'b-', Tmin = None, Tmax = None, ncontour = 20, eraseifoutofbound = False ):
        """ plot a field T defined at the node of the mesh (linear per element) """
        if len(T) != self.ntriangles :
            print('len(d) != ntri')
            raise
        if ax is None :
            fig, ax = plt.subplots()
            
        if eraseifoutofbound and Tmax is not None:
            tris = self.triangles
            filt = np.where(T < Tmax)[0]
            tris = tris[filt ,:]
            T = T[filt]
            mpl_tri = matplotlib.tri.triangulation.Triangulation(self.xy[:,0], self.xy[:,1], tris)
        else:
            tris = self.triangles
            mpl_tri=self.getMPLTriangulation() 
        
        if disp is not None:
            mpl_tri = matplotlib.tri.triangulation.Triangulation(self.xy[:,0], self.xy[:,1], tris)            
            mpl_tri.x =  mpl_tri.x + disp[:,0]
            mpl_tri.y =  mpl_tri.y + disp[:,1]
            
        else : mpl_tri=self.getMPLTriangulation()
        
        
        c = ax.tripcolor(mpl_tri, T, vmin = Tmin, vmax =Tmax, cmap=plt.cm.rainbow)
        #clabel = True
#        if clabel:
#            ax.clabel(c, inline=True,  fontsize=10)
        #fig.colorbar(c, ax=ax)
        if showmesh :
            self.plot(fig, ax, style=meshplotstyle)
        ax.axis('equal') 
        return c, fig, ax
    
    def plotScalarField(self, T,  disp = None, fig =None, ax = None, showmesh =True, meshplotstyle = 'b-', Tmin = None, Tmax = None, ncontour = 20 ):
        """ plot a field T defined at each element of the mesh (contant per element )  or at each node of the mesh (linear par elemenent )
            T must be convertible to a 1d np.array using T = np.array(T).squeeze()
            reformated T must be of len == number of element | number of vertices
            option :
            disp :  a 2 d displacement field at the node of the mesh : used to deform by moving the nodes.
            fig, ax :  figure and ax on which to draw the figure
            showmesh = True|False  : draw the mesh or not
            meshplotstyle : string to define how to plot the edges of the mesh
            Tmin, Tmax : min and max value of the field used to define the contour
            ncontour : number of contour to draw
        """
        T = np.array(T).squeeze()
        if len(T) == self.ntriangles :
            return self.plotScalarElemField(T, disp, fig, ax, showmesh,  meshplotstyle , Tmin, Tmax, ncontour = 20 )
        if len(T) == self.nvertices :
            return self.plotScalarVertexField(T, disp, fig, ax, showmesh,  meshplotstyle , Tmin, Tmax, ncontour = 20 )
        
    def plotVectorNodalField(self, v, scale = 1., fig =None, ax = None, pivot='tail', showmesh =True):
        """ plot in fig, ax the vector nodal field u on the mesh, using small arrows """
        if v.shape != (self.nvertices,2) :
            print('u.shape != (nvert,2)')
            raise
        if ax is None :  fig, ax = plt.subplots()
        mpltri = self.getMPLTriangulation()
        ax.quiver(mpltri.x, mpltri.y, v[:,0], v[:,1], scale = scale, pivot = pivot)
        if showmesh :
            plt.triplot(mpltri, 'b-', lw=1)
        ax.axis('equal') 
        return fig, ax
    
    def plotVectorElemField(self, v, scale = 1., fig =None, ax = None, pivot='tail', showmesh =True):
        """ plot in fig, ax the vector nodal field v on the mesh, using small arrows
            v must be  an np.array of shape (ntriangles, 2), so that v[i,:] = [vx,vy] where vx, vy are the component of v at element i
            option :
            scale :  a scaling factor for the size of the vector, as defined in pylab.quiver
            fig, ax :  figure and ax on which to draw the figure
            pivot :  string ('tail', 'head' ... as defined in pylab.quiver : what point of the arraow is centered on the cog of the trangle)
            showmesh = True|False  : draw the mesh or not
        """
        if v.shape != (self.ntriangles,2) :
            print('u.shape != (ntriangles,2)')
            raise
        if ax is None :  fig, ax = plt.subplots()
        xyg = self.getTopCOG()
        print (scale)
        ax.quiver(xyg[:,0], xyg[:,1], v[:,0], v[:,1], scale = scale, pivot = pivot)
        if showmesh :
            mpltri = self.getMPLTriangulation()
            plt.triplot(mpltri, 'b-', lw=1)
        return fig, ax
    
    def plotSymTensor2ElemField(self, eps, scale = 1., fig =None, ax = None, showmesh =True):
        ''' plot in fig, ax scaled eigenvector of the symetric tensor eps (represented in voigt format strain convention)
            eps[i,:] = [epsxx, epsyy, 2epsxy] where epsxx,yy,xy are the component of a symetic tensor defined at the center of element i.
            option :
            scale :  a scaling factor for the size of the vector, as defined in pylab.quiver
            fig, ax :  figure and ax on which to draw the figure
            showmesh = True|False  : draw the mesh or not
        '''
        import lip2d.material_laws_planestrain as mlaws
        l0, l1, N0, N1 = mlaws.eigenSim2D_voigt(eps,  vector = True)
        ne = l0.shape[0]
        l0N0 =  np.broadcast_to(l0.reshape(ne,1), (ne,2))*N0
        l1N1 =  np.broadcast_to(l1.reshape(ne,1), (ne,2))*N1
    
        l0p = np.where(np.broadcast_to(l0.reshape(ne,1), (ne,2)) > 0., l0N0, 0.)
        l0m = np.where(np.broadcast_to(l0.reshape(ne,1), (ne,2)) < 0., l0N0, 0.)
    
        l1p = np.where(np.broadcast_to(l1.reshape(ne,1), (ne,2)) > 0., l1N1, 0.)
        l1m = np.where(np.broadcast_to(l1.reshape(ne,1), (ne,2)) < 0., l1N1, 0.)

        fig, ax  = self.plotVectorElemField( l0p, scale = scale, fig =fig, ax=ax, showmesh =showmesh)
        self.plotVectorElemField( -l0p, scale = scale, fig =fig, ax=ax, showmesh =False)
    
        self.plotVectorElemField( l0m, scale = scale, fig =fig, ax=ax, showmesh = False, pivot= 'tip')
        self.plotVectorElemField( -l0m, scale = scale, fig =fig, ax=ax, showmesh = False, pivot = 'tip')
    
        self.plotVectorElemField( l1p, scale = scale, fig =fig, ax=ax, showmesh = False)
        self.plotVectorElemField( -l1p, scale = scale, fig =fig, ax=ax, showmesh = False)
    
        self.plotVectorElemField( l1m, scale = scale, fig =fig, ax=ax, showmesh = False, pivot= 'tip')
        self.plotVectorElemField( -l1m, scale = scale, fig =fig, ax=ax, showmesh = False, pivot = 'tip')
    
        return fig, ax
    
    def getTriFinder(self) :
        trifinder = self._cached.get("trifinder")
        if trifinder is None :
             trifinder = matplotlib.tri.TrapezoidMapTriFinder(self.getMPLTriangulation())
             self._cached["trifinder"] = trifinder
        return trifinder
    
    def extractScalarVertexFieldOnLine(self, T, P0, P1,npt):
        """ interpolate a nodal field Talong the segment P0,P1 on npt equidistant points """
        if len(T) != self.nvertices:
            print('len(d) != nverts')
            raise
        x = np.linspace(P0[0], P1[0], npt)
        y = np.linspace(P0[1], P1[1], npt)
        s=  np.linspace(0., np.linalg.norm(P1-P0), npt )
        trifinder =self.getTriFinder()
        interp = matplotlib.tri.LinearTriInterpolator(self.getMPLTriangulation(), T, trifinder=trifinder)
        v = interp(x,y)
        return {'x':x, 'y': y, 's':s, 'v': v}
    
    def plotScalarVertexFieldOnLine(self, T, P0, P1,npt,fig =None, ax = None):
        """ plot in fig, ax a nodal field T defined on the mesh along the line joining P0, P1 by interpolating npt equidistant points along [P0,P1] """
        if ax is None :
            fig, ax = plt.subplots()
        res = self.extractScalarVertexFieldOnLine(T, P0, P1,npt)
        s = res['s']
        v = res['v']
        ax.plot(s, v)   
        return fig, ax
    
    def extractScalarElemFieldOnLine(self, T, P0, P1, npt): 
        """ extract a constant element field T along the segment P0,P1 on npt equidistant points """
        if len(T) != self.ntriangles:
            print('len(d) != nverts')
            raise
        trifinder = self.getTriFinder()           
        x = np.linspace(P0[0], P1[0], npt)
        y = np.linspace(P0[1], P1[1], npt)
        s=  np.linspace(0., np.linalg.norm(P1-P0), npt )
        ie = trifinder(x,y)
        return {'x':x, 'y': y, 's':s, 'v': T[ie]}
    
    def plotScalarElemFieldOnLine(self, T, P0, P1,npt, fig =None, ax = None):
        """ plot in fig, a elemental field T defined on the mesh along the line joining P0, P1 by interpolating npt equidistant points along [P0,P1] """
        res = self.extractScalarElemFieldOnLine( T, P0, P1, npt)
        if ax is None :
            fig, ax = plt.subplots()
        s = res['s']
        v = res['v']
        ax.plot(s, v)
        return fig, ax
    
    def plotScalarFieldOnLine(self, T, P0, P1,npt,fig =None, ax = None):
        """ plot in fig, ax a nodal or elemental field T defined on the mesh along the line joining P0, P1 by interpolating npt equidistant points along [P0,P1] """
        if len(T) == self.ntriangles :
            return self.plotScalarElemFieldOnLine(T,P0,P1,npt, fig= fig, ax = ax)
        if len(T) == self.nvertices :
            return self.plotScalarVertexFieldOnLine(T,P0,P1,npt, fig= fig, ax = ax)
        raise

def partitionTri(mesh, tris):
    '''given a list tris of triangle and a mesh, return a partition of the connected tris'''
    tris = set(tris)
    parts = []
    while len(tris)> 0 :
      start = tris.pop()
      queue = [start]
      part = set([start])
      while len(queue) > 0:
          start = queue.pop()
          for neib in  mesh.getTri2NeibTri(start):
              if neib >= 0 and neib in tris :
                  tris.remove(neib)
                  queue.append(neib)
                  part.add(neib)
      parts.append( list(part))
    return parts

def partitionGraph(vertices, v2v):
    '''given a list of vertices and a graph return a partition of the vertices, each partition contains connected vertices'''
    vertices = set(vertices)
    parts = []
    while len(vertices)> 0 :
      start = vertices.pop()
      queue = [start]
      part = set([start])
      while len(queue) > 0:
          start = queue.pop()
          for neib in  v2v[start]:
              if neib >= 0 and neib in vertices :
                  vertices.remove(neib)
                  queue.append(neib)
                  part.add(neib)
      parts.append( list(part))
    return parts
     
def dualLipMesh(mesh):
    cogs = mesh.getTopCOG()
    neigh = mesh.getMPLTriangulation().neighbors
    triangles = []
    topedges = []
   # print(len(neigh[1] > 0))
    for i, neib in enumerate(neigh) :
        nneib = sum(neigh[i] >= 0)
        if (nneib == 3):
            for k in range(3) : 
                t = [i, neib[k], neib[(k+1)%3]]
                triangles.append(t)
        elif (nneib == 2):  
            neib = neib[neib>=0]
            t= [i, neib[0], neib[1]]
            triangles.append(t)   
        elif (nneib == 1) : 
            neib = neib[neib>=0]
            e = [i, neib[0]]
            topedges.append(e)
        else :   raise
    if len(topedges) == 0 : 
        topedges = None
    else:
        topedges = np.array(topedges)
    return simplexMesh(cogs, np.array(triangles), topedges= topedges)


def triangle2mpl(triangle_mesh):
    xy = triangle_mesh['vertices']
    t =  triangle_mesh['triangles']
    return matplotlib.tri.triangulation.Triangulation(xy[:,0], xy[:,1], t)

def triangle2simplexMesh(triangle_mesh):
    xy = triangle_mesh['vertices']
    t =  triangle_mesh['triangles']
    return simplexMesh(xy,t)


def dualLipMeshTriangleFromCOG(cogs, boundaryvertices=[], boundaryedges=[], triopt ='p'):
    if len(boundaryedges) == 0 :
         lipmesh = triangle.triangulate({'vertices': cogs})
         return simplexMesh(cogs, lipmesh['triangles'])
         
    nvkeep = cogs.shape[0]
    bnodes = dict()
    bnodeid = nvkeep
    nedges = []
    bv= []
    for b in boundaryedges :
        v0id = b[0]
        v1id = b[1]
        v0nid  = bnodes.get(v0id)
        if v0nid is None :
            v0nid = bnodeid
            bnodeid +=1
            bnodes[v0id] = v0nid
            bv.append(boundaryvertices[v0id])
        v1nid  = bnodes.get(v1id)
        if v1nid is None :
            v1nid = bnodeid
            bnodeid +=1
            bnodes[v1id] = v1nid
            bv.append(boundaryvertices[v1id])
        nedges.append([v0nid, v1nid])
  
    v= np.vstack((cogs, np.array(bv)))
    lipmesh_triangle = triangle.triangulate({'vertices':v, 'segments':nedges},triopt)
    tris = lipmesh_triangle['triangles']
    filt = np.ones(tris.shape[0], dtype='bool')
    for (i, t) in enumerate(tris) :
        if np.max(tris[i]) >= nvkeep :
            filt[i] = False
    tris = tris[filt]
    return simplexMesh(cogs, tris)


def dualLipMeshTriangle(mesh, triopt ='p'):
    cogs = mesh.getTopCOG()
    boundaryedges = mesh.getClassifiedEdges()
    boundaryvertices = mesh.xy
    return dualLipMeshTriangleFromCOG(cogs, boundaryvertices, boundaryedges, triopt ='p')
   



def dualLipMeshEdges(mesh, triopt ='p'):
    mtris  =  mesh.triangles 
    e2v = set()
    for (v0, v1, v2) in mtris :
        e0 = (min(v0, v1 ), max(v0, v1) )
        e1 = (min(v1, v2 ), max(v1,v2) )
        e2 = (min(v2, v0 ), max(v2,v0) )
        e2v.update(  [e0, e1, e2])
    
    boundary = mesh.getClassifiedEdges()
    e2vb = set()
  
    for (ev0, ev1) in boundary:
        e = (min(ev0, ev1), max(ev0,ev1) )
        e2vb.update( [e])

    nedges = len(e2v)
    cogs =np.empty((nedges,2))
    onboundary = False*np.ones(nedges, dtype='bool')
    for (i, e) in enumerate(e2v):
        x0 = mesh.xy[e[0]]
        x1 = mesh.xy[e[1]]
        cogs[i] = (x0+x1)/2.
        onboundary[i] = e in e2vb
    lipmesh_triangle = triangle.triangulate({'vertices':cogs})
    tris = lipmesh_triangle['triangles']
    filt = np.ones(tris.shape[0], dtype='bool')
    for (i, (i0,i1,i2)) in enumerate(tris) :
        if(onboundary[i0] and onboundary[i1] and onboundary[i2]) :    filt[i] = False
    
    return simplexMesh(cogs, tris[filt])     
    
