# -*- coding: utf-8 -*-
# @Time    : 2024/9/27 15:18
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : auto_fine_mesh.py
# @Software: PyCharm
"""
自适应网格细化
"""

import enum

# import cst.results
import typing

import numpy

import matplotlib
matplotlib.use('tkagg')

import pandas
import scipy
import scipy.constants as C
from scipy.fft import ifft
from scipy.optimize import curve_fit
from shapely.geometry import LineString
from refinement import refine_mesh
from _logging import logger




import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
f_target = 9.3e9
def xi(f,f0, Q):
    return Q*(f/f0-f0/f)
def f0(x,y,):
    f_target_ = f_target
    fx = 1e9 / 1e-3
    fy = 0.2e9 / 1e-3
    x0 ,y0 = 0, 0
    # return f_target_  + fx * (x-x0) +fy*(y-y0)
    r = 1e-3
    a = (((x - x0) ** 2 + (y - y0) ** 2 - r ** 2) * fx ** 2)
    return f_target_ +numpy.sign(a) * (numpy.abs(a)**0.5)

def dummy_func(x,y,sigma_x , sigma_y,):
    return numpy.exp(-1/2*(x/sigma_x)**2) * numpy.exp(-1/2*(y/sigma_y)**2)

def dummy_func2 (x,y, Df ,f = f_target ):
    # f =f_target
    return 1/(1+xi(f,f0(x,y) ,f/Df)**2)**0.5
if __name__ == '__main__':
    plt.ion()
    xs,ys = numpy.linspace(-3e-3, 3e-3, 200),numpy.linspace(-3e-3, 3e-3, 200)
    X,Y = numpy.meshgrid(xs,ys)
    Df= 2.e9
    plt.figure()
    cf = plt.contourf(X,Y, dummy_func2( X,Y,Df))
    plt.colorbar(cf)

    plt.figure()
    plt.plot(xs, dummy_func2(xs, 0 ,Df))


    X_amr,Y_amr = numpy.meshgrid( numpy.linspace(-2.5e-3, 2.5e-3,12),numpy.linspace(-2.5e-3, 2.5e-3, 12))
    plt.figure()
    plt.contourf(X_amr,Y_amr, dummy_func2(X_amr, Y_amr,Df))
    plt.scatter(X_amr,Y_amr,s = 1)


    tri = Delaunay(numpy.array([X_amr,Y_amr]).transpose((1,2,0)).reshape((-1, 2)),incremental=True,)
    # tri.add_points([[0.7e-3, 0.7e-3]])


    # examined_edges = {}


    plt.figure(constrained_layout=True)
    plt.triplot(tri.points[:, 0],tri.points[:, 1],tri.simplices,c= 'k',#label = "original",
                 alpha = 0.2)
    cf = plt.tricontourf(tri.points[:,0], tri.points[:,1],tri.simplices,     dummy_func2(tri.points[:,0], tri.points[:,1],Df),zorder = -1 )
    plt.colorbar(cf)
    plt.title("npts = %d" % (tri.npoints,))
    # plt.legend()
    plt.gca().set_aspect('equal')
    logger.info("npts = %d"%(tri.npoints))

    npts = [[0, tri.npoints]]
    for i in range(10):
        tri = refine_mesh(tri,lambda x,y,f:dummy_func2(x,y,Df,f),f_target)
        plt.figure(constrained_layout=True)
        plt.triplot(tri.points[:, 0], tri.points[:, 1], tri.simplices,c = 'k',#label = "refined (round %d)"%(i+1),
                    alpha = 0.2)
        plt.title("npts = %d (refined, round %d)"%(tri.npoints,i+1))
        cf =         plt.tricontourf(tri.points[:,0], tri.points[:,1],tri.simplices,
                       dummy_func2(tri.points[:,0], tri.points[:,1],Df),zorder = -1 )
        plt.colorbar(cf)

        # plt.legend()
        plt.gca().set_aspect('equal')
        plt.title("npts = %d (refined, round %d)"%(tri.npoints,i+1))
        npts.append([i+1, tri.npoints])
    plt.figure()
    npts = numpy.array(npts)
    plt.plot(npts[:,0],npts[:,1])

    from scipy.interpolate import griddata,LinearNDInterpolator
    li = LinearNDInterpolator(tri, dummy_func2(tri.points[:,0],tri.points[:,1],Df, ),)
    plt.figure()
    plt.plot(xs,li(xs,0,))
    plt.plot(xs, dummy_func2(xs, 0 ,Df))
