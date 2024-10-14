# -*- coding: utf-8 -*-
# @Time    : 2024/10/8 10:15
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : refinement.py
# @Software: PyCharm

from itertools import combinations


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

from _logging import logger

import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
def get_all_edges(tri: Delaunay):
    edges = set()
    combins = [c for c in combinations(range(tri.simplices.shape[-1]), 2)]

    def min_and_max(*a):
        return min(a), max(a)

    for simplex in tri.simplices:
        for combin in combins:
            edges.add(min_and_max(simplex[combin[0]], simplex[combin[1]]))

    return edges


def new_points(tri: Delaunay, edge: numpy.ndarray):
    return (tri.points[edge[0]] + tri.points[edge[1]]) / 2


def refine_mesh(tri: Delaunay,get_S_parameter:typing.Callable[[float,float,float],float], f_target, ):
    """

    :param tri:
    :param get_S_parameter: 调用形式：get_S_parameter(x, y, f), return S21 at frequency f
    :param f_target:
    :return:
    """
    edges = numpy.array(list(get_all_edges(tri)))
    S21_at_f_target = numpy.vectorize((lambda i: numpy.abs(get_S_parameter(*tri.points[int(i)],
                                                                 f_target
                                                                 ))),  # signature= "()->()"
                                      )(edges)
    Delta_S21 = numpy.abs(S21_at_f_target[:, 1] - S21_at_f_target[:, 0])

    def dS_df():

        f = f_target
        df = 0.05 / 9.3 * f
        fs = f + numpy.linspace(-1, 2, 3) * df
        S = numpy.vectorize((lambda i: numpy.abs(get_S_parameter(
            *tri.points[int(i)],
            fs
        ))), signature="()->(n)")(edges)
        _dS_df = (S[..., 2] - S[..., 0]) / (2 * df)
        return _dS_df

    dS_df_ = dS_df()
    dS_df_with_diffrent_signal = dS_df_[..., 0] * dS_df_[..., 1] < -(0.1 / (0.6)) ** 2  # 一条edge的左右dS/df异号，表明此edge中间的一点很可能是最大值
    _filter = (Delta_S21 > 0.35) | (dS_df_with_diffrent_signal)
    edges_to_refine = edges[_filter]
    new_points_ = numpy.array([])
    if len(edges_to_refine):
        new_points_ = numpy.vectorize(new_points, signature="(),(2)->(2)")(tri, edges_to_refine)
        # tri.add_points(new_points, restart=True)
        return Delaunay(numpy.vstack((tri.points, new_points_))),new_points_
    return tri,new_points_