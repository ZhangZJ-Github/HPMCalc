# -*- coding: utf-8 -*-
# @Time    : 2024/11/21 20:28
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : get_zeros.py
# @Software: PyCharm
import os.path

import matplotlib.pyplot as plt
import numpy

from  theory.rfCompressor.time_dependent_output import *
import scipy.constants as C

projs_charging: cst.results.ProjectFile = cst.results.ProjectFile(
    r"E:\CSTprojects\rfCompressor\cascadedHT\single_coupling_hole\SES_trivial8.noout.cst",
    allow_interactive=True)
run_id = 0
run_id_charging = 0

hdata  =numpy.array( projs_charging.get_3d().get_result_item('Tables\\1D Results\\h-field (f=9.3) (3)_Z (Z)').get_data())
import shapely
from shapely.geometry import LineString
l = LineString([[1,0],[305,0]])
line_hdata = LineString(hdata)
intersection_pts = l.intersection(line_hdata)
pts_ = []
xs = []
ys = []
for pt in intersection_pts.geoms:
    xs+= list(pt.xy[0])
    ys+=list(pt.xy[1])
    # if isinstance(pt,shapely.geometry.point.Point):    pts_.append((pt.x,pt.y) )
intersection_pts_ = numpy.array([xs,ys]).T
intersection_pts_ = intersection_pts_[numpy.argsort(intersection_pts_[:,0])]
plt.figure()
plt.plot(*line_hdata.xy)
plt.scatter(*intersection_pts_.T)

plt.figure()
plt.plot(intersection_pts_[1:,0], numpy.diff(intersection_pts_[:,0]))



