# -*- coding: utf-8 -*-
# @Time    : 2024/9/27 14:31
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : test_tri.py
# @Software: PyCharm
from  scipy.spatial import Delaunay

import enum

import cst.results
import matplotlib
import numpy
import pandas
import scipy
import scipy.constants as C
from scipy.fft import ifft
from scipy.optimize import curve_fit
from shapely.geometry import LineString

from _logging import logger
matplotlib.use('tkagg')
import matplotlib.pyplot as plt


import numpy as np
from scipy.spatial import Delaunay

points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])
tri = Delaunay(points)

points_3D =np.array([[0, 0,0], [0, 1.1,0], [1, 0,0], [1, 1,0] , [1, 1,1]])
tri_3D = Delaunay(points_3D)



plt.ion()
plt.triplot(points[:,0], points[:,1], tri.simplices)
plt.plot(points[:,0], points[:,1], 'o')




