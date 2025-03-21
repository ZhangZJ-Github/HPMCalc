# -*- coding: utf-8 -*-
# @Time    : 2025/3/18 20:08
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : generate_fake_macroparticles.py
# @Software: PyCharm
import pandas


import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt

plt.ion()
import common
import numpy
import scipy
from scipy.integrate import quad
import scipy.constants as C
from scipy.integrate import solve_ivp

ln = numpy.log

import cst.results
import cst.results
import matplotlib

matplotlib.use('tkagg')
import skrf
from skrf.media.rectangularWaveguide import RectangularWaveguide

from _logging import logger

from scipy.optimize import minimize

import numpy
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

plt.ion()
df1= pandas.read_csv(r"E:\CSTprojects\Genac2\pic 2d monitor 1.pit",sep = '\s+')