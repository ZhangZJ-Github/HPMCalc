# -*- coding: utf-8 -*-
# @Time    : 2024/5/9 17:42
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : gap_coupling_coefficient_M.py
# 计算谐振腔的间隙耦合系数M（定义见速调管领域教材/AJDISK帮助文档）
# @Software: PyCharm

import cst.results
import matplotlib

import common

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy
import scipy.constants as C
from _logging import logger
from scipy.optimize import curve_fit

plt.ion()

proj: cst.results.ProjectFile = cst.results.ProjectFile(
    r'E:\CSTprojects\GeneratorAccelerator\klystron5045.cst', allow_interactive=True)
res3d: cst.results.ResultModule = proj.get_3d()
runid = 0
Ez: cst.results.ResultItem = res3d.get_result_item(r'Tables\1D Results\e_Abs (Z)', run_id=runid)
Ezdata = numpy.array(Ez.get_data())


def Gauss(x, a, x0, k):
    return a * numpy.exp(-(k * (x - x0)) ** 2)


max_idx = numpy.argmax(Ezdata[:, 1])
_Gauss_known_peak = lambda x,x0, k:Gauss(x,Ezdata[max_idx, 1],x0,k )
popt, pcov = curve_fit(_Gauss_known_peak, Ezdata[:, 0], Ezdata[:, 1],
                       p0=[#Ezdata[max_idx, 1],
                           Ezdata[max_idx, 0], 1 / 20])

logger.info(popt)
k_SI = popt[1] * 1e3
f_reference = 2856e6
M = numpy.exp(- (2 * numpy.pi * f_reference / (common.Ek_to_beta(350e3) * C.c) / (2 * k_SI)) ** 2)
# logger.info('M = %s' % (M))
plt.figure()
plt.plot(*Ezdata.T, label='original data')
plt.plot(Ezdata[:, 0], _Gauss_known_peak(Ezdata[:, 0], *popt), '--', label='Gauss func. (M = %.2f)' % (M))
plt.xlabel('z / mm')
plt.ylabel('Ez / arb. unit')
plt.legend()


logger.info("CST计算结果：\nM = %s\nR/Q: %s\nf_target: %s"%(M,res3d.get_result_item(r'Tables\1D Results\R over Q (Multiple Modes)', run_id=runid).get_data(),res3d.get_result_item(r'1D Results\Mode Frequencies\Mode 1', run_id=runid).get_data()))
# logger.info("f_target: %s"%res3d.get_result_item(r'1D Results\Mode Frequencies\Mode 1', run_id=runid).get_data())
