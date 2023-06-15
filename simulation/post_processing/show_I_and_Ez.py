# -*- coding: utf-8 -*-
# @Time    : 2023/6/7 16:57
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : show_I_and_Ez.py
# @Software: PyCharm
"""
将电流与电场绘制在同一图中
"""
import os.path
import typing

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from _logging import logger

import grd_parser
from total_parser import ExtTool



def plot_range_Ez_JDA(grd: grd_parser.GRD, t, Ez_title, JDA_title, axs: typing.List[plt.Axes]):
    t_actual, Ez_data, _ = grd.get_data_by_time(t, Ez_title)
    logger.info("t_actual = %.2e" % t_actual)
    t_actual, JDA_data, _ = grd.get_data_by_time(t_actual, JDA_title)
    logger.info("t_actual = %.2e" % t_actual)
    axs[0].plot(*Ez_data.values.T, label=Ez_title)
    axs[1].plot(*JDA_data.values.T, label=JDA_title)
    for ax in axs:
        ax.legend()
        ax.grid()

    plt.suptitle("t = %.2e" % t_actual)


def plot_obs_Ez_JDA(grd: grd_parser.GRD, Ez_title, JDA_title, axs: typing.List[plt.Axes]):
    JDA_data = grd.obs[JDA_title]['data']
    Ez_data = grd.obs[Ez_title]['data']
    axs[0].plot(*Ez_data.values.T, label=Ez_title)
    axs[1].plot(*JDA_data.values.T, label=JDA_title)
    for ax in axs:
        ax.legend()
        ax.grid()



if __name__ == '__main__':
    plt.ion()
    et = ExtTool(os.path.splitext(r"D:\MagicFiles\HPM\12.5GHz\自动化\test01\0610\23.m2d")[0])
    grd = grd_parser.GRD(et.get_name_with_ext(ExtTool.FileType.grd))

    range_Ez_title = r' FIELD EZ @LINE_PARTICLE_MOVING$,FFT #4.1'
    range_JDA_title = r' FIELD_INTEGRAL J.DA @OSYS$AREA,FFT #5.1'
    obs_Ez_title = r' FIELD E1 @OBSP.EXTRACTOR,FFT-#17.1'
    obs_JDA_title = r' FIELD_INTEGRAL J.DA @LINE_NEAR_EXTRACTOR,FFT-#18.1'

    t = 20e-9
    plot_range_Ez_JDA(grd, t, range_Ez_title, range_JDA_title,
                      plt.subplots(2, 1, sharex=True, constrained_layout=True)[1])
    plot_obs_Ez_JDA(grd, obs_Ez_title, obs_JDA_title,
                    plt.subplots(2, 1, sharex=True,
                                 constrained_layout=True)[1])
    plt.show()
