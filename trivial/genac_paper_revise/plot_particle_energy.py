# -*- coding: utf-8 -*-
# @Time    : 2025/4/17 16:09
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : plot_particle_energy.py
# @Software: PyCharm

import typing

import matplotlib.pyplot as plt

import cst.results

import common
from _logging import logger
import numpy
import pandas
import scipy.constants as C
from scipy.interpolate import interp1d

# -*- coding: utf-8 -*-
# @Time    : 2024/2/27 11:21
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : CST_PIC2D_monitor_data.py
# @Software: PyCharm
"""
用于处理CST PIC2D monitor的数据
"""
import os
import typing

import matplotlib
import numpy
import pandas
import scipy.constants as C

import common
import matplotlib.colors as  mcolors
from shapely.geometry import Point
from  _logging import logger

matplotlib.use('tkagg')
import matplotlib.pyplot as plt


def average(PIC2D_df: pandas.DataFrame, key, get_filter: typing.Callable[[pandas.DataFrame], pandas.Series] = None):
    if get_filter:    PIC2D_df = PIC2D_df[get_filter(PIC2D_df)]
    logger.info(len(PIC2D_df))

    return numpy.average(PIC2D_df[key], weights=PIC2D_df['nmacro'])


def std_and_avg(PIC2D_df: pandas.DataFrame, key,
                get_filter: typing.Callable[[pandas.DataFrame], pandas.Series] = None):
    if get_filter:    PIC2D_df = PIC2D_df[get_filter(PIC2D_df)]

    avg = average(PIC2D_df, key)
    return numpy.average((PIC2D_df[key] - avg) ** 2, weights=PIC2D_df['nmacro']) ** 0.5, avg


def std(PIC2D_df: pandas.DataFrame, key, get_filter: typing.Callable[[pandas.DataFrame], pandas.Series] = None):
    return std_and_avg(PIC2D_df, key, get_filter)[0]


# cache_dir = r'E:\CSTprojects\GeneratorAccelerator\StandingWaveAccelerator_PICSolver\Result\Cache'
# _pitfilename = 'pic 2d monitor 2.pit'
# for _dir in os.listdir(cache_dir):
#     _dir =os.path.join( cache_dir,_dir)
#     if _pitfilename in  os.listdir(_dir):
#         proj_discharging: cst.results.ProjectFile = cst.results.ProjectFile(os.path.join(_dir,'StandingWaveAccelerator_PICSolver.cst'), allow_interactive=True)
#         res3d: cst.results.ResultModule = proj_discharging.get_3d()
#         aaaa

res_index = []
res_dir = r'E:\CSTprojects\GeneratorAccelerator\StandingWaveAccelerator_PICSolver\zzj\choosed_for_plot'
for _filename in os.listdir(res_dir):
    if _filename .endswith('.pit'):
        res_index.append([float(os.path.splitext(_filename)[0][2:]), os.path.join(res_dir, _filename)])
res_index.sort(key=lambda x: x[0])
plt.ion()
fig1 = plt.figure(figsize=(4, 3), constrained_layout=True)
ax1 = plt.gca()
# fig2 = plt.figure(figsize=(2.5, 2.2), constrained_layout=True)
# fig2,ax2s = plt.subplots(len(res_index),1,figsize=(4, 2),
#                                     constrained_layout=True)
# ax2: plt.Axes = plt.gca()
blues = plt.get_cmap('Blues')
Ek_spectra = pandas.DataFrame()
PIC2D_dfs = {}
sigmaxy = {}

Ek_bins = None
Ek_counts = None
Ek_counts_on_mesh =[]

def build_PIC2D_df(filename):
    PIC2D_df = pandas.read_csv(filename, header=None, sep=r'\s+')

    PIC2D_df.columns = 'x y zs GBx GBy GBz m q qmacro time'.split(' ')

    PIC2D_df['Ek_MeV'] = (common.gammabeta_to_gamma(
        (PIC2D_df['GBx'] ** 2 + PIC2D_df['GBy'] ** 2 + PIC2D_df['GBz'] ** 2) ** 0.5) - 1) * PIC2D_df['m'] * C.c ** 2 / (
                                 1e6 * C.eV)
    PIC2D_df['nmacro'] = PIC2D_df['qmacro'] / PIC2D_df['q']
    return PIC2D_df

fig_to_plot_spectra,axs_to_plot_spectra = plt.subplots(len(res_index),1,figsize = (4,1.*len(res_index),),
                                  constrained_layout=True,sharex =True,sharey = True)

for i in range(len(res_index)):
    B = res_index[i][0]
    # PIC2D_df = pandas.read_csv(r"E:\CSTprojects\GeneratorAccelerator\pic 2d monitor 2.pit", header=None, sep=r'\s+')
    PIC2D_df = PIC2D_dfs.get(B,build_PIC2D_df(res_index[i][1] ))
    # PIC2D_df = PIC2D_dfs.get(B,pandas.read_csv(res_index[i][1], header=None, sep=r'\s+'))
    #
    # PIC2D_df.columns = 'x y zs GBx GBy GBz m q qmacro time'.split(' ')
    #
    # PIC2D_df['Ek_MeV'] = (common.gammabeta_to_gamma(
    #     (PIC2D_df['GBx'] ** 2 + PIC2D_df['GBy'] ** 2 + PIC2D_df['GBz'] ** 2) ** 0.5) - 1) * PIC2D_df['m'] * C.c ** 2 / (
    #                              1e6 * C.eV)
    # PIC2D_df['nmacro'] = PIC2D_df['qmacro'] / PIC2D_df['q']


    PIC2D_dfs[B] =            PIC2D_df

    sigmaxy [B] =[std(PIC2D_df, 'x', ),std(PIC2D_df, 'y', )]
    get_filter = lambda df: df['Ek_MeV'] > 4
    Ek_avg = average(PIC2D_df, 'Ek_MeV', get_filter)
    Ek_std = std(PIC2D_df, 'Ek_MeV', get_filter)

    # 出口能谱
    # plt.figure(figsize=(2, 2), constrained_layout=True)
    # plt.hist(PIC2D_df['Ek_MeV'], weights=PIC2D_df['nmacro'], bins=100,label='B=%.2f'%B,#color=blues(i)
    #          )
    Ek_counts, Ek_bins = numpy.histogram(PIC2D_df['Ek_MeV'], bins=1000, weights=PIC2D_df['nmacro'])

    Ek_counts = numpy.append(Ek_counts, 0)

    Ek_counts_on_mesh.append(Ek_counts)

    # Ek_spectra['Ek_bins'] = Ek_bins
    # Ek_spectra['B=%.2fT'%(B)] = Ek_counts
    # plt.sca(ax1)
    plt.sca(axs_to_plot_spectra[i])

    plt.plot(Ek_bins, Ek_counts  # /histdata[0].max()
             , label='B=%.2f T' % B,
             # color=blues(((B+0.1)/(0.9+0.1))                         )
             )


    plt.xlabel("particle energy (MeV)")
    # plt.ylabel('count (arb. unit)')
    # plt.xlim(4, Ek_bins.max()*1.1)
    plt.xlim(4.1, 4.7)
    plt.ylim(0, Ek_counts.max()* 1.05)
    plt.legend()
    # plt.grid()
    # plt.gca().xaxis.set_major_locator(matplotlib. ticker.MultipleLocator(1))
    # plt.gca().yaxis.set_major_locator(matplotlib. ticker.MultipleLocator(0.2e8))
    # plt.savefig("electron_energy_spectrum.png", dpi=400)

    # plt.sca(axs_to_plot_spectra)


    # 出口束斑
    # plt.scatter(PIC2D_df['x'], PIC2D_df['y'], s=0.001, c='k')
    # # plt.gca().set_aspect('equal')
    # plt.gca().xaxis.set_major_formatter(lambda x, pos: "%.1f" % (x / 1e-3))
    # plt.gca().yaxis.set_major_formatter(lambda x, pos: "%.1f" % (x / 1e-3))
    # plt.xlabel("x / mm")
    # plt.ylabel("y / mm")
    # colors = list(mcolors.TABLEAU_COLORS.keys())
    # alpha = 1
    #
    # plt.plot(*Point(0, 0).buffer(0.7e-3).exterior.xy, '--', c=colors[1], alpha=alpha)
    # # plt.axvline(-0.7e-3,c = c[1],alpha = alpha)
    # # plt.axvline(0.7e-3,c=c[1],alpha = alpha)
    # # plt.axhline(-0.7e-3,c = c[1],alpha = alpha)
    # # plt.axhline(0.7e-3,c=c[1],alpha = alpha)
    #
    # rmax = PIC2D_df['x'].max() * 1.2
    # plt.xlim(-rmax, rmax)
    # plt.ylim(-rmax, rmax)
    # ax: plt.Axes = plt.gca()
    #
    # twinx: plt.Axes = ax.twinx()
    # twiny: plt.Axes = ax.twiny()
    #
    # PIC2D_df_x_histdata = numpy.histogram(PIC2D_df['x'], bins=100, weights=PIC2D_df['nmacro'])
    # PIC2D_df_y_histdata = numpy.histogram(PIC2D_df['y'], bins=100, weights=PIC2D_df['nmacro'])
    # _ratio_hist_max = 1 / 5
    # twinx.bar(PIC2D_df_x_histdata[1][:-1], (PIC2D_df_x_histdata[0] / (PIC2D_df_x_histdata[0].max())),  # align = 'edge'
    #           alpha=0.5,
    #           # bottom = 0  ,
    #           width=numpy.diff(PIC2D_df_x_histdata[1])[0]
    #           )
    # twinx.set_ylim(0, 1 / _ratio_hist_max)
    # twinx.set_yticks([])
    # twiny.barh(PIC2D_df_y_histdata[1][:-1], (PIC2D_df_y_histdata[0] / (PIC2D_df_y_histdata[0].max())),  # align = 'edge'
    #            alpha=0.5,
    #            # left=0.001,
    #            height=numpy.diff(PIC2D_df_y_histdata[1])[0])
    # twiny.set_xlim(0, 1 / _ratio_hist_max)
    # twiny.set_xticks([])
    # # ax.set_aspect('equal', 'datalim')
    # # ax.set_aspect('auto')
    # z_screen = PIC2D_df['zs'][0]
    # # plt.title("screen at zs = %.0f mm"%(z_screen*1e3))
    # plt.savefig("Beam_spot_out.png", dpi=400)
    # logger.info('B = %s' % B)

    #     出口束斑2
    if False:
        fig2 = plt.figure(figsize=(1,1), constrained_layout=True)
        ax2: plt.Axes = plt.gca()
        ax2.scatter(PIC2D_df['x'], PIC2D_df['y'], s=0.0001,# c='k'
                    )

        ax2.set_aspect('equal')
        ax2.xaxis.set_major_formatter(lambda x, pos: "%.0f" % (x / 1e-3))
        ax2.yaxis.set_major_formatter(lambda x, pos: "%.0f" % (x / 1e-3))
        plt.xlim(-2e-3,2e-3)
        plt.ylim(-2e-3,2e-3)
        plt.savefig('B=%.2fT.png'%(B),dpi = 400)
fig_to_plot_spectra:plt.Figure
fig_to_plot_spectra.supylabel('count (arb. unit)')
plt.savefig("electron_energy_spectrum.png", dpi=400)

# Ek_counts_on_mesh = numpy.array(Ek_counts_on_mesh)
# B_mesh, Ek_bin_mesh = numpy.meshgrid(list(PIC2D_dfs.keys()), Ek_bins,indexing= 'ij')
# plt.figure()
# plt.contourf(Ek_bin_mesh,B_mesh,Ek_counts_on_mesh)
# for B in PIC2D_dfs: