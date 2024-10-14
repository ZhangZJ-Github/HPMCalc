# -*- coding: utf-8 -*-
# @Time    : 2024/4/6 22:52
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : phasespace.py
# @Software: PyCharm
import matplotlib
import numpy
import pandas
import shapely.geometry
from _logging import logger
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

df = pandas.read_excel(r"F:\papers\Genac\aps\figs\raw\Bzs=0.35T z_Ek@1.98008ns.xlsx", header=None)
plt.ion()
plt.figure(figsize=(2, 2), constrained_layout=True)
cnts, bins = numpy.histogram(df[df[0] >= 130][0], bins=300, )
cnts = numpy.append(cnts, 0)
cnts = cnts / cnts.max()
plt.plot(bins, cnts, linewidth=1)
# plt.gca().yaxis.set_major_formatter(lambda y,pos:"%.2f"%(y/cnts.max()))
plt.fill_between(bins, 0, cnts, facecolor='blue', alpha=0.3)
plt.grid()
plt.xlabel('zs / mm')
plt.ylabel('count /arb. unit')
plt.savefig('z_hist.png', dpi=400)
hist_line = shapely.geometry.LineString(numpy.array((bins, cnts)).T)
intersection_pts = hist_line.intersection(shapely.geometry.LineString([[bins[0], 0.5], [bins[-1], 0.5]]))
dz_FWHM = intersection_pts[-1].xs - intersection_pts[0].xs
logger.info('脉宽dz_FWHM=%.2e' % (dz_FWHM))

plt.figure(figsize=(2, 2), constrained_layout=True)
plt.scatter(df[0], df[1], s=0.00005)
