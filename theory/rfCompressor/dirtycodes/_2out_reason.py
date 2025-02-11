# -*- coding: utf-8 -*-
# @Time    : 2024/12/14 15:20
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : _2out_reason.py
# @Software: PyCharm

import pandas

import enum
import typing

import cst.results
import matplotlib
import numpy
import pandas
import scipy
import skrf
from scipy.fft import ifft
from skrf.network import Network

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RegularGridInterpolator
from _logging import logger
import scipy.constants as C
plt.ion()
color_table = list(matplotlib.colors.TABLEAU_COLORS.keys())

def get_interpolator (E_data):
    return interp1d(E_data[:,0],E_data[:,1])
cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\why_2hole\2hole.002.cst"
# cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\SES_switch.TapperedSwitchCav.2.paramsweep.cst"
# cst_proj_path =         r"E:\CSTprojects\rfCompressor\cascadedHT\SES_allCST.cst"
proj: cst.results.ProjectFile = cst.results.ProjectFile(cst_proj_path,
                                                                    allow_interactive=True)

Hz_data = numpy.array(proj.get_3d().get_result_item('Tables\\1D Results\\Hz(Y=-15)').get_data())
S11_data = numpy.array(proj.get_3d().get_result_item( '1D Results\\S-Parameters\\S1(1),1(1)').get_data())
Hz_interpolater = get_interpolator(Hz_data)
S11_interpolater = get_interpolator(S11_data)
phi_0 = numpy.angle(1+S11_interpolater(9.3))
phi_H_reaches_max = phi_0 + numpy.pi / 2
logger.info(numpy.rad2deg(phi_0))
def get_Hdata_at_pahse(Hz_data,phi):
    return    numpy.array([ Hz_data[:, 0],
                            (Hz_data[:, 1] * numpy.exp(1j * phi))]).T


phis = numpy.linspace(0, 2* numpy.pi,200)
zs = Hz_data[:,0].real
dz_out = 18

core_of_convolve = numpy.piecewise(zs, [numpy.abs(zs  -(zs.max()+zs.min())/2 )<dz_out/2 , ], [1, 0])
Hz_integral = scipy.signal.convolve(core_of_convolve, Hz_data[:,1])[core_of_convolve.shape[0]//2:
              core_of_convolve.shape[0]//2 + zs.shape[0]]

Hz_integral_interpolator = get_interpolator(numpy.array([zs, Hz_integral]).T)


plt.figure()
plt.plot(*get_Hdata_at_pahse(Hz_data, phi_0).T.real, )




from shapely.geometry import LineString
l = LineString([[1,0],[295,0]])
line_hdata = LineString(numpy.array([zs, Hz_integral * numpy.exp(1j*phi_0)]).T.real)
# line_hdata = LineString(Hz_data.real)
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


fig,axs = plt.subplots(2,1,sharex=True,constrained_layout= True)
axs[0] .plot(Hz_data[:, 0], Hz_data[:, 1] * numpy.exp(1j*phi_0), )
# axs[0].scatter(intersection_pts_[:, 0], Hz_interpolater(intersection_pts_[:, 0]) * numpy.exp(1j*phi_0))
axs[0].set_ylabel (r"$H_z (z)$ (A/m)")
max_Hz = numpy.abs(Hz_data[:,1]).max()
axs[0].vlines(intersection_pts_[:,0],max_Hz, -max_Hz,linestyles=':',alpha = 0.5)

# right_ax= plt.twinx(axs[0]                    )
#
#
# right_ax.plot(line_hdata.xy[0],numpy.array( line_hdata.xy[1]) *1e-3   , c= color_table[1])
# right_ax.scatter(*intersection_pts_.T,  c= color_table[1])
# right_ax.set_ylabel (r"$\int H_z (z)\rm{d}z$ (arb. unit)")
# right_ax= plt.twinx(axs[0]                    )
eta_0 = 377
f0 = 9.3e9
Z_out = eta_0 / (1-( C.c / f0 / (2* 18e-3))**2)**0.5
Hz_int_max = (Hz_integral_interpolator(intersection_pts_[:, 0]) .__abs__()).max()
axs[1].bar(intersection_pts_[:, 0], ((Hz_integral_interpolator(intersection_pts_[:, 0]) ).__abs__()  / Hz_int_max )**2   , width = 5)
axs[1].set_xlabel ("z (mm)")
axs[1].set_ylabel (r"normalized $\left[\int H_z (z)\rm{d}z\right]_{max}^2$ (arb. unit)")
# plt.figure()
# plt.plot(intersection_pts_[1:,0], numpy.diff(intersection_pts_[:,0]))




Z,Phi = numpy.meshgrid(zs, phis)


# right_ax= plt.twinx(axs[0]                    )
# right_ax.plot(zs, Hz_integral * numpy.exp(1j* phi_0),ls = '--')
# plt.figure()
dH_cf_level = 0.8
fig,axs = plt.subplots(2,1,sharex=True,constrained_layout= True)

# cf = axs[1].contourf(Z,Phi,numpy.abs(( Hz_interpolater(Z) * numpy.exp(1j*Phi)).real),cmap = plt.get_cmap('jet') ,levels = numpy.arange(-max_Hz*0, max_Hz+dH_cf_level,dH_cf_level))
cf = axs[1].contourf(Z,Phi,numpy.abs(( Hz_integral_interpolator(Z) * numpy.exp(1j*Phi)).real),cmap = plt.get_cmap('jet') ,
                     # levels = numpy.arange(-max_Hz*0, max_Hz+dH_cf_level,dH_cf_level)
                     # levels = 20
                     # levels = numpy.linspace(0, 40, 10)
                     )
axs:typing.Tuple[plt.Axes]
axs[1].vlines(intersection_pts_[:,0],phis[0], phis[-1],linestyles='dashed')
axs[1].axhline((phi_0+2*numpy.pi)%(2*numpy.pi),ls=':')

# plt.figure()
l = 0.92
b = 0.12
w = 0.015
h = 0.5 - 1*b
rect = [l,b,w,h]
cbar_ax = fig.add_axes(rect)
cb = plt.colorbar(cf, cax=cbar_ax)
# plt.colorbar(cf)


plt.figure()
# fig,axs = plt.subplots(len(intersection_pts_), 1,sharex = True, sharey   = True)
for i,pt in enumerate(intersection_pts_):
    plt.plot(phis , numpy.abs((Hz_integral_interpolator(pt[0]) * numpy.exp(1j *phis)).real),label = "%d"%i)
# plt.legend()
