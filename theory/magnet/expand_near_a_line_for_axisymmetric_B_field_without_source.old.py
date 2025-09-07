# -*- coding: utf-8 -*-
# @Time    : 2025/9/5 16:27
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : expand_near_a_line_for_axisymmetric_B_field_without_source.py
# @Software: PyCharm
# 轴对称（且不含Bphi分量)的磁场在无散无旋区的展开
import matplotlib

import common

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import scipy.integrate
import cst.results
import numpy
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import scipy.constants as C


proj = cst.results.ProjectFile(r"E:\SharingDirOnIntranet\TTO_01\CST\Eguns\EGunForCoaxialSource\GyroLike\Magnet_PCM.cst",allow_interactive=True)
Bz_data = numpy.array(proj.get_3d().get_result_item('Tables\\1D Results\\B-Field (Ms)_Z (Z)').get_data())
Br_data = numpy.array(proj.get_3d().get_result_item('Tables\\1D Results\\B-Field (Ms)_Y (Z)').get_data())

def build_interpolator(Bz_data):
    return interp1d(Bz_data[:,0].real * 1e-3,Bz_data[:,1],fill_value=0.0,bounds_error=False)
Bz_interp = build_interpolator(Bz_data)
Br_interp = build_interpolator(Br_data)
mm =1e-3
pBr_pz_data = numpy.array(
    [
(Bz_data[1:, 0] + Bz_data[:-1,0])/ 2 ,
numpy.diff(Br_data[:, 1]) / (numpy.diff(Bz_data[:,0]) * mm)
    ]
).T
pBz_pz_data =numpy.array(
    [
(Bz_data[1:, 0] + Bz_data[:-1,0]) / 2,
numpy.diff(Bz_data[:, 1]) / (numpy.diff(Bz_data[:,0]) * mm)
    ]
).T

p2Bz_pz2_data =numpy.array(
    [
(pBz_pz_data[1:, 0] + pBz_pz_data[:-1,0]) / 2,
numpy.diff(pBz_pz_data[:, 1]) / (numpy.diff(pBz_pz_data[:,0]) * mm)
    ]
).T
p2Br_pz2_data =numpy.array(
    [
(pBr_pz_data[1:, 0] + pBr_pz_data[:-1,0]) / 2,
numpy.diff(pBr_pz_data[:, 1]) / (numpy.diff(pBr_pz_data[:,0]) * mm)
    ]
).T

pBr_pz_interpolator =  build_interpolator(pBr_pz_data                                          )
pBz_pz_interpolator =  build_interpolator(pBz_pz_data                                          )
p2Bz_pz2_interpolator =  build_interpolator(p2Bz_pz2_data    )
p2Br_pz2_interpolator =  build_interpolator(p2Br_pz2_data    )

# Bz_expand = lambda r,z :Bz_interp +
r_center = 40e-3
dr_interested_region = 20e-3

def Bz_expand(r, z):
    Dr = r - r_center
    return Bz_interp(z) + pBr_pz_interpolator(z) * Dr + 1/2. * ( - p2Bz_pz2_interpolator(z) - 2  / r * pBr_pz_interpolator(z)) * Dr **2

def Br_expand_old(r, z):
    Dr = r - r_center
    return (
            Br_interp(z)
            +( - pBz_pz_interpolator(z) - Br_interp(z)/r ) * Dr
            + 1/2. *( - p2Bz_pz2_interpolator(z) + 2/ r **2 * Br_interp(z) - 1/r * pBz_pz_interpolator(z)) * Dr **2
            )
def Br_expand(r, z):
    Dr = r - r_center
    return (
            Br_interp(z)
            +( - pBz_pz_interpolator(z) ) * Dr
            + 1/2. *( - p2Br_pz2_interpolator(z) - 1/r * pBz_pz_interpolator(z)) * Dr **2
            ) / (1 + Dr/ r - (Dr/r) **2 )
rs,zs = numpy.linspace(r_center - dr_interested_region/ 2 ,r_center + dr_interested_region/ 2 ,30),Bz_data[:,0] * mm
R,Z = numpy.meshgrid(rs,zs)
fig, axs = plt.subplots(2,1 ,sharex=True,sharey=True)
plt.sca(axs[0])
cf = plt.contourf(Z / mm,R / mm,     Bz_expand(R,Z),cmap = plt.get_cmap('jet'),levels = 20)
plt.colorbar(cf,label = "$B_z$")
plt.xlabel("z (mm)")
plt.ylabel("r (mm)")
plt.gca().set_aspect('equal')


plt.sca(axs[1])
cf = plt.contourf(Z / mm,R / mm,     Br_expand(R,Z),cmap = plt.get_cmap('jet'),levels = 20)
plt.colorbar(cf,label = "$B_r$")
plt.xlabel("z (mm)")
plt.ylabel("r (mm)")
plt.gca().set_aspect('equal')
plt.xlim(0,120)

plt.figure()
plt.plot(Bz_data[:,0],Bz_data[:,1],label = "$B_z$")
plt.plot(Bz_data[:,0],Br_data[:,1],label = "$B_r$")
plt.legend()
plt.xlabel("z (mm)")
plt.ylabel("B-field (T)")