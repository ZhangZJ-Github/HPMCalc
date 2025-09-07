# -*- coding: utf-8 -*-
# @Time    : 2025/9/4 20:28
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : beam_dynamics.py
# @Software: PyCharm
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
plt.figure()
plt.plot(Bz_data[:,0],Bz_data[:,1],)
plt.plot(Bz_data[:,0],Bz_interp(Bz_data[:,0]*1e-3),':')
plt.plot(Bz_data[:,0],Br_data[:,1],)

def _dr_dphi_dz_ddr_ddphi_ddz(t, r,theta,z,dr,dtheta,dz,q,m0,gamma,
                              Br_interp, Bz_interp,
                              Esr_interp, Bsphi_interp):
    # Each interp is called as interp(t, r,theta,z)
    # phi = r * theta

    Bz = Bz_interp(t,r,theta,z)
    Br = Br_interp(t,r,theta,z)
    Esr =Esr_interp(t,r,theta,z)
    Bsphi =Bsphi_interp(t,r,theta,z)
    ddr = q/(gamma*m0) * (Esr - dz * Bsphi + (#dr * theta +
                                              dtheta * r ) * Bz)
    ddphi =         q/(gamma*m0) * (dz*Br - dr * Bz)

    return  numpy.array([
        dr,
        dtheta,
        dz,
        ddr,
        1/r * (    q/(gamma*m0) * (dz*Br - dr * Bz)-  2 *dr * dtheta ),
        q/(gamma*m0) * (-(#dr * theta +
                          dtheta * r ) * Br),
    ])
def _dr_dphi_dz_ddr_ddphi_ddz_wrap(t, arr_ ,q,m0,gamma,
                              Br_interp, Bz_interp,
                              Esr_interp, Bsphi_interp ):
    return _dr_dphi_dz_ddr_ddphi_ddz(t,*arr_,q,m0,gamma,
                              Br_interp, Bz_interp,
                              Esr_interp, Bsphi_interp )
Ek = 50e3
r_beam_center = 40e-3

ts= numpy.linspace(0,0.5e-9)

def dummy_interp(t,r,phi,z):
    return numpy.zeros(numpy.broadcast(t,r,phi,z).shape)

Br_interp_t_r_phi_z = lambda t,r,phi,z: Br_interp(z)
Bz_interp_t_r_phi_z = lambda t,r,phi,z: Bz_interp(z)
mm = 1e-3
dr_channel = 6e-3



plt.figure()

r_phi_z_dr_dphi_dz_initial = [r_beam_center,0, 0,0,0,common.Ek_to_beta(Ek)*C.c]

sol = solve_ivp(_dr_dphi_dz_ddr_ddphi_ddz_wrap,y0 = r_phi_z_dr_dphi_dz_initial,
          t_eval= ts,t_span= [ts[0],ts[-1]],args=( - C.e,C.m_e,common.Ek_to_gamma(Ek),
Br_interp_t_r_phi_z,Bz_interp_t_r_phi_z,dummy_interp,dummy_interp
),)




plt.plot(sol.y[2,:] / mm, sol.y[0,:] / mm,)
plt.axhspan(*numpy.array((r_beam_center - dr_channel /2  ,r_beam_center + dr_channel /2 )) / mm,alpha = 0.1)
plt.ylim(0,None)

plt.gca().set_aspect('equal')



