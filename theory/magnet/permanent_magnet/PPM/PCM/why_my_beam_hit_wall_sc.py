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
import cst.results
import numpy
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import scipy.constants as C

from theory.magnet.expand_near_a_line_for_axisymmetric_B_field_without_source import \
    NoDivNoCurlNoAngularComponentAxisSymmetricFieldExtrapolator

# 将一条线上的B field map外推，用于粒子追踪，并考虑空间电荷效应

proj = cst.results.ProjectFile(r"E:\SharingDirOnIntranet\TTO_01\CST\Eguns\EGunForCoaxialSource\GyroLike\Magnet_PCM.cst",
                               allow_interactive=True)

if 0:
    B_data = pandas.read_csv(r"E:\SharingDirOnIntranet\TTO_01\CST\Eguns\EGunForCoaxialSource\GyroLike\Bdata.txt",
                             header=None, skiprows=2, sep=r'\s+')
    # # len_y_data , len_z_data = len(B_data[1].unique()),len(B_data[2].unique())
    #
    B_data_interp = LinearNDInterpolator(B_data[[1, 2]].values, B_data[[4, 5]].values, fill_value=0., )

    plt.figure()
    cf = plt.tricontourf(
        B_data_interp.points[:, 1],
        B_data_interp.points[:, 0], B_data_interp(B_data_interp.points)[:, 1], cmap=plt.get_cmap('jet'), levels=20)
    plt.colorbar(cf)
    plt.xlabel("z (mm)")
    plt.ylabel("r (mm)")
    plt.gca().set_aspect('equal')

Ek = 50e3
r_beam_center = 40e-3
dr_channel = 6e-3
dr_beam = 1.8e-3
Ibeam = -300
ve_ref = common.Ek_to_beta(Ek) * C.c
mm = 1e-3

ts = numpy.linspace(0, 1.5e-9, 500)
# ts= numpy.linspace(0,1.5e-9)


Bz_data = numpy.array(proj.get_3d().get_result_item('Tables\\1D Results\\B-Field (Ms)_Z (Z)').get_data())
Br_data = numpy.array(proj.get_3d().get_result_item('Tables\\1D Results\\B-Field (Ms)_Y (Z)').get_data())
if 0:
    # Bz_data[:, 1] = 0.
    Br_data[:, 1] = 0.

plt.figure()
plt.plot(Br_data[:, 0], Br_data[:, 1], label="$B_r$")
plt.plot(Bz_data[:, 0], Bz_data[:, 1], label="$B_z$")
plt.legend()
plt.xlabel("z (mm)")
plt.ylabel("$B$ (T)")
plt.figure()
plt.plot(Br_data[:, 0], numpy.cumsum(Br_data[:, 1] * numpy.diff(Br_data[:, 0], append=numpy.nan)))
plt.xlabel("z (mm)")
plt.ylabel("$\int B_r dz$ (T mm)")

B_extrapolator = NoDivNoCurlNoAngularComponentAxisSymmetricFieldExtrapolator(Bz_data, Br_data, r_beam_center, mm)

from theory.ebeam_dynamic_in_HPM_device.ebeam_transverse_.ebeam_transverse import \
    AnnularBeamInsideCoaxialDriftWeiYuanZhang


def build_interpolator(Bz_data):
    return interp1d(Bz_data[:, 0].real * 1e-3, Bz_data[:, 1], fill_value=0.0, bounds_error=False)


# Bz_interp = build_interpolator(Bz_data)
# Br_interp = build_interpolator(Br_data)


# plt.figure()
# plt.plot(Bz_data[:,0],Bz_data[:,1],)
# plt.plot(Bz_data[:,0],Bz_interp(Bz_data[:,0]*1e-3),':')
# plt.plot(Bz_data[:,0],Br_data[:,1],)

def _dr_dphi_dz_ddr_ddphi_ddz(t, r, theta, z, dr, dtheta, dz, q, m0, gamma,
                              Br_interp, Bz_interp,
                              Esr_interp, Bsphi_interp):
    # Each interp is called as interp(t, r,theta,z)
    # phi = r * theta

    Bz = Bz_interp(t, r, theta, z)
    Br = Br_interp(t, r, theta, z)
    Esr = Esr_interp(t, r, theta, z)
    Bsphi = Bsphi_interp(t, r, theta, z)
    ddr = q / (gamma * m0) * (Esr - dz * Bsphi + (  # dr * theta +
            dtheta * r) * Bz)
    # ddphi =         q/(gamma*m0) * (dz*Br - dr * Bz)

    return numpy.array([
        dr,
        dtheta,
        dz,
        ddr,
        1 / r * (q / (gamma * m0) * (dz * Br - dr * Bz) - 2 * dr * dtheta),
        q / (gamma * m0) * (-(  # dr * theta +
                dtheta * r) * Br),
    ])


def _dr_dphi_dz_ddr_ddphi_ddz_wrap(t, arr_, q, m0, gamma,
                                   Br_interp, Bz_interp,
                                   Esr_interp, Bsphi_interp, N_state):
    return _dr_dphi_dz_ddr_ddphi_ddz(
        t,
        *[arr_[i::N_state] for i in range(N_state)],
        q, m0, gamma,
        Br_interp, Bz_interp,
        Esr_interp, Bsphi_interp).T


def dummy_interp(t, r, phi, z):
    return numpy.zeros(numpy.broadcast(t, r, phi, z).shape)


Br_interp_t_r_phi_z = lambda t, r, phi, z: B_extrapolator.Br_expand(r, z)
Bz_interp_t_r_phi_z = lambda t, r, phi, z: B_extrapolator.Bz_expand(r, z)

abwei = AnnularBeamInsideCoaxialDriftWeiYuanZhang(
    r_beam_center + dr_channel / 2,
    r_beam_center + dr_beam / 2,
    r_beam_center - dr_beam / 2,
    r_beam_center - dr_channel / 2, )

if 1:
    plt.figure(  # layout = "constrained"
    )
    __rs_to_check_Esc = numpy.linspace(r_beam_center - dr_channel / 2, r_beam_center + dr_channel / 2, 100, )
    plt.plot(__rs_to_check_Esc / mm, abwei.E_sc_r(__rs_to_check_Esc, Ibeam, ve_ref) / 1e6)
    plt.xlabel("r (mm)")
    plt.ylabel("$E_{sc,r}$ (MV/m)")

Esc_r_interp_t_r_phi_z = lambda t, r, phi, z: abwei.E_sc_r(r, Ibeam, ve_ref)

N_macropar = 20
r_phi_z_dr_dphi_dz_initial = [r_beam_center,
                              0, 0, 0, 0, common.Ek_to_beta(Ek) * C.c] * numpy.ones((N_macropar, 1))
dr_beam = 1.8e-3
r_phi_z_dr_dphi_dz_initial[:, 0] += numpy.linspace(-dr_beam / 2, dr_beam / 2, N_macropar)
r_phi_z_dr_dphi_dz_initial = r_phi_z_dr_dphi_dz_initial.reshape((-1,))
N_state = int(r_phi_z_dr_dphi_dz_initial.size / N_macropar)

sol = solve_ivp(_dr_dphi_dz_ddr_ddphi_ddz_wrap, y0=r_phi_z_dr_dphi_dz_initial,
                t_eval=ts,
                t_span=[ts[0], ts[-1]], vectorized=True,
                args=(- C.e, C.m_e, common.Ek_to_gamma(Ek),
                      Br_interp_t_r_phi_z, Bz_interp_t_r_phi_z, Esc_r_interp_t_r_phi_z, dummy_interp, N_state
                      ), )

fig, axs = plt.subplots(3, 1, sharex=True, sharey=True,
                        figsize=(6, 4), layout='constrained')
plt.sca(axs[0,])

# plt.figure()
# for i_par in range(N_macropar):
plt.plot(sol.y[2:(N_macropar // 2) * N_state:N_state, :].T / mm, sol.y[0:(N_macropar // 2) * N_state:N_state, :].T / mm,
         c='r')
plt.plot(sol.y[2 + (N_macropar // 2) * N_state::N_state, :].T / mm,
         sol.y[0 + (N_macropar // 2) * N_state::N_state, :].T / mm, c='b')

plt.axhspan(*numpy.array((r_beam_center - dr_channel / 2, r_beam_center + dr_channel / 2)) / mm, alpha=0.1)
plt.ylim(*(numpy.array([r_beam_center - 2 * dr_channel, r_beam_center + 2 * dr_channel, ]) / mm))
plt.xlabel("z (mm)")
plt.ylabel("r (mm)")
plt.gca().set_aspect('equal')

Z, R = numpy.meshgrid(numpy.linspace(-50e-3, 200e-3, 100), numpy.linspace(35e-3, 45e-3, 101))

plt.sca(axs[1,])
cf = plt.contourf(Z / mm, R / mm, B_extrapolator.Bz_expand(R, Z), cmap=plt.get_cmap('jet'), levels=20)
# plt.colorbar(cf, label="$B_z$ (T)",cax = axs[1,1])
plt.colorbar(cf, label="$B_z$ (T)", ax=axs[1]
             , shrink=0.6, ticks=numpy.linspace(cf.zmin, cf.zmax, 3)
             # cax = axs[2,1]
             )
plt.xlabel("z (mm)")
plt.ylabel("r (mm)")
plt.gca().set_aspect('equal')

plt.sca(axs[2,])
cf = plt.contourf(Z / mm, R / mm, B_extrapolator.Br_expand(R, Z), cmap=plt.get_cmap('jet'), levels=20)
plt.colorbar(cf, label="$B_r$ (T)", ax=axs[2]
             , shrink=0.6, ticks=numpy.linspace(cf.zmin, cf.zmax, 3)
             # cax = axs[2,1]
             )
plt.xlabel("z (mm)")
plt.ylabel("r (mm)")
plt.gca().set_aspect('equal')
plt.xlim(0, 120)
