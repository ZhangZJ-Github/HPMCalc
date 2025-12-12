# -*- coding: utf-8 -*-
# @Time    : 2025/9/22 15:46
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : my_comsol_B_field.py
# @Software: PyCharm
import re

import matplotlib
import numpy
import pandas
import scipy.constants as C
from scipy.integrate import solve_ivp

import common
from simulation.task_manager.simulator import df_to_gdf
from theory.ebeam_dynamic_in_HPM_device.ebeam_transverse_.ebeam_transverse import \
    AnnularBeamInsideCoaxialDriftWeiYuanZhang
from theory.magnet.expand_near_a_line_for_axisymmetric_B_field_without_source import \
    NoDivNoCurlNoAngularComponentAxisSymmetricFieldExtrapolator

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

r_beam_center = 65e-3  # 55e-3
dr_channel = 6e-3
dr_beam = 0.5e-3 * 1 + 1.8e-3 * 0 + 6e-3 * 0
Ibeam = -300
Ek = 50e3
ve_ref = common.Ek_to_beta(Ek) * C.c

ts = numpy.linspace(0,
                    280e-3 / ve_ref * 1.5,
                    # 1.5e-9,
                    500)

df = pandas.read_csv(
    # r'E:\SharingDirOnIntranet\TTO_01\COMSOL\B_export.txt'
    r'E:\SharingDirOnIntranet\TTO_01\COMSOL\B_export_all_regular_grid.txt'
    , skiprows=9, sep=r'\s+', header=None)

lenunit_in_COMSOL_exported = mm = 1e-3


def build_B_interp(df: pandas.DataFrame):
    B_interp = LinearNDInterpolator(df[[0, 1]].values, df[[2, 3]].values, fill_value=0.0)
    return B_interp


B_interp = build_B_interp(df)
# 对称性
if 1:
    z_sym_unit_in_m = 88.5e-3  # df[1].max()
    df_generated_by_symmetry = df.copy()
    df_generated_by_symmetry[1] = 2 * z_sym_unit_in_m / lenunit_in_COMSOL_exported - df[1][::-1].values
    B_data_generated_by_symmetry = B_interp(df_generated_by_symmetry[0].values, df[1][::-1].values)
    df_generated_by_symmetry[2] = B_data_generated_by_symmetry[..., 0]
    df_generated_by_symmetry[3] = -B_data_generated_by_symmetry[..., 1]
    df = pandas.concat([df, df_generated_by_symmetry], axis=0)
    B_interp = build_B_interp(df)

rs, zs = numpy.arange(0, 75e-3, 1e-3), numpy.arange(-60e-3, 270e-3, 2e-3)
R, Z = numpy.meshgrid(rs, zs)

# 强行消除束流中心位置上的Br分量
Br_ = B_interp(r_beam_center / mm * numpy.ones(zs.shape), zs / mm, )[:, 0]*0#* 1e-1
# Br_[(zs > 0) #& (zs <182e-3)
# ] *= 0.1
B_extrapolator = NoDivNoCurlNoAngularComponentAxisSymmetricFieldExtrapolator(
    numpy.array([zs / mm, (0.2 / 0.45) ** 0 * B_interp(r_beam_center / mm * numpy.ones(zs.shape), zs / mm, )[:, 1]]).T,
    numpy.array([zs / mm, Br_]
                ).T,
    r_beam_center, mm  # 1.0
)

interpolated_B = B_interp(R / lenunit_in_COMSOL_exported, Z / lenunit_in_COMSOL_exported)
plt.figure()
cf = plt.contourf(Z / mm, R / mm, interpolated_B[:, :, 1], cmap='jet', levels=100)
plt.streamplot(Z.T / mm, R.T / mm, interpolated_B[:, :, 1].T, interpolated_B[:, :, 0].T,
               # cmap = 'jet',levels=  100
               )
plt.colorbar(cf)
plt.gca().set_aspect("equal")

plt.figure()
plt.plot(zs / mm, B_interp(r_beam_center / mm * numpy.ones(zs.shape), zs / mm, )[:, 1],label = "$B_z(r = %.1f~\mathrm{mm}, z)$"%(r_beam_center/mm))
plt.plot(zs / mm, B_interp(r_beam_center / mm * numpy.ones(zs.shape), zs / mm, )[:, 0],label = "$B_r(r = %.1f~\mathrm{mm}, z)$"%(r_beam_center/mm))
plt.xlabel("z (mm)")
plt.ylabel("magnetic induction intensity (T)")
# plt.plot(zs / mm, B_interp(0 * numpy.ones(zs.shape), zs / mm, )[:, 1],label = "$B_r(r = 0, z)$")
plt.legend()
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

N_macropar = 21
r_phi_z_dr_dphi_dz_initial = [r_beam_center + 0.2e-3 * 0,
                              0,
                              # 以下为z初始位置
                              33e-3 * 0 + 20e-3 * 0 + 29e-3 * 0 + 13e-3 * 0 + -13e-3,  # 20e-3,
                              0, 0, common.Ek_to_beta(Ek) * C.c] * numpy.ones((N_macropar, 1))

r_phi_z_dr_dphi_dz_initial[:, 0] += numpy.linspace(-dr_beam / 2, dr_beam / 2, N_macropar)
r_phi_z_dr_dphi_dz_initial = r_phi_z_dr_dphi_dz_initial.reshape((-1,))
N_state = int(r_phi_z_dr_dphi_dz_initial.size / N_macropar)


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


# Br_interp_t_r_phi_z = lambda t, r, phi, z: B_interp(r / mm, z/ mm)[...,0]
# Bz_interp_t_r_phi_z = lambda t, r, phi, z: B_interp(r / mm, z/ mm)[...,1]
Br_interp_t_r_phi_z = lambda t, r, phi, z: B_extrapolator.Br_expand(r, z)
Bz_interp_t_r_phi_z = lambda t, r, phi, z: B_extrapolator.Bz_expand(r, z)


def dummy_interp(t, r, phi, z):
    return numpy.zeros(numpy.broadcast(t, r, phi, z).shape)


sol = solve_ivp(_dr_dphi_dz_ddr_ddphi_ddz_wrap, y0=r_phi_z_dr_dphi_dz_initial,
                t_eval=ts,
                t_span=[ts[0], ts[-1]], vectorized=True,
                args=(- C.e, C.m_e, common.Ek_to_gamma(Ek),
                      Br_interp_t_r_phi_z, Bz_interp_t_r_phi_z,
                      # dummy_interp,
                      Esc_r_interp_t_r_phi_z,
                      dummy_interp, N_state
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
# plt.gca().set_aspect('equal')

Z, R = numpy.meshgrid(numpy.linspace(-100e-3, 200e-3, 100), numpy.linspace(60e-3, 70e-3, 11))

plt.sca(axs[1,])
cf = plt.contourf(Z / mm, R / mm, B_extrapolator.Bz_expand(R, Z), cmap=plt.get_cmap('jet'), levels=20)
# plt.colorbar(cf, label="$B_z$ (T)",cax = axs[1,1])
plt.colorbar(cf, label="$B_z$ (T)", ax=axs[1]
             , shrink=0.6, ticks=numpy.linspace(cf.zmin, cf.zmax, 3)
             # cax = axs[2,1]
             )
plt.streamplot(Z / mm, R / mm, B_extrapolator.Bz_expand(R, Z), B_extrapolator.Br_expand(R, Z), )
plt.xlabel("z (mm)")
plt.ylabel("r (mm)")
# plt.gca().set_aspect('equal')

plt.sca(axs[2,])
cf = plt.contourf(Z / mm, R / mm, B_extrapolator.Br_expand(R, Z), cmap=plt.get_cmap('jet'), levels=20)
plt.colorbar(cf, label="$B_r$ (T)", ax=axs[2]
             , shrink=0.6, ticks=numpy.linspace(cf.zmin, cf.zmax, 3)
             # cax = axs[2,1]
             )
plt.xlabel("z (mm)")
plt.ylabel("r (mm)")
# plt.gca().set_aspect('equal')
# plt.xlim(0, 120)


# plt.sca(axs[3,])
# cf = plt.plot(zs, B_extrapolator.Br_expand(r_beam_center*numpy.ones(zs.shape), zs),# cmap=plt.get_cmap('jet'), levels=20
#               )
# # plt.colorbar(cf, label="$B_r$ (T)", ax=axs[2]
# #              , shrink=0.6, ticks=numpy.linspace(cf.zmin, cf.zmax, 3)
# #              # cax = axs[2,1]
# #              )
# plt.xlabel("z (mm)")
# plt.ylabel("r (mm)")
plt.ylim(60, 70)

if 0:
    for ax in axs:    ax.set_aspect('equal')


# plt.suptitle("$B_p$ = %.2f T, $L$ = %.2f mm"%(Bpeak, lambda_pm / mm ))


# Export to CST readable
def export_to_CST_readable():
    rmin_fine_mesh = 30
    # x=  mm * numpy.array([   1, (rmin_fine_mesh-1) / 2**0.5,*numpy.arange((rmin_fine_mesh) / 2**0.5,46,0.5),    ])
    # x = numpy.hstack([-x[::-1],x])
    x = numpy.arange(-70e-3, 70e-3, 1e-3)
    ZZZ, YYY, XXX = numpy.meshgrid(
        mm * numpy.array([*numpy.arange(-60, 250, 2), ]),
        x,
        x,

        indexing='ij'
    )
    R = (XXX ** 2 + YYY ** 2) ** 0.5
    if 0:  # Use true data
        B_interpolated_res = B_interp(R / mm, ZZZ / mm)
        Br = B_interpolated_res[..., 0]
        Bz = B_interpolated_res[..., 1]
    if 1:  # Use fake data
        Br = B_extrapolator.Br_expand(R, ZZZ)
        Bz = B_extrapolator.Bz_expand(R, ZZZ)
    _filter = (R < 50e-3) | (R > 70e-3)
    Br[_filter] = 0.0
    Bz[_filter] = 0.0

    B_extrapolator2 = NoDivNoCurlNoAngularComponentAxisSymmetricFieldExtrapolator(
        numpy.array(
            [zs / mm,
             B_interp(0 / mm * numpy.ones(zs.shape), zs / mm, )[:, 1]]).T,
        numpy.array([zs / mm, numpy.zeros(zs.shape)]
                    ).T,
        0, mm  # 1.0
    )

    _filter2 = (R < 16e-3)
    if 0:
        _data = B_interp(R / mm, ZZZ / mm)
        Br[_filter2] = _data[..., 0][_filter2]
        Bz[_filter2] = _data[..., 1][_filter2]
    if 0:
        Br[_filter2] = B_extrapolator2.Br_expand(R, ZZZ)[_filter2]
        Bz[_filter2] = B_extrapolator2.Bz_expand(R, ZZZ)[_filter2]

    Bx = Br * XXX / (XXX ** 2 + YYY ** 2) ** 0.5
    By = Br * YYY / (XXX ** 2 + YYY ** 2) ** 0.5

    plt.figure()
    i = 30
    cf = plt.contourf(XXX[i], YYY[i], R[i], cmap='jet', levels=20)
    plt.scatter(XXX[i], YYY[i], s=0.1)
    plt.gca().set_aspect("equal")

    plt.figure()
    i = 30
    cf = plt.contourf(XXX[i], YYY[i], ((Br ** 2 + Bz ** 2) ** .5)[i], cmap='jet', levels=20)
    plt.scatter(XXX[i], YYY[i], s=0.1)
    plt.colorbar(cf)
    plt.gca().set_aspect("equal")

    df = pandas.DataFrame({
        "x": XXX.ravel() / mm,
        "y": YYY.ravel() / mm,
        "z": ZZZ.ravel() / mm,
        "Bx": Bx.ravel(),
        "By": By.ravel(),
        "Bz": Bz.ravel(),
    })
    df_to_gdf(df, "GPT_B_map.gdf", True)
    df.columns = re.split(r'\s{2,}',
                          "x [mm]           y [mm]           z [mm]      x [V.s/m^2]      y [V.s/m^2]      z [V.s/m^2]", )
    csv_path = "B_exported.txt"
    df.to_csv(csv_path, index=False,
              #               header="""           x [mm]           y [mm]           z [mm]      x [V.s/m^2]      y [V.s/m^2]      z [V.s/m^2]
              # ------------------------------------------------------------------------------------------------------""",
              sep='\t',
              # float_format = "%.12e"
              )

    with open(csv_path, 'r') as f:
        s = f.read()
    new_s = s.replace('\t', '     ')

    with open(csv_path, 'w') as f:
        f.write(new_s)


if 0:
    export_to_CST_readable()


if 1:
    from theory.magnet.to_MAGIC.COMSOL_B_map_to_MAGIC_readable import df_B_map_from_COMSOL_to_MAGIC_readable
    df_generated_B_data_in_COMSOL_style = pandas.DataFrame(
        numpy.array((    R.ravel(order = 'F')/  mm, Z.ravel(order = 'F') / mm,
                         B_extrapolator.Br_expand(R.ravel(order = 'F'), Z.ravel(order = 'F')),
                         B_extrapolator.Bz_expand(R.ravel(order = 'F'), Z.ravel(order = 'F')),)).T
    )
    df_generated_B_data_in_COMSOL_style.to_csv("df_generated_B_data_in_COMSOL_style.csv",index=False)
    df_B_map_from_COMSOL_to_MAGIC_readable(df_generated_B_data_in_COMSOL_style)


